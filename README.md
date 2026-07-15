# Pipeline de Avaliação ARES

Sistema modular de mineração de literatura científica focado na conferência **ARES** (Availability, Reliability and Security). O pipeline coleta artigos, avalia-os qualitativamente via ensemble de LLMs e quantitativamente via bibliometria, produzindo um ranking final dos trabalhos mais relevantes para os domínios de pesquisa: **CPS, Blockchain, IoT, Fault Tolerance e Distributed Systems**.

## Uso rápido

```bash
python pipeline.py --stage coleta         # Coleta DBLP + OpenAlex
python pipeline.py --stage preprocess     # Pré-processamento NLP
python pipeline.py --stage embeddings     # Embeddings (SentenceTransformer)
python pipeline.py --stage llm            # Ensemble de LLMs (filtro sim_max >= 60%)
python pipeline.py --stage agreement      # Concordância entre modelos
python pipeline.py --stage bibliometria   # Scores quantitativos
python pipeline.py --stage ranking        # Ranking final + outputs
python pipeline.py                        # Pipeline completo
python pipeline.py --list                 # Listar etapas
```

## Arquitetura

```
Coleta                    Preprocess               Embeddings
dataset_raw.csv      →    dataset_preprocessed.csv  →  dataset_embeddings.pkl
                                                         cache/embeddings.pkl
                             ↓
                        LLM Ensemble ──→ cache/responses/*.json (183 arquivos)
                             ↓                    ↓
                        S_Quali              Agreement
                             ↓                    ↓
                        Bibliometria         agreement.csv
                        (C_Norm, V_Norm,
                         S_Quant, Score_Global)
                             ↓
                        Ranking → outputs/*
```

---

## Etapa 1 — Coleta (DBLP + OpenAlex)

**Arquivo:** `pipeline.py:174-285` | **Comando:** `python pipeline.py --stage coleta`

### Fluxo

```
DBLP API                          OpenAlex API
─────────                         ────────────
Busca venue:ARES: + termo    →    DOI → enriquece
6 termos × 4 anos = 24 buscas    autores, abstract,
com exponential backoff          citações
```

1. **Configura sessões HTTP** com `User-Agent` customizado e `verify=False` para ambientes com certificados desatualizados (ex: macOS Homebrew)
2. **Para cada ano** (2022–2025) e **cada termo** (Blockchain, IoT, Cyber, Fault, Distributed, Resilience):
   - Consulta `https://dblp.org/search/publ/api` com query `venue:ARES: year:{ano} {termo}`
   - Extrai DOIs do campo `ee` (electronic edition) de cada artigo
   - **Resiliência:** até 5 retries com exponential backoff (2s → 4s → 8s → 16s → 32s) para HTTP 429/500+
   - Sleep de 1.5s entre buscas (politeness)
3. **Deduplica DOIs** via `set()` e envia batch para `https://api.openalex.org/works?filter=doi:...`
4. **Reconstrói abstracts** do formato `inverted_index` do OpenAlex (dicionário palavra → [posições]) para texto linear
5. **Estrutura final:** id, title, venue, authors, year, abstract, citations, influential_citations

### Limitação
O DBLP indexa apenas artigos com DOI registrado. A ARES publica ~70–100 artigos/ano. O limite de 500 é um **teto**, não uma garantia.

### Saída
`outputs/dataset_raw.csv`

---

## Etapa 2 — Pré-processamento NLP

**Arquivo:** `preprocessing.py` | **Comando:** `python pipeline.py --stage preprocess`

### Pipeline de limpeza textual (6 passos, aplicados a title + abstract)

| Passo | Operação | Exemplo |
|---|---|---|
| 1 | Normalização Unicode NFKD | `café` → `cafe` (decompõe acentos) |
| 2 | Decodificação HTML | `&amp;` → `&` |
| 3 | Remoção de tags HTML | `<p>texto</p>` → `texto` |
| 4 | Remoção de caracteres de controle | Remove `\x00`–`\x1f` (exceto `\t`, `\n`) |
| 5 | Colapso de espaços múltiplos | `texto    com   espaços` → `texto com espaços` |
| 6 | Strip de bordas | Remove espaços iniciais/finais |

**Importante:** NÃO remove stopwords, NÃO aplica stemming, NÃO aplica lemmatization — o texto mantém estrutura semântica completa para os LLMs.

### Concatenação
`clean_text = normalize(title) + " " + normalize(abstract)`

### Saída
`outputs/dataset_preprocessed.csv` (adiciona coluna `clean_text`)

---

## Etapa 3 — Embeddings (SentenceTransformer)

**Arquivo:** `embeddings.py` | **Comando:** `python pipeline.py --stage embeddings`

### Funcionamento

1. **Carrega modelo:** `BAAI/bge-large-en-v1.5` (768 dimensões) com fallback para `all-mpnet-base-v2`
2. **Cache inteligente:** Verifica `cache/embeddings.pkl` — se existir, apenas artigos novos são processados
3. **Geração em batch:** `model.encode(texts, normalize_embeddings=True)` — vetores normalizados para similaridade por produto escalar
4. **Embeddings NUNCA vão para o DataFrame** — apenas `embedding_id` (índice numérico) é adicionado
5. Os vetores ficam no dicionário `{paper_id: np.ndarray}` salvo em pickle

### Saídas
- `outputs/dataset_embeddings.pkl` — DataFrame + embeddings
- `cache/embeddings.pkl` — cache reutilizável

---

## Etapa 4 — Ensemble de LLMs (com filtro semântico)

**Arquivos:** `pipeline.py:583-685` + `llm.py` | **Comando:** `python pipeline.py --stage llm`

### Arquitetura do ensemble

```
Artigo → 3 modelos em paralelo → Agregação (mediana + remoção de outliers)
         ├── qwen (qwen2.5:3b via Ollama)
         ├── gemma3 (gemma3 via Ollama)
         └── llama3.2 (llama3.2:3b via Ollama)
```

### Filtro de similaridade (ANTES do ensemble)

1. Carrega embeddings do cache (`cache/embeddings.pkl`)
2. Gera embeddings dos **5 domínios de pesquisa** a partir das descrições textuais em `config.py`
3. Calcula `sim_max` = produto escalar artigo × cada domínio → maior valor (já normalizados, range 0–1)
4. **Corte em 60%:** artigos com `sim_max < 0.6` recebem `S_Quali = 0` automaticamente e NÃO são enviados aos LLMs

### Para cada artigo que passa o filtro

1. **Preenche prompt** (`prompts/relevance.txt`) com `{title}` e `{abstract}`
2. **Chama cada modelo** via Ollama local (`/api/chat` com `format=json`)
3. **Extrai JSON** da resposta — score (0–10), confidence (0–1), domains, reason
4. **Cache individual:** `cache/responses/{paper_id}_{model}.json`
5. **Sleep 1.5s** entre artigos (politeness, modo produção)

### Agregação inline

- **Mediana** dos 3 scores por artigo
- **Remove outliers:** scores com `|score - mediana| > 2` são descartados
- **S_Quali = média** dos scores restantes (ou mediana se todos removidos)

### Modo Mock
Com `MODO_PRODUCAO=False`, usa `_mock_response()` que gera scores determinísticos via `hash(paper_id + model)` — sem custo, para teste.

### Backends suportados

| Backend | Modelo padrão | Via |
|---|---|---|
| Ollama (local) | qwen2.5:3b, gemma3, llama3.2:3b | `http://localhost:11434` |
| OpenAI | gpt-4o-mini | API key |
| Gemini | gemini-1.5-flash | API key |
| Grok (xAI) | grok-2-latest | API key |

### Saída
- `cache/responses/*.json` — um arquivo por artigo por modelo
- `outputs/dataset_preprocessed.csv` atualizado com colunas `sim_max` + `S_Quali`

---

## Etapa 4b — Agreement (concordância entre modelos)

**Arquivo:** `agreement.py` | **Comando:** `python pipeline.py --stage agreement`

### Métricas calculadas

| Métrica | Descrição |
|---|---|
| **S_Quali** | Score qualitativo final (mediana sem outliers) |
| **Desvio padrão** | Dispersão entre scores dos modelos por artigo |
| **Variância** | Quadrado do desvio padrão |
| **Spearman** | Correlação de rankings entre pares de modelos |
| **ICC(2,k)** | Intraclass Correlation — concordância absoluta |

### Fórmula do ICC
```
ICC = (MSR - MSE) / MSR
MSR = variância entre artigos (between-subjects)
MSE = variância residual (within-subjects)
```

### Saída
`outputs/agreement.csv`

---

## Etapa 5 — Bibliometria (Score Quantitativo)

**Arquivo:** `pipeline.py:293-313` | **Comando:** `python pipeline.py --stage bibliometria`

### Fórmulas

```
S_Quant = (citations / max_citations) × 10
Score_Global = 0.6 × S_Quali + 0.4 × S_Quant
```

### Pesos configuráveis

| Peso | Valor | Significado |
|---|---|---|
| `S_QUANT` | 1.0 | Peso de citações brutas |
| `QUALI_WEIGHT` | 0.6 | Peso do score qualitativo (LLM) |
| `QUANT_WEIGHT` | 0.4 | Peso do score quantitativo (bibliometria) |

### Segregação
- **Aprovados:** `S_Quali ≥ 5.0` → calcula scores quantitativos
- **Rejeitados:** `S_Quali < 5.0` → excluídos do ranking

### Saída
`outputs/dataset_preprocessed.csv` atualizado com `C_Norm`, `V_Norm`, `S_Quant`, `Score_Global`

---

## Etapa 6 — Ranking + Outputs

**Arquivo:** `pipeline.py:942-1023` | **Comando:** `python pipeline.py --stage ranking`

### Fluxo final

1. **Ordena** artigos aprovados por `Score_Global` decrescente
2. **Seleciona TOP_K** (config: `TOP_K = 5`)
3. ~~Perfis semânticos~~ — DESABILITADO (Gemini API retornando 404)

### Outputs gerados

| Arquivo | Formato | Conteúdo |
|---|---|---|
| `outputs/ranking.csv` | CSV | Todos os artigos aprovados ordenados |
| `outputs/aprovados.csv` | CSV | Artigos com S_Quali ≥ 5.0 |
| `outputs/rejeitados.csv` | CSV | Artigos com S_Quali < 5.0 |
| `outputs/top5.csv` | CSV | Apenas os 5 melhores |
| `outputs/dataset_final.xlsx` | Excel | 3 abas (Top 5, Interesse, Rejeitados) |
| `outputs/experiment.json` | JSON | Metadados do experimento |
| `outputs/pipeline.log` | Log | Log completo em DEBUG |

### Garantia de persistência
CSVs são salvos **sempre**, independentemente do sucesso do Excel. Se o Excel falhar, os CSVs já foram escritos antes.

---

## Configuração

Todas as constantes em `config.py`:

```python
# Scores e pesos
TOP_K: int = 5
QUALI_WEIGHT: float = 0.6    # Peso qualitativo no Score Global
QUANT_WEIGHT: float = 0.4    # Peso quantitativo no Score Global
C_WEIGHT: float = 0.7        # Peso citações no S_Quant
V_WEIGHT: float = 0.3        # Peso citações influentes no S_Quant
NOTA_CORTE: float = 5.0      # S_Quali mínimo para aprovação

# Coleta
LIMITE_COLETA: int = 500
ANOS_COLETA: list = [2022, 2023, 2024, 2025]

# Embeddings
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
EMBEDDING_FALLBACK: str = "all-mpnet-base-v2"

# LLMs
LLM_MODELS: list = ["qwen", "gemma3", "llama3.2"]
LLM_TEMPERATURE: float = 0.0

# Similariade mínima (etapa LLM)
SIM_THRESHOLD: float = 0.6   # 60%

# Termos de busca DBLP
TERMOS_PESQUISA: list = [
    "Blockchain", "IoT", "Cyber", "Fault", "Distributed", "Resilience"
]

# Domínios de pesquisa
DOMAINS: list = [
    "CPS", "Blockchain", "IoT", "Fault Tolerance", "Distributed Systems"
]
```

---

## Estrutura do projeto

```
classification_ares/
├── pipeline.py              # Orquestrador principal + CLI
├── config.py                # Configurações centralizadas
├── preprocessing.py         # Normalização de texto
├── embeddings.py            # SentenceTransformer + similaridade
├── llm.py                   # Interface unificada para LLMs
├── agreement.py             # Concordância entre modelos
├── requirements.txt         # Dependências
├── README.md                # Este arquivo
├── prompts/
│   ├── relevance.txt        # Template de prompt de relevância
│   ├── relevance_embed.txt  # Template com embeddings
│   └── semantic_profile.txt # Template de perfil semântico
├── cache/
│   ├── embeddings.pkl       # Cache de embeddings
│   ├── responses/           # Cache de respostas LLM
│   └── responses_embed/     # Cache com embeddings
└── outputs/
    ├── dataset_raw.csv
    ├── dataset_preprocessed.csv
    ├── dataset_embeddings.pkl
    ├── ranking.csv
    ├── aprovados.csv
    ├── rejeitados.csv
    ├── top5.csv
    ├── agreement.csv
    ├── dataset_final.xlsx
    ├── experiment.json
    └── pipeline.log
```
