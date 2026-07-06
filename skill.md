Você é um engenheiro de software sênior e pesquisador em NLP, Recuperação da Informação e Engenharia de Software.

Sua tarefa NÃO é criar um novo projeto.

Sua tarefa é refatorar completamente o script existente, preservando toda a lógica de coleta e bibliometria.

O objetivo é transformar o código em um pipeline científico modular, reproduzível e escalável para mineração de literatura científica.

====================================================================

REGRAS IMPORTANTES

====================================================================

NÃO altere a lógica de coleta.

Mantenha exatamente:

- DBLP
- OpenAlex
- Retry
- Backoff
- Sessions
- Enriquecimento dos metadados

Essa etapa já está validada.

====================================================================

ARQUITETURA

====================================================================

Implemente exatamente o seguinte pipeline:

Coleta
(DBLP + OpenAlex)

↓

Pré-processamento NLP

↓

Embeddings (Sentence Transformers)

↓

Ensemble de LLMs
(OpenAI + Gemini + Grok + Ollama)

↓

Agregação das respostas

↓

Score Qualitativo

↓

Bibliometria

↓

Score Quantitativo

↓

Score Global

↓

TOP_K configurável

====================================================================

FASE 1

====================================================================

Não modificar.

====================================================================

FASE 2

PRÉ-PROCESSAMENTO

====================================================================

Criar um novo módulo

preprocessing.py

Ele deve receber

Title
Abstract
Keywords

e produzir

clean_text

O pré-processamento deve incluir:

- Unicode normalization
- remoção de HTML
- remoção de múltiplos espaços
- remoção de caracteres inválidos
- strip()

Não remover stopwords.

Não fazer stemming.

Não fazer lemmatization.

O objetivo é apenas normalizar.

Adicionar uma nova coluna

clean_text

no DataFrame.

====================================================================

FASE 3

EMBEDDINGS

====================================================================

Criar

embeddings.py

Utilizar

SentenceTransformer

Modelo padrão

BAAI/bge-large-en-v1.5 

Caso indisponível

usar

all-mpnet-base-v2

Para cada artigo gerar

embedding

Salvar automaticamente em

cache/embeddings.pkl

Se o cache existir

não recalcular.

Adicionar uma coluna

embedding_id

no DataFrame.

NUNCA armazenar embeddings no Excel.

====================================================================

FASE 4

LLMs

====================================================================

Criar

llm.py

Criar uma interface única

evaluate_with_model(model_name,text)

Implementar suporte para

OpenAI

Gemini

Grok

Ollama

Ollama deverá suportar

llama3

qwen2.5

mistral

deepseek-r1

Todos devem usar a mesma interface.

====================================================================

PROMPTS

====================================================================

Remover completamente prompts hardcoded.

Criar

prompts/

relevance.txt

semantic_profile.txt

Carregar automaticamente.

====================================================================

PROMPT DE RELEVÂNCIA

====================================================================

O prompt deverá instruir o modelo a responder APENAS JSON.

Formato:

{

"score":0-10,

"confidence":0-1,

"domains":[
"CPS",
"Blockchain",
"IoT",
"Fault Tolerance",
"Distributed Systems"
],

"reason":"..."

}

Nenhum texto adicional.

====================================================================

TEMPERATURA

====================================================================

Todos os modelos devem utilizar

temperature=0

quando disponível.

====================================================================

ENSEMBLE

====================================================================

Executar

GPT

Gemini

Grok

Llama3

Qwen

Para cada artigo.

Cada resposta deve ser salva

cache/responses/

paperID_model.json

====================================================================

AGREGAÇÃO

====================================================================

Criar

agreement.py

Receber

todos os scores

Calcular

mediana

remover outliers

(mesma regra existente)

calcular

média

Produzir

SQuali

Também calcular

desvio padrão

variância

Spearman

ICC

Salvar

agreement.csv

====================================================================

SCORE QUALITATIVO

====================================================================

Manter exatamente a lógica atual.

Apenas substituir

nota

por

score

do JSON.

====================================================================

FASE 5

BIBLIOMETRIA

====================================================================

NÃO ALTERAR.

Manter

Citações

Normalização

SQuant

Exatamente iguais.

====================================================================

FASE 6

SCORE GLOBAL

====================================================================

Manter

Score_Global

=

0.6*SQuali

+

0.4*SQuant

Os pesos devem ficar em

config.py

====================================================================

TOP K

====================================================================

Nunca utilizar

head(5)

Criar

config.py

TOP_K

O pipeline deverá utilizar

TOP_K

====================================================================

CACHE

====================================================================

Criar

cache/

embeddings.pkl

responses/

====================================================================

OUTPUTS

====================================================================

Gerar automaticamente

dataset_raw.csv

dataset_preprocessed.csv

dataset_embeddings.pkl

agreement.csv

ranking.csv

dataset_final.xlsx

experiment.json

pipeline.log

====================================================================

experiment.json

====================================================================

Salvar

modelo de embedding

LLMs utilizados

temperatura

TOP_K

pesos

data

hora

tempo de execução

====================================================================

LOGGING

====================================================================

Substituir completamente

print()

por

logging

====================================================================

CÓDIGO

====================================================================

Utilizar

typing

dataclasses

pathlib

logging

requests.Session

cache

tratamento de exceções

docstrings

type hints

====================================================================

OBJETIVO

====================================================================

O resultado deve ser um software científico modular e reproduzível, adequado para publicação acadêmica.

Não alterar a lógica da coleta nem da bibliometria.

Refatorar apenas a camada de processamento semântico e avaliação qualitativa.