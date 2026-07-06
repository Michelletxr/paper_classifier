# -*- coding: utf-8 -*-
"""Módulo de interface unificada para LLMs.

Fornece uma interface única evaluate_with_model(model_name, text, paper_id, ...)
que despacha para o backend apropriado: OpenAI, Gemini, Grok ou Ollama.

Cada resposta é cacheada em cache/responses/{paper_id}_{model}.json.

Suporte a modelos:
- OpenAI: gpt-4o-mini (configurável)
- Gemini: gemini-1.5-flash
- Grok: grok-2-1212
- Ollama: llama3, qwen2.5, mistral, deepseek-r1
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from typing import Any, Optional

import pandas as pd

from config import API_KEYS, config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Resposta estruturada de um modelo LLM.

    Attributes:
        score: Nota de relevância (0-10).
        confidence: Confiança da avaliação (0-1).
        domains: Lista de domínios identificados.
        reason: Justificativa da avaliação.
        model: Nome do modelo que gerou a resposta.
        paper_id: Identificador do artigo avaliado.
        timestamp: Timestamp ISO 8601 da avaliação.
    """

    score: float
    confidence: float
    domains: list[str]
    reason: str
    model: str
    paper_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompt(prompt_path: Path) -> str:
    """Carrega template de prompt do disco.

    Args:
        prompt_path: Caminho para o arquivo .txt do prompt.

    Returns:
        Conteúdo do template de prompt.

    Raises:
        FileNotFoundError: Se o arquivo de prompt não existir.
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt não encontrado: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def fill_prompt(template: str, **kwargs: str) -> str:
    """Preenche placeholders {key} no template com os valores fornecidos.

    Args:
        template: String do template com placeholders {chave}.
        **kwargs: Valores para substituir os placeholders.

    Returns:
        Prompt preenchido.
    """
    return template.format(**kwargs)


# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    """Extrai um objeto JSON da saída textual de um LLM.

    Lida com markdown code fences, texto antes/depois do JSON,
    e caracteres de controle.

    Args:
        text: Saída bruta do LLM.

    Returns:
        Dicionário parseado do JSON.

    Raises:
        ValueError: Se não for possível extrair JSON válido.
    """
    if not text:
        raise ValueError("Resposta vazia do modelo.")

    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Encontra o primeiro { e o último }
    start = text.find("{")
    end = text.rfind("}")

    if start == -1:
        raise ValueError(f"Não foi possível localizar JSON na resposta: {text[:200]}")

    # Se JSON está truncado (sem }), tenta fechar
    if end == -1 or end < start:
        json_str = text[start:]
        logger.warning("JSON parece truncado (sem chave de fechamento). Tentando recuperar...")
        # Remove trailing incompleto após última vírgula ou aspa
        last_comma = json_str.rfind(",")
        if last_comma > 0:
            json_str = json_str[:last_comma]
        # Fecha arrays/objetos abertos
        open_braces = json_str.count("{") - json_str.count("}")
        open_brackets = json_str.count("[") - json_str.count("]")
        json_str += "]" * open_brackets + "}" * open_braces
    else:
        json_str = text[start : end + 1]

    # Tenta parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Correção 1: fecha strings não terminadas (truncamento)
    # Padrão: "chave": "valor sem aspa final}
    fixed = re.sub(r'(:\s*")([^"]*?)(\s*})', r'\1\2"\3', json_str)
    if fixed != json_str:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Correção 2: trailing commas + fecha strings
    try:
        cleaned = re.sub(r",\s*}", "}", json_str)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Correção 3: trailing commas + fecha strings + aspas finais
    try:
        cleaned = re.sub(r",\s*}", "}", fixed)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(
            f"Falha ao parsear JSON após correção: {json_str[:300]}"
        )


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------


def _call_openai(prompt: str, model: str, api_key: str) -> dict:
    """Chama a API da OpenAI (Chat Completions).

    Args:
        prompt: Texto completo do prompt.
        model: Nome do modelo (ex: gpt-4o-mini).
        api_key: Chave de API da OpenAI.

    Returns:
        Dicionário parseado da resposta JSON.
    """
    try:
        import openai
        import httpx
    except ImportError:
        raise ImportError(
            "openai não está instalado. Instale com: pip install openai"
        )

    http_client = httpx.Client(verify=False)
    client = openai.OpenAI(api_key=api_key, http_client=http_client)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a JSON-only responder. You must output valid JSON and nothing else.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=500,
    )

    raw = response.choices[0].message.content or ""
    return extract_json(raw)


def _call_gemini(prompt: str, model: str, api_key: str) -> dict:
    """Chama a API Gemini via REST.

    Args:
        prompt: Texto completo do prompt.
        model: Nome do modelo (ex: gemini-1.5-flash).
        api_key: Chave de API do Gemini.

    Returns:
        Dicionário parseado da resposta JSON.
    """
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0},
    }

    resp = requests.post(url, json=payload, timeout=30, verify=False)
    resp.raise_for_status()

    data = resp.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return extract_json(raw)


def _call_grok(prompt: str, model: str, api_key: str) -> dict:
    """Chama a API do Grok (xAI) via REST.

    Args:
        prompt: Texto completo do prompt.
        model: Nome do modelo (ex: grok-2-1212).
        api_key: Chave de API do xAI.

    Returns:
        Dicionário parseado da resposta JSON.
    """
    import requests

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a JSON-only responder. You must output valid JSON and nothing else.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30, verify=False)
    resp.raise_for_status()

    data = resp.json()
    raw = data["choices"][0]["message"]["content"].strip()
    return extract_json(raw)


def _call_ollama(prompt: str, model: str, base_url: str) -> dict:
    """Chama Ollama local via REST API (chat endpoint com format=json).

    Usa o endpoint /api/chat com o parâmetro format=json, que força
    o motor do Ollama a gerar JSON sintaticamente válido — nunca trunca
    no meio de uma string ou deixa chaves sem fechar.

    Args:
        prompt: Texto completo do prompt.
        model: Nome do modelo no Ollama (ex: llama3, qwen2.5).
        base_url: URL base do servidor Ollama.

    Returns:
        Dicionário parseado da resposta JSON.
    """
    import requests

    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a scientific paper reviewer. "
                    "Output ONLY a valid JSON object with fields: "
                    "score (integer 0-10), confidence (float 0-1), "
                    "domains (array of strings from: CPS, Blockchain, IoT, "
                    "Fault Tolerance, Distributed Systems), "
                    "reason (short string, max 1 sentence)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 1024,
        },
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    # /api/chat retorna message.content; /api/generate retorna response
    raw = (
        data.get("message", {}).get("content", "")
        or data.get("response", "")
    ).strip()
    logger.debug("Ollama raw response (%d chars): %s...", len(raw), raw[:150])
    return extract_json(raw)


# ---------------------------------------------------------------------------
# Mock mode (deterministic, for testing without API credits)
# ---------------------------------------------------------------------------


def _mock_response(model: str, paper_id: str) -> dict:
    """Gera resposta mockada determinística para testes.

    Usa hash do paper_id para produzir scores consistentes por artigo,
    simulando variabilidade entre modelos.

    Args:
        model: Nome do modelo.
        paper_id: Identificador do artigo.

    Returns:
        Dicionário simulado no formato da resposta LLM.
    """
    # Score determinístico baseado no hash do paper_id + model
    seed = hash(paper_id + model) % 100
    score = 3 + (seed % 8)  # score entre 3 e 10
    confidence = 0.5 + (seed % 50) / 100  # confidence entre 0.5 e 0.99

    domain_options = ["CPS", "Blockchain", "IoT", "Fault Tolerance", "Distributed Systems"]
    n_domains = 1 + (seed % 4)  # 1 a 4 domínios
    domains = domain_options[:n_domains]

    return {
        "score": float(score),
        "confidence": round(confidence, 2),
        "domains": domains,
        "reason": f"Mock evaluation by {model} for paper {paper_id}.",
    }


# ---------------------------------------------------------------------------
# Cache de respostas
# ---------------------------------------------------------------------------


def _load_cached_response(responses_dir: Path, paper_id: str, model: str) -> Optional[dict]:
    """Carrega resposta cacheada do disco.

    Args:
        responses_dir: Diretório de cache de respostas.
        paper_id: Identificador do artigo.
        model: Nome do modelo.

    Returns:
        Dicionário da resposta ou None se não existir.
    """
    safe_paper_id = paper_id.replace("/", "_")
    cache_file = responses_dir / f"{safe_paper_id}_{model}.json"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Falha ao ler cache %s: %s", cache_file, e)
        return None


def _save_cached_response(responses_dir: Path, paper_id: str, model: str, response: dict) -> None:
    """Salva resposta no cache em disco.

    Args:
        responses_dir: Diretório de cache de respostas.
        paper_id: Identificador do artigo.
        model: Nome do modelo.
        response: Dicionário da resposta a cachear.
    """
    responses_dir.mkdir(parents=True, exist_ok=True)
    safe_paper_id = paper_id.replace("/", "_")
    cache_file = responses_dir / f"{safe_paper_id}_{model}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def evaluate_with_model(
    model_name: str,
    title: str,
    abstract: str,
    paper_id: str,
    prompt_template: str,
    responses_dir: Optional[Path] = None,
    **extra_template_kwargs: str,
) -> LLMResponse:
    """Interface unificada para avaliação por LLM.

    Verifica cache primeiro. Se não encontrado, chama o backend apropriado
    e salva a resposta no cache.

    Args:
        model_name: Identificador do modelo ("gpt", "gemini", "grok", "llama3", "qwen",
                    "mistral", "deepseek-r1").
        title: Título do artigo.
        abstract: Resumo do artigo.
        paper_id: Identificador único do artigo.
        prompt_template: Template de prompt com placeholders {title} e {abstract}.
        responses_dir: Diretório para cache de respostas.
        **extra_template_kwargs: Placeholders extras para o template
                                (ex: similarity_context).

    Returns:
        LLMResponse com score, confidence, domains, reason.
    """
    # Verifica cache
    if responses_dir:
        cached = _load_cached_response(responses_dir, paper_id, model_name)
        if cached is not None:
            logger.debug("Cache hit: %s / %s", paper_id, model_name)
            return LLMResponse(
                score=cached["score"],
                confidence=cached["confidence"],
                domains=cached.get("domains", []),
                reason=cached.get("reason", ""),
                model=model_name,
                paper_id=paper_id,
                timestamp=cached.get("timestamp", ""),
            )

    # Preenche o prompt
    prompt = fill_prompt(prompt_template, title=title, abstract=abstract, **extra_template_kwargs)

    # Mock mode: retorna dado determinístico
    if not config.MODO_PRODUCAO:
        logger.debug("Mock mode: %s / %s", paper_id, model_name)
        data = _mock_response(model_name, paper_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        data["timestamp"] = timestamp

        if responses_dir:
            _save_cached_response(responses_dir, paper_id, model_name, data)

        return LLMResponse(
            score=data["score"],
            confidence=data["confidence"],
            domains=data["domains"],
            reason=data["reason"],
            model=model_name,
            paper_id=paper_id,
            timestamp=timestamp,
        )

    # Produção: chama o backend apropriado
    logger.info("Chamando %s para artigo %s...", model_name, paper_id)

    try:
        if model_name == "gpt":
            api_key = API_KEYS.get("openai", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não configurada.")
            data = _call_openai(prompt, config.OPENAI_MODEL, api_key)

        elif model_name == "gemini":
            api_key = API_KEYS.get("gemini", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY não configurada.")
            data = _call_gemini(prompt, config.GEMINI_MODEL, api_key)

        elif model_name == "grok":
            api_key = API_KEYS.get("grok", "")
            if not api_key:
                raise ValueError("GROK_API_KEY não configurada.")
            data = _call_grok(prompt, config.GROK_MODEL, api_key)

        elif model_name in config.OLLAMA_MODELS:
            ollama_model = config.OLLAMA_MODELS[model_name]
            data = _call_ollama(prompt, ollama_model, config.OLLAMA_BASE_URL)

        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

    except Exception as e:
        logger.error(
            "Erro ao chamar %s para artigo %s: %s",
            model_name,
            paper_id,
            e,
        )
        # Retorna resposta de erro com score 0
        return LLMResponse(
            score=0.0,
            confidence=0.0,
            domains=[],
            reason=f"Erro: {str(e)[:200]}",
            model=model_name,
            paper_id=paper_id,
        )

    # Adiciona timestamp e salva cache
    timestamp = datetime.now(timezone.utc).isoformat()
    data["timestamp"] = timestamp

    if responses_dir:
        _save_cached_response(responses_dir, paper_id, model_name, data)

    return LLMResponse(
        score=float(data.get("score", 0)),
        confidence=float(data.get("confidence", 0)),
        domains=data.get("domains", []),
        reason=data.get("reason", ""),
        model=model_name,
        paper_id=paper_id,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Ensemble runner
# ---------------------------------------------------------------------------


def run_ensemble(
    df: pd.DataFrame,
    models: Optional[list[str]] = None,
    prompt_path: Optional[Path] = None,
    responses_dir: Optional[Path] = None,
    id_column: str = "id",
    title_column: str = "title",
    abstract_column: str = "abstract",
) -> list[LLMResponse]:
    """Executa o ensemble de LLMs para todos os artigos do DataFrame.

    Para cada artigo, chama cada modelo configurado.
    Salva cada resposta individual em cache/responses/{paper_id}_{model}.json.

    Args:
        df: DataFrame com artigos a avaliar.
        models: Lista de modelos a usar. Se None, usa config.LLM_MODELS.
        prompt_path: Caminho para o template de prompt de relevância.
        responses_dir: Diretório para cache de respostas.
        id_column: Coluna com identificador único do artigo.
        title_column: Coluna com título.
        abstract_column: Coluna com abstract (ou clean_text).

    Returns:
        Lista de LLMResponse, uma por artigo por modelo.
    """
    if models is None:
        models = config.LLM_MODELS

    if prompt_path is None:
        prompt_path = config.PROMPT_RELEVANCE

    if responses_dir is None:
        responses_dir = config.RESPONSES_DIR

    prompt_template = load_prompt(prompt_path)

    total = len(df) * len(models)
    logger.info(
        "Iniciando ensemble: %d artigos × %d modelos = %d avaliações.",
        len(df),
        len(models),
        total,
    )

    all_responses: list[LLMResponse] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        paper_id = str(row[id_column])
        title = str(row.get(title_column, ""))
        abstract = str(row.get(abstract_column, ""))

        if abstract == "Resumo não disponível" or not abstract.strip():
            logger.info(
                "Artigo %s sem abstract disponível — atribuindo score 0.",
                paper_id,
            )
            for model_name in models:
                all_responses.append(
                    LLMResponse(
                        score=0.0,
                        confidence=1.0,
                        domains=[],
                        reason="Resumo não disponível.",
                        model=model_name,
                        paper_id=paper_id,
                    )
                )
            continue

        for model_name in models:
            response = evaluate_with_model(
                model_name=model_name,
                title=title,
                abstract=abstract,
                paper_id=paper_id,
                prompt_template=prompt_template,
                responses_dir=responses_dir,
            )
            all_responses.append(response)

        # Throttle entre artigos no modo produção
        if config.MODO_PRODUCAO and idx < len(df) - 1:
            time.sleep(1.5)

    logger.info("Ensemble concluído: %d respostas coletadas.", len(all_responses))
    return all_responses


# ---------------------------------------------------------------------------
# Ensemble runner — versão com embeddings (similaridade artigo–domínio)
# ---------------------------------------------------------------------------


def run_ensemble_with_embeddings(
    df: pd.DataFrame,
    models: Optional[list[str]] = None,
    prompt_path: Optional[Path] = None,
    responses_dir: Optional[Path] = None,
    id_column: str = "id",
    title_column: str = "title",
    abstract_column: str = "abstract",
) -> list[LLMResponse]:
    """Executa o ensemble de LLMs com contexto de similaridade semântica.

    DIFERE de run_ensemble():
    - Usa prompt de relevance_embed.txt (inclui similaridade artigo–domínio)
    - Cache em cache/responses_embed/ (separado da versão sem embeddings)
    - Injeta similarity_context no prompt para cada artigo

    O DataFrame deve conter colunas de similaridade (sim_max, sim_CPS, etc.)
    geradas por embeddings.compute_all_similarities().

    Args:
        df: DataFrame com colunas de similaridade.
        models: Lista de modelos. Se None, usa config.LLM_MODELS.
        prompt_path: Caminho para relevance_embed.txt.
        responses_dir: Diretório de cache. Padrão: cache/responses_embed/.
        id_column: Coluna com ID do artigo.
        title_column: Coluna com título.
        abstract_column: Coluna com abstract.

    Returns:
        Lista de LLMResponse.
    """
    if models is None:
        models = config.LLM_MODELS

    if prompt_path is None:
        prompt_path = config.PROMPT_RELEVANCE_EMBED

    if responses_dir is None:
        responses_dir = config.RESPONSES_EMBED_DIR

    prompt_template = load_prompt(prompt_path)

    total = len(df) * len(models)
    logger.info(
        "Iniciando ensemble COM embeddings: %d artigos × %d modelos = %d avaliações.",
        len(df),
        len(models),
        total,
    )
    logger.info("Cache separado: %s", responses_dir)

    all_responses: list[LLMResponse] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        paper_id = str(row[id_column])
        title = str(row.get(title_column, ""))
        abstract = str(row.get(abstract_column, ""))

        if abstract == "Resumo não disponível" or not abstract.strip():
            for model_name in models:
                all_responses.append(
                    LLMResponse(
                        score=0.0,
                        confidence=1.0,
                        domains=[],
                        reason="Resumo não disponível.",
                        model=model_name,
                        paper_id=paper_id,
                    )
                )
            continue

        # Constrói contexto de similaridade para o prompt
        similarity_context = _build_similarity_context(row)

        for model_name in models:
            response = evaluate_with_model(
                model_name=model_name,
                title=title,
                abstract=abstract,
                paper_id=paper_id,
                prompt_template=prompt_template,
                responses_dir=responses_dir,
                similarity_context=similarity_context,
            )
            all_responses.append(response)

        if config.MODO_PRODUCAO and idx < len(df) - 1:
            time.sleep(1.5)

    logger.info(
        "Ensemble com embeddings concluído: %d respostas coletadas.",
        len(all_responses),
    )
    return all_responses


def _build_similarity_context(row: pd.Series) -> str:
    """Constrói string de contexto de similaridade para o prompt.

    Exemplo de saída:
        - CPS: 0.9234
        - Blockchain: 0.8102
        - IoT: 0.7560
        - Fault Tolerance: 0.6891
        - Distributed Systems: 0.8345
        → Max similarity: 0.9234 (CPS)

    Args:
        row: Linha do DataFrame com colunas de similaridade.

    Returns:
        String formatada com similaridades por domínio.
    """
    from config import DOMAINS

    lines: list[str] = []
    max_sim = 0.0
    max_domain = ""

    for domain in DOMAINS:
        col_name = domain  # nome da coluna é igual ao nome do domínio
        sim = float(row.get(col_name, 0.0))
        lines.append(f"  - {domain}: {sim:.4f}")
        if sim > max_sim:
            max_sim = sim
            max_domain = domain

    lines.append(f"  → Max similarity: {max_sim:.4f} ({max_domain})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Semantic profile (preserved from original)
# ---------------------------------------------------------------------------


def semantic_profile(
    abstract: str,
    prompt_path: Optional[Path] = None,
) -> str:
    """Gera perfil semântico de tópicos para um abstract.

    Args:
        abstract: Texto do resumo a analisar.
        prompt_path: Caminho para o template de prompt de perfil semântico.

    Returns:
        String com a distribuição de tópicos (ou mensagem de erro).
    """
    if prompt_path is None:
        prompt_path = config.PROMPT_SEMANTIC_PROFILE

    prompt_template = load_prompt(prompt_path)
    prompt = fill_prompt(prompt_template, abstract=abstract)

    if not config.MODO_PRODUCAO:
        # Mock determinístico
        return (
            "Blockchain: 25%, IoT: 25%, Fault Tolerance: 25%, "
            "Distributed Systems: 15%, Other: 10% (Mock)"
        )

    # Modo produção: usa Gemini para perfil semântico
    try:
        api_key = API_KEYS.get("gemini", "")
        if not api_key:
            return "Erro: GEMINI_API_KEY não configurada"

        data = _call_gemini(prompt, config.GEMINI_MODEL, api_key)
        # Formata como string legível
        parts = [f"{k}: {v}%" for k, v in data.items()]
        return ", ".join(parts)

    except Exception as e:
        logger.error("Erro no perfil semântico: %s", e)
        return f"Erro na avaliação semântica: {e}"
