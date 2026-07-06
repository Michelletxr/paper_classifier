# -*- coding: utf-8 -*-
"""Pipeline de Avaliação ARES — Orquestrador Principal.

Pipeline científico modular para mineração de literatura científica:
  1. Coleta (DBLP + OpenAlex)        — PRESERVADO
  2. Pré-processamento NLP
  3. Embeddings (SentenceTransformer)
  4. Ensemble de LLMs + Agregação
  5. Bibliometria                     — PRESERVADO
  6. Score Global + TOP_K + Outputs

Uso:
  python pipeline.py                          # Executa o pipeline completo
  python pipeline.py --stage coleta           # Executa apenas a coleta
  python pipeline.py --stage preprocess       # Executa apenas o pré-processamento
  python pipeline.py --stage embeddings       # Executa apenas os embeddings
  python pipeline.py --stage llm              # Executa apenas o ensemble de LLMs (SEM embeddings)
  python pipeline.py --stage llm-embed        # Executa apenas o ensemble de LLMs (COM embeddings)
  python pipeline.py --stage agreement        # Executa apenas a agregação
  python pipeline.py --stage bibliometria     # Executa apenas a bibliometria
  python pipeline.py --stage ranking          # Executa apenas ranking + outputs
  python pipeline.py --list                   # Lista todas as etapas disponíveis
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import API_KEYS, TERMOS_PESQUISA, config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(log_path: Optional[Path] = None) -> logging.Logger:
    """Configura logging para arquivo e console.

    Args:
        log_path: Caminho para o arquivo de log. Se None, usa config.PIPELINE_LOG.

    Returns:
        Logger raiz configurado.
    """
    if log_path is None:
        log_path = config.PIPELINE_LOG

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # Evita handlers duplicados
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log = setup_logging()


# ==============================================================================
# FASE 1: EXTRAÇÃO DE DADOS (PRESERVADO — NÃO ALTERAR)
# ==============================================================================


def reconstruir_abstract(indice_invertido: Optional[dict]) -> str:
    """Reconstrói abstract a partir de inverted index do OpenAlex.
    """
    if not indice_invertido:
        return "Resumo não disponível"
    try:
        tamanho = (
            max([pos for posicoes in indice_invertido.values() for pos in posicoes]) + 1
        )
        palavras = [""] * tamanho
        for palavra, posicoes in indice_invertido.items():
            for pos in posicoes:
                palavras[pos] = palavra
        return " ".join(palavras).strip()
    except Exception:
        return "Resumo não disponível"


def obter_dois_dblp_tematicos(
    ano: int, limite_por_ano: int, session: requests.Session
) -> list[str]:
    """Obtém DOIs de artigos da conferência ARES via DBLP.

    Garante 100% de exatidão na conferência com Exponential Backoff.
    """
    dois_coletados: set[str] = set()
    url = "https://dblp.org/search/publ/api"

    log.info("Sincronizando metadados DBLP para %d...", ano)

    for termo in TERMOS_PESQUISA:
        if len(dois_coletados) >= limite_por_ano:
           break

        params = {
            "q": f"venue:ARES: year:{ano} {termo}",
            "h": limite_por_ano,
            "format": "json",
        }

        max_retries = 5
        base_delay = 2

        for tentativa in range(max_retries):
            try:
                resp = session.get(url, params=params, timeout=15)

                if resp.status_code == 429 or resp.status_code >= 500:
                    sleep_time = base_delay * (2 ** tentativa)
                    log.warning(
                        "DBLP HTTP %d. Backoff: %ds...", resp.status_code, sleep_time
                    )
                    time.sleep(sleep_time)
                    continue

                resp.raise_for_status()
                hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])

                for hit in hits:
                    ee = hit.get("info", {}).get("ee", "")
                    if isinstance(ee, list):
                        ee = ee[0] if ee else ""
                    if "doi.org/" in ee:
                        doi_puro = ee.split("doi.org/")[1]
                        dois_coletados.add(doi_puro)

                time.sleep(1.5)
                break

            except requests.exceptions.RequestException:
                sleep_time = base_delay * (2 ** tentativa)
                log.warning(
                    "Queda de conexão DBLP ('%s'). Backoff: %ds...", termo, sleep_time
                )
                time.sleep(sleep_time)

    return list(dois_coletados)[:limite_por_ano]


def buscar_artigos_ares(limite: int = 100) -> pd.DataFrame:
    """Coleta artigos via arquitetura dual DBLP + OpenAlex.
    """
    log.info("Iniciando extração Federada (DBLP -> OpenAlex) para %d artigos...", limite)
    url_openalex = "https://api.openalex.org/works"

    anos = config.ANOS_COLETA
    limite_por_ano = max(1, limite // len(anos))
    artigos: list[dict] = []

    # Desabilita verificação SSL para ambientes com certificados desatualizados
    # (ex: macOS com Python do Homebrew). O Google Colab não tem esse problema.
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session_dblp = requests.Session()
    session_dblp.verify = False
    session_dblp.headers.update(
        {
            "User-Agent": "ARES-Auditor/1.0 (auditoria.academica@universidade.edu)",
            "Accept": "application/json",
        }
    )

    session_openalex = requests.Session()
    session_openalex.verify = False
    session_openalex.headers.update(
        {"User-Agent": "ARES-Auditor/1.0 (auditoria.academica@universidade.edu)"}
    )

    for ano in anos:
        log.info("Operação no ano %d:", ano)

        dois_alvo = obter_dois_dblp_tematicos(ano, limite_por_ano, session_dblp)

        if not dois_alvo:
            log.info("Nenhum artigo temático encontrado no DBLP para %d.", ano)
            continue

        log.info("%d DOIs identificados. Requisitando resumos ao OpenAlex...", len(dois_alvo))

        filtro_doi = "|".join(dois_alvo)
        params = {
            "filter": f"doi:{filtro_doi}",
            "per-page": limite_por_ano,
            "mailto": "auditoria.academica@universidade.edu",
        }

        max_retries = 5
        base_delay = 2

        for tentativa in range(max_retries):
            try:
                response = session_openalex.get(url_openalex, params=params, timeout=20)

                if response.status_code == 429 or response.status_code >= 500:
                    sleep_time = base_delay * (2 ** tentativa)
                    log.warning(
                        "OpenAlex HTTP %d. Backoff: %ds...",
                        response.status_code,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                resultados = response.json().get("results", [])

                for item in resultados:
                    autores_lista = item.get("authorships", [])
                    autores = (
                        ", ".join(
                            [
                                a.get("author", {}).get("display_name", "")
                                for a in autores_lista
                            ]
                        )
                        if autores_lista
                        else "Autor Desconhecido"
                    )
                    abstract_reconstruido = reconstruir_abstract(
                        item.get("abstract_inverted_index")
                    )

                    artigos.append(
                        {
                            "id": item.get("id"),
                            "title": item.get("title") or "Título Desconhecido",
                            "venue": "ARES (Validação DBLP)",
                            "authors": autores,
                            "year": item.get("publication_year"),
                            "abstract": abstract_reconstruido,
                            "citations": item.get("cited_by_count") or 0,
                            "influential_citations": item.get("cited_by_count") or 0,
                        }
                    )

                log.info("Ano %d enriquecido com sucesso.", ano)
                time.sleep(1)
                break

            except requests.exceptions.RequestException:
                sleep_time = base_delay * (2 ** tentativa)
                log.warning(
                    "Falha de conexão OpenAlex para %d. Backoff: %ds...",
                    ano,
                    sleep_time,
                )
                time.sleep(sleep_time)

    log.info("Extração concluída. %d artigos validados e indexados.", len(artigos))
    return pd.DataFrame(artigos)


# ==============================================================================
# FASE 5: BIBLIOMETRIA (PRESERVADO — NÃO ALTERAR)
# ==============================================================================


def calcular_scores_quantitativos(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula scores quantitativos: C_Norm, V_Norm, S_Quant, Score_Global.
    """
    log.info("Processando ranqueamento quantitativo...")

    max_cit = df["citations"].max() if df["citations"].max() > 0 else 1
    df["C_Norm"] = (df["citations"] / max_cit) * 10

    max_inf = (
        df["influential_citations"].max()
        if df["influential_citations"].max() > 0
        else 1
    )
    df["V_Norm"] = (df["influential_citations"] / max_inf) * 10

    df["S_Quant"] = (df["C_Norm"] * config.C_WEIGHT) + (df["V_Norm"] * config.V_WEIGHT)
    df["Score_Global"] = (df["S_Quali"] * config.QUALI_WEIGHT) + (
        df["S_Quant"] * config.QUANT_WEIGHT
    )

    return df.round(2)


# ==============================================================================
# EXPORT
# ==============================================================================


def export_results(
    df_top: pd.DataFrame,
    df_interesse: pd.DataFrame,
    df_rejeitados: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> None:
    """Exporta resultados finais para Excel (3 abas) + CSVs individuais como fallback.

    CSVs são salvos SEMPRE, independentemente do sucesso do Excel.
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    cols_to_drop = ["clean_text", "embedding_id", "_norm_title", "_norm_abstract"]

    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # --- CSVs (sempre salvos — garantia contra falha do Excel) ---
    csv_paths = {
        "aprovados": output_dir / "aprovados.csv",
        "rejeitados": output_dir / "rejeitados.csv",
        "top_k": output_dir / f"top{config.TOP_K}.csv",
    }

    try:
        _clean_df(df_interesse).to_csv(csv_paths["aprovados"], index=False)
        log.info("CSV salvo: %s (%d artigos)", csv_paths["aprovados"], len(df_interesse))
    except Exception as e:
        log.error("Falha ao salvar CSV de aprovados: %s", e)

    try:
        _clean_df(df_rejeitados).to_csv(csv_paths["rejeitados"], index=False)
        log.info("CSV salvo: %s (%d artigos)", csv_paths["rejeitados"], len(df_rejeitados))
    except Exception as e:
        log.error("Falha ao salvar CSV de rejeitados: %s", e)

    try:
        _clean_df(df_top).to_csv(csv_paths["top_k"], index=False)
        log.info("CSV salvo: %s (%d artigos)", csv_paths["top_k"], len(df_top))
    except Exception as e:
        log.error("Falha ao salvar CSV do TOP_%d: %s", config.TOP_K, e)

    # --- Excel (3 abas em um único arquivo) ---
    xlsx_path = output_dir / "dataset_final.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            _clean_df(df_top).to_excel(
                writer, sheet_name=f"Artigos Finais (Top {config.TOP_K})", index=False
            )
            _clean_df(df_interesse).to_excel(
                writer, sheet_name="Artigos de Interesse", index=False
            )
            _clean_df(df_rejeitados).to_excel(
                writer, sheet_name="Artigos Rejeitados", index=False
            )
        log.info("Excel exportado: %s", xlsx_path)
    except Exception as e:
        log.warning("Falha na escrita do Excel: %s (os CSVs foram salvos normalmente)", e)


def export_experiment_json(
    start_time: float,
    stats: dict,
    output_path: Optional[Path] = None,
) -> None:
    """Exporta metadados do experimento em JSON."""
    if output_path is None:
        output_path = config.EXPERIMENT_JSON

    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    execution_time = time.time() - start_time

    experiment = {
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_models": config.LLM_MODELS,
        "temperature": config.LLM_TEMPERATURE,
        "top_k": config.TOP_K,
        "weights": {
            "quali": config.QUALI_WEIGHT,
            "quant": config.QUANT_WEIGHT,
        },
        "nota_corte": config.NOTA_CORTE,
        "date": now.strftime("%Y-%m-%d"),
        "time_utc": now.strftime("%H:%M:%S"),
        "execution_time_seconds": round(execution_time, 2),
        "execution_time_minutes": round(execution_time / 60, 2),
        "total_papers_collected": stats.get("total_coletados", 0),
        "papers_above_threshold": stats.get("aprovados", 0),
        "papers_rejected": stats.get("rejeitados", 0),
        "agreement_stats": stats.get("agreement", {}),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2, ensure_ascii=False)

    log.info("Metadados do experimento exportados: %s", output_path)


# ==============================================================================
# ETAPAS INDIVIDUAIS DO PIPELINE
# ==============================================================================


def stage_coleta(limite: Optional[int] = None) -> pd.DataFrame:
    """Etapa 1 — Coleta de artigos via DBLP + OpenAlex.

    Busca artigos da conferência ARES usando busca temática no DBLP
    e enriquece os metadados via OpenAlex.

    Entrada: nenhuma (requisições HTTP)
    Saída:   outputs/dataset_raw.csv

    Returns:
        DataFrame com os artigos coletados.
    """
    if limite is None:
        limite = config.LIMITE_COLETA

    _print_stage_header("COLETA", "DBLP + OpenAlex — Extração federada de artigos ARES")
    t0 = time.time()

    df = buscar_artigos_ares(limite=limite)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_RAW, index=False)

    _print_stage_footer(len(df), t0, "dataset_raw.csv")
    return df


def stage_preprocess(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Etapa 2 — Pré-processamento e normalização de texto.

    Aplica limpeza textual: normalização Unicode, remoção de HTML,
    colapso de espaços, remoção de caracteres de controle.
    Adiciona a coluna 'clean_text' ao dataset.

    Entrada: outputs/dataset_raw.csv (se df não fornecida)
    Saída:   outputs/dataset_preprocessed.csv

    Args:
        df: DataFrame opcional da etapa anterior. Se None, carrega do disco.

    Returns:
        DataFrame com a coluna 'clean_text' adicionada.
    """
    from preprocessing import preprocess_dataframe

    _print_stage_header("PRÉ-PROCESSAMENTO", "Normalização de texto (Unicode, HTML, espaços)")
    t0 = time.time()

    if df is None:
        df = _load_or_die(config.DATASET_RAW, "coleta")

    df = preprocess_dataframe(df, title_col="title", abstract_col="abstract")
    df.to_csv(config.DATASET_PREPROCESSED, index=False)

    _print_stage_footer(len(df), t0, "dataset_preprocessed.csv")
    return df


def stage_embeddings(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Etapa 3 — Geração de embeddings via SentenceTransformer.

    Utiliza BAAI/bge-large-en-v1.5 (fallback: all-mpnet-base-v2).
    Embeddings cacheados em cache/embeddings.pkl para evitar recomputação.
    Adiciona a coluna 'embedding_id' (índice numérico).

    Entrada: outputs/dataset_preprocessed.csv (se df não fornecida)
    Saída:   outputs/dataset_embeddings.pkl
             cache/embeddings.pkl

    Args:
        df: DataFrame opcional com coluna 'clean_text'.

    Returns:
        DataFrame com coluna 'embedding_id' adicionada.
    """
    from embeddings import compute_embeddings

    _print_stage_header(
        "EMBEDDINGS",
        f"SentenceTransformer: {config.EMBEDDING_MODEL}",
    )
    t0 = time.time()

    if df is None:
        df = _load_or_die(config.DATASET_PREPROCESSED, "preprocess")

    df, embeddings_dict = compute_embeddings(
        df,
        model_name=config.EMBEDDING_MODEL,
        fallback=config.EMBEDDING_FALLBACK,
        cache_path=config.EMBEDDINGS_CACHE,
        text_column="clean_text",
    )

    # Salva dataset + embeddings
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DATASET_EMBEDDINGS, "wb") as f:
        pickle.dump({"dataframe": df, "embeddings": embeddings_dict}, f)

    _print_stage_footer(len(df), t0, "dataset_embeddings.pkl")
    return df


# Limiar de similaridade mínima para artigo ser avaliado pelo LLM
SIM_THRESHOLD: float = 0.6


def _compute_sim_max_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna sim_max ao DataFrame usando embeddings cacheados.

    Se os embeddings não existirem ou ocorrer erro, sim_max = 1.0 para todos
    (modo degradado: avalia todos os artigos).

    Args:
        df: DataFrame com coluna 'id'.

    Returns:
        DataFrame com coluna 'sim_max' adicionada.
    """
    embeddings_dict = _load_article_embeddings()
    if embeddings_dict is None:
        log.warning(
            "Embeddings não encontrados — todos os artigos serão avaliados (sem filtro)."
        )
        df["sim_max"] = 1.0
        return df

    from embeddings import build_domain_embeddings, compute_domain_similarity, load_model
    from config import DOMAIN_DESCRIPTIONS

    try:
        model = load_model(config.EMBEDDING_MODEL, config.EMBEDDING_FALLBACK)
        domain_embeddings = build_domain_embeddings(model, DOMAIN_DESCRIPTIONS)

        sims: dict[str, float] = {}
        for _, row in df.iterrows():
            paper_id = str(row["id"])
            article_vec = embeddings_dict.get(paper_id)
            if article_vec is not None:
                sim_values = compute_domain_similarity(article_vec, domain_embeddings)
                sims[paper_id] = max(sim_values.values()) if sim_values else 0.0
            else:
                sims[paper_id] = 1.0  # sem embedding → avalia por segurança
                log.debug("Artigo %s sem embedding — será avaliado.", paper_id)

        df["sim_max"] = df["id"].apply(lambda pid: sims.get(str(pid), 1.0))
        log.info("Similaridade média (sim_max): %.4f", df["sim_max"].mean())
    except Exception as e:
        log.warning("Erro ao calcular similaridades: %s. Avaliando todos os artigos.", e)
        df["sim_max"] = 1.0

    return df


def stage_llm(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Etapa 4 — Ensemble de LLMs para avaliação qualitativa.

    Executa GPT, Gemini, Grok, Llama3 e Qwen para cada artigo.
    Cada resposta é cacheada em cache/responses/{paper_id}_{model}.json.
    Com MODO_PRODUCAO=False, usa mocks determinísticos.

    FILTRO DE SIMILARIDADE: apenas artigos com sim_max >= 0.6 são enviados
    ao ensemble. Os demais recebem S_Quali = 0 automaticamente.
    Requer que a etapa 'embeddings' tenha sido executada antes.

    Entrada: outputs/dataset_preprocessed.csv (se df não fornecida)
    Saída:   cache/responses/*.json (uma resposta por artigo por modelo)

    Args:
        df: DataFrame opcional com colunas 'id', 'title', 'abstract'.

    Returns:
        DataFrame com coluna 'S_Quali' adicionada.
    """
    from llm import run_ensemble

    _print_stage_header(
        "ENSEMBLE DE LLMs",
        f"Modelos: {', '.join(config.LLM_MODELS)} | "
        f"Temperatura: {config.LLM_TEMPERATURE} | "
        f"Filtro: sim_max ≥ {SIM_THRESHOLD:.0%} | "
        f"Cache: cache/responses/",
    )
    t0 = time.time()

    if df is None:
        df = _load_or_die(config.DATASET_PREPROCESSED, "preprocess")

    # --- Filtro de similaridade semântica ---
    df = _compute_sim_max_column(df)

    df_valid = df[df["sim_max"] >= SIM_THRESHOLD].copy()
    df_skip = df[df["sim_max"] < SIM_THRESHOLD].copy()

    log.info(
        "Filtro de similaridade: %d artigos com sim_max ≥ %.0f%% → LLM | "
        "%d artigos com sim_max < %.0f%% → S_Quali = 0",
        len(df_valid), SIM_THRESHOLD * 100,
        len(df_skip), SIM_THRESHOLD * 100,
    )

    # Artigos abaixo do limiar recebem S_Quali = 0 automaticamente
    if not df_skip.empty:
        df_skip["S_Quali"] = 0.0

    # Executa ensemble apenas nos artigos que passaram o filtro
    if not df_valid.empty:
        responses = run_ensemble(
            df_valid,
            models=config.LLM_MODELS,
            prompt_path=config.PROMPT_RELEVANCE,
            responses_dir=config.RESPONSES_DIR,
            id_column="id",
            title_column="title",
            abstract_column="abstract",
        )

        # Agrega scores brutos por artigo
        scores_map: dict[str, list[float]] = {}
        for r in responses:
            scores_map.setdefault(str(r.paper_id), []).append(r.score)

        # Calcula S_Quali inline (mediana + remoção de outliers)
        squali_map: dict[str, float] = {}
        for paper_id, scores in scores_map.items():
            median_val = float(np.median(scores))
            filtered = [s for s in scores if abs(s - median_val) <= 2.0]
            squali_map[paper_id] = round(float(np.mean(filtered)) if filtered else median_val, 2)

        df_valid["S_Quali"] = df_valid["id"].apply(
            lambda pid: squali_map.get(str(pid), 0.0)
        )
    else:
        log.warning("Nenhum artigo atingiu o limiar mínimo de similaridade (%.0f%%).", SIM_THRESHOLD * 100)

    # Recombina os DataFrames
    df = pd.concat([df_valid, df_skip], ignore_index=True)

    # Salva DataFrame com S_Quali para uso pelas próximas etapas
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_PREPROCESSED, index=False)

    n_aprovados = int((df["S_Quali"] >= config.NOTA_CORTE).sum())
    n_rejeitados = len(df) - n_aprovados
    n_avaliados = len(df_valid)
    n_pulados = len(df_skip)
    elapsed = time.time() - t0
    log.info(
        "Ensemble concluído: %d avaliados + %d pulados = %d artigos em %.1fs | "
        "Aprovados: %d (≥%.1f) | Rejeitados: %d",
        n_avaliados, n_pulados, len(df),
        elapsed, n_aprovados, config.NOTA_CORTE, n_rejeitados,
    )
    log.info("Respostas cache: %s", config.RESPONSES_DIR)
    log.info("=" * 60)

    return df


def stage_llm_embed(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Etapa 4-EMBED — Ensemble de LLMs COM similaridade semântica artigo–domínio.

    DIFERE de stage_llm():
    - Calcula similaridade por cosseno entre cada artigo e os 5 domínios
    - Injeta essas similaridades no prompt do LLM como contexto auxiliar
    - Usa prompt/prompts/relevance_embed.txt
    - Cache separado em cache/responses_embed/
    - FILTRO: apenas artigos com sim_max >= 0.6 são avaliados

    Entrada: outputs/dataset_embeddings.pkl (precisa dos embeddings)
    Saída:   cache/responses_embed/*.json
             coluna 'S_Quali' no DataFrame

    Args:
        df: DataFrame opcional com colunas 'id', 'title', 'abstract', 'clean_text'.

    Returns:
        DataFrame com colunas de similaridade e 'S_Quali'.
    """
    from embeddings import compute_all_similarities
    from llm import run_ensemble_with_embeddings
    from config import DOMAIN_DESCRIPTIONS, DOMAINS

    _print_stage_header(
        "ENSEMBLE DE LLMs (COM EMBEDDINGS)",
        f"Modelos: {', '.join(config.LLM_MODELS)} | "
        f"Similaridade artigo–domínio via {config.EMBEDDING_MODEL} | "
        f"Filtro: sim_max ≥ {SIM_THRESHOLD:.0%} | "
        f"Cache: cache/responses_embed/",
    )
    t0 = time.time()

    if df is None:
        df = _load_or_die(config.DATASET_PREPROCESSED, "preprocess")

    # Carrega embeddings dos artigos (precisa ter rodado etapa 3 antes)
    embeddings_dict = _load_article_embeddings()

    if embeddings_dict is None:
        log.error(
            "Embeddings não encontrados. Execute a etapa 'embeddings' primeiro:\n"
            "  python pipeline.py --stage embeddings"
        )
        sys.exit(1)

    # Calcula similaridade artigo–domínio
    log.info("Calculando similaridade artigo–domínio para %d artigos...", len(df))
    df, _domain_embeddings = compute_all_similarities(
        df,
        article_embeddings=embeddings_dict,
        domain_descriptions=DOMAIN_DESCRIPTIONS,
        model_name=config.EMBEDDING_MODEL,
        fallback=config.EMBEDDING_FALLBACK,
        id_column="id",
    )

    # Salva similaridades
    sim_cols = ["sim_max"] + [d for d in DOMAINS]
    for col in sim_cols:
        if col in df.columns:
            log.info(
                "  Similaridade média %s: %.4f", col, df[col].mean()
            )

    # --- Filtro de similaridade semântica ---
    df_valid = df[df["sim_max"] >= SIM_THRESHOLD].copy()
    df_skip = df[df["sim_max"] < SIM_THRESHOLD].copy()

    log.info(
        "Filtro de similaridade: %d artigos com sim_max ≥ %.0f%% → LLM | "
        "%d artigos com sim_max < %.0f%% → S_Quali = 0",
        len(df_valid), SIM_THRESHOLD * 100,
        len(df_skip), SIM_THRESHOLD * 100,
    )

    # Artigos abaixo do limiar recebem S_Quali = 0 automaticamente
    if not df_skip.empty:
        df_skip["S_Quali"] = 0.0

    # Executa ensemble COM embeddings apenas nos artigos que passaram o filtro
    if not df_valid.empty:
        responses = run_ensemble_with_embeddings(
            df_valid,
            models=config.LLM_MODELS,
            prompt_path=config.PROMPT_RELEVANCE_EMBED,
            responses_dir=config.RESPONSES_EMBED_DIR,
            id_column="id",
            title_column="title",
            abstract_column="abstract",
        )

        # Agrega scores brutos por artigo
        scores_map: dict[str, list[float]] = {}
        for r in responses:
            scores_map.setdefault(str(r.paper_id), []).append(r.score)

        squali_map: dict[str, float] = {}
        for paper_id, scores in scores_map.items():
            median_val = float(np.median(scores))
            filtered = [s for s in scores if abs(s - median_val) <= 2.0]
            squali_map[paper_id] = round(float(np.mean(filtered)) if filtered else median_val, 2)

        df_valid["S_Quali"] = df_valid["id"].apply(
            lambda pid: squali_map.get(str(pid), 0.0)
        )
    else:
        log.warning("Nenhum artigo atingiu o limiar mínimo de similaridade (%.0f%%).", SIM_THRESHOLD * 100)

    # Recombina os DataFrames
    df = pd.concat([df_valid, df_skip], ignore_index=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_PREPROCESSED, index=False)

    n_aprovados = int((df["S_Quali"] >= config.NOTA_CORTE).sum())
    n_rejeitados = len(df) - n_aprovados
    n_avaliados = len(df_valid)
    n_pulados = len(df_skip)
    elapsed = time.time() - t0
    log.info(
        "Ensemble COM embeddings concluído: %d avaliados + %d pulados = %d artigos em %.1fs | "
        "Aprovados: %d (≥%.1f) | Rejeitados: %d",
        n_avaliados, n_pulados, len(df),
        elapsed, n_aprovados, config.NOTA_CORTE, n_rejeitados,
    )
    log.info("Cache (embed): %s", config.RESPONSES_EMBED_DIR)
    log.info("=" * 60)

    return df


def _load_article_embeddings() -> Optional[dict]:
    """Carrega embeddings de artigos do cache.

    Returns:
        Dicionário {paper_id: np.ndarray} ou None.
    """
    cache_path = config.EMBEDDINGS_CACHE
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        log.warning("Erro ao carregar embeddings: %s", e)
        return None


def stage_agreement(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Etapa 4b — Agregação e estatísticas de concordância entre LLMs.

    Calcula mediana, remove outliers (|score - mediana| > 2), produz SQuali.
    Calcula desvio padrão, variância, correlação de Spearman e ICC.

    Entrada: cache/responses/*.json
    Saída:   outputs/agreement.csv

    Args:
        df: DataFrame opcional para merge do S_Quali.

    Returns:
        DataFrame com coluna 'S_Quali' atualizada.
    """
    from agreement import compute_agreement_stats, compute_squali, export_agreement
    from llm import LLMResponse

    _print_stage_header(
        "AGREGAÇÃO",
        "Mediana, remoção de outliers, Spearman, ICC",
    )
    t0 = time.time()

    # Reconstrói respostas do cache
    responses = _load_responses_from_cache(config.RESPONSES_DIR, config.LLM_MODELS)

    if not responses:
        log.error("Nenhuma resposta de LLM encontrada em %s.", config.RESPONSES_DIR)
        log.error("Execute a etapa 'llm' primeiro: python pipeline.py --stage llm")
        if df is not None:
            return df
        return _load_or_die(config.DATASET_PREPROCESSED, "llm")

    # Calcula SQuali por artigo
    agreement_df = compute_squali(responses, outlier_threshold=2.0)
    export_agreement(agreement_df, config.AGREEMENT_CSV)

    # Estatísticas globais
    agreement_stats = compute_agreement_stats(responses, model_names=config.LLM_MODELS)
    log.info("ICC: %.4f | Spearman médio: %.4f",
             agreement_stats.get("icc", float("nan")),
             agreement_stats.get("spearman_mean", float("nan")))

    # Merge SQuali no DataFrame
    squali_map = dict(zip(agreement_df["paper_id"], agreement_df["squali"]))
    if df is None:
        df = _load_or_die(config.DATASET_PREPROCESSED, "llm")
    df["S_Quali"] = df["id"].map(squali_map).fillna(0.0)
    df.to_csv(config.DATASET_PREPROCESSED, index=False)

    _print_stage_footer(len(df), t0, "agreement.csv")
    return df


def stage_bibliometria(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Etapa 5 — Cálculo de scores bibliométricos (quantitativos).

    Normaliza citações (C_Norm, V_Norm), calcula S_Quant e Score_Global.
    PRESERVADO do script original — NÃO ALTERAR.

    Entrada: outputs/dataset_preprocessed.csv (com S_Quali)
    Saída:   DataFrame com C_Norm, V_Norm, S_Quant, Score_Global

    Args:
        df: DataFrame opcional. Deve conter 'citations', 'S_Quali'.

    Returns:
        DataFrame com scores quantitativos e Score_Global.
    """
    _print_stage_header(
        "BIBLIOMETRIA",
        f"Citações (peso {config.C_WEIGHT}) + Citações influentes (peso {config.V_WEIGHT})",
    )
    t0 = time.time()

    if df is None:
        df = _load_or_die(config.DATASET_PREPROCESSED, "llm (ou agreement)")

    if "S_Quali" not in df.columns:
        log.error("Coluna 'S_Quali' não encontrada. Execute 'llm' ou 'agreement' primeiro.")
        sys.exit(1)

    # Segrega por nota de corte
    df_rejeitados = df[df["S_Quali"] < config.NOTA_CORTE].copy()
    df_interesse = df[df["S_Quali"] >= config.NOTA_CORTE].copy()
    log.info("Artigos acima da nota de corte (%.1f): %d", config.NOTA_CORTE, len(df_interesse))
    log.info("Artigos rejeitados: %d", len(df_rejeitados))

    if df_interesse.empty:
        log.warning("Nenhum artigo superou a nota de corte. Abortando.")
        return df

    df_interesse = calcular_scores_quantitativos(df_interesse)

    # Salva ambos para a próxima etapa (ranking precisa dos rejeitados também)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_result = pd.concat([df_interesse, df_rejeitados], ignore_index=True)
    df_result.to_csv(config.DATASET_PREPROCESSED, index=False)

    _print_stage_footer(len(df_interesse), t0, "(S_Quant, Score_Global)")
    return df_interesse


def stage_ranking(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Etapa 6 — Ranking final, TOP_K, perfis semânticos e exportação.

    Ordena artigos por Score_Global, seleciona TOP_K, extrai perfis
    semânticos e gera todos os outputs finais.

    Entrada: DataFrame com Score_Global (da etapa bibliometria)
    Saída:   outputs/ranking.csv
             outputs/dataset_final.xlsx
             outputs/experiment.json

    Args:
        df: DataFrame opcional com Score_Global, S_Quali, etc.

    Returns:
        DataFrame do TOP_K.
    """
    # from llm import semantic_profile  # DESABILITADO — Gemini API fora do ar

    _print_stage_header(
        "RANKING + OUTPUTS",
        f"TOP_{config.TOP_K} | "
        f"Score Global = {config.QUALI_WEIGHT}×SQuali + {config.QUANT_WEIGHT}×SQuant",
    )
    t0 = time.time()

    if df is None:
        # Tenta carregar do bibliometria. Se falhar, carrega do preprocess
        df = _load_or_die(config.DATASET_PREPROCESSED, "bibliometria")
        if "Score_Global" not in df.columns:
            log.error(
                "Coluna 'Score_Global' não encontrada. Execute 'bibliometria' primeiro."
            )
            sys.exit(1)
        # Segrega novamente
        df = df[df["S_Quali"] >= config.NOTA_CORTE].copy()

    # Ranking
    df_ranking = df.sort_values(by="Score_Global", ascending=False)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_ranking.to_csv(config.RANKING_CSV, index=False)
    log.info("Ranking completo: %s (%d artigos)", config.RANKING_CSV, len(df_ranking))

    # TOP_K
    df_top = df_ranking.head(config.TOP_K).copy()
    log.info("TOP_%d selecionados:", config.TOP_K)
    for i, (_, row) in enumerate(df_top.iterrows(), 1):
        log.info(
            "  %d. [Score: %.2f] %s",
            i,
            row["Score_Global"],
            str(row.get("title", ""))[:80],
        )

    # Perfis semânticos do TOP_K — DESABILITADO (Gemini 404)
    # log.info("Extraindo perfis semânticos...")
    # perfis = []
    # for _, row in df_top.iterrows():
    #     perfil = semantic_profile(
    #         abstract=str(row.get("abstract", "")),
    #         prompt_path=config.PROMPT_SEMANTIC_PROFILE,
    #     )
    #     perfis.append(perfil)
    # df_top["Perfil_Keywords"] = perfis
    log.info("Perfis semânticos: DESABILITADOS (Gemini API indisponível)")
    df_top["Perfil_Keywords"] = ""

    # Recupera rejeitados
    df_all = _load_or_die(config.DATASET_PREPROCESSED, "llm")
    df_rejeitados = df_all[df_all["S_Quali"] < config.NOTA_CORTE].copy()

    # Exportações finais
    export_results(df_top, df_ranking, df_rejeitados)
    export_experiment_json(t0, {
        "total_coletados": len(df_all),
        "aprovados": len(df_ranking),
        "rejeitados": len(df_rejeitados),
        "agreement": {},
    })

    _print_stage_footer(len(df_ranking), t0, "ranking.csv, dataset_final.xlsx, experiment.json")
    return df_top


def executar_pipeline() -> None:
    """Executa o pipeline completo (todas as etapas em sequência).

    Os dados são passados em memória entre etapas para máxima eficiência.
    Cada etapa também salva seu output intermediário em disco.
    """
    start_time = time.time()
    log.info("=" * 60)
    log.info("=== PIPELINE COMPLETO DE AUDITORIA ARES ===")
    log.info("Modo produção: %s | TOP_K: %d | Nota de corte: %.1f",
             config.MODO_PRODUCAO, config.TOP_K, config.NOTA_CORTE)
    log.info("=" * 60)

    # Etapa 1: Coleta
    df = stage_coleta()

    if df.empty:
        log.error("Dataset vazio. Abortando.")
        return

    total_coletados = len(df)

    # Etapa 2: Pré-processamento
    df = stage_preprocess(df)

    # Etapa 3: Embeddings
    df = stage_embeddings(df)

    # Etapa 4: Ensemble de LLMs
    df = stage_llm(df)

    # Etapa 4b: Agregação
    df = stage_agreement(df)

    # Etapa 5: Bibliometria
    df_interesse = stage_bibliometria(df)

    if df_interesse.empty or "Score_Global" not in df_interesse.columns:
        log.warning("Nenhum artigo após filtro. Finalizando.")
        elapsed = time.time() - start_time
        log.info("Tempo total: %.2f minutos.", elapsed / 60)
        return

    # Etapa 6: Ranking + Outputs
    stage_ranking(df_interesse)

    # Conclusão
    elapsed = time.time() - start_time
    log.info("=" * 60)
    log.info("=== PIPELINE CONCLUÍDO ===")
    log.info("Total coletado: %d | Aprovados: %d | TOP_%d selecionados",
             total_coletados, len(df_interesse), config.TOP_K)
    log.info("Tempo total: %.2f minutos.", elapsed / 60)
    log.info("Outputs: %s", config.OUTPUT_DIR)
    log.info("=" * 60)


# ==============================================================================
# CLI
# ==============================================================================

# Mapeamento nome → (função, descrição curta, descrição longa)
STAGES: dict[str, tuple] = {
    "coleta": (
        lambda: stage_coleta(),
        "Coleta DBLP + OpenAlex",
        "Busca artigos da conferência ARES via DBLP (validação de venue) e "
        "enriquece metadados via OpenAlex (abstracts, citações, autores). "
        "Usa Exponential Backoff e sessões HTTP persistentes.",
    ),
    "preprocess": (
        lambda: stage_preprocess(),
        "Pré-processamento NLP",
        "Normalização de texto: Unicode NFKD, remoção de tags HTML, "
        "colapso de múltiplos espaços, remoção de caracteres de controle. "
        "NÃO remove stopwords, NÃO aplica stemming ou lemmatization. "
        "Adiciona coluna 'clean_text'.",
    ),
    "embeddings": (
        lambda: stage_embeddings(),
        "Embeddings (SentenceTransformer)",
        f"Gera embeddings via {config.EMBEDDING_MODEL} "
        f"(fallback: {config.EMBEDDING_FALLBACK}). "
        "Cache automático em cache/embeddings.pkl — não recalcula se já existir. "
        "Adiciona coluna 'embedding_id' (índice numérico).",
    ),
    "llm": (
        lambda: stage_llm(),
        "Ensemble de LLMs (filtro sim_max ≥ 60%)",
        f"Avalia cada artigo com {', '.join(config.LLM_MODELS)}. "
        "Filtra por similaridade semântica — apenas artigos com sim_max ≥ 60% "
        "são enviados ao ensemble. Os demais recebem S_Quali = 0. "
        "Requer etapa 'embeddings' executada antes. "
        "Cada resposta é cacheada em cache/responses/. "
        "Com MODO_PRODUCAO=False, usa mocks determinísticos (sem custo). "
        "Calcula S_Quali preliminar (mediana + remoção de outliers).",
    ),
    "llm-embed": (
        lambda: stage_llm_embed(),
        "Ensemble de LLMs COM embeddings (filtro sim_max ≥ 60%)",
        f"Avalia cada artigo com {', '.join(config.LLM_MODELS)}, "
        "enriquecendo o prompt com a similaridade semântica artigo–domínio "
        "(cosseno entre embedding do artigo e embeddings dos 5 domínios: "
        "CPS, Blockchain, IoT, Fault Tolerance, Distributed Systems). "
        "Filtra por sim_max ≥ 60% — artigos abaixo do limiar recebem S_Quali = 0. "
        "Cache separado em cache/responses_embed/. "
        "REQUER etapa 'embeddings' executada antes.",
    ),
    "agreement": (
        lambda: stage_agreement(),
        "Agregação e concordância",
        "Calcula S_Quali final, desvio padrão, variância, correlação de Spearman "
        "e ICC (Intraclass Correlation Coefficient) entre os modelos. "
        "Remove outliers com |score - mediana| > 2. "
        "Salva agreement.csv.",
    ),
    "bibliometria": (
        lambda: stage_bibliometria(),
        "Bibliometria (Score Quantitativo)",
        "Calcula C_Norm e V_Norm (citações normalizadas), S_Quant "
        f"(pesos: {config.C_WEIGHT} citações + {config.V_WEIGHT} influentes) "
        f"e Score_Global ({config.QUALI_WEIGHT}×SQuali + {config.QUANT_WEIGHT}×SQuant). "
        "PRESERVADO do script original.",
    ),
    "ranking": (
        lambda: stage_ranking(),
        "Ranking + TOP_K + Outputs finais",
        f"Ordena por Score_Global, seleciona TOP_{config.TOP_K}, "
        "extrai perfis semânticos dos melhores artigos e gera todos os outputs: "
        "ranking.csv, dataset_final.xlsx, experiment.json.",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos da CLI."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Pipeline ARES — Mineração científica modular de literatura",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(),
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        metavar="ETAPA",
        help="Etapa a executar. Use 'all' para pipeline completo. "
             "Use --list para ver todas as etapas disponíveis.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista todas as etapas disponíveis com descrição e sai.",
    )

    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        metavar="N",
        help="Limite de artigos a coletar (apenas para --stage coleta ou all). "
             f"Padrão: {config.LIMITE_COLETA}.",
    )

    return parser


def _build_epilog() -> str:
    """Constrói o texto de exemplos para o help."""
    return (
        "Exemplos:\n"
        "  python pipeline.py                          Pipeline completo\n"
        "  python pipeline.py --stage coleta           Apenas coleta DBLP + OpenAlex\n"
        "  python pipeline.py --stage preprocess       Apenas pré-processamento\n"
        "  python pipeline.py --stage embeddings       Apenas embeddings\n"
        "  python pipeline.py --stage llm              Apenas ensemble de LLMs (SEM embeddings)\n"
        "  python pipeline.py --stage llm-embed        Apenas ensemble de LLMs (COM embeddings)\n"
        "  python pipeline.py --stage agreement        Apenas agregação\n"
        "  python pipeline.py --stage bibliometria     Apenas bibliometria\n"
        "  python pipeline.py --stage ranking          Apenas ranking + outputs\n"
        "  python pipeline.py --list                   Listar etapas disponíveis\n"
    )


def _print_stage_header(stage_name: str, detail: str) -> None:
    """Imprime cabeçalho visual de etapa."""
    log.info("")
    log.info("=" * 60)
    log.info("  ETAPA: %s", stage_name)
    log.info("  %s", detail)
    log.info("=" * 60)


def _print_stage_footer(n_items: int, t0: float, output_label: str) -> None:
    """Imprime rodapé de etapa com estatísticas."""
    elapsed = time.time() - t0
    log.info("---")
    log.info("Concluído: %d itens processados em %.1fs", n_items, elapsed)
    log.info("Output: %s", output_label)
    log.info("=" * 60)


def _load_or_die(filepath: Path, stage_name: str) -> pd.DataFrame:
    """Carrega arquivo de etapa anterior ou aborta com mensagem clara.

    Args:
        filepath: Caminho do arquivo a carregar.
        stage_name: Nome da etapa que deveria ter gerado o arquivo.

    Returns:
        DataFrame carregado.

    Raises:
        SystemExit: Se o arquivo não existir.
    """
    if not filepath.exists():
        log.error(
            "Arquivo não encontrado: %s\n"
            "Execute a etapa '%s' primeiro:\n"
            "  python pipeline.py --stage %s",
            filepath,
            stage_name,
            stage_name,
        )
        sys.exit(1)

    if filepath.suffix == ".pkl":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict) and "dataframe" in data:
                return data["dataframe"]
            return data
    elif filepath.suffix == ".csv":
        return pd.read_csv(filepath)
    else:
        return pd.read_csv(filepath)


def _load_responses_from_cache(
    responses_dir: Path, models: list[str]
) -> list:
    """Reconstrói lista de LLMResponse a partir do cache em disco.

    Args:
        responses_dir: Diretório com os arquivos JSON de cache.
        models: Lista de nomes de modelos.

    Returns:
        Lista de objetos LLMResponse.
    """
    from llm import LLMResponse

    if not responses_dir.exists():
        return []

    responses: list[LLMResponse] = []
    for cache_file in sorted(responses_dir.glob("*.json")):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extrai paper_id e model do nome do arquivo: {paper_id}_{model}.json
            stem = cache_file.stem
            # O paper_id pode conter underscores, então pegamos o model do final
            matched = False
            for model in models:
                if stem.endswith(f"_{model}"):
                    paper_id = stem[: -(len(model) + 1)]
                    matched = True
                    break
            if not matched:
                # Fallback: usa o nome completo como paper_id
                paper_id = stem

            responses.append(
                LLMResponse(
                    score=float(data.get("score", 0)),
                    confidence=float(data.get("confidence", 0)),
                    domains=data.get("domains", []),
                    reason=data.get("reason", ""),
                    model=data.get("model", model if matched else "unknown"),
                    paper_id=paper_id,
                    timestamp=data.get("timestamp", ""),
                )
            )
        except Exception as e:
            log.warning("Erro ao ler cache %s: %s", cache_file, e)

    log.info("Respostas carregadas do cache: %d", len(responses))
    return responses


def _cmd_list_stages() -> None:
    """Lista todas as etapas disponíveis com descrições detalhadas."""
    print()
    print("=" * 70)
    print("  PIPELINE ARES — Etapas Disponíveis")
    print("=" * 70)
    print()

    stage_order = ["coleta", "preprocess", "embeddings", "llm", "llm-embed", "agreement", "bibliometria", "ranking"]

    for i, key in enumerate(stage_order, 1):
        _, name, desc = STAGES[key]
        print(f"  [{i}] {name}")
        print(f"      Comando:  python pipeline.py --stage {key}")
        print(f"      {desc}")
        print()

    print("  [0] Pipeline completo")
    print("      Comando:  python pipeline.py")
    print("      Executa todas as etapas em sequência, com dados em memória.")
    print()
    print("=" * 70)
    print(f"  Configuração atual: TOP_K={config.TOP_K} | "
          f"MODO_PRODUCAO={config.MODO_PRODUCAO} | "
          f"LLMs={config.LLM_MODELS}")
    print("=" * 70)
    print()


def _run_single_stage(stage_key: str, limite: Optional[int] = None) -> None:
    """Executa uma única etapa do pipeline.

    Args:
        stage_key: Chave da etapa no dicionário STAGES.
        limite: Limite de artigos (apenas para etapa coleta).
    """
    if stage_key not in STAGES:
        log.error("Etapa desconhecida: '%s'", stage_key)
        log.info("Etapas disponíveis: %s", ", ".join(STAGES.keys()))
        log.info("Use --list para ver detalhes de cada etapa.")
        sys.exit(1)

    # Etapa de coleta permite override de limite
    if stage_key == "coleta" and limite is not None:
        func = lambda: stage_coleta(limite=limite)
    else:
        func = STAGES[stage_key][0]

    log.info("Modo: etapa única | MODO_PRODUCAO=%s", config.MODO_PRODUCAO)
    result = func()

    if isinstance(result, pd.DataFrame):
        log.info("Etapa concluída. %d linhas no DataFrame.", len(result))


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # --list: mostrar etapas e sair
    if args.list:
        _cmd_list_stages()
        sys.exit(0)

    # Pipeline completo
    if args.stage == "all":
        # Override de limite se especificado
        if args.limite is not None:
            old_limite = config.LIMITE_COLETA
            config.LIMITE_COLETA = args.limite
            log.info("Limite de coleta alterado: %d → %d", old_limite, args.limite)
        executar_pipeline()

    # Etapa única
    else:
        _run_single_stage(args.stage, limite=args.limite)
