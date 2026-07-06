# -*- coding: utf-8 -*-
"""Módulo de geração de embeddings via SentenceTransformer.

Utiliza o modelo BAAI/bge-large-en-v1.5 como padrão, com fallback para
all-mpnet-base-v2. Implementa cache automático em disco para evitar
recomputação desnecessária.

NUNCA armazena os vetores de embedding no DataFrame — apenas adiciona
a coluna 'embedding_id' como índice para o dicionário de embeddings.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_sentence_transformer():
    """Importa SentenceTransformer (lazy import para evitar carga desnecessária)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers não está instalado. "
            "Instale com: pip install sentence-transformers"
        )


def load_model(
    model_name: str = "BAAI/bge-large-en-v1.5",
    fallback: str = "all-mpnet-base-v2",
) -> "SentenceTransformer":
    """Carrega o modelo SentenceTransformer, com fallback automático.

    Args:
        model_name: Nome do modelo primário no HuggingFace Hub.
        fallback: Nome do modelo de fallback se o primário falhar.

    Returns:
        Instância de SentenceTransformer carregada e pronta para uso.

    Raises:
        RuntimeError: Se ambos os modelos falharem ao carregar.
    """
    SentenceTransformer = _get_sentence_transformer()

    for name in [model_name, fallback]:
        try:
            logger.info("Carregando modelo de embeddings: %s", name)
            model = SentenceTransformer(name)
            logger.info("Modelo %s carregado com sucesso.", name)
            return model
        except Exception as e:
            if name == model_name:
                logger.warning(
                    "Falha ao carregar %s: %s. Tentando fallback %s...",
                    model_name,
                    e,
                    fallback,
                )
            else:
                raise RuntimeError(
                    f"Falha ao carregar ambos os modelos de embedding: "
                    f"{model_name} e {fallback}"
                ) from e

    # unreachable, mas o type checker não sabe disso
    raise RuntimeError("Nenhum modelo de embedding disponível.")


def load_embeddings_cache(cache_path: Path) -> Optional[dict]:
    """Carrega embeddings cacheados do disco.

    Args:
        cache_path: Caminho para o arquivo .pkl de cache.

    Returns:
        Dicionário {paper_id: np.ndarray} ou None se o cache não existir.
    """
    if not cache_path.exists():
        logger.info("Cache de embeddings não encontrado em %s.", cache_path)
        return None

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        logger.info(
            "Cache de embeddings carregado: %d entradas de %s.",
            len(cache),
            cache_path,
        )
        return cache
    except Exception as e:
        logger.warning("Falha ao carregar cache de embeddings: %s. Recomputando.", e)
        return None


def save_embeddings_cache(cache_path: Path, embeddings: dict) -> None:
    """Salva embeddings em disco no formato pickle.

    Args:
        cache_path: Caminho onde salvar o arquivo .pkl.
        embeddings: Dicionário {paper_id: np.ndarray}.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(
        "Cache de embeddings salvo: %d entradas em %s.",
        len(embeddings),
        cache_path,
    )


def compute_embeddings(
    df: pd.DataFrame,
    model_name: str = "BAAI/bge-large-en-v1.5",
    fallback: str = "all-mpnet-base-v2",
    cache_path: Optional[Path] = None,
    text_column: str = "clean_text",
) -> tuple[pd.DataFrame, dict]:
    """Gera embeddings para cada artigo no DataFrame, com cache em disco.

    Se o cache existir, apenas artigos sem embedding são processados.
    Adiciona a coluna 'embedding_id' (índice inteiro) ao DataFrame retornado.
    Os vetores de embedding NUNCA são armazenados no DataFrame.

    Args:
        df: DataFrame com coluna de texto limpo.
        model_name: Modelo SentenceTransformer primário.
        fallback: Modelo de fallback.
        cache_path: Caminho do arquivo .pkl de cache. Se None, não usa cache.
        text_column: Nome da coluna com texto para gerar embeddings.

    Returns:
        Tuple de (DataFrame com coluna embedding_id, dicionário de embeddings).
    """
    SentenceTransformer = _get_sentence_transformer()
    df = df.copy()

    # Garante que existe uma coluna de ID única para cache
    if "id" not in df.columns:
        df["_paper_idx"] = range(len(df))
        id_column = "_paper_idx"
    else:
        id_column = "id"

    # Tenta carregar cache
    embeddings: dict = {}
    if cache_path:
        cached = load_embeddings_cache(cache_path)
        if cached is not None:
            embeddings = cached

    # Identifica artigos que precisam de embedding
    missing_ids = []
    for _, row in df.iterrows():
        paper_id = str(row[id_column])
        if paper_id not in embeddings:
            missing_ids.append(paper_id)

    if missing_ids:
        logger.info(
            "Gerando embeddings para %d artigos (de %d total)...",
            len(missing_ids),
            len(df),
        )

        # Carrega modelo
        model = load_model(model_name, fallback)

        # Extrai textos para os artigos faltantes
        df_missing = df[df[id_column].astype(str).isin(missing_ids)]
        texts = df_missing[text_column].fillna("").tolist()

        # Gera embeddings em batch
        new_embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Armazena no dicionário
        for paper_id, embedding in zip(
            df_missing[id_column].astype(str), new_embeddings
        ):
            embeddings[paper_id] = embedding

        # Atualiza cache se configurado
        if cache_path:
            save_embeddings_cache(cache_path, embeddings)
    else:
        logger.info(
            "Todos os %d artigos já possuem embeddings cacheados.", len(df)
        )

    # Cria mapeamento paper_id -> índice numérico sequencial
    id_to_idx = {}
    for idx, paper_id in enumerate(embeddings.keys()):
        id_to_idx[paper_id] = idx

    # Adiciona coluna embedding_id ao DataFrame
    df["embedding_id"] = df[id_column].astype(str).map(id_to_idx)

    # Verifica se algum artigo ficou sem embedding_id
    null_count = df["embedding_id"].isna().sum()
    if null_count > 0:
        logger.warning(
            "%d artigos ficaram sem embedding_id. Verifique os dados de entrada.",
            null_count,
        )

    return df, embeddings


# ---------------------------------------------------------------------------
# Similaridade artigo–domínio (embeddings enriquecem a avaliação LLM)
# ---------------------------------------------------------------------------


def compute_domain_similarity(
    article_embedding: np.ndarray,
    domain_embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """Calcula similaridade por cosseno entre um artigo e cada domínio de pesquisa.

    Como os embeddings já estão normalizados (normalize_embeddings=True),
    a similaridade do cosseno é simplesmente o produto escalar (dot product).

    Args:
        article_embedding: Vetor de embedding do artigo (já normalizado).
        domain_embeddings: Dicionário {nome_domínio: vetor_embedding}.

    Returns:
        Dicionário {nome_domínio: similaridade (0-1)}.
    """
    similarities: dict[str, float] = {}
    for domain_name, domain_vec in domain_embeddings.items():
        sim = float(np.dot(article_embedding, domain_vec))
        similarities[domain_name] = round(max(0.0, min(1.0, sim)), 4)
    return similarities


def build_domain_embeddings(
    model: "SentenceTransformer",
    domain_descriptions: dict[str, str],
) -> dict[str, np.ndarray]:
    """Cria embeddings para as descrições textuais de cada domínio de pesquisa.

    Args:
        model: Instância de SentenceTransformer carregada.
        domain_descriptions: Dicionário {nome_domínio: texto_descritivo}.

    Returns:
        Dicionário {nome_domínio: vetor_embedding normalizado}.
    """
    logger.info("Gerando embeddings para %d domínios de pesquisa...", len(domain_descriptions))
    domain_embeddings: dict[str, np.ndarray] = {}

    for name, description in domain_descriptions.items():
        vec = model.encode([description], normalize_embeddings=True)[0]
        domain_embeddings[name] = vec
        logger.debug("  Domínio '%s': embedding gerado (%d dimensões)", name, len(vec))

    return domain_embeddings


def compute_all_similarities(
    df: pd.DataFrame,
    article_embeddings: dict,
    domain_descriptions: dict[str, str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    fallback: str = "all-mpnet-base-v2",
    id_column: str = "id",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Calcula similaridade de cada artigo com cada domínio de pesquisa.

    Adiciona colunas ao DataFrame:
      - sim_max: maior similaridade entre todos os domínios
      - sim_{dominio}: similaridade com cada domínio específico

    Args:
        df: DataFrame com coluna de ID.
        article_embeddings: Dicionário {paper_id: np.ndarray}.
        domain_descriptions: Dicionário {nome_domínio: texto_descritivo}.
        model_name: Modelo para gerar embeddings de domínio.
        fallback: Modelo de fallback.
        id_column: Coluna com ID do artigo.

    Returns:
        Tuple de (DataFrame com colunas de similaridade, dicionário de embeddings de domínio).
    """
    model = load_model(model_name, fallback)
    domain_embeddings = build_domain_embeddings(model, domain_descriptions)

    df = df.copy()
    similarities_list: list[dict[str, float]] = []

    for _, row in df.iterrows():
        paper_id = str(row[id_column])
        article_vec = article_embeddings.get(paper_id)

        if article_vec is None:
            logger.warning("Embedding não encontrado para artigo %s. Similaridades = 0.", paper_id)
            sims = {name: 0.0 for name in domain_descriptions}
            sims["sim_max"] = 0.0
        else:
            sims = compute_domain_similarity(article_vec, domain_embeddings)
            sims["sim_max"] = max(sims.values()) if sims else 0.0

        similarities_list.append(sims)

    # Adiciona colunas de similaridade ao DataFrame
    sim_df = pd.DataFrame(similarities_list)

    for col in sim_df.columns:
        df[col] = sim_df[col].values

    # Ordena colunas: sim_max primeiro, depois domínios
    domain_cols = [c for c in sim_df.columns if c != "sim_max"]
    ordered = ["sim_max"] + sorted(domain_cols)
    for col in ordered:
        if col not in df.columns:
            df[col] = sim_df.get(col, 0.0)

    logger.info(
        "Similaridades calculadas. SimMax médio: %.4f",
        df["sim_max"].mean(),
    )
    return df, domain_embeddings
