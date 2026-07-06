# -*- coding: utf-8 -*-
"""Módulo de pré-processamento e normalização de texto.

Aplica limpeza textual para preparar os campos de título, abstract e keywords
para as fases posteriores de embeddings e avaliação por LLMs.

Regras:
- Normalização Unicode (NFKD)
- Remoção de tags HTML
- Remoção de múltiplos espaços
- Remoção de caracteres de controle
- strip()

NÃO remove stopwords, NÃO faz stemming, NÃO faz lemmatization.
"""

import html
import re
import unicodedata
from typing import Optional

import pandas as pd


def normalize_text(text: Optional[str]) -> str:
    """Aplica normalização completa a uma string.

    Pipeline:
    1. Normalização Unicode (NFKD) + remoção de acentos combinados
    2. Decodificação de entidades HTML
    3. Remoção de tags HTML
    4. Remoção de caracteres de controle (0x00-0x1f, 0x7f-0x9f), exceto \t, \n
    5. Colapso de múltiplos espaços em um único espaço
    6. strip()

    Args:
        text: String de entrada. Se None ou vazia, retorna string vazia.

    Returns:
        Texto normalizado.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Normalização Unicode NFKD (decompõe caracteres acentuados)
    text = unicodedata.normalize("NFKD", text)

    # 2. Decodifica entidades HTML (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(text)

    # 3. Remove tags HTML
    text = re.sub(r"<[^>]+>", "", text)

    # 4. Remove caracteres de controle, mantendo tabs e newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # 5. Substitui tabs e newlines por espaço, colapsa múltiplos espaços
    text = text.replace("\t", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # 6. Remove espaços nas bordas
    text = text.strip()

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    title_col: str = "title",
    abstract_col: str = "abstract",
    keywords_col: Optional[str] = None,
) -> pd.DataFrame:
    """Aplica normalização de texto ao DataFrame e adiciona coluna clean_text.

    A coluna clean_text é construída como:
        normalize_text(title) + " " + normalize_text(abstract)
    Se keywords_col for fornecida, as keywords também são concatenadas.

    Args:
        df: DataFrame com os dados brutos da coleta.
        title_col: Nome da coluna de título.
        abstract_col: Nome da coluna de abstract.
        keywords_col: Nome da coluna de keywords (opcional).

    Returns:
        DataFrame com a coluna adicional 'clean_text'.
    """
    df = df.copy()

    # Normaliza título
    df["_norm_title"] = df[title_col].apply(normalize_text)

    # Normaliza abstract
    df["_norm_abstract"] = df[abstract_col].apply(normalize_text)

    # Constrói clean_text
    df["clean_text"] = df["_norm_title"] + " " + df["_norm_abstract"]

    # Adiciona keywords se disponíveis
    if keywords_col and keywords_col in df.columns:
        df["_norm_keywords"] = df[keywords_col].apply(normalize_text)
        df["clean_text"] = (
            df["clean_text"] + " " + df["_norm_keywords"]
        )

    # Remove colunas auxiliares
    df = df.drop(columns=["_norm_title", "_norm_abstract"], errors="ignore")
    if "_norm_keywords" in df.columns:
        df = df.drop(columns=["_norm_keywords"], errors="ignore")

    return df
