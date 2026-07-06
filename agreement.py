# -*- coding: utf-8 -*-
"""Módulo de agregação e estatísticas de concordância entre LLMs.

Recebe as respostas do ensemble de LLMs e calcula:
- SQuali (escore qualitativo após remoção de outliers)
- Mediana, média, desvio padrão, variância
- Correlação de Spearman entre rankings dos modelos
- ICC (Intraclass Correlation Coefficient)

Produz o arquivo agreement.csv com as estatísticas por artigo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import config

logger = logging.getLogger(__name__)


@dataclass
class AgreementResult:
    """Resultado da agregação para um artigo.

    Attributes:
        paper_id: Identificador do artigo.
        scores: Lista de scores brutos de cada modelo.
        median: Mediana dos scores.
        filtered_scores: Scores após remoção de outliers.
        mean: Média dos scores filtrados.
        std_dev: Desvio padrão dos scores filtrados.
        variance: Variância dos scores filtrados.
        squali: Score qualitativo final (SQuali).
        num_models: Número total de modelos que avaliaram.
        num_outliers_removed: Quantos scores foram removidos como outliers.
    """

    paper_id: str
    scores: list[float]
    median: float
    filtered_scores: list[float]
    mean: float
    std_dev: float
    variance: float
    squali: float
    num_models: int
    num_outliers_removed: int


def compute_squali(
    responses: list,
    outlier_threshold: float = 2.0,
) -> pd.DataFrame:
    """Calcula SQuali e estatísticas de concordância para cada artigo.

    Para cada artigo:
    1. Extrai scores de todos os modelos
    2. Calcula a mediana
    3. Remove scores onde |score - median| > outlier_threshold
    4. Se houver scores restantes: SQuali = mean(filtered_scores)
       Senão: SQuali = median
    5. Calcula std_dev, variance

    Args:
        responses: Lista de objetos LLMResponse (de llm.py).
        outlier_threshold: Threshold para remoção de outliers (distância da mediana).

    Returns:
        DataFrame com uma linha por artigo e colunas:
        paper_id, median, mean, std_dev, variance, squali,
        num_models, num_outliers_removed, scores_raw
    """
    # Agrupa respostas por paper_id
    from collections import defaultdict

    paper_scores: dict[str, list[float]] = defaultdict(list)

    for r in responses:
        paper_scores[r.paper_id].append(r.score)

    results: list[dict] = []

    for paper_id, scores in paper_scores.items():
        scores_arr = np.array(scores, dtype=float)
        n_models = len(scores_arr)

        # Mediana
        median = float(np.median(scores_arr))

        # Remove outliers: |score - median| > outlier_threshold
        mask = np.abs(scores_arr - median) <= outlier_threshold
        filtered = scores_arr[mask].tolist()
        n_removed = n_models - len(filtered)

        # SQuali
        if filtered:
            squali = float(np.mean(filtered))
        else:
            squali = median

        # Estatísticas
        if len(filtered) > 1:
            std_dev = float(np.std(filtered, ddof=1))
            variance = float(np.var(filtered, ddof=1))
        else:
            std_dev = 0.0
            variance = 0.0

        results.append(
            {
                "paper_id": paper_id,
                "median": round(median, 4),
                "mean": round(float(np.mean(filtered)) if filtered else median, 4),
                "std_dev": round(std_dev, 4),
                "variance": round(variance, 4),
                "squali": round(squali, 2),
                "num_models": n_models,
                "num_outliers_removed": n_removed,
                "scores_raw": str([round(s, 2) for s in scores]),
            }
        )

    df = pd.DataFrame(results)
    logger.info(
        "SQuali calculado para %d artigos. Média SQuali: %.2f",
        len(df),
        df["squali"].mean(),
    )
    return df


def compute_spearman(scores_matrix: np.ndarray) -> np.ndarray:
    """Calcula matriz de correlação de Spearman entre rankings dos modelos.

    Args:
        scores_matrix: Matriz (n_artigos × n_modelos) com os scores.

    Returns:
        Matriz de correlação (n_modelos × n_modelos).
    """
    n_models = scores_matrix.shape[1]
    corr = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Remove NaNs pairwise
            valid = ~(np.isnan(scores_matrix[:, i]) | np.isnan(scores_matrix[:, j]))
            if valid.sum() >= 3:  # precisa de pelo menos 3 pontos
                r, _ = scipy_stats.spearmanr(
                    scores_matrix[valid, i], scores_matrix[valid, j]
                )
                corr[i, j] = r
                corr[j, i] = r
            else:
                corr[i, j] = np.nan
                corr[j, i] = np.nan

    return corr


def compute_icc(scores_matrix: np.ndarray) -> float:
    """Calcula ICC(2,k) — two-way random effects, average measures.

    Usa a fórmula: ICC = (MSR - MSE) / MSR
    onde MSR = mean square entre artigos, MSE = mean square residual.

    Args:
        scores_matrix: Matriz (n_artigos × n_modelos) com os scores.

    Returns:
        Valor de ICC no intervalo [-1/(k-1), 1], ou NaN se não for possível calcular.
    """
    n_articles, n_models = scores_matrix.shape

    if n_articles < 2 or n_models < 2:
        return float("nan")

    # Remove artigos com NaNs
    valid = ~np.isnan(scores_matrix).any(axis=1)
    if valid.sum() < 2:
        return float("nan")

    scores = scores_matrix[valid]

    # Grand mean
    grand_mean = scores.mean()

    # MSR: between-subjects
    article_means = scores.mean(axis=1)
    msr = n_models * np.sum((article_means - grand_mean) ** 2) / (len(article_means) - 1)

    # MSE: residual (within)
    residuals = scores - article_means.reshape(-1, 1) - scores.mean(axis=0) + grand_mean
    mse = np.sum(residuals**2) / ((len(article_means) - 1) * (n_models - 1))

    if mse == 0:
        return 1.0 if msr > 0 else 0.0

    icc = (msr - mse) / msr
    return float(icc)


def build_scores_matrix(responses: list, paper_ids: list[str], model_names: list[str]) -> np.ndarray:
    """Constrói matriz de scores (artigos × modelos) a partir das respostas.

    Args:
        responses: Lista de LLMResponse.
        paper_ids: Lista de IDs de artigos (ordem das linhas).
        model_names: Lista de nomes de modelos (ordem das colunas).

    Returns:
        Matriz numpy (n_artigos × n_modelos).
    """
    # Mapeia (paper_id, model) -> score
    score_map: dict[tuple[str, str], float] = {}
    for r in responses:
        score_map[(r.paper_id, r.model)] = r.score

    matrix = np.full((len(paper_ids), len(model_names)), np.nan)
    for i, pid in enumerate(paper_ids):
        for j, model in enumerate(model_names):
            matrix[i, j] = score_map.get((pid, model), np.nan)

    return matrix


def compute_agreement_stats(
    responses: list,
    model_names: Optional[list[str]] = None,
) -> dict:
    """Calcula estatísticas globais de concordância do ensemble.

    Args:
        responses: Lista de LLMResponse.
        model_names: Lista de nomes de modelos. Se None, extrai dos responses.

    Returns:
        Dicionário com spearman_matrix, icc, e métricas agregadas.
    """
    if model_names is None:
        model_names = sorted(set(r.model for r in responses))

    paper_ids = sorted(set(r.paper_id for r in responses))

    if not paper_ids or len(model_names) < 2:
        logger.warning("Dados insuficientes para estatísticas de concordância.")
        return {
            "spearman_matrix": None,
            "icc": float("nan"),
            "spearman_mean": float("nan"),
        }

    matrix = build_scores_matrix(responses, paper_ids, model_names)

    spearman_corr = compute_spearman(matrix)
    icc = compute_icc(matrix)

    # Média das correlações de Spearman (triangular superior)
    upper_tri = spearman_corr[np.triu_indices_from(spearman_corr, k=1)]
    spearman_mean = float(np.nanmean(upper_tri)) if len(upper_tri) > 0 else float("nan")

    logger.info("ICC: %.4f | Spearman médio: %.4f", icc, spearman_mean)

    return {
        "spearman_matrix": spearman_corr.tolist(),
        "icc": icc,
        "spearman_mean": spearman_mean,
        "model_names": model_names,
    }


def export_agreement(agreement_df: pd.DataFrame, output_path: Path) -> None:
    """Salva DataFrame de concordância em CSV.

    Args:
        agreement_df: DataFrame com estatísticas por artigo.
        output_path: Caminho do arquivo CSV de saída.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agreement_df.to_csv(output_path, index=False)
    logger.info("Agreement exportado: %s (%d linhas)", output_path, len(agreement_df))
