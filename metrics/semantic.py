from typing import Tuple, List
import re
import numpy as np


def _tokens(text: str):
    return re.findall(r"\w+", (text or "").lower())


# ============================================================
# MÉTRICAS PARA LA RESPUESTA DEL AGENTE (LLM)
# ============================================================

def compute_semantic_precision_recall(
    pred: str,
    expected: str,
    embedder,
    threshold: float = 0.75
) -> Tuple[float, float]:
    """
    Precision y Recall semánticos entre la respuesta generada
    por el LLM y la respuesta esperada.

    Se comparan embeddings de palabras (token-level).
    """

    pred_tokens = _tokens(pred)
    expected_tokens = _tokens(expected)

    if not pred_tokens and not expected_tokens:
        return 1.0, 1.0

    if not pred_tokens or not expected_tokens:
        return 0.0, 0.0

    pred_emb = np.asarray(
        embedder.encode(pred_tokens),
        dtype=np.float32
    )

    exp_emb = np.asarray(
        embedder.encode(expected_tokens),
        dtype=np.float32
    )

    sims = np.matmul(pred_emb, exp_emb.T)

    pred_max = sims.max(axis=1)
    exp_max = sims.max(axis=0)

    tp_pred = np.sum(pred_max >= threshold)
    tp_exp = np.sum(exp_max >= threshold)

    precision = tp_pred / len(pred_tokens)
    recall = tp_exp / len(expected_tokens)

    return round(float(precision), 4), round(float(recall), 4)


# ============================================================
# MÉTRICAS PARA EL RETRIEVER
# ============================================================

def compute_retrieval_precision_recall(
    retrieved_chunks: List[str],
    ground_truth_context: List[str],
    embedder,
    threshold: float = 0.80
) -> Tuple[float, float]:
    """
    Calcula Precision y Recall del Retriever.

    Cada chunk recuperado se compara contra cada contexto del
    Ground Truth mediante similitud coseno.

    TP:
        Chunk recuperado cuya similitud con algún contexto GT
        supera el umbral.

    FP:
        Chunk recuperado que no coincide con ningún contexto GT.

    FN:
        Contexto GT que no fue recuperado.

    Parameters
    ----------
    retrieved_chunks:
        Lista de textos recuperados por el retriever.

    ground_truth_context:
        Lista de contextos esperados.

    threshold:
        Similitud mínima para considerar un match.

    Returns
    -------
    (precision, recall)
    """

    if len(retrieved_chunks) == 0:
        return 0.0, 0.0

    if len(ground_truth_context) == 0:
        return 0.0, 0.0

    retrieved_emb = np.asarray(
        embedder.encode(retrieved_chunks),
        dtype=np.float32
    )

    gt_emb = np.asarray(
        embedder.encode(ground_truth_context),
        dtype=np.float32
    )

    # Matriz de similitud
    sims = np.matmul(retrieved_emb, gt_emb.T)

    #
    # Precision
    #
    # Para cada chunk recuperado:
    # ¿coincide con algún contexto GT?
    #

    retrieved_max = sims.max(axis=1)

    tp = int(np.sum(retrieved_max >= threshold))

    fp = len(retrieved_chunks) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    #
    # Recall
    #
    # Para cada contexto GT:
    # ¿algún chunk recuperado lo cubre?
    #

    gt_max = sims.max(axis=0)

    fn = int(np.sum(gt_max < threshold))

    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return round(float(precision), 4), round(float(recall), 4)