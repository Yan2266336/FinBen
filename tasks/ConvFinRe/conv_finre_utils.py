import json
import math
from typing import Any, Dict, List, Tuple


# -----------------------------
# doc -> chat messages
# -----------------------------
def doc_to_text_chat(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    # lm-eval-harness + --apply_chat_template expects list[{role,content}]
    msgs = doc["messages"]
    return [{"role": m["role"], "content": m["content"]} for m in msgs]

# -----------------------------
# dummy target (required by some harness forks)
# -----------------------------
def doc_to_target_dummy(doc: Dict[str, Any]) -> str:
    """
    Some forks of lm-eval-harness require doc_to_target(doc) to return a string
    during task initialization, even if scoring is done via process_results.
    """
    return ""

# -----------------------------
# JSON parsing + validation
# -----------------------------
def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty output")
    s = text.strip()
    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("No JSON object span found")
    return json.loads(s[l : r + 1])


def _validate_rank(rank: Any, candidates: List[str]) -> Tuple[bool, str]:
    if not isinstance(rank, list):
        return False, "final_rank is not list"
    if len(rank) != len(candidates):
        return False, "length mismatch"
    if set(rank) != set(candidates):
        return False, "not permutation"
    return True, "ok"


# -----------------------------
# Metrics
# -----------------------------
def _dcg_from_relevance(items: List[str], rel: Dict[str, float]) -> float:
    s = 0.0
    for k, t in enumerate(items, start=1):
        s += rel[t] / math.log2(k + 1)
    return s


def _u_ndcg(pred_rank: List[str], utility_rank: List[str]) -> float:
    """
    uNDCG proxy: use utility_rank as preference ordering and assign
    graded relevance rel(t) = n - position_in_utility_rank (higher = better).
    """
    n = len(utility_rank)
    rel = {t: float(n - i) for i, t in enumerate(utility_rank)}  # n, n-1, ..., 1
    dcg = _dcg_from_relevance(pred_rank, rel)
    idcg = _dcg_from_relevance(utility_rank, rel)
    return dcg / idcg if idcg > 0 else 0.0


def _rank_position(rank: List[str], item: str) -> int:
    # 1-based; return large if missing
    try:
        return rank.index(item) + 1
    except ValueError:
        return 10**9


def _mrr(rank: List[str], chosen: str) -> float:
    pos = _rank_position(rank, chosen)
    return 1.0 / pos if pos < 10**8 else 0.0


def _hr_at_k(rank: List[str], chosen: str, k: int) -> float:
    return 1.0 if _rank_position(rank, chosen) <= k else 0.0


def _kendall_tau(a: List[str], b: List[str]) -> float:
    """
    Kendall's tau for permutations (no ties).
    tau = 1 - 2 * discordant / (n*(n-1)/2)
    """
    n = len(a)
    pos_b = {x: i for i, x in enumerate(b)}
    discord = 0
    total = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            if pos_b[a[i]] > pos_b[a[j]]:
                discord += 1
    return 1.0 - 2.0 * discord / total if total > 0 else 0.0


# -----------------------------
# process_results (single task)
# -----------------------------
def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """
    Returns per-example metrics; aggregation handled by harness (mean).
    You can optionally postprocess to match "avg over steps per user, then over users".
    """
    out = results[0] if results else ""
    candidates = list(doc["meta"]["candidate_tickers"])

    utility_rank = list(doc["labels"]["utility_rank"])
    momentum_rank = list(doc["labels"]["momentum_rank"])
    safety_rank = list(doc["labels"]["safety_rank"])
    user_choice = doc["labels"]["user_choice"]

    m = {
        "json_valid": 0.0,
        "perm_valid": 0.0,

        # paper metrics
        "uNDCG": 0.0,
        "MRR": 0.0,
        "HR@1": 0.0,
        "HR@3": 0.0,

        # EAS
        "tau_utility": 0.0,
        "tau_momentum": 0.0,
        "tau_safety": 0.0,
    }

    try:
        obj = _extract_json_object(out)
        if set(obj.keys()) != {"final_rank"}:
            return m

        rank = obj["final_rank"]
        m["json_valid"] = 1.0

        ok, _ = _validate_rank(rank, candidates)
        if not ok:
            return m

        m["perm_valid"] = 1.0

        # uNDCG proxy against utility preference ordering
        m["uNDCG"] = _u_ndcg(rank, utility_rank)

        # user recovery
        m["MRR"] = _mrr(rank, user_choice)
        m["HR@1"] = _hr_at_k(rank, user_choice, 1)
        m["HR@3"] = _hr_at_k(rank, user_choice, 3)

        # EAS
        m["tau_utility"] = _kendall_tau(rank, utility_rank)
        m["tau_momentum"] = _kendall_tau(rank, momentum_rank)
        m["tau_safety"] = _kendall_tau(rank, safety_rank)

        return m

    except Exception:
        return m
