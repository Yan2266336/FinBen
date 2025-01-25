import evaluate

# bertscore
def bertscore(items, lang="en"):
    """
    Calculate BERTScore for a list of (reference, candidate) pairs.
    """
    refs = [item[0] for item in items]
    preds = [item[1] for item in items]

    # Load BERTScore metric
    bertscore_scorer = evaluate.load("bertscore")

    # Compute BERTScore
    scores = bertscore_scorer.compute(predictions=preds, references=refs, lang=lang)
    return scores

def bertscore_agg(items, lang="en"):
    """
    Aggregate BERTScore F1 scores for a list of items.
    Higher is better.
    """
    # Compute individual scores
    scores = bertscore(items, lang=lang)

    # Use the F1 scores for aggregation
    return sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0

