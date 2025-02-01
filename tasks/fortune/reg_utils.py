import evaluate
import re
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# bertscore
def bertscore(items):
    """
    Calculate BERTScore for a list of (reference, candidate) pairs.
    passthrough for efficiency
    """
   
    return items

def bertscore_agg(items):
    """
    Aggregate BERTScore F1 scores for a list of items.
    Higher is better.
    """

    refs = [item[0] for item in items]
    preds = [item[1] for item in items]

    # Load BERTScore metric
    bertscore_scorer = evaluate.load("evaluate-metric/bertscore",device=DEVICE)

    # Compute BERTScore
    scores = bertscore_scorer.compute(predictions=preds, references=refs, lang='en')
    
    # Use the F1 scores for aggregation
    return sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0




# rouge1
def rouge1(items):
    """
    # passthrough for efficiency
    """
    return items


def rouge1_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]


## exact_match

def normalize_text(text, ignore_case=False):
    cleaned_text = re.sub(r'[\n\t"\']+', '', text)
    if ignore_case:
        return cleaned_text.lower().strip()
    else:
        return cleaned_text.strip()

def exact_match(items):
    """
    # passthrough for efficiency
    """
    return items


def exact_match_agg(items):
    """
    Higher is better
    """
    ig_case = True
    refs = [normalize_text(item[0],ig_case) for item in items]
    preds = [normalize_text(item[1],ig_case) for item in items]
    
    exact_match = evaluate.load("exact_match")
    results = exact_match.compute(predictions=preds, references=refs)
    return results
