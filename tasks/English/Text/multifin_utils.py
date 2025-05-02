import evaluate
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import datetime
from collections import defaultdict
from FactScoreLite.factscore import FactScore
import string
from seqeval.metrics import f1_score as entity_score

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

    refs = [normalize_bertscore_text(item[0]) for item in items]
    preds = [normalize_bertscore_text(item[1]) for item in items]

    # Load BERTScore metric
    bertscore_scorer = evaluate.load("evaluate-metric/bertscore",device=DEVICE)

    # Compute BERTScore
    scores = bertscore_scorer.compute(predictions=preds, references=refs, lang='en')
    
    # Use the F1 scores for aggregation
    return sum(scores["f1"]) / len(scores["f1"]) if scores["f1"] else 0.0

def normalize_bertscore_text(text):
    
    """
    You can achieve the specific functions
    """
    
    ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')

    text = text.lower()

    # Remove stock tickers
    text = ticker_pattern.sub('', text)

    # Remove hyphens/dashes
    text = re.sub(r'[-–—]', ' ', text)

    # Remove punctuation
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\_`~()"]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


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



## FActScore
def FActScore(items):

    return items

def FActScore_agg(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]

    ## change this path to your Finben directory
    # path1 = "~/inference/FinBen/decisions.json"
    # path2 = "~/inference/FinBen/facts.json"
    # if os.path.exists(path1):
    #     os.remove(path1)

    # if os.path.exists(path2):
    #     os.remove(path2)
        
    fact_scorer = FactScore()
    scores, _ = fact_scorer.get_factscore(generations=preds, knowledge_sources=refs)

    return scores


## Accuracy
def math_acc(items):
    return items


def math_acc_agg(items):
    true_answer = [extract_first_number(item[0]) for item in items]
    pred_answer = [extract_first_number(item[1]) for item in items]

    # Define tolerance percentage (5% allowed deviation)
    tolerance = 0.05  # 5%

    correct = 0
    for true_number, pred_number in zip(true_answer, pred_answer):
        if true_number is not None and pred_number is not None:
            difference = abs(true_number-pred_number)
            allowed_difference = true_number*tolerance

            if difference <= allowed_difference:
                correct += 1
        elif true_number is None and pred_number is None:
            correct += 1
        else:
            continue

    accuracy = correct/len(true_answer)
    return accuracy

def extract_first_number(value):
    """
    Extracts the first numeric value from a given string.
    Ignores any explanations or additional text after the number.
    Returns the number as a float or None if not found.
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return None
    
    match = re.search(r"-?\d+(\.\d+)?", value)
    return float(match.group(0)) if match else None 


#--------------------------------------------------
    
## rouge1 for ECTSUM task
def get_sum(labels, texts):
    summaries = []
    for label, text in zip(labels, texts):
        lines = text.split("\n")
        selected = [
            lines[i] for i in range(len(lines)) if i < len(label) and label[i] == 1
        ]
        summaries.append("\n".join(selected))
    return summaries
        

def ect_rouge1(items):
    # items: list of (doc_to_target, pred)
    return items  # passthrough

def ect_rouge1_agg(items):
    # items: List[Tuple[str (target_json), str (pred)]]
    golds, preds = zip(*items)

    label_list, text_list = [], []
    for gold_json in golds:
        data = json.loads(gold_json)
        label_list.append(data["label"])
        text_list.append(data["text"])

    pred_label_list = [p.split("\n") for p in preds]

    ref_summaries = get_sum(label_list, text_list)
    pred_summaries = get_sum(pred_label_list, text_list)

    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=pred_summaries, references=ref_summaries)

    return result["rouge1"]





# # summarizaiton
# def rouge1(items):
#     """
#     # passthrough for efficiency
#     """
#     return items


# def rouge1_agg(items):
#     """
#     Higher is better
#     """
#     refs = list(zip(*items))[0]
#     preds = list(zip(*items))[1]
#     rouge_scorer = evaluate.load("rouge")
#     return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]


# extractive summarization ROUGE-1 evaluation
# def parse_prediction_indices(s):
#     s = s.strip()
#     try:
#         items = json.loads(s)
#     except:
#         delim = ';' if ';' in s else ',' if ',' in s else None
#         items = s.split(delim) if delim else [s]
#     return [int(x) for x in items if x.strip().isdigit()]


# def process_results_for_es(doc, results):
#     choices = doc["choices"]

#     ground_truth_indices = doc["gold"]
#     print(f"* ground_truth_indices: {ground_truth_indices}")
#     ground_truth = " ".join([choices[i] for i in ground_truth_indices])
#     print(f"* ground_truths: {ground_truth}")

#     print(f"* output: {results[0].strip()}")

#     prediction_indices = parse_prediction_indices(results[0].strip())
#     print(f"* prediction_indices: {prediction_indices}")
#     prediction = " ".join([choices[i] for i in prediction_indices])
#     print(f"* prediction: {prediction}")
    
#     rouge_scorer = evaluate.load("rouge")
#     return {"rouge1": rouge_scorer.compute(predictions=[prediction], references=[ground_truth])["rouge1"]}


# ner

LMAP = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
}

def process_result(pred, tokens):
    format_pred = ["O"] * len(tokens)
    pred_lines = pred.strip().split("\n")
    for index, line in enumerate(pred_lines[: len(tokens)]):
        try:
            word, label = line.split(":")
        except ValueError:
            continue
        if word.strip() == tokens[index] and label.strip() in LMAP:
            format_pred[index] = label.strip()
    return format_pred

def process_results(doc, results):
    gold_labels = doc["label"]
    tokens = doc["token"]

    prediction_string = results[0]

    entity_pred = process_result(prediction_string, tokens)
    
    entity_f1_score = entity_score
    
    if isinstance(entity_f1_score, tuple):
        entity_f1_score = entity_f1_score[-1]

    return {"f1": float(entity_f1_score)}





# def process_text(entity_string, text):
#     # Initialize
#     entity_list = [(", ".join(val.split(":")[:-1]), val.split(":")[-1]) for val in entity_string.split("\n")]
#     # text_words = list(filter(None, re.split(r'(\s+|[' + re.escape(string.punctuation).replace('%', '') + r'«»‘’“”€])', text)))
#     text_words = text
#     # print(text_words)
#     labels = ['O'] * len(text_words)
#     # text_lower = text.lower()
#     text_lower = text

#     # Create a list to store the start index of each word
#     word_indices = [0]
#     for word in text_words[:-1]:
#         word_indices.append(word_indices[-1] + len(word))

#     # Iterate over the entity list
#     # print (entity_list)
#     for entity, entity_type in entity_list:
#         entity_words = entity.split()
#         entity_lower = entity

#         # Find start and end index of each occurrence of the entity in the text
#         start = 0
#         while True:
#             start = text_lower.find(entity_lower, start)
#             if not entity or start == -1: break  # No more occurrence
#             end = start + len(entity) - 1

#             # Find the words included in this occurrence
#             try:
#                 start_word = next(i for i, ind in enumerate(word_indices) if ind >= start)
#                 end_word = next(i for i, ind in enumerate(word_indices) if ind > end)

#                 # Label the words
#                 labels[start_word] = 'B-' + entity_type
#                 for i in range(start_word+1, end_word):
#                     labels[i] = 'I-' + entity_type

#                 # Move to the next character after the occurrence
#             except Exception:
#                 pass
#             start = end + 1

#     _, filtered_labels = bio_filter(text_words, labels)

#     return filtered_labels


# def bio_filter(text_list, label_list):
#     processed_text = []
#     processed_label = []

#     for text, label in zip(text_list, label_list):
#         if not re.search(r'(\s+|[' + re.escape(string.punctuation).replace('%', '') + r'«»‘’“”€])', text):
#             processed_text.append(text)
#             processed_label.append(label)

#     # print(processed_text)
#     return processed_text, processed_label


# def process_results(doc, results):
#     text = doc["token"]
#     text = json.loads(text)
#     # print("\n" + text)

#     ground_truths_string = doc["answer"]
#     # print(ground_truths_string)
#     ground_truths = process_text(ground_truths_string, text)
#     # print(len(ground_truths))
#     # print(ground_truths)

#     prediction_string = results[0].strip()
#     # print(prediction_string)
#     prediction = process_text(prediction_string, text)
#     # print(len(prediction))
#     # print(prediction)

#     f1 = entity_score([ground_truths], [prediction])
#     return {"f1": f1}








