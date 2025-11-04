import json
import sys
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    if len(common) == 0:
        return 0
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate(dataset_path, preds_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(preds_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    total, em, f1 = 0, 0, 0
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qid = qa["id"]
                if qid not in preds:
                    continue
                pred = preds[qid]
                answers = [a["text"] for a in qa["answers"]] if not qa.get("is_impossible", False) else [""]
                em += max(exact_match(pred, a) for a in answers)
                f1 += max(f1_score(pred, a) for a in answers)
                total += 1

    em = 100 * em / total
    f1 = 100 * f1 / total
    print(f"\nEvaluation results:")
    print(f"Total questions: {total}")
    print(f"Exact Match (EM): {em:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simple_eval.py <dataset_path> <predictions_path>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
