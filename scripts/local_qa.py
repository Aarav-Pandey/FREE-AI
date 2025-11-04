# D:\FREEAI\scripts\local_qa.py
"""
Multi-mode local QA tool (interactive / CSV / SQuAD JSON).
Usage examples:
  # Interactive mode
  python local_qa.py --mode interactive --model_dir "D:\FREEAI\models\local_distilbert"

  # Batch from CSV (expects columns: 'context' and 'question')
  python local_qa.py --mode csv --input "D:\FREEAI\data\myqa.csv" --output "D:\FREEAI\output\csv_preds.json"

  # Batch from SQuAD dev JSON (produces predictions.json mapping qid->answer)
  python local_qa.py --mode squad_json --input "D:\FREEAI\data\dev-v2.0.json" --output "D:\FREEAI\scripts\predictions.json"

Notes:
 - model_dir should point to the folder with saved model/tokenizer (model.save_pretrained path).
 - The script automatically uses GPU if available.
"""
import os
import json
import argparse
import csv
import sys
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, pipeline

def load_pipeline(model_dir):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading model from: {model_dir}  (using {'GPU' if device==0 else 'CPU'})")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForQuestionAnswering.from_pretrained(model_dir)
    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
    return qa_pipe

def run_interactive(qa_pipe):
    print("\n=== Local QA (interactive) ===")
    print("Type 'exit' to quit at any prompt.\n")
    while True:
        context = input("Context (or 'exit'):\n")
        if context.strip().lower() == "exit":
            break
        question = input("\nQuestion (or 'exit'):\n")
        if question.strip().lower() == "exit":
            break
        try:
            res = qa_pipe(question=question, context=context)
        except Exception as e:
            print("Error running pipeline:", e)
            continue
        answer = res.get("answer", "")
        score = res.get("score", 0.0)
        print(f"\nAnswer: {answer}\nScore: {score:.4f}\n{'-'*60}\n")

def run_csv(qa_pipe, input_path, output_path=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"CSV file not found: {input_path}")
    print(f"Reading CSV: {input_path}")
    rows_out = []
    with open(input_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'context' not in reader.fieldnames or 'question' not in reader.fieldnames:
            raise ValueError("CSV must contain 'context' and 'question' columns.")
        for i, row in enumerate(reader):
            context = row['context']
            question = row['question']
            try:
                res = qa_pipe(question=question, context=context)
                answer = res.get("answer", "")
                score = res.get("score", 0.0)
            except Exception as e:
                answer, score = "", 0.0
                print(f"Error on row {i}: {e}", file=sys.stderr)
            out = {
                "index": i,
                "question": question,
                "context": context,
                "answer": answer,
                "score": score
            }
            rows_out.append(out)
            if (i+1) % 100 == 0:
                print(f"Processed {i+1} rows...")
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows_out, f, ensure_ascii=False, indent=2)
        print(f"Saved CSV predictions to: {output_path}")
    else:
        print(json.dumps(rows_out, ensure_ascii=False, indent=2))

def run_squad_json(qa_pipe, input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"SQuAD JSON not found: {input_path}")
    print(f"Loading SQuAD JSON: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        squad = json.load(f)

    predictions = {}
    total = 0
    for article in squad.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qid = qa.get("id")
                question = qa.get("question", "")
                try:
                    res = qa_pipe(question=question, context=context)
                    answer = res.get("answer", "")
                except Exception as e:
                    answer = ""
                    print(f"Error predicting qid={qid}: {e}", file=sys.stderr)
                predictions[qid] = answer
                total += 1
                if total % 500 == 0:
                    print(f"Predicted {total} questions...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved SQuAD-style predictions (qid->answer) to: {output_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Local QA tool (interactive / csv / squad_json)")
    p.add_argument("--mode", choices=["interactive", "csv", "squad_json"], required=True, help="Run mode")
    p.add_argument("--model_dir", required=True, help="Path to local model directory (saved model/tokenizer)")
    p.add_argument("--input", help="Input file path (CSV or SQuAD JSON)")
    p.add_argument("--output", help="Output JSON path for batch modes (CSV or SQuAD predictions)")
    return p.parse_args()

def main():
    args = parse_args()
    qa_pipe = load_pipeline(args.model_dir)

    if args.mode == "interactive":
        run_interactive(qa_pipe)
    elif args.mode == "csv":
        if not args.input:
            raise ValueError("CSV mode requires --input <path/to/file.csv>")
        run_csv(qa_pipe, args.input, args.output)
    elif args.mode == "squad_json":
        if not args.input or not args.output:
            raise ValueError("squad_json mode requires --input <dev.json> and --output <predictions.json>")
        run_squad_json(qa_pipe, args.input, args.output)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
