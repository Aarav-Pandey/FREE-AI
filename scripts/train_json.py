# D:\FREEAI\scripts\train_json.py
import os
import json
import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments
)

# ==============================
# Config
# ==============================
TRAIN_JSON = r"D:\FREEAI\data\train-v2.0.json"
DEV_JSON = r"D:\FREEAI\data\dev-v2.0.json"
SAVE_DIR = r"D:\FREEAI\models\local_distilbert"

MAX_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 4   # safe for GTX 1650
EPOCHS = 2
LEARNING_RATE = 2e-5

USE_SUBSET = False    # set True for quick testing
SUBSET_SIZE = 5000

# ==============================
# Device
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==============================
# Helper function: load SQuAD JSON
# ==============================
def load_squad_json(path):
    print(f"Loading dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        squad = json.load(f)

    contexts, questions, answers_text = [], [], []
    for article in squad["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if qa.get("is_impossible", False) or len(qa.get("answers", [])) == 0:
                    answer_text = ""
                else:
                    answer_text = qa["answers"][0]["text"]
                contexts.append(context)
                questions.append(question)
                answers_text.append(answer_text)

    dataset = Dataset.from_dict({
        "context": contexts,
        "question": questions,
        "answer_text": answers_text
    })
    print(f"Loaded {len(dataset)} examples from {os.path.basename(path)}")
    return dataset

# ==============================
# Load datasets
# ==============================
train_ds = load_squad_json(TRAIN_JSON)
dev_ds = load_squad_json(DEV_JSON)

if USE_SUBSET:
    train_ds = train_ds.shuffle(seed=42).select(range(SUBSET_SIZE))
    dev_ds = dev_ds.shuffle(seed=42).select(range(200))
    print(f"Using subset: {len(train_ds)} train examples, {len(dev_ds)} dev examples")

# ==============================
# Load tokenizer & model
# ==============================
print("Loading tokenizer and model (distilbert-base-uncased)...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased").to(device)

# ==============================
# Preprocessing for QA
# ==============================
def prepare_train_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answer_text = examples["answer_text"][sample_index]
        context = examples["context"][sample_index]

        cls_index = tokenized_examples["input_ids"][i].index(tokenizer.cls_token_id)

        if answer_text == "" or answer_text is None:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            answer_start_char = context.find(answer_text)
            answer_end_char = answer_start_char + len(answer_text)

            token_start_index = None
            token_end_index = None
            for idx, (off_start, off_end) in enumerate(offsets):
                if off_start is None or off_end is None:
                    continue
                if off_start <= answer_start_char < off_end:
                    token_start_index = idx
                if off_start < answer_end_char <= off_end:
                    token_end_index = idx
                if token_start_index is not None and token_end_index is not None:
                    break

            if token_start_index is None or token_end_index is None:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_positions.append(token_start_index)
                end_positions.append(token_end_index)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

# ==============================
# Tokenize datasets
# ==============================
print("Tokenizing training dataset...")
tokenized_train = train_ds.map(
    prepare_train_features,
    batched=True,
    remove_columns=train_ds.column_names
)

print("Tokenizing dev dataset...")
tokenized_dev = dev_ds.map(
    prepare_train_features,
    batched=True,
    remove_columns=dev_ds.column_names
)

# ==============================
# Training Arguments
# ==============================
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=50,
)

# ==============================
# Trainer
# ==============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
)

# ==============================
# Train and Save
# ==============================
print("Starting training...")
trainer.train()
print("Training finished.")

os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"✅ Training complete! Model & tokenizer saved at {SAVE_DIR}")
