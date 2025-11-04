import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# ----------------------------
# 1. Load CSV Dataset
# ----------------------------
df = pd.read_csv(r"D:\FREEAI\data\qa_dataset.csv")  # CSV must have columns: question, context, answer
dataset = Dataset.from_pandas(df)

# ----------------------------
# 2. Load Tokenizer and Model
# ----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# ----------------------------
# 3. Preprocess Function
# ----------------------------
def preprocess(examples):
    encodings = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

    start_positions = []
    end_positions = []

    for i, answer_text in enumerate(examples['answer']):
        context_text = examples['context'][i]
        start_char = context_text.find(answer_text)
        if start_char == -1:
            start_positions.append(0)
            end_positions.append(0)
        else:
            end_char = start_char + len(answer_text)
            # Map char positions to token positions
            token_start_index = encodings.char_to_token(i, start_char)
            token_end_index = encodings.char_to_token(i, end_char - 1)
            if token_start_index is None:
                token_start_index = 0
            if token_end_index is None:
                token_end_index = 0
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)

    encodings.update({
        'start_positions': start_positions,
        'end_positions': end_positions
    })

    return encodings

# ----------------------------
# 4. Tokenize Dataset
# ----------------------------
tokenized_dataset = dataset.map(preprocess, batched=True)

# Convert to torch tensors
tokenized_dataset.set_format(
    type='torch', 
    columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions']
)

# ----------------------------
# 5. Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=r"D:\FREEAI\models\local_distilbert",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=10,
    logging_dir=r"D:\FREEAI\models\logs"
)

# ----------------------------
# 6. Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# ----------------------------
# 7. Train the Model
# ----------------------------
trainer.train()

# ----------------------------
# 8. Save Model & Tokenizer
# ----------------------------
trainer.save_model(r"D:\FREEAI\models\local_distilbert")
tokenizer.save_pretrained(r"D:\FREEAI\models\local_distilbert")

print("Training complete. Model saved in D:\\FREEAI\\models\\local_distilbert")
