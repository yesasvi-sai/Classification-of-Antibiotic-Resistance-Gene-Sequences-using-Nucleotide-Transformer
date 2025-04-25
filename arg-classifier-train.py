import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "/Users/prasanthkumar/Desktop/Course_Work/Spring_2025/ML/Project_Final/Data"
model_path = "/Users/prasanthkumar/Desktop/Course_Work/Spring_2025/ML/Project_Final/Hugging_Face_Model"

train_df = pd.read_csv(f"{data_dir}/train.csv")
val_df = pd.read_csv(f"{data_dir}/val.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    num_labels=2,
    trust_remote_code=True
).to(device)

def tokenize_function(examples):
    return tokenizer(
        examples["sequence"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir=f"{data_dir}/results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,    
    per_device_eval_batch_size=4,
    num_train_epochs=1,               
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=f"{data_dir}/logs",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    fp16=False                        
)


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(eval_pred.label_ids, predictions),
        "f1": f1_score(eval_pred.label_ids, predictions, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

final_metrics = trainer.evaluate(test_dataset)
print(f"Test Accuracy: {final_metrics['eval_accuracy']:.2f}")
print(f"Test F1 Score: {final_metrics['eval_f1']:.2f}")