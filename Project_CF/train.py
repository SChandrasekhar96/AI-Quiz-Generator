import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_from_disk

print("Loading dataset...")
dataset = load_from_disk("./processed_dataset")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print("Loading T5 tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4,
    num_train_epochs=3,  
    max_steps=2000, 
    save_total_limit=2, 
    save_steps=500, 
    evaluation_strategy="steps",
    eval_steps=500, 
    logging_steps=200, 
    logging_dir="./logs",
    learning_rate=3e-4, 
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False,  
    save_strategy="steps",
    report_to="none" 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

print("Saving the final model...")
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

print("Training complete. Model saved in './final_model'.")
