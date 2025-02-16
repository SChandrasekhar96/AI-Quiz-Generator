from datasets import load_dataset
from transformers import T5Tokenizer

dataset = load_dataset("sciq")

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    inputs = ["Generate an MCQ from: " + passage for passage in examples["support"]]
    outputs = [
        f"Question: {q} Choices: {d1}, {d2}, {d3}, {ca} Answer: {ca}"
        for q, d1, d2, d3, ca in zip(
            examples["question"], examples["distractor1"],
            examples["distractor2"], examples["distractor3"], examples["correct_answer"]
        )
    ]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=256, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

tokenized_dataset.save_to_disk("./processed_dataset")

