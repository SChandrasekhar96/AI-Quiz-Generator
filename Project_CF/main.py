import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "./final_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_mcq(text, max_length=150):
    """Generate MCQs from input text using the trained model."""
    input_text = f"generate_mcq: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,  
        top_k=50,        
        top_p=0.95,      
        temperature=0.7  
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    parts = output_text.split("Choices:")
    if len(parts) < 2:
        return {"question": "Failed to generate a proper MCQ", "choices": [], "answer": ""}

    question_part = parts[0].strip()
    choices_part = parts[1].split("Answer:")

    if len(choices_part) < 2:
        return {"question": question_part, "choices": [], "answer": ""}

    choices = [c.strip() for c in choices_part[0].split(",")]
    answer = choices_part[1].strip()

    choices = list(set(choices))

    return {"question": question_part, "choices": choices, "answer": answer}

if __name__ == "__main__":
    sample_text = "The internet is a global network that connects millions of computers worldwide, allowing communication and information sharing."
    mcq_output = generate_mcq(sample_text)
    
    print("Generated MCQ:\n")
    print("Question:", mcq_output["question"])
    print("Choices:", mcq_output["choices"])
    print("Answer:", mcq_output["answer"])
