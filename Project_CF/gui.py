import tkinter as tk
from tkinter import scrolledtext, ttk
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "./final_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_mcqs(text, max_length=150, num_mcqs=5):
    """Generate multiple MCQs from input text using the trained model."""
    input_text = f"generate_mcq: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_mcqs,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    mcq_list = []
    for output in output_ids:
        output_text = tokenizer.decode(output, skip_special_tokens=True)

        parts = output_text.split("Choices:")
        if len(parts) < 2:
            continue

        question_part = parts[0].strip()
        choices_part = parts[1].split("Answer:")

        if len(choices_part) < 2:
            continue

        choices = [c.strip() for c in choices_part[0].split(",")]
        answer = choices_part[1].strip()

        choices = list(set(choices))

        mcq_list.append({"question": question_part, "choices": choices, "answer": answer})

    return mcq_list

def generate_and_display_mcqs():
    """Handles MCQ generation and clears previous content before displaying new ones."""
    global mcq_data
    text = input_text_box.get("1.0", tk.END).strip()
    
    if not text:
        return

    for widget in mcq_frame.winfo_children():
        widget.destroy()
    for widget in answer_frame.winfo_children():
        widget.destroy()

    mcq_data = generate_mcqs(text)

    selected_options.clear()
    for idx, mcq in enumerate(mcq_data):
        question_label = tk.Label(mcq_frame, text=f"{idx+1}. {mcq['question']}", font=("Arial", 12, "bold"), wraplength=600, justify="left")
        question_label.pack(anchor="w", pady=(5, 2))

        selected_option = tk.StringVar(value="None")
        selected_options.append(selected_option)

        for choice in mcq["choices"]:
            rb = tk.Radiobutton(mcq_frame, text=choice, variable=selected_option, value=choice, font=("Arial", 11))
            rb.pack(anchor="w")

    show_answers_button.pack(pady=5)
    show_answers_button.config(state=tk.NORMAL)

    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

def show_answers():
    """Displays the correct answers below the questions."""
    for widget in answer_frame.winfo_children():
        widget.destroy()

    for idx, mcq in enumerate(mcq_data):
        answer_label = tk.Label(answer_frame, text=f"Correct Answer {idx+1}: {mcq['answer']}", font=("Arial", 12, "bold"), fg="green")
        answer_label.pack()

    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

root = tk.Tk()
root.title("AI-Powered Quiz Generator")
root.geometry("750x500")

input_text_box = scrolledtext.ScrolledText(root, width=80, height=5, font=("Arial", 11))
input_text_box.pack(pady=10)

generate_button = tk.Button(root, text="Generate MCQs", command=generate_and_display_mcqs, font=("Arial", 12), bg="blue", fg="white")
generate_button.pack(pady=5)

container = tk.Frame(root)
container.pack(fill="both", expand=True, padx=10, pady=10)

canvas = tk.Canvas(container)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind("<Configure>", lambda e: canvas.config(scrollregion=canvas.bbox("all")))

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.config(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

mcq_frame = tk.Frame(scrollable_frame)
mcq_frame.pack(pady=10)

selected_options = []

show_answers_button = tk.Button(root, text="Show Answers", command=show_answers, font=("Arial", 12), bg="green", fg="white")
show_answers_button.pack(pady=5)
show_answers_button.config(state=tk.DISABLED)
answer_frame = tk.Frame(scrollable_frame)
answer_frame.pack(pady=10)

root.mainloop()
