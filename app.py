from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load your model and tokenizer
model_path = "text_summarization_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def format_summary(summary):
    lines = summary.split('<n>')
    formatted_summary = ""
    for line in lines:
        if ':' in line:
            speaker, dialogue = line.split(':', 1)
            formatted_summary += f"{speaker.strip()}:\n    {dialogue.strip()}\n\n"
        else:
            formatted_summary += f"{line.strip()}\n\n"
    return formatted_summary.strip()

@app.route('/', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['text']
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        formatted_summary = format_summary(summary)
        return render_template('index.html', summary=formatted_summary)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)