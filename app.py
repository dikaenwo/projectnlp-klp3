from flask import Flask, request, render_template
from transformers import BertTokenizer, EncoderDecoderModel
import torch

app = Flask(__name__)

# Load tokenizer dan model dari folder lokal
tokenizer = BertTokenizer.from_pretrained("model")  # Pastikan folder "model" berisi model yang sudah di-download
model = EncoderDecoderModel.from_pretrained("model")

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        text = request.form["text"]
        min_length = int(request.form["min_length"])
        max_length = int(request.form["max_length"])

        input_ids = tokenizer.encode(text, return_tensors="pt")
        
        summary_ids = model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            num_beams=10,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=2,
            use_cache=True,
            do_sample=True,
            temperature=0.3,
            top_k=50,
            top_p=0.95
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
