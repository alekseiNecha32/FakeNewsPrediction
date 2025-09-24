from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Calculate the absolute path to the project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load model and vectorizer from root directory
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news_text = request.form.get("news")
        if news_text:
            transformed_text = vectorizer.transform([news_text])
            raw_score = model.decision_function(transformed_text)[0]
            prediction_label = model.predict(transformed_text)[0]
            confidence = round(abs(raw_score) * 10, 2)
            confidence = min(confidence, 99.9)

            # Build label + confidence
            label = "✅ Real News" if prediction_label == 1 else "❌ Fake News"
            prediction = f"{label} (Confidence: {confidence}%)"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
