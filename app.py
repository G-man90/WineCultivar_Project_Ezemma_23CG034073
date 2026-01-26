import os
from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            values = [
                float(request.form["alcohol"]),
                float(request.form["malic_acid"]),
                float(request.form["ash"]),
                float(request.form["alcalinity"]),
                float(request.form["phenols"]),
                float(request.form["color"])
            ]

            scaled_values = scaler.transform([values])
            result = model.predict(scaled_values)

            prediction = f"Cultivar {int(result[0]) + 1}"

        except Exception as e:
            prediction = "Invalid input. Please enter valid numbers."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
