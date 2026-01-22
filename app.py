from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        values = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["ash"]),
            float(request.form["alcalinity"]),
            float(request.form["phenols"]),
            float(request.form["color"])
        ]

        scaled = scaler.transform([values])
        result = model.predict(scaled)
        prediction = f"Cultivar {result[0] + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
