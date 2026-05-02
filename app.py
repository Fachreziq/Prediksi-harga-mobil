from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')
le_name = joblib.load('encoder_name.pkl')
car_names = joblib.load('car_names.pkl')


@app.route('/')
def index():
    return render_template('index.html', car_names=car_names)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        km = int(request.form['km'])
        fuel = int(request.form['fuel'])
        trans = int(request.form['trans'])
        name = request.form['name']

        # encoding
        name_encoded = le_name.transform([name])[0]

        data = np.array([[year, km, fuel, trans, name_encoded]])
        data = scaler.transform(data)

        pred = model.predict(data)

        # konversi ke rupiah
        result_lakh = pred[0][0]
        result_rupiah = result_lakh * 15000000

        result = f"Rp {int(result_rupiah):,}".replace(",", ".")

        return render_template(
            'result.html',
            prediction=result,
            car_name=name  # ✅ kirim nama mobil
        )

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}")


if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)