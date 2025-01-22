# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo, el scaler y los label encoders
model = joblib.load('credit_risk_model.joblib')
scaler = joblib.load('credit_risk_scaler.joblib')
le_dict = joblib.load('credit_risk_le_dict.joblib')

# Definir las características del modelo
features = [
    'loan_amnt', 'term', 'int_rate', 'grade',
    'home_ownership', 'annual_inc', 'verification_status',
    'purpose', 'dti', 'delinq_2yrs', 'open_acc',
    'revol_util'
]

# Función para hacer predicciones
def predict_credit_risk(sample_data):
    # Crear DataFrame con una fila
    input_data = pd.DataFrame([sample_data])

    # Procesar variables categóricas
    for col in ['grade', 'home_ownership', 'verification_status', 'purpose']:
        input_data[col] = le_dict[col].transform(input_data[col].astype(str))

    # Procesar variables numéricas
    numeric_cols = ['loan_amnt', 'term', 'int_rate', 'annual_inc',
                   'dti', 'delinq_2yrs', 'open_acc', 'revol_util']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Realizar predicción
    risk_prob = model.predict(input_data)[0][0]
    risk_label = "Alto Riesgo" if risk_prob > 0.5 else "Bajo Riesgo"

    probabilidad = risk_prob * 100

    return {
        'probabilidad': probabilidad,
        'clasificacion': risk_label
    }

# Ruta para la página principal y mostrar el formulario
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        data = {
            'loan_amnt': float(request.form['loan_amnt']),
            'term': int(request.form['term']),
            'int_rate': float(request.form['int_rate']),
            'grade': request.form['grade'],
            'home_ownership': request.form['home_ownership'],
            'annual_inc': float(request.form['annual_inc']),
            'verification_status': request.form['verification_status'],
            'purpose': request.form['purpose'],
            'dti': float(request.form['dti']),
            'delinq_2yrs': int(request.form['delinq_2yrs']),
            'open_acc': int(request.form['open_acc']),
            'revol_util': float(request.form['revol_util'])
        }

        # Realizar la predicción
        prediction = predict_credit_risk(data)

        # Mostrar el resultado en result.html
        return render_template('result.html', prediction=prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)