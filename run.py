import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Cargar el modelo y los objetos guardados
model = load_model('credit_risk_model.joblib')
scaler = joblib.load('credit_risk_scaler.joblib')
le_dict = joblib.load('credit_risk_le_dict.joblib')

# Función para hacer predicciones
def predict_credit_risk(sample_data):
    """
    Realiza predicción para un nuevo caso.

    Args:
        sample_data: Diccionario con los datos del préstamo
    """
    # Asegurarnos de que las características del nuevo dato están en el orden correcto
    features = [
        'loan_amnt', 'term', 'int_rate', 'grade',
        'home_ownership', 'annual_inc', 'verification_status',
        'purpose', 'dti', 'delinq_2yrs', 'open_acc',
        'revol_util'
    ]

    # Crear DataFrame con una fila
    input_data = pd.DataFrame([sample_data])

    # Procesar las variables categóricas usando LabelEncoders
    for col in ['grade', 'home_ownership', 'verification_status', 'purpose']:
        input_data[col] = le_dict[col].transform(input_data[col].astype(str))

    # Normalizar las variables numéricas usando el scaler
    numeric_cols = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 
                    'dti', 'delinq_2yrs', 'open_acc', 'revol_util']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Realizar la predicción
    risk_prob = model.predict(input_data)[0][0]
    risk_label = "Alto Riesgo" if risk_prob > 0.5 else "Bajo Riesgo"

    return {
        'probabilidad': float(risk_prob),
        'clasificacion': risk_label
    }

# Ejemplo de uso
sample_loan = {
    'loan_amnt': 10000,
    'term': 36,
    'int_rate': 10.5,
    'grade': 'B',
    'home_ownership': 'RENT',
    'annual_inc': 50000,
    'verification_status': 'Verified',
    'purpose': 'debt_consolidation',
    'dti': 15.5,
    'delinq_2yrs': 0,
    'open_acc': 3,
    'revol_util': 50.5
}

# Realizar predicción
prediction = predict_credit_risk(sample_loan)
print("\n=== Predicción para nuevo préstamo ===")
print(f"Probabilidad de riesgo: {prediction['probabilidad']:.2%}")
print(f"Clasificación: {prediction['clasificacion']}")