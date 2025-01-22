import pandas as pd
import joblib

def predict_credit_risk(model, scaler, le_dict, sample_data):
    """
    Realiza predicción para un nuevo caso.

    Args:
        model: Modelo entrenado
        scaler: StandardScaler ajustado
        le_dict: Diccionario de LabelEncoders
        sample_data: Diccionario con los datos del préstamo
    """
    # Preparar los datos en el orden correcto
    features = [
        'loan_amnt', 'term', 'int_rate', 'grade',
        'home_ownership', 'annual_inc', 'verification_status',
        'purpose', 'dti', 'delinq_2yrs', 'open_acc',
        'revol_util'
    ]

    # Crear DataFrame con una fila
    input_data = pd.DataFrame([sample_data])

    # Procesar variables categóricas
    for col in ['grade', 'home_ownership', 'verification_status', 'purpose']:
        input_data[col] = le_dict[col].transform(input_data[col].astype(str))

    # Procesar variables numéricas
    numeric_cols = ['loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'open_acc', 'revol_util']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Realizar predicción
    risk_prob = model.predict(input_data)[0][0]
    risk_label = "Alto Riesgo" if risk_prob > 0.5 else "Bajo Riesgo"

    return {
        'probabilidad': float(risk_prob),
        'clasificacion': risk_label
    }