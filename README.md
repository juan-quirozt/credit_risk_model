# Credit Risk Prediction App
Esta aplicación permite predecir el riesgo crediticio de un cliente mediante un modelo de aprendizaje automático. La interfaz está diseñada para recibir los datos de entrada del usuario y mostrar la probabilidad de riesgo y su clasificación.

## Requisitos previos
Antes de comenzar, asegúrate de tener instalado lo siguiente en tu sistema:

Python 3.12.3
pip (administrador de paquetes de Python)

## Instalación y configuración
Sigue los pasos a continuación para ejecutar la aplicación en tu máquina local:

1. Clonar el repositorio
Clona este repositorio en tu máquina local usando el siguiente comando:

```bash
git clone https://github.com/juan-quirozt/credit_risk_model.git
```

2. Crear y activar un entorno virtual
Navega al directorio del repositorio y crea un entorno virtual para aislar las dependencias del proyecto:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Instalar dependencias
Instala las dependencias necesarias para el proyecto ejecutando:

```bash
pip install -r requirements.txt
```

4. Ejecutar la aplicación
Inicia la aplicación localmente con el siguiente comando:

```bash
python app.py
```

Por defecto, la aplicación estará disponible en http://127.0.0.1:5000/.

5. Acceder a la aplicación
Abre tu navegador y ve a la URL mencionada para interactuar con la aplicación.

## Estructura del proyecto
app.py: Archivo principal que contiene el código para la ejecución de la aplicación Flask.
templates/: Carpeta que contiene los archivos HTML para la interfaz del usuario.
static/: Carpeta con los estilos CSS.
credit_risk_model.joblib: Archivo del modelo de predicción.
credit_risk_scaler.joblib: Escalador para las variables numéricas.
credit_risk_le_dict.joblib: Diccionario de codificadores de etiquetas para las variables categóricas.
Notas importantes
Si deseas modificar el modelo o los datos, asegúrate de entrenar nuevamente el modelo y reemplazar el archivo credit_risk_model.joblib.
Si encuentras algún problema, revisa el archivo requirements.txt para asegurarte de que las dependencias estén correctamente instaladas.
¡Disfruta usando la aplicación! Si tienes alguna pregunta o sugerencia, no dudes en abrir un issue en el repositorio. 🚀