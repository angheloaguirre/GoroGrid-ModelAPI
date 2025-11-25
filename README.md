# GoroGrid Model API (Backend)

## Descripción

El **backend de GoroGrid** proporciona una API robusta que se encarga de gestionar el procesamiento de datos y la comunicación con el modelo de predicción de consumo energético. Esta API permite la integración de modelos de Machine Learning para prever y optimizar el uso de energía en edificios inteligentes. 

Este repositorio contiene los archivos necesarios para el servidor API, que se ejecuta en **Flask** y permite la interacción con modelos preentrenados en **Python**.

## Componentes del Proyecto

El proyecto se organiza de la siguiente manera:

- **.github/workflows**: Archivos de configuración para la integración continua y despliegue automático.
- **__pycache__**: Archivos generados automáticamente por Python para optimizar el rendimiento.
- **data**: Contiene los datos de entrada necesarios para las predicciones y entrenamiento de los modelos.
- **static**: Archivos estáticos utilizados en la aplicación.
- **app.py**: Archivo principal de la aplicación Flask que gestiona las rutas y la lógica del backend.
- **export_weights.py**: Script para exportar los pesos entrenados del modelo.
- **modelo_rlm.pkl**: Modelo entrenado en formato `pkl` que se utiliza para hacer las predicciones.
- **modelo_rlm_schema.json**: Esquema del modelo para la validación de los datos de entrada.
- **requirements.txt**: Lista de dependencias necesarias para ejecutar el backend.
- **vercel.json**: Configuración para el despliegue de la aplicación en Vercel.
- **weights.json**: Pesos del modelo en formato JSON.
- **weights_scaled.json**: Pesos del modelo escalados en formato JSON.

## Requisitos

Para ejecutar este proyecto, necesitas tener instalado:

- **Python 3.x** (recomendado Python 3.8 o superior)
- **pip** (gestor de paquetes de Python)
- **Flask** (framework para crear la API)
- **Scikit-learn** y otras librerías necesarias para el modelo

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/GoroGrid-ModelAPI.git
   ```
   
2. Navega al directorio del proyecto:
  ```bash
  cd GoroGrid-ModelAPI
  ```

3. Crea un entorno virtual y activa:
  ```bash
  python -m venv venv
  source venv/bin/activate  # En Windows: venv\Scripts\activate
  ```

4. Instala las dependencias:
  ```bash
  pip install -r requirements.txt
  ```

5. Ejecuta el servidor Flask:
  ```bash
  python app.py
  ```

La API estará disponible en **http://localhost:5000**.
