# 🧠 Clasificación del Dataset Iris con PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

Este proyecto implementa una Red Neuronal Artificial (MLP) construida desde cero con **PyTorch** para clasificar el famoso dataset de flores Iris. El modelo es capaz de predecir la especie de una flor (Setosa, Versicolor o Virginica) basándose en las 4 medidas de sus sépalos y pétalos.

## 🚀 Características del Proyecto

Durante el desarrollo de este modelo, se han aplicado buenas prácticas de Deep Learning y resolución de problemas específicos del entorno:

* **Gestión de Datos por Lotes (Batches):** Implementación de `DataLoader` manejando correctamente el multiprocesamiento (`num_workers=0` para compatibilidad en entornos Windows/Jupyter).
* **Entrenamiento y Validación:** Estructuración limpia del bucle de entrenamiento, separando correctamente la fase de optimización de la fase de validación con `torch.no_grad()` para maximizar el rendimiento de la memoria.
* **Cálculo de Precisión Vectorial:** Uso avanzado de álgebra de tensores (`torch.argmax` con `dim=1`) para comparar correctamente matrices 2D de predicciones por lotes contra etiquetas con codificación *One-Hot*.
* **Visualización de Curvas de Aprendizaje:** Integración con `matplotlib` para graficar la evolución de la pérdida (*Loss*) y el porcentaje de precisión (*Accuracy*) a lo largo de las 100 épocas.
* **Monitorización Profesional:** Registro de métricas binarias exportadas a **TensorBoard** (`SummaryWriter`) para análisis y auditoría del entrenamiento en tiempo real.
* **Persistencia del Modelo:** Extracción y guardado del `state_dict` (pesos y sesgos optimizados) en un archivo `.pth` para permitir predicciones futuras sin necesidad de reentrenar.

## 📁 Estructura del Proyecto

```text
├── modelos/
│   └── modelo_iris_entrenado.pth  # Archivo con los pesos finales de la red (~96% precisión)
├── runs/
│   └── experimento_iris/          # Logs binarios generados para TensorBoard
├── red_neuronal_iris.ipynb        # Código fuente interactivo con el modelo y los bucles
└── README.md                      # Documentación del proyecto
```
## 💻 Acceso Rápido al Código

Puedes explorar el código completo, las explicaciones paso a paso y las gráficas generadas directamente en el cuaderno principal del proyecto haciendo clic en el siguiente enlace:

👉 **[Abrir el Jupyter Notebook: `red_neuronal_iris.ipynb`](/notebooks/red_neuronal_iris.ipynb)**

> 💡 **Nota:** GitHub renderiza los archivos `.ipynb` de forma nativa, por lo que puedes visualizar todo el código, los comentarios y los resultados de las ejecuciones directamente desde tu navegador sin necesidad de descargar nada.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alexisfpy/Red-Neuronal-Iris/blob/master/notebooks/red_neuronal_iris.ipynb)
## ⚙️ Instalación y Uso

Sigue estos pasos para clonar el proyecto y ejecutar el modelo en tu máquina local:

### 1. Requisitos previos (`venv`)

El módulo `venv` viene instalado por defecto al instalar Python en Windows y macOS. Sin embargo, si usas una distribución de Linux basada en Debian/Ubuntu y te da error al intentar crearlo, instálalo manualmente con este comando:

```bash
sudo apt update
sudo apt install python3-venv
```
### 2. Clonar el repositorio

Abre tu terminal y descarga el código:

```bash
git clone https://github.com/Alexisfpy/iris-ai-classification-pytorch.git
cd Red-Neuronal-Iris

```
### 3. Crear y activar el entorno virtual
```bash
python -m venv .venv
```
#### En Windows
```bash
.venv\Scripts\activate
```
#### En Linux/macOs
```bash
source .venv/bin/activate
```
### 4. Instalar dependencias
Este proyecto utiliza un archivo pyproject.toml para gestionar sus paquetes. Con el entorno virtual activado, instala todas las dependencias automáticamente ejecutando:
```bash
pip install .
```
o
```bash
uv sync
```
### 4. Lanzar el cuaderno interactivo
Una vez instalado todo, arranca el entorno de Jupyter para ver el código fuente y ejecutar la red neuronal:
```bash
jupyter notebook red_neuronal_iris.ipynb
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - mira el archivo [LICENSE](LICENSE) para más detalles.
