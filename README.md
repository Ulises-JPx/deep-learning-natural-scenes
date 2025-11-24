# Intel Image Classification – Deep Learning Natural Scenes Project

Este repositorio contiene la implementación completa de un modelo de clasificación de escenas naturales utilizando redes neuronales profundas. El proyecto se desarrolló en dos versiones:

- **Versión Colab (comprobada y funcional)**
- **Versión Local (adaptada pero no probada debido a limitaciones del equipo utilizado)**

La solución emplea TensorFlow y EfficientNetB0, aplicando técnicas modernas de deep learning como transfer learning, data augmentation y fine-tuning.

---

## Objetivo del Proyecto

Entrenar un modelo capaz de clasificar imágenes del dataset Intel Image Classification en categorías como:

- buildings  
- forest  
- glacier  
- mountain  
- sea  
- street

Este proyecto implementa un pipeline de visión por computadora completo, desde la preparación del dataset hasta la inferencia final con imágenes personalizadas.

---

## Estructura del Repositorio

```
deep-learning-natural-scenes/
│
├── colab/
│   └── scenes_classification.ipynb    # Versión funcional probada en Google Colab
│
├── local/
│   ├── main_local.py                  # Versión local (no probada por limitaciones de hardware)
│   ├── data/                          # Dataset local (ignorado en GitHub)
│   ├── models/                        # Modelos locales (ignorado en GitHub)
│   └── images/
│
└── README.md
```

---

## Motivo por el cual la versión local no fue probada

Aunque la carpeta **local/** contiene una versión completamente adaptada para ejecutarse de forma local, **no fue posible probar su funcionamiento** debido a limitaciones del equipo utilizado durante el desarrollo. Entre las restricciones principales se encuentran:

- Falta de GPU dedicada  
- Insuficiente capacidad de procesamiento para entrenar EfficientNetB0 localmente  
- Limitaciones de memoria RAM durante la carga y preprocesamiento del dataset  
- Tiempos de ejecución demasiado altos en el entorno local

Por esta razón, se optó por utilizar **Google Colab**, ya que proporciona:

- GPU gratuitas  
- Mayor capacidad de RAM  
- Entorno ya configurado para TensorFlow  
- Mejor desempeño general para tareas de deep learning

Esto permitió completar la solución con éxito y evaluar el modelo de forma estable y eficiente.

---

## Tecnologías Utilizadas

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-Learn  
- Google Colab (GPU)

---

## Pipeline Implementado

1. Preparación del entorno y dependencias  
2. Descarga y preparación del dataset  
3. Generación de splits: train / validation / test  
4. Carga del dataset mediante `image_dataset_from_directory`  
5. Construcción del modelo EfficientNetB0  
6. Entrenamiento fase 1 (base congelada)  
7. Fine-tuning fase 2 (últimas capas descongeladas)  
8. Evaluación en test (accuracy, classification report, matriz de confusión)  
9. Curvas de entrenamiento (accuracy y loss)  
10. Guardado del modelo final  
11. Inferencia con imágenes personalizadas  

---

## Resultados

El modelo entrenado en Colab logró:

- Un rendimiento sólido en el conjunto de prueba  
- Buen nivel de generalización  
- Predicciones consistentes sobre imágenes externas  

Las gráficas, métricas y reportes pueden encontrarse en el notebook dentro de `colab/`.

---

## Ejecución de la Versión Colab (Recomendada)

1. Abrir:
   ```
   colab/scenes_classification.ipynb
   ```

2. Ejecutar las celdas en orden  
3. Subir `kaggle.json` cuando se solicite  
4. Ejecutar el pipeline completo

---

## Ejecución de la Versión Local (No probada)

La versión local se encuentra en:
```
local/main_local.py
```

Para ejecutarla:

```bash
pip install -r requirements.txt
python local/main_local.py
```

Sin embargo:

- No pudo ser probada debido a restricciones del hardware disponible  
- Puede requerir una GPU para entrenar EfficientNetB0  
- Su desempeño puede variar significativamente en equipos sin aceleración

---

## Conclusión

El uso de Google Colab permitió completar el proyecto de manera eficiente, evitando las limitaciones del entorno local y asegurando un entrenamiento exitoso del modelo. Aunque la versión local está preparada y estructurada correctamente, la versión probada y verificada es la que se encuentra en la carpeta `colab/`.

---

## Autor

**Ulises Jaramillo Portilla** — *Matrícula:* A01798380

**Tecnológico de Monterrey** - Campus Estado de México

*Proyecto desarrollado como parte del portafolio de implementación del **_módulo #2 TC3007C.501_** de Deep Learning para clasificación de imágenes.*

## Licencia

Este proyecto está disponible bajo la licencia MIT.
