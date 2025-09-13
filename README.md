# Proyecto CUCEI-NLP: Sistema de Análisis de Opiniones Docentes

## Resumen del Proyecto
Este proyecto utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y Deep Learning para analizar y extraer insights a partir de reseñas de estudiantes sobre profesores del Centro Universitario de Ciencias Exactas e Ingenierías (CUCEI) de la Universidad de Guadalajara. El objetivo final era sentar las bases para un sistema de recomendación inteligente que pudiera ayudar a los estudiantes en la selección de materias y profesores.

## Arquitectura y Pipeline
El pipeline del proyecto consiste en los siguientes pasos:
1.  **Extracción de Datos:** Procesamiento de documentos PDF y web scraping inicial para crear un dataset base a partir de fuentes públicas.
2.  **Limpieza y Estructuración:** Un pipeline robusto utilizando `pandas` para limpiar, normalizar y consolidar los datos crudos, resolviendo inconsistencias y preparando el corpus para el modelado.
3.  **Generación de Embeddings:** Uso de modelos `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) para convertir el texto de las reseñas en representaciones vectoriales de alta dimensión (embeddings), capturando el significado semántico del texto.
4.  **Modelado:** Implementación de un modelo de red neuronal (MLP) en PyTorch (Arquitectura V6) que fusiona los embeddings de texto con características contextuales (departamento, división) para predecir la calificación y el sentimiento de una reseña.

## Resultados Finales
El modelo final (V6) fue entrenado con el dataset curado de **461 reseñas válidas**. Las métricas de rendimiento obtenidas en el conjunto de validación son las siguientes:
* **Correlación (Rating):** 0.6700
* **RMSE (Error):** 0.1973
* **Accuracy de Sentimiento:** 69.90%

*(Los resultados completos, las gráficas de entrenamiento y el análisis detallado se pueden encontrar en el notebook `4_Entrenamiento_y_Resultados.ipynb`)*.

## Desafíos y Aprendizajes Clave
La fase más desafiante del proyecto fue la adquisición de datos a gran escala. Se realizaron múltiples intentos de scraping automatizado en redes sociales (Facebook), lo que llevó a la conclusión de que la naturaleza dinámica y las robustas medidas de seguridad de la plataforma hacen que este enfoque sea inviable y poco fiable a largo plazo para un proyecto de esta escala.

Este proceso subrayó la importancia crítica de la calidad y accesibilidad de los datos en cualquier proyecto de Machine Learning. El principal hallazgo fue que, con un dataset limitado, el modelo sufre de un **sobreajuste severo**, lo que impide su generalización para un uso en producción. Esta conclusión es el aprendizaje más valioso del proyecto.

## Estado del Proyecto
**Finalizado.** El proyecto culmina con un pipeline de NLP de extremo a extremo completamente funcional y un modelo entrenado que demuestra la viabilidad del enfoque propuesto. El código y los hallazgos sirven como una base sólida y un caso de estudio para futuros trabajos que puedan tener acceso a datasets más grandes y de origen oficial.