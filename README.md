## Arquitectura y Pipeline
El pipeline del proyecto se estructura en una arquitectura modular de última generación, combinando NLP, bases de datos vectoriales y aprendizaje profundo supervisado.

### 1. Extracción y Limpieza de Datos
Se recopilaron datos desde diversas fuentes públicas, incluyendo documentos PDF y scraping web.
Los datos fueron sometidos a un proceso riguroso de limpieza, normalización y consolidación utilizando pandas, resolviendo duplicidades, errores de formato y ambigüedades en nombres de profesores.
El resultado fue un dataset curado de alta calidad, adecuado para tareas de modelado semántico y fine-tuning.

### 2. Base de Datos Vectorial (RAG)

#### Generación de Embeddings
Las reseñas limpias fueron transformadas en representaciones vectoriales de alta dimensión (embeddings) utilizando el modelo paraphrase-multilingual-MiniLM-L12-v2 de sentence-transformers.
Estos embeddings capturan el significado semántico del texto, permitiendo comparar reseñas por similitud conceptual y no solo por coincidencia léxica.

#### Indexación en ChromaDB
Los embeddings generados fueron indexados en una base de datos vectorial ChromaDB, optimizada para búsqueda semántica en tiempo real.
Esto permite al sistema localizar las reseñas más relevantes según el contexto y la intención de la consulta del usuario, sirviendo como base de conocimiento para la capa de generación.

### 3. Fine-Tuning del Modelo de Lenguaje (LLM)
Se utilizó como base el modelo microsoft/Phi-3-mini-4k-instruct, un modelo preentrenado optimizado para tareas conversacionales.
Se generó un dataset sintético en formato de conversación (pregunta, contexto, respuesta) y se aplicó un proceso de Fine-Tuning con LoRA, especializando al modelo en la tarea de responder preguntas sobre profesores de CUCEI con un tono natural y coherente.

### 4. Sistema de Recomendación (Chatbot)
El chatbot integra todas las capas del sistema.
Al recibir una pregunta, primero busca en ChromaDB las reseñas más relevantes mediante RAG, y luego utiliza el modelo fine-tuned para generar una respuesta fundamentada y fluida en lenguaje natural.
De esta manera, el sistema combina recuperación de información con generación contextualizada, ofreciendo resultados de alta precisión y valor práctico.

## Desafíos y Aprendizajes Clave
El principal desafío fue la adquisición de un dataset grande y de calidad, ya que los intentos de scraping automatizado a gran escala resultaron inviables por las restricciones de las plataformas.
Esto confirmó que, con un conjunto de datos limitado, el modelo tiende a sobreajustarse severamente, dificultando su generalización para un uso en producción.

No obstante, la combinación RAG + Fine-Tuning demostró ser la arquitectura más robusta y escalable, y su desempeño puede mejorar significativamente con acceso a un volumen mayor de datos verificados y de origen institucional.

## Estado del Proyecto
Finalizado.
El proyecto culmina con un pipeline de NLP de extremo a extremo completamente funcional y un modelo entrenado que valida la viabilidad del enfoque RAG + Fine-Tuning.
El código y los hallazgos obtenidos constituyen una base sólida y un caso de estudio para futuras investigaciones y desarrollos que busquen optimizar la recomendación docente con datos de mayor escala y calidad.