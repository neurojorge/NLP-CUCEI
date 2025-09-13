import re
import pandas as pd
import numpy as np

def get_optimal_weight_iic(departamento, division=None):
    """
    Función optimizada para obtener pesos específicos de IIC
    Combina peso de departamento (70%) + división (30%)
    """
    dept_key = str(departamento).upper().strip()
    dept_weight = PESOS_DEPARTAMENTOS_IIC.get(dept_key, 0.10)
    
    if division:
        div_key = str(division).upper().strip()
        div_weight = PESOS_DIVISIONES.get(div_key, 0.10)
        final_weight = 0.7 * dept_weight + 0.3 * div_weight
    else:
        final_weight = dept_weight
        
    return min(final_weight, 1.0)  # Cap máximo 1.0

PESOS_DEPARTAMENTOS_IIC = {
    'DEPARTAMENTO DE CIENCIAS COMPUTACIONALES': 1.0, 
    'DEPTO. DE CIENCIAS COMPUTACIONALES': 1.0, 
    'CIENCIAS COMPUTACIONALES': 1.0, 
    'DEPARTAMENTO DE INNOVACIÓN BASADA EN LA INFORMACIÓN Y EL CONOCIMIENTO': 0.95, 
    'INNOVACION BASADA EN LA INFORMACION Y EL CONOCIMIENTO': 0.95, 
    'DEPTO. DE INNOVACIÓN BASADA EN LA INFORMACIÓN Y EL CONOCIMIENTO': 0.95, 
    'DEPARTAMENTO DE MATEMÁTICAS': 0.88, 
    'DEPTO. DE MATEMATICAS': 0.88, 
    'MATEMATICAS': 0.88, 
    'DEPTO. DE MATEMÁTICAS': 0.88, 
    'DEPARTAMENTO DE INGENIERÍA ELECTRO-FOTÓNICA': 0.82, 
    'DEPARTAMENTO DE FÍSICA': 0.75, 
    'DEPARTAMENTO DE INGENIERÍA INDUSTRIAL': 0.7, 
    'DEPARTAMENTO DE INGENIERÍA DE PROYECTOS': 0.65, 
    'DEPARTAMENTO DE INGENIERÍA MECÁNICA ELÉCTRICA': 0.62, 
    'DEPARTAMENTO DE INGENIERÍA CIVIL Y TOPOGRAFÍA': 0.55, 
    'DEPARTAMENTO DE INGENIERÍA QUÍMICA': 0.5, 
    'DEPARTAMENTO DE BIOINGENIERÍA TRASLACIONAL': 0.45, 
    'DEPARTAMENTO DE QUÍMICA': 0.35, 
    'DEPARTAMENTO DE FARMACOBIOLOGÍA': 0.3, 
    'DEPARTAMENTO DE MADERA, CELULOSA Y PAPEL': 0.25, 
    'No encontrado': 0.1, 
    'División no encontrada': 0.1, 
    'DEPTO. NO ENCONTRADO': 0.1, 
    'Sin departamento': 0.1
    # (Se omiten alias redundantes por brevedad)
}
PESOS_DIVISIONES = {
    'DIVISION DE TECNOLOGIAS PARA LA INTEGRACION CIBER-HUMANA': 1.0, 
    'División de Tecnologías para la Integración Ciber-Humana': 1.0, 
    'TECNOLOGIAS PARA LA INTEGRACION CIBER-HUMANA': 1.0, 
    'DIVISION DE CIENCIAS BASICAS': 0.65, 
    'División de Ciencias Básicas': 0.65, 
    'CIENCIAS BASICAS': 0.65, 
    'DIVISION DE INGENIERIAS': 0.7, 
    'División de Ingenierías': 0.7, 
    'INGENIERIAS': 0.7, 
    'Sin división': 0.1, 
    'División no encontrada': 0.1
}
def clean_text(text):
    if pd.isna(text) or text == "": return ""
    text = str(text)
    text = re.sub(r'Calificación:\s*[\d\.]+/10\s*-\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_rating(text):
    if pd.isna(text): return None
    match = re.search(r'Calificación:\s*([\d\.]+)/10', str(text))
    if match:
        try: return float(match.group(1))
        except: return None
    return None

def create_enriched_text(row):
    parts = []
    if pd.notna(row['DEPARTAMENTO']) and row['DEPARTAMENTO'] != "":
        parts.append(f"Departamento: {str(row['DEPARTAMENTO']).replace('DEPTO. DE ', '').strip()}")
    if pd.notna(row['DIVISION']) and row['DIVISION'] != "":
        parts.append(f"División: {str(row['DIVISION']).replace('División de ', '').strip()}")
    comment = clean_text(row['COMENTARIOS'])
    if comment: parts.append(f"Comentario: {comment}")
    return " | ".join(parts)

# (Funciones de Materia (V4) omitidas intencionalmente ya que los datos son NaN)
def get_subject_weight(subject_name):
    # Función dummy para evitar errores de importación si algo la llama por error
    return 0.1

# =============================================================================
# SOLUCIÓN V5/V6: RESCATE DE SENTIMIENTO TEXTUAL
# =============================================================================
PALABRAS_MUY_POSITIVAS = ['excelente', 'increible', 'mejor', 'recomendadisimo', 'ame', 'perfecto', 'fascinante']
PALABRAS_POSITIVAS = ['buen', 'buena', 'recomiendo', 'explica bien', 'claro', 'aprobe', 'facil', 'ayuda', 'paciente', 'aprendes']
PALABRAS_MUY_NEGATIVAS = ['pesimo', 'horrible', 'peor', 'terrible', 'nefasto', 'evitenlo', 'cuidado', 'barco', 'basura']
PALABRAS_NEGATIVAS = ['malo', 'aburrido', 'confuso', 'dificil', 'no aprendi', 'no sabe', 'reprobe', 'grosero', 'prepotente']

def get_sentiment_score(comment_text):
    if pd.isna(comment_text): return 0.0 # Neutro
    text = str(comment_text).lower()
    score = 0
    for word in PALABRAS_MUY_POSITIVAS: score += 2 * text.count(word)
    for word in PALABRAS_POSITIVAS: score += 1 * text.count(word)
    for word in PALABRAS_MUY_NEGATIVAS: score -= 2 * text.count(word)
    for word in PALABRAS_NEGATIVAS: score -= 1 * text.count(word)
    # Normalizar (Clamp)
    normalized_score = max(-1.0, min(1.0, score / 5.0))
    return normalized_score