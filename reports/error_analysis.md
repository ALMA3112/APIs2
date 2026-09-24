# Análisis de errores — sentiment140

Total de errores en test: 41697 de 240000 (17.37%).

Casos analizados (muestra aleatoria, semilla 42): 20.

## Frecuencia por categoría

| Categoría | Frecuencia |
|---|---|
| other | 6 |
| elongation | 6 |
| negation | 3 |
| informal | 2 |
| hashtag | 1 |
| contrast | 1 |
| intensification | 1 |

## Interpretación

Los dos patrones más frecuentes son **other** (6 casos) y **elongation** (6 casos).

Los dos patrones más frecuentes son **elongation** (6 casos) y **other** (5 casos).

En los casos de "elongation", el alargamiento de caracteres (ej. "sooooo", "haha") 
casi nunca es la causa real del error: actúa como una señal tonal (ironía, cariño, 
desahogo) que el modelo no puede interpretar porque TF-IDF con Logistic Regression 
solo cuenta la frecuencia de palabras, sin captar el tono ni el orden de las frases. 
Por ejemplo, en el caso 154690 ("driving me out of my mind... can't get it out of 
my mind... haha") el tweet es positivo en la etiqueta real, pero el modelo lo predice 
como negativo porque el vocabulario dominante ("driving out of my mind") es negativo, 
sin captar que el "haha" final suaviza el tono a una queja jocosa.

Los casos de "other" comparten un patrón: son tweets breves y neutros en superficie 
(ej. "off to geometry & earth science", "is the only 1 that works around here") donde 
la etiqueta original de Sentiment140 parece depender de contexto externo al texto 
(emoticonos ASCII eliminados en la limpieza, o el tono real del usuario) que el 
pipeline no puede recuperar solo con las palabras presentes.

En conjunto, estos errores muestran la limitación estructural de un modelo bag-of-words: 
funciona bien con vocabulario emocional explícito, pero falla con sarcasmo, ironía y 
sentimiento implícito que depende del contexto o de señales no léxicas.
