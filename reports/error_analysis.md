# Análisis de errores del modelo final

Modelo: TF-IDF (unigramas y bigramas) + Logistic Regression, entrenado con los 1.360.000 registros de `train` (macro-F1 en `test`: 0.8255; 41875 errores en total).
Muestra aleatoria (semilla 42) de 40 errores sobre `test`, con errores de ambas clases (23 negativos y 17 positivos). Una primera clasificación automática por reglas se revisó manualmente, caso por caso; cada texto se asignó a la categoría del fenómeno predominante.

| Categoría | Frecuencia |
|---|---|
| other | 16 |
| mixed | 8 |
| negation | 5 |
| informal | 4 |
| contrast | 3 |
| sarcasm | 2 |
| elongation | 1 |
| hashtag | 1 |

Con 40 casos las frecuencias son aproximadas y la asignación a una categoría implica cierto juicio.

## Interpretación de los dos patrones más frecuentes

**1. `other` (16 de 40): tweets sin señal de sentimiento clara o con etiqueta dudosa.** Son textos casi neutros (preguntas, noticias, comentarios cortos como "i spill the limonade") o casos donde la etiqueta parece contradecir el texto. Sentiment140 etiqueta los tweets de forma automática a partir de emoticonos, y esa supervisión es ruidosa: parte de estos errores no se debe al modelo sino a la etiqueta. Un modelo basado en n-gramas solo puede apoyarse en el vocabulario presente, y cuando este no expresa polaridad, la predicción es casi arbitraria. Es un techo de rendimiento más que un defecto corregible con preprocesamiento.

**2. `mixed` (8 de 40): polaridad mezclada.** El tweet combina expresiones positivas y negativas ("happy Fathers Day... u sure are missed", "thanks for letting me know! ... rude people saying shitty things"). El modelo suma el peso de cada n-grama sin determinar cuál domina ni la intención del autor, así que depende de qué palabras tienen coeficientes más grandes. Representaciones que modelen el orden o el contexto, o más datos con ese tipo de ejemplos, podrían ayudar.

Patrones menores: negación (5), donde los bigramas capturan frases como "nothing can bring me down" solo si aparecen en el entrenamiento, y escritura informal (4).
