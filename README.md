# Laboratorio II — Análisis de Sentimientos (Sentiment140)

**Procesamiento de Lenguaje Natural — Universidad Sergio Arboleda, 2026 S02**
**Autor:** Carlos Cardona

## Resumen

Sistema de análisis binario de sentimientos sobre el dataset Sentiment140, con
experimentación registrada en MLflow y despliegue de una API FastAPI que resuelve
el modelo final desde MLflow Model Registry (`sentiment140@champion`).

## URLs de acceso

- **API:** `http://54.87.148.165:8000` — documentación interactiva en `/docs`
- **MLflow Tracking Server:** `http://54.87.148.165:5000`
- **Experimento MLflow:** `nlp-lab2-sentiment140`

> Nota: la instancia EC2 no tiene una IP fija (Elastic IP) asignada, por lo que la
> IP pública puede cambiar si la instancia se detiene y se vuelve a iniciar. La IP
> vigente al momento de la entrega se comunica por el canal definido para la
> evaluación.

## Configuración final seleccionada

Tras las comparaciones obligatorias de las Secciones 3 y 4 de la guía:

| Componente | Decisión |
|---|---|
| Preprocesamiento | Igual a B0, salvo `elongation=normalize` (normalización de alargamientos, ej. "sooo" → "soo") |
| Representación | TF-IDF con unigramas + bigramas |
| Clasificador | Logistic Regression |

**Resultados de validación cruzada (3 folds, muestra de 200.000 registros):**

| Configuración | macro_f1_mean |
|---|---|
| T0 (referencia trivial) | 0.3334 |
| B0 (baseline) | 0.7831 |
| Pipeline candidato (C_LOGREG) | 0.8010 |

**Ablación:** el candidato difiere de B0 en dos decisiones (elongation y
representation). Ambas ablaciones dieron `macro_f1_delta` positivo, confirmando
que ambas decisiones aportan valor; se mantuvo el candidato sin cambios.

**Modelo final:** reentrenado con los 1.360.000 registros de `train` y evaluado
una sola vez sobre los 240.000 registros de `test`.

- `test_macro_f1 = 0.8263`
- Registrado en MLflow Model Registry como `sentiment140`, alias `champion`

## Arquitectura de despliegue

Una única instancia EC2 (Ubuntu 24.04) aloja:

- **MLflow Tracking Server** (puerto 5000), con `--backend-store-uri sqlite` local
  y artefactos servidos desde un bucket S3 (`--artifacts-destination`,
  `--serve-artifacts`).
- **API FastAPI** (puerto 8000), que resuelve el modelo en tiempo de ejecución
  desde `sentiment140@champion` — no incluye ningún archivo serializado local.

La experimentación (protocolo, T0, B0, comparaciones obligatorias, ablación y
reentrenamiento final) se ejecutó desde un **SageMaker Notebook Instance**
(`ml.t3.xlarge`), tal como exige la Sección 6 de la guía. La procedencia de cada
run queda registrada mediante el archivo `/opt/ml/metadata/resource-metadata.json`
de esa instancia, copiado sin editar como `provenance/sagemaker-resource-metadata.json`
en cada run experimental y en el run final.

## Endpoints de la API

| Método y ruta | Descripción |
|---|---|
| `POST /api/v1/predict` | Inferencia con el modelo registrado |
| `GET /audit/protocol` | Protocolo experimental |
| `GET /audit/runs` | Todos los runs presentados |
| `GET /audit/contributions` | Contribución individual |
| `GET /audit/model` | Modelo desplegado y su trazabilidad |
| `GET /health` | Estado del servicio |

Contratos exactos de entrada, salida y códigos de error en el Anexo A.5 de la guía.

## Herramientas de inteligencia artificial utilizadas

Se utilizó **Claude** (Anthropic) como asistente conversacional durante todo el
desarrollo del laboratorio, para:

- Diseñar y escribir el código de la API FastAPI (routers, validación del
  contrato del Anexo A, capa de acceso a MLflow).
- Diseñar y escribir los scripts de experimentación (protocolo, T0, B0,
  comparaciones de preprocesamiento/representación/clasificador, ablación,
  reentrenamiento final, análisis de errores).
- Guiar la configuración de la infraestructura en AWS Academy (EC2, S3,
  SageMaker Notebook Instance, systemd, Security Groups).
- Diagnosticar y corregir errores de entorno (incompatibilidades de versión de
  Python/librerías, memoria insuficiente, espacio en disco, configuración de
  red y CORS de MLflow).
- Redactar este README y el notebook de auditoría.

Todo el código generado fue ejecutado, revisado y verificado por el autor antes
de incorporarse a la solución. El clasificador final fue entrenado por el autor
con los datos del laboratorio, sin usar APIs externas de clasificación ni
modelos preentrenados para la tarea (solo embeddings preentrenados de spaCy se
evaluaron como una de las representaciones comparadas, según lo permite la guía).

No se compartieron contraseñas, claves, tokens ni información sensible de AWS
Academy con ninguna herramienta externa.
