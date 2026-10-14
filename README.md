# Laboratorio II — Análisis de sentimientos sobre Sentiment140

Sistema de clasificación binaria de sentimientos (`negative` / `positive`) sobre tweets, con experimentación trazable en **MLflow**, modelo publicado en el **Model Registry** y servido mediante una **API FastAPI** desplegada en AWS.

Curso de Procesamiento de Lenguaje Natural · Universidad Sergio Arboleda · 2026 S02.

## Contenido

1. [Resumen](#resumen)
2. [Direcciones y entregables](#direcciones-y-entregables)
3. [Arquitectura](#arquitectura)
4. [Datos y protocolo experimental](#datos-y-protocolo-experimental)
5. [Experimentos y resultados](#experimentos-y-resultados)
6. [Pipeline final](#pipeline-final)
7. [Análisis de errores](#análisis-de-errores)
8. [Modelo registrado](#modelo-registrado)
9. [API](#api)
10. [Auditoría y verificación](#auditoría-y-verificación)
11. [Estructura del repositorio](#estructura-del-repositorio)
12. [Reproducción y despliegue](#reproducción-y-despliegue)
13. [Procedencia](#procedencia)
14. [Limitaciones y notas](#limitaciones-y-notas)
15. [Uso de inteligencia artificial generativa](#uso-de-inteligencia-artificial-generativa)
16. [Seguridad](#seguridad)

## Resumen

| | |
|---|---|
| **Dataset** | `adilbekovich/Sentiment140Twitter`, revisión `b6037e127257d95b9b23d31f78b264b9ebe697fd` |
| **Modelo final** | Preprocesamiento de B0 + TF-IDF (unigramas y bigramas) + Logistic Regression |
| **Macro-F1 en validación cruzada** | 0.79979 (3 folds, 200.000 registros) |
| **Macro-F1 en `test`** | **0.8255** (240.000 registros; ver la nota en *Limitaciones y notas*) |
| **Entrenamiento final** | 1.360.000 registros (`train` completo) |
| **Registro** | `sentiment140@champion` (versión 1) |
| **Versiones clave** | scikit-learn 1.5.2 · MLflow 3.17.0 |

El flujo seguido fue: referencias (T0, B0) → comparaciones de preprocesamiento → comparaciones de representación → comparaciones de clasificador → pipeline candidato → ablación → reentrenamiento con todo `train` → evaluación en `test` → análisis de errores → registro y despliegue.

## Direcciones y entregables

Las direcciones públicas dependen de la IP de la instancia EC2, que cambia al reiniciar el laboratorio. La IP vigente se informa junto con la entrega. En los ejemplos de este documento se usan estas variables:

```bash
export API=http://<IP_PUBLICA_EC2>:8000      # API FastAPI
export MLFLOW_TRACKING_URI=http://<IP_PUBLICA_EC2>:5000   # MLflow Tracking Server
```

| Entregable | Ubicación |
|---|---|
| API pública | `$API` (puerto 8000 de la EC2 `nlp-lab2-mlflow-api`) |
| MLflow Tracking Server | `$MLFLOW_TRACKING_URI` (puerto 5000), experimento `nlp-lab2-sentiment140` |
| Modelo | `sentiment140@champion` en el Model Registry |
| Cuaderno de auditoría | [`notebooks/experiment_audit.ipynb`](notebooks/experiment_audit.ipynb) |
| Análisis de errores | [`reports/error_analysis.csv`](reports/error_analysis.csv) y [`reports/error_analysis.md`](reports/error_analysis.md), también registrados como artefactos del run final |
| Repositorio | https://github.com/ALMA3112/APIs2 |

## Arquitectura

```
                     ┌───────────────────────────────────────────┐
                     │ SageMaker Notebook Instance               │
                     │ nlp-lab2-cardona                          │
                     │  - experimentos (T0, B0, P_*, R_*, C_*)   │
                     │  - entrenamiento y registro del modelo    │
                     └──────────────────┬────────────────────────┘
                                        │ registra runs, métricas y artefactos
                                        ▼
┌───────────────────────────── EC2 nlp-lab2-mlflow-api ─────────────────────────────┐
│                                                                                    │
│  MLflow Tracking Server :5000          API FastAPI :8000                           │
│   - backend: SQLite                     - POST /api/v1/predict                      │
│   - artefactos: S3                      - GET  /health                              │
│   - Model Registry + alias              - GET  /audit/{protocol,runs,               │
│                  ▲                                contributions,model}              │
│                  └─────── consulta en cada petición ──────┘                         │
└────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                         S3 bucket-carlos-nlp-2026 (artefactos y respaldos)
```

- Los runs se ejecutan **desde el SageMaker Notebook Instance** asignado por el curso, y la procedencia queda registrada en cada run.
- El Tracking Server y la API corren en la misma EC2 como servicios `systemd` (`mlflow.service` y `sentiment-api.service`), por lo que arrancan solos tras un reinicio.
- La API **no contiene el modelo**: lo resuelve en cada petición desde `sentiment140@champion` y lo guarda en memoria por versión.
- Los endpoints de auditoría consultan MLflow en tiempo de ejecución, sin copias locales.

## Datos y protocolo experimental

- **Dataset:** `adilbekovich/Sentiment140Twitter` cargado con `load_dataset(..., revision="b6037e127257d95b9b23d31f78b264b9ebe697fd")`. `train`: 1.360.000 registros; `test`: 240.000. Etiqueta 0 = `negative`, 1 = `positive`.
- **Muestra:** 200.000 registros de `train`, estratificada, semilla 42. El índice original de `train` se conservó.
- **Validación cruzada:** 3 folds estratificados, con mezcla aleatoria y semilla 42 (`StratifiedKFold`).
- **Fuente de verdad del protocolo:** el artefacto `protocol/partitions.csv` del run de protocolo (200.000 filas con `index,fold`, índices únicos y ascendentes). Todo el código posterior lee los folds de ahí, sin regenerarlos.
- **Métrica:** Macro-F1 por fold, su media y la desviación estándar poblacional (`ddof=0`). Cada configuración conserva los mismos hiperparámetros en los tres folds.
- **Run de protocolo (único):** `99482e00b9c94bdc9755afde981c331e`. Registra los parámetros comunes y los artefactos `protocol/partitions.csv` y `protocol/members.csv`.
- **`test` se reservó** para la evaluación final y no se usó para ninguna decisión.

## Experimentos y resultados

Todos los resultados son Macro-F1 en validación cruzada de 3 folds. Cada fila es un run experimental de MLflow (`lab_run_type=experiment`).

| `lab_experiment_id` | Etapa | Descripción | Media | Desv. | Run |
|---|---|---|---|---|---|
| `T0` | referencia | Predice la clase más frecuente | 0.33338 | 0.00000 | `ec33e9da` |
| `B0` | baseline | Minúsculas, URL→`url`, menciones→`user`, espacios; BoW unigramas; Logistic Regression | 0.78200 | 0.00089 | `03e4eb1c` |
| `P_STOPWORDS` | preprocesamiento | B0 + eliminar stopwords | 0.75833 | 0.00146 | `6b12c3dc` |
| `P_STOPWORDS_NEGATION` | preprocesamiento | B0 + eliminar stopwords preservando negadores | 0.76753 | 0.00055 | `33f423d0` |
| `P_LEMMA` | preprocesamiento | B0 + lematización | 0.77890 | 0.00156 | `d517b957` |
| `P_ELONGATION` | preprocesamiento | B0 + normalizar alargamientos | 0.78245 | 0.00055 | `c1115730` |
| `P_EMOJI` | preprocesamiento | B0 + emojis/emoticonos a texto | 0.78165 | 0.00082 | `dca0d382` |
| `R_BOW` | representación | BoW unigramas | 0.78200 | 0.00089 | `b3687e9a` |
| `R_TFIDF_UNI` | representación | TF-IDF unigramas | 0.78800 | 0.00024 | `629ee090` |
| `R_TFIDF_UNI_BI` | representación | **TF-IDF unigramas + bigramas** | **0.79979** | 0.00038 | `0cc7f8b5` |
| `R_SPACY` | representación | Embeddings de spaCy (vectores preentrenados) | 0.68673 | 0.00084 | `fc7607d9` |
| `C_LOGREG` | clasificador | **Logistic Regression** | **0.79979** | 0.00038 | `4b34cf05` |
| `C_LINEAR_SVM` | clasificador | Linear SVM | 0.79706 | 0.00074 | `ba8c13b4` |
| `C_SGD` | clasificador | SGDClassifier | 0.78004 | 0.00063 | `9819d8cd` |
| `ABLATION` | ablación | Revierte `representation` a BoW de B0 | 0.78200 | 0.00089 | `76f7071f` |

Los detalles exactos de cada configuración (lista de negadores, regla de alargamientos, conversión de emojis, modelo de spaCy y método de vector por documento) están en el artefacto `run/configuration.json` de cada run, con el esquema del Anexo A.3.

<!-- CONFIG-TABLE:START -->
### Detalle de las configuraciones

Valores extraídos de `run/configuration.json` de cada run; se muestran solo los campos que difieren de B0 (esquema del Anexo A.3).

| Experimento | `configuration_id` | Diferencias frente a B0 |
|---|---|---|
| `T0` | `CFG_T0` | clasificador `most_frequent`, sin preprocesamiento ni representación |
| `B0` | `CFG_B0` | configuración base (ver [Pipeline final](#pipeline-final)) |
| `P_STOPWORDS` | `CFG_P_STOPWORDS` | `preprocessing.resources.spacy` = `"3.8.2"`; `preprocessing.resources.spacy_model` = `"en_core_web_sm"`; `preprocessing.resources.spacy_model_version` = `"3.8.0"`; `preprocessing.stopwords` = `"remove"` |
| `P_STOPWORDS_NEGATION` | `CFG_P_STOPWORDS_NEGATION` | `preprocessing.negators` = `["cannot", "n't", "neither", "never", "no", "nobody", "none", "nor", "not", "nothing", "nowhere", "n’t"]`; `preprocessing.resources.spacy` = `"3.8.2"`; `preprocessing.resources.spacy_model` = `"en_core_web_sm"`; `preprocessing.resources.spacy_model_version` = `"3.8.0"`; `preprocessing.stopwords` = `"remove_preserve_negation"` |
| `P_LEMMA` | `CFG_P_LEMMA` | `preprocessing.lemmatize` = `true`; `preprocessing.resources.spacy` = `"3.8.2"`; `preprocessing.resources.spacy_model` = `"en_core_web_sm"`; `preprocessing.resources.spacy_model_version` = `"3.8.0"` |
| `P_ELONGATION` | `CFG_P_ELONGATION` | `preprocessing.elongation` = `"normalize"`; `preprocessing.elongation_spec` = `"reduce_repeated_chars_to_2"` |
| `P_EMOJI` | `CFG_P_EMOJI` | `preprocessing.emoji` = `"text"`; `preprocessing.emoji_spec` = `"emoji.demojize"`; `preprocessing.resources.emoji` = `"2.14.0"` |
| `R_BOW` | `CFG_B0` | sin diferencias: misma configuración que B0 |
| `R_TFIDF_UNI` | `CFG_R_TFIDF_UNI` | `representation.type` = `"tfidf"` |
| `R_TFIDF_UNI_BI` | `CFG_R_TFIDF_UNI_BI` | `representation.ngram_range` = `[1, 2]`; `representation.type` = `"tfidf"` |
| `R_SPACY` | `CFG_R_SPACY` | `representation.library` = `"spacy"`; `representation.library_version` = `"3.8.2"`; `representation.ngram_range` = `[]`; `representation.parameters.document_vector_method` = `"mean_token_vectors"`; `representation.spacy_model` = `"en_core_web_md"`; `representation.spacy_model_version` = `"3.8.0"`; `representation.type` = `"spacy_embedding"` |
| `C_LOGREG` | `CFG_R_TFIDF_UNI_BI` | `representation.ngram_range` = `[1, 2]`; `representation.type` = `"tfidf"` |
| `C_LINEAR_SVM` | `CFG_C_LINEAR_SVM` | `classifier.type` = `"linear_svm"`; `representation.ngram_range` = `[1, 2]`; `representation.type` = `"tfidf"` |
| `C_SGD` | `CFG_C_SGD` | `classifier.parameters.random_state` = `42`; `classifier.type` = `"sgd"`; `representation.ngram_range` = `[1, 2]`; `representation.type` = `"tfidf"` |
| `ABLATION` | `CFG_B0` | sin diferencias: misma configuración que B0 |
<!-- CONFIG-TABLE:END -->

### Lectura de los resultados

- **Preprocesamiento.** Ninguna transformación mejoró a B0 más de lo que varían los folds: la mejor, `P_ELONGATION`, supera a B0 por 0.00045, menos que la desviación entre folds. Eliminar stopwords empeora el resultado, y conservar los negadores recupera parte de la pérdida (0.7583 → 0.7675). Por eso se mantuvo el preprocesamiento de B0.
- **Representación.** TF-IDF mejora a BoW, y añadir bigramas aporta casi 0.012 más (0.7880 → 0.7998). Los vectores de spaCy rindieron bastante peor (0.6867); es probable que se deba a que un vector promedio de palabras pierde información de n-gramas y de orden que los modelos dispersos sí conservan.
- **Clasificador.** Logistic Regression supera a Linear SVM por 0.003 y a SGD por casi 0.02. Los tres usan la misma representación y el mismo preprocesamiento.
- **Candidato.** Preprocesamiento de B0 + TF-IDF (1, 2) + Logistic Regression, que coincide con `C_LOGREG` y `R_TFIDF_UNI_BI` (`CFG_R_TFIDF_UNI_BI`).

### Ablación

El candidato difiere de B0 en una sola decisión ablacionable (`representation`), por lo que se evaluó esa reversión:

| | Macro-F1 medio |
|---|---|
| Candidato (`4b34cf05`) | 0.79979 |
| Ablación: BoW de B0 (`76f7071f`) | 0.78200 |
| `macro_f1_delta` | **0.017786** |

Se conservó el candidato como configuración final.

## Pipeline final

```text
texto crudo
  └─ minúsculas
  └─ URL → " url "
  └─ @menciones → " user "
  └─ espacios normalizados
  └─ TfidfVectorizer(ngram_range=(1, 2))      # 3.441.998 características al entrenar con train completo
  └─ LogisticRegression()                      # hiperparámetros por defecto
  └─ etiqueta: negative | positive
```

| Componente | Valor |
|---|---|
| Biblioteca | scikit-learn 1.5.2 |
| Entrenamiento | 1.360.000 registros (todo `train`) |
| Evaluación | 240.000 registros de `test` (ver nota en *Limitaciones y notas*) |
| Macro-F1 en `test` | 0.8255187 |
| Errores en `test` | 41.875 de 240.000 |

La configuración efectiva está en `run/configuration.json` del run final y es equivalente a la del run experimental seleccionado (mismo `configuration_id`: `CFG_R_TFIDF_UNI_BI`).

## Análisis de errores

Muestra aleatoria (semilla 42) de 40 errores de `test`, con 23 negativos y 17 positivos. Una primera clasificación automática se revisó manualmente, caso por caso.

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

Los dos patrones más frecuentes:

1. **`other` (16 de 40):** tweets sin una señal de sentimiento clara o con una etiqueta dudosa. Sentiment140 se etiquetó automáticamente a partir de emoticonos, por lo que parte de estos errores se debe al ruido de la etiqueta y no al modelo.
2. **`mixed` (8 de 40):** tweets con polaridad mezclada. Un modelo de n-gramas suma el peso de cada término sin determinar cuál domina.

La interpretación completa está en [`reports/error_analysis.md`](reports/error_analysis.md).

## Modelo registrado

| | |
|---|---|
| Nombre en el Model Registry | `sentiment140` |
| Alias | `champion` (no se usan los *stages* heredados) |
| Versión | 1 |
| Tipo | `mlflow.pyfunc` con el preprocesamiento, la vectorización y el clasificador encapsulados |
| Entrada | Texto sin procesar |
| Salida | `negative` o `positive` |
| Run final | `a6bb89d7fd4d4657946bc1cbadf605bc` (`lab_run_type=final`) |
| Run experimental seleccionado | `4b34cf05ac4041978fa0cc28efc95f4c` (`C_LOGREG`) |

Cadena de trazabilidad: `POST /api/v1/predict` → `sentiment140@champion` → run final → run experimental seleccionado.

El run final registra `training_size=1360000`, la métrica `test_macro_f1` y los artefactos `run/configuration.json`, `provenance/sagemaker-resource-metadata.json`, `reports/error_analysis.csv` y `reports/error_analysis.md`, además del modelo.

## API

Implementada con FastAPI (`api/main.py`). Todas las respuestas son `application/json`.

| Método y ruta | Descripción |
|---|---|
| `POST /api/v1/predict` | Inferencia con el modelo registrado |
| `GET /health` | Resuelve `sentiment140@champion` y verifica una inferencia |
| `GET /audit/protocol` | Protocolo experimental (200 solo si existe exactamente un protocolo) |
| `GET /audit/runs` | Todos los runs presentados, con parámetros, métricas, etiquetas, artefactos y configuración |
| `GET /audit/contributions` | Contribución por integrante y runs inválidos o sin atribuir |
| `GET /audit/model` | Modelo desplegado y su trazabilidad |

### Ejemplos

```bash
# Un texto
curl -s -X POST $API/api/v1/predict -H "Content-Type: application/json" \
     -d '{"text": "this movie was surprisingly good"}'
# {"model_run_id":"a6bb89d7fd4d4657946bc1cbadf605bc","predictions":["positive"]}

# Un lote (entre 1 y 32 textos, se conserva el orden)
curl -s -X POST $API/api/v1/predict -H "Content-Type: application/json" \
     -d '{"text": ["i loved it", "worst day ever", "not bad at all"]}'
# {"model_run_id":"...","predictions":["positive","negative","positive"]}

# Salud
curl -s $API/health
# {"status":"ok","model_run_id":"a6bb89d7fd4d4657946bc1cbadf605bc"}
```

### Validación y errores

| Situación | Respuesta |
|---|---|
| `text` ausente, `null`, vacío o solo espacios; lista vacía; más de 32 elementos; texto de más de 1.000 caracteres; tipos distintos de `string` o lista; algún elemento inválido; JSON mal formado | HTTP 4xx (422), sin resultados parciales |
| MLflow no disponible en `/audit/*` | HTTP 503, `detail = mlflow_unavailable` |
| `/audit/protocol` sin exactamente un protocolo presentado | HTTP 409, `detail = protocol_not_unique` |
| `/audit/model` sin alias `champion` | HTTP 404, `detail = champion_not_found` |
| `/audit/model` con alias que no apunta a un run final válido | HTTP 409, `detail = champion_invalid` |
| `/health` con el modelo no disponible | HTTP 503, `{"status": "unavailable", "model_run_id": null}` |

`/api/v1/predict`, `/health`, `/audit/model` y el alias `champion` identifican el mismo run final. Las listas de identificadores y rutas se devuelven ordenadas lexicográficamente; `predictions` conserva el orden de entrada.

## Auditoría y verificación

| Herramienta | Qué verifica |
|---|---|
| [`notebooks/experiment_audit.ipynb`](notebooks/experiment_audit.ipynb) | Reconstruye desde el Tracking Server el protocolo, los runs, la consistencia aritmética de las métricas, la ablación, las contribuciones y la cadena alias → run final → run seleccionado |
| [`src/audit_contract.py`](src/audit_contract.py) | Valida los runs frente al Anexo A: etiquetas, métricas (media y desviación poblacional), artefactos, procedencia, esquema de `run/configuration.json`, diferencias frente a B0 y contribución individual |
| [`api/smoke_test.py`](api/smoke_test.py) | Pruebas de contrato de la API contra una URL: entradas válidas e inválidas, límites, orden del lote, tiempos y coherencia del run final |
| [`src/reproduce.py`](src/reproduce.py) | Recalcula B0 y el candidato con `partitions.csv` y compara con las métricas registradas (diferencia máxima: 0) |

```bash
python api/smoke_test.py $API
python src/audit_contract.py
```

## Estructura del repositorio

```text
.
├── README.md
├── api/
│   ├── main.py                 # API FastAPI
│   ├── requirements.txt
│   └── smoke_test.py           # pruebas de contrato de la API
├── deploy/
│   └── sentiment-api.service   # unidad systemd de la API
├── notebooks/
│   └── experiment_audit.ipynb  # cuaderno de auditoría
├── reports/
│   ├── error_analysis.csv
│   └── error_analysis.md
└── src/
    ├── common.py               # constantes, preprocesamiento B0, carga de datos y particiones
    ├── model_def.py            # modelo pyfunc (SentimentModel)
    ├── reproduce.py            # reproduce B0 y el candidato
    ├── train_final.py          # entrenamiento final, evaluación en test y muestra de errores
    ├── register_final.py       # run final y registro de sentiment140@champion
    ├── audit_contract.py       # validador del contrato
    └── build_notebook.py       # genera el cuaderno de auditoría
```

## Reproducción y despliegue

### Entorno del notebook de SageMaker

```bash
conda create -y -p ~/SageMaker/lab2/envs/lab2 python=3.11
source activate ~/SageMaker/lab2/envs/lab2
pip install "mlflow==3.17.0" "scikit-learn==1.5.2" pandas numpy joblib datasets fastapi uvicorn ipykernel
export MLFLOW_TRACKING_URI=http://<IP_PUBLICA_EC2>:5000
```

El entorno se crea dentro de `~/SageMaker` porque es la única ruta que persiste cuando el Notebook Instance se detiene y se vuelve a iniciar.

### Verificar el protocolo y el código

```bash
python src/audit_contract.py     # contrato del Anexo A sobre los runs registrados
python src/reproduce.py          # recalcula B0 y el candidato (usa solo train, no toca test)
```

### Modelo final

```bash
python src/train_final.py        # entrena con todo train, evalúa test UNA vez y genera la muestra de errores
python src/register_final.py     # crea el run final y registra sentiment140@champion
```

`train_final.py` se niega a ejecutarse de nuevo si ya existe `artifacts/final_results.json`, para no evaluar `test` más de una vez.

### Tracking Server (EC2)

El servidor corre como `mlflow.service` con SQLite como almacén de metadatos y S3 para los artefactos:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:////home/ubuntu/mlflow/mlflow.db \
  --artifacts-destination s3://bucket-carlos-nlp-2026/mlflow-artifacts \
  --allowed-hosts "*" --cors-allowed-origins "*"
```

### API (EC2)

```bash
mkdir -p ~/api && cd ~/api
python3 -m venv venv
./venv/bin/pip install fastapi==0.143.0 uvicorn==0.54.0 starlette==1.7.0 pydantic==2.14.1 \
  mlflow==3.17.0 scikit-learn==1.5.2 numpy==2.4.6 scipy==1.17.1 pandas==3.0.6 \
  joblib==1.6.0 cloudpickle==3.1.2 requests==2.34.2
cp <repositorio>/api/main.py .
sudo cp <repositorio>/deploy/sentiment-api.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now sentiment-api
curl -s localhost:8000/health
```

La API se configura con la variable `MLFLOW_TRACKING_URI`; dentro de la EC2 apunta a `http://127.0.0.1:5000`, por lo que el cambio de IP pública no afecta su funcionamiento.

## Procedencia

Todos los runs experimentales y el run final se ejecutaron desde el SageMaker Notebook Instance `nlp-lab2-cardona` (integrante `E01`). Cada run registra:

- `notebook_arn` (el `ResourceArn` del notebook, con la forma `arn:aws:sagemaker:us-east-1:<cuenta>:notebook-instance/nlp-lab2-cardona`);
- `lab_member_id`;
- una copia sin editar de `/opt/ml/metadata/resource-metadata.json` como artefacto `provenance/sagemaker-resource-metadata.json`, cuyo `ResourceArn` coincide con `notebook_arn`.

Contribución individual (`GET /audit/contributions`): 11 configuraciones válidas distribuidas en las etapas de ablación, clasificador, preprocesamiento y representación.

## Limitaciones y notas

- **Scripts de experimentos.** Los scripts originales de las comparaciones `P_*`, `R_SPACY`, `C_*` y de la ablación estaban en una carpeta fuera de `~/SageMaker` y se perdieron cuando el Notebook Instance se reinició. Su evidencia (métricas por fold, configuración efectiva y procedencia) permanece íntegra en MLflow. Para `B0` y el candidato, el código se reescribió en `src/reproduce.py` y reproduce los Macro-F1 registrados con diferencia máxima de 0.
- **Evaluación en `test`.** El modelo final se evaluó en `test` una primera vez. Al perderse los archivos locales por el reinicio del notebook, el entrenamiento final se repitió para regenerar el modelo y la muestra de errores. Ambas ejecuciones dieron exactamente el mismo Macro-F1 (0.8255187213071261); la configuración no se modificó en ningún momento a partir de ese resultado y se eligió solo con validación cruzada.
- **Python.** El modelo se entrenó con Python 3.11 y la API corre con Python 3.12; las versiones de scikit-learn, NumPy y SciPy son idénticas y las predicciones coinciden.
- **Ruido de etiquetas.** Sentiment140 se etiquetó automáticamente a partir de emoticonos, lo que impone un techo al rendimiento y explica parte de los errores.
- **Alcance del análisis de errores.** La muestra es de 40 casos, por lo que las frecuencias son indicativas y la asignación a una categoría implica criterio humano.
- **Representación con spaCy.** El promedio de vectores de palabras rindió mal frente a TF-IDF; no se exploró otra forma de obtener el vector del documento.
- **IP pública.** La IP de la EC2 cambia al reiniciar el laboratorio; la API y el Tracking Server deben estar encendidos durante la ventana de evaluación.

## Uso de inteligencia artificial generativa

Se usó **únicamente Claude (Anthropic)**, con los siguientes propósitos:

| Propósito | Detalle |
|---|---|
| Programación | Código de entrenamiento final, registro en MLflow, API FastAPI y servicio `systemd` |
| Experimentación y pruebas | Reescritura del código de verificación (`reproduce.py`), validador del contrato del Anexo A y pruebas de contrato de la API |
| Análisis | Revisión del cumplimiento de la guía y apoyo en el análisis de errores (las categorías se revisaron manualmente) |
| Despliegue y depuración | Configuración de la EC2, diagnóstico de errores (por ejemplo, el experimento borrado en MLflow) |
| Documentación | Redacción de este README y del cuaderno de auditoría |

El clasificador final fue entrenado por el equipo con los datos del laboratorio. No se usaron APIs externas ni modelos previamente ajustados para esta tarea. Los resultados, el código y la configuración se comprobaron contra lo registrado en MLflow.

## Seguridad

El repositorio no contiene credenciales, claves ni tokens. El acceso del servidor a S3 se hace con el rol de la instancia, y el Tracking Server no requiere autenticación en este laboratorio.
