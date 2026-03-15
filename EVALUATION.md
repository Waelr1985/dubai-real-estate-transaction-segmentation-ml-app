# End-to-End Evaluation Guide

This guide shows how to evaluate the app across **all critical dimensions**: data quality, model quality, product behavior, deployment/API, performance, reliability, and operations.

## 1) Evaluation Dimensions (what “all aspects” means)

1. **Business fit**: Are discovered segments actionable for real-estate stakeholders?
2. **Data quality**: Schema validity, null handling, drift, and representativeness.
3. **Model quality**: Cluster separation/stability and profile interpretability.
4. **App behavior (Streamlit)**: Upload, segmentation, chart rendering, and error states.
5. **API/deployment quality**: Scoring correctness, schema handling, and latency.
6. **Non-functional quality**: Runtime, memory, reproducibility, and reliability.
7. **MLOps readiness**: Monitoring, retraining triggers, versioning, and rollback.

---

## 2) Fast Evaluation Plan (90-minute audit)

### A. Environment + smoke checks
```bash
pip install -r requirements.txt
python -m src.model_training --help  # or run training entrypoint directly
streamlit run src/app.py
```

Expected outcome:
- app boots without import/runtime errors
- model artifacts in `models/` are readable by app

### B. Data/schema validation
Use the same validation logic used by training and scoring:
```bash
python - <<'PY'
import pandas as pd
from src.data_validation import validate_data

df = pd.read_csv('sample_transactions.csv')
clean = validate_data(df)
print('rows_in=', len(df), 'rows_out=', len(clean), 'cols=', len(clean.columns))
PY
```

Pass criteria:
- required columns survive validation
- invalid rows are dropped/cleaned as expected
- no silent schema corruption

### C. Cluster quality metrics
Run intrinsic clustering metrics (already implemented):
```bash
python - <<'PY'
import pickle, pandas as pd
from src.data_validation import validate_data
from src.data_preprocessing import apply_target_encoding
from src.model_evaluation import evaluate_clusters

df = pd.read_csv('sample_transactions.csv')
df = validate_data(df)
df = apply_target_encoding(df)

pre = pickle.load(open('models/preprocessor.pkl','rb'))
km = pickle.load(open('models/kmeans_model.pkl','rb'))

sil, db = evaluate_clusters(df.copy(), pre, km)
print('silhouette=', sil, 'davies_bouldin=', db)
PY
```

Primary thresholds (suggested):
- Silhouette: avoid material regression vs baseline
- Davies-Bouldin: lower is better; alert on sustained increase
- Also track Calinski-Harabasz from logs for trend comparison

### D. Segment stability test
Evaluate if cluster assignments remain stable under random resampling:
- draw multiple stratified samples
- re-score each sample
- compute consistency (e.g., ARI between runs)

Pass criteria:
- high stability for same model/data distribution
- instability should trigger data-drift or retraining investigation

### E. Business interpretability
For each segment, verify:
- median `actual_worth`, `procedure_area`, property type mix, area mix
- names still match profile reality (e.g., “Premium Villa & Land Investors”)

Pass criteria:
- each cluster has clear economic meaning
- no duplicate/indistinguishable segments

---

## 3) Streamlit App Evaluation

Run:
```bash
streamlit run src/app.py
```

Checklist:
1. Upload valid CSV and verify row count/preview.
2. Upload malformed CSV and confirm user-friendly error.
3. Click **Segment Transactions** and confirm results are persisted in session state.
4. Verify all tabs render:
   - distribution,
   - PCA view,
   - evaluation metrics,
   - centroid heatmap,
   - profiles,
   - radar comparison.
5. Confirm “Azure API” mode warns when endpoint/key are missing.

Pass criteria:
- no crashes across menu flow
- charts render within acceptable time for expected dataset sizes
- warnings/errors are actionable for users

---

## 4) API/Deployment Evaluation

Local emulation of `deployment/score.py` behavior:
```bash
python - <<'PY'
import json, pandas as pd
import deployment.score as s

s.init()
df = pd.read_csv('sample_transactions.csv').head(100)
resp = s.run(df.to_json(orient='records'))
print(resp[:200])
PY
```

API quality checks:
- valid payload returns `clusters` list with same length as input
- invalid JSON returns structured error
- p95 latency target defined and measured
- endpoint authentication enforced

Production checks (Azure):
- readiness/liveness probes
- autoscaling policy
- model version pinning + rollback procedure tested

---

## 5) Robustness & Reliability Testing

### Data robustness
Test corner cases:
- unseen categories in categorical columns
- extreme numeric outliers
- missing optional columns
- mixed date formats

### Service robustness
- repeated burst scoring requests
- large-batch payload stress test
- dependency restart/redeploy without data loss

Pass criteria:
- controlled degradation (clear errors, no silent bad predictions)
- no memory leaks under sustained load

---

## 6) Performance & Cost Evaluation

Measure:
1. **Training runtime** (full data and sample-based runs)
2. **Inference throughput/latency** (local and endpoint)
3. **Memory footprint** (preprocessor + PCA + KMeans)
4. **Cost per 1k predictions** on target infra

Simple timing example:
```bash
python - <<'PY'
import time, pickle, pandas as pd
from src.data_validation import validate_data
from src.data_preprocessing import apply_target_encoding

df = pd.read_csv('sample_transactions.csv')
df = apply_target_encoding(validate_data(df))
with open('models/segmentation_pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)

t0=time.time(); y=pipe.predict(df); dt=time.time()-t0
print('rows=',len(df),'seconds=',round(dt,4),'rows_per_sec=',round(len(df)/dt,2))
PY
```

---

## 7) Monitoring & Drift (ongoing evaluation)

Use `models/baseline_stats.json` as reference:
- compare current medians/modes with baseline each scoring window
- define alert thresholds per feature
- trigger retraining workflow when drift persists

Operational KPIs:
- % requests failed
- p95 latency
- schema validation failure rate
- drift alert count/week
- segment distribution shift vs baseline

---

## 8) Suggested Exit Criteria (release gate)

Promote a model/app version only if all are true:
1. No critical UI/API defects in smoke + regression checks.
2. Cluster metrics are non-regressive vs current production baseline.
3. Segment profiles are interpretable and approved by domain stakeholders.
4. Drift monitors + alerting are active.
5. Rollback test passed for previous model version.

---

## 9) Recommended CI Evaluation Pipeline

1. Lint + import checks
2. Data validation tests on fixture CSVs
3. Offline scoring contract test (length match, schema, JSON format)
4. Cluster-metric regression test on fixed holdout sample
5. Streamlit smoke test (headless)
6. Build/deploy stage with post-deploy canary scoring

This gives you full-spectrum confidence: **data -> model -> app -> API -> operations**.
