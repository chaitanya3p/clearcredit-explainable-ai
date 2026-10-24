# ClearCredit

ClearCredit is a full-stack demonstration of explainable AI for data-driven
loan decisions. It combines a FastAPI service with a React/Vite interface to
show:

- a reproducible synthetic loan dataset;
- a Random Forest approval classifier;
- a local LIME explanation for each application;
- an approval probability, decision, risk band, and responsible-use notes.

> **Important:** ClearCredit is a research/demo application. It is not a
> credit decisioning system and must not be used for real lending without
> appropriate validation, governance, security controls, fairness and bias
> testing, adverse-action review, and qualified human oversight.

## Project structure

```text
type-2/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py          # FastAPI app, model, validation, and API routes
│   ├── scripts/
│   │   └── simulate_reviewers.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.jsx         # React application and API integration
│   │   └── styles.css       # Responsive UI styles
│   ├── index.html
│   └── package.json
└── README.md
```

## Architecture

```text
Applicant
   │
   ▼
React/Vite frontend ── POST /api/decide ──► FastAPI backend
   │                                          │
   │                                          ├─ Validate request with Pydantic
   │                                          ├─ Train/load Random Forest pipeline
   │                                          ├─ Calculate approval probability
   │                                          └─ Generate LIME local explanation
   │
   └────────────── Render decision, drivers, review capture, and dashboard
```

The backend creates the model lazily on the first decision request and caches
the resulting model and explainer for the process lifetime. The model is
trained from synthetic data generated with NumPy seed `42`, so the training
dataset and model setup are reproducible for a given dependency environment.

## Model and explainability

### Input features

The model uses the following features:

| Feature | Type | Description |
|---|---|---|
| `annual_income` | number | Applicant annual income |
| `credit_score` | integer | Credit score from 300 to 850 |
| `debt_to_income` | number | Debt-to-income ratio from 0 to 1 |
| `employment_years` | number | Years of employment |
| `loan_amount` | number | Requested loan amount |
| `savings_balance` | number | Applicant savings balance |
| `employment_type` | category | Salaried, Self-employed, or Contract |
| `home_ownership` | category | Own, Mortgage, or Rent |
| `loan_purpose` | category | Home improvement, Education, Vehicle, or Debt consolidation |

Protected attributes are not included in this demonstration model. Excluding
them does not guarantee that the model is fair, because other features may
act as proxies.

### Training pipeline

1. Generate 1,600 synthetic applications.
2. Generate a synthetic approval target from feature-based scoring plus noise.
3. Standardize numeric features with `StandardScaler`.
4. One-hot encode categorical features with `OneHotEncoder`.
5. Train a `RandomForestClassifier` with 180 trees, maximum depth 8, balanced
   class weights, and random state `42`.
6. Use the positive-class probability as the approval probability.

The decision thresholds are:

- `Approved` when approval probability is at least `0.5`;
- `Declined` otherwise;
- `Low risk` at `>= 0.7`;
- `Moderate risk` at `>= 0.5` and `< 0.7`;
- `Higher risk` below `0.5`.

LIME explains the numeric feature contribution around the submitted
application. The explanation is a local approximation of model behavior; it
is not causal, does not prove fairness, and is not a regulatory adverse-action
notice.

## Human-AI trust-calibration study flow

The interface supports a two-stage review:

1. **Stage 1 — human judgment:** the reviewer sees the applicant profile,
   selects Approve or Decline, records confidence from 1 (low) to 5 (high),
   and enters a reviewer/session ID. The participant does not choose the
   condition.
2. **Stage 2 — AI-assisted judgment:** the AI decision and probability are
   revealed. When the condition is `Prediction + LIME explanation`, the local
   drivers are also shown. The reviewer can keep or change the decision,
   records final confidence, and saves the review.

The two conditions are:

- `prediction_only`: AI prediction and probability without LIME drivers;
- `prediction_lime`: AI prediction, probability, and LIME drivers.

The server assigns one condition per session by taking a deterministic,
balanced SHA-256 hash of the session ID and persists that assignment in the
`study_sessions` SQLite table. Use
`GET /api/session/{session_id}/condition` to retrieve it. Researchers can
inspect an alternate condition with the query-only override
`?condition=prediction_lime`; the override does not change the persisted
assignment. The protocol panel displays consent/instructions and the current
case number, for example, `Case 3 of 12`.

Reviews are stored in `backend/reviews.db`, which is created automatically.
Only synthetic data should be used for this demonstration. The application
does not authenticate reviewer IDs and is not suitable for collecting
identifiable participant data.

### Held-out study cases

At startup, ClearCredit creates a 50-case held-out pool in the
`study_cases` SQLite table using a different random seed from model training.
Each case stores its synthetic ground-truth label server-side; approximately
30% are deliberately selected where the trained model is wrong so
over-reliance can be measured. Each session receives exactly 4 model-wrong and 8 model-right cases selected
from the pool with a deterministic SHA-256 seed derived from the session ID,
then shuffled into a session-specific order. Study clients request the next
unseen profile with:

```text
GET /api/study/next-case?session_id=...
```

The response includes the `case_id` and applicant profile but never the label.
Study reviews must submit that `case_id`; the backend obtains the label from
the pool and rejects unknown or repeated cases. Explore mode remains available
for manually entered profiles, but Explore reviews have no ground-truth label
and are excluded from study metrics.

### Review endpoints

`POST /api/reviews` is a strict study endpoint. The client may submit only
`session_id`, `case_id`, the initial and final decisions/confidences, and
`initial_seconds`/`final_seconds`. Session IDs must match
`reviewer-XXXXXXXX` (eight letters or digits). The server looks up the case
profile and ground-truth label, assigns the persisted condition, and computes
the AI decision and probability itself. A duplicate `(session_id, case_id)` or
an unknown case returns HTTP `409`. Explore mode does not call this endpoint.
For arbitrary manually
entered profiles, the ground-truth label is `null` because no observed label
exists.

`GET /api/reviews/export` downloads all stored reviews as
`clearcredit-reviews.csv`. The applicant profile is included as a JSON column.

### Trust metrics

`GET /api/metrics/trust` returns metrics split by study condition:

- **Agreement rate:** final human decision agrees with the AI decision.
- **Switch rate:** the reviewer changes the initial decision after seeing AI.
- **Over-reliance:** the reviewer follows an incorrect AI decision.
- **Under-reliance:** the reviewer rejects a correct AI decision.
- **Appropriate reliance:** `(reject wrong AI advice + follow correct AI advice) / counted reviews`.
- **Human accuracy before/after:** initial and final decisions match the
  available synthetic ground-truth label.
- **AI-alone accuracy:** correct AI decisions divided by counted reviews.
- **Human+AI team accuracy:** final post-AI reviewer accuracy, used as the
  operational team outcome in this protocol.
- **Confidence ECE:** for confidence bins 1–5, the weighted absolute
  difference between `0.5 + 0.5 * (confidence - 1) / 4` and empirical
  correctness in that bin.
  Lower is better.

### Threats to validity

- Ground-truth labels are random draws from the synthetic approval
  probability, so some “model wrong” cases are partly irreducible noise rather
  than model defects.
- Review observations are clustered within reviewers; reviewer-level
  permutation tests preserve that clustering.
- All applicants, labels, reviews, and demographic groups in this project are
  synthetic and do not represent real lending populations or outcomes.

Over-reliance is `follow AI when AI is wrong / AI-wrong reviews`.
Under-reliance is `reject AI when AI is right / AI-right reviews`.
Accuracy, reliance, and confidence metrics exclude reviews without a
ground-truth label. Agreement and switch rate use all reviews in the
condition. Bootstrap intervals use 1,000 deterministic resamples. The condition
comparison uses a permutation test that shuffles condition labels across
reviewers rather than individual reviews; estimates warn when either condition
has fewer than 10 reviewers.

The **Trust dashboard** link in the frontend displays a Recharts comparison
of these metrics and the underlying condition counts. Simulated reviewer rows
are excluded by default. The dashboard toggle
**Include simulated reviewers in dashboard metrics** requests
`GET /api/metrics/trust?include_simulated=true` when enabled.

### Honest reviewer simulation

Run the local simulation with:

```powershell
cd backend
py -3.13 scripts\simulate_reviewers.py
```

The script creates 30 simulated reviewers, each assigned a condition using the
same persisted session-hash assignment as the application, and runs each
reviewer over the held-out `study_cases`. Each simulated reviewer has explicit
skill, AI-reliance probability, and confidence-noise parameters. Initial and
final decisions are sampled from the applicant profile and model probability;
the ground-truth label is never used to select a decision or confidence. Rows
are marked `is_simulated=1`, so they are available for an opt-in dashboard
comparison but do not affect the default participant metrics.

## Model probability calibration

The Random Forest is wrapped in a sigmoid `CalibratedClassifierCV` while
remaining inside the existing preprocessing pipeline. The classifier is fit
on 75% of the deterministic synthetic data, with the remaining 25% held out
for calibration evaluation.

`GET /api/model/calibration` returns:

- the calibration method (`sigmoid`);
- held-out sample count;
- Brier score;
- Expected Calibration Error (ECE);
- ten fixed probability bins containing the mean predicted probability,
  observed approval frequency, and count.

The Brier score is the mean squared error of predicted approval probabilities
against the binary target. ECE is the count-weighted mean absolute difference
between predicted probability and observed frequency across non-empty bins.
Lower values indicate better probability calibration.

The dashboard reliability diagram plots held-out predicted probability against
observed approval frequency, with a perfect-calibration reference line. Empty
bins are omitted from the chart. These metrics describe this synthetic
demonstration dataset and should not be interpreted as evidence of production
model calibration.

## LIME explanation quality

LIME runs on the full original feature space. Numeric values remain numeric,
while `employment_type`, `home_ownership`, and `loan_purpose` are passed as
named categorical features. A single `LimeTabularExplainer` is built with the
model artifacts and reused for all requests and stability runs. Its prediction
callback reconstructs typed application frames before calling the complete
fitted pipeline, so categorical drivers remain visible without exposing
one-hot implementation details.

Each `POST /api/decide` response includes `explanation_reliability`:

- **Fidelity:** the local surrogate score (R²) for the explanation instance.
- **Stability:** five repeat runs with different seeds and 2,000 samples report mean pairwise Spearman
  rank correlation of absolute contributions and mean top-five overlap.
  Stability combines rank correlation with chance-corrected top-five overlap;
  the chance baseline is approximately `0.56` for five of nine features.
- **Warning:** a plain-language warning when fidelity is below `0.70` or
  chance-corrected stability is below `0.50`.

The frontend displays these values under **Explanation reliability**. A low
score means the local explanation should be treated as uncertain and reviewed
by a human; it is not evidence that the underlying model is accurate or fair.

Adverse-action-style `reason_codes` are deliberately conservative: they are
returned only for a declined decision when LIME fidelity is at least `0.50`
and the mapped negative driver has magnitude at least `0.01`. Otherwise the
response returns an empty list. Approved decisions use
`reason_codes_status: "not_applicable"`; declined decisions with insufficient
explanation quality use `"suppressed_low_fidelity"`, and other declined
decisions use `"ok"`. The status is included in saved decision records.

Study conditions are assigned deterministically from the session ID and are
balanced by hashing the ID. Reviewers cannot choose their condition. Researchers
may inspect or override the assignment with
`GET /api/session/{session_id}/condition?condition=prediction_lime`.
The `prediction_only` condition shows only the AI decision, probability, and
risk band; the `prediction_lime` condition additionally shows counterfactuals,
explanation reliability, and local drivers.

## Counterfactuals

Each `POST /api/decide` response also includes up to three
`counterfactuals`. These answer **“What would change the decision?”** by
searching for the smallest bounded changes to actionable numeric features:

- credit score: `300` to `850`;
- debt-to-income ratio: `0` to `1`;
- requested loan amount: `$0` to `$2,000,000`;
- savings balance: `$0` to `$5,000,000`.

The search first tests one-feature changes, then paired changes when no
single-feature change flips the model threshold. Candidate changes move
toward the relevant bound in 5% increments and are ranked by normalized
movement, so dollars and score points are comparable. The response reports
the changed feature values, resulting decision, approval probability, and
normalized effort.

Counterfactuals are model simulations, not recommendations or guarantees.
They do not account for underwriting policy, affordability, fraud, fairness,
or causal relationships. An empty list means no permitted search candidate
crossed the required margin: at least `0.55` approval probability for an
approval suggestion or at most `0.45` for a decline suggestion.

To populate the dashboard with synthetic observations:

```powershell
cd backend
py -3.13 scripts\simulate_reviewers.py
```

The script creates 30 simulated reviewers over the 50 held-out cases. Each
reviewer has skill, reliance probability, and confidence-noise parameters.
Ground truth is not read by the simulation's decision process. Rows are marked
`is_simulated=1` and excluded from the dashboard unless explicitly enabled.

## Regulatory compliance support

Declined decisions map the most important negative local drivers to
configuration-backed adverse-action-style reason codes in `REASON_CODES` in
`backend/app/main.py`. These are demonstration reason phrases, not legal
adverse-action notices.

Every decision receives a `decision_id` and is persisted as a decision record.
The frontend can download a JSON record or open a printable view containing
applicant inputs, model version, data version, library versions, decision and probability, LIME explanation,
fidelity/stability, reason codes, and reviewer decisions/confidence when
available. The API exposes `GET /api/decisions/{decision_id}` for the same
server-side record.

Every successful `/api/decide` and `/api/reviews` call is written as an
append-only JSON line to `backend/audit.log.jsonl`. Failed calls to those
endpoints are also recorded with their status code. This local log is a demo
artifact, but each entry now contains a SHA-256 `entry_hash` and the
`previous_hash` of the preceding entry. Successful decision and review entries
contain the `decision_id`, a SHA-256 hash of the applicant payload, and outcome
fields without storing the full applicant profile in the audit log. Use
`GET /api/audit/verify` to validate the chain; it returns `valid`, the number
of verified entries, and an error description when tampering or malformed
entries are found. This remains a local demonstration control, not a
replacement for an immutable production audit service.

Decision records include:

- `model_version` for the model implementation;
- `data_version` for the synthetic training-data generation contract; and
- `library_versions` for the installed FastAPI, LIME, NumPy, pandas, Pydantic,
  scikit-learn, and SciPy distributions.

### Fairness screening

`backend/scripts/fairness_check.py` and `GET /api/fairness` run an audit using
an independent seeded synthetic demographic group that is never passed to the
model. Evaluation uses the held-out split only. The report includes approval
rate, true-positive rate, false-positive rate, demographic parity difference,
equal opportunity difference, and proxy correlations. Numeric proxies use
Pearson correlation; categorical proxies use Cramer's V. The optional
`proxy_strength` query parameter creates a controlled synthetic correlation
for test/demo purposes. The **Fairness** page displays these results.

This screening does not establish regulatory compliance or fairness. It uses
synthetic data and configurable demo thresholds and requires expert review.
No real personal data is used.

## Mapping to the project title

The implementation maps to
**“Trust Calibration Between Human Decision-makers and AI Systems via LIME
Local Surrogate Model Explanations, Applied to Loan/Credit Approval
Explainability for Regulatory Compliance”** as follows:

| Project-title concept | ClearCredit feature |
|---|---|
| Trust calibration | Two-stage human decisions, confidence capture, reliance metrics, confidence ECE, bootstrap intervals, and condition comparison |
| Human decision-makers and AI systems | Reviewer judgment before and after a server-computed AI prediction |
| LIME local surrogate model explanations | Raw-feature LIME drivers, local fidelity, stability, and low-reliability warnings |
| Loan/credit approval explainability | Approval probability, risk band, counterfactuals, and synthetic applicant profiles |
| Regulatory compliance support | Conservative reason codes, decision records, model/data/library provenance, audit hash-chain verification, and fairness screening |

These mappings describe a research demonstration, not evidence that the
system satisfies lending regulation or is suitable for production decisions.

## Limitations and Threats to Validity

**Data.** All data is synthetic. The approval label is a random draw from a scoring formula, so some cases where the model is "wrong" are irreducible noise that no reviewer could resolve from the profile. Results show how the pipeline behaves, not how real lending behaves.

**Explanation quality.** LIME fits a linear surrogate around a Random Forest, so local fidelity (R^2) is often low, about 30% in typical cases. The app reports fidelity and stability, shows a warning when reliability is low, and suppresses reason codes below a fidelity threshold. LIME explanations are local approximations, not causal reasons.

**Study design.** The sample is small. The condition (prediction only vs prediction + LIME) is assigned per reviewer, and each reviewer completes several cases, so the analysis is run at the reviewer level, which reduces statistical power. With fewer than 10 reviewers per condition, the dashboard warns that estimates are unreliable. Simulated reviewers demonstrate that the pipeline works but are not evidence about human trust.

**Technical limits.** Condition gating is enforced when a session is given; calling /api/decide directly without a session returns the full output. There is no authentication, rate limiting, or production data handling. The model is trained in memory on a fixed seed, and the audit log is tamper-evident but not access-controlled.

**Compliance.** Reason codes and decision records are demonstrations of the format of adverse-action support. They are not legal notices, do not satisfy ECOA/Reg B or the EU AI Act by themselves, and must not be used for real credit decisions. Excluding protected attributes does not guarantee fairness, because other features can act as proxies. The fairness audit uses an independent synthetic group.

## How to run the study

1. Start the backend and frontend using the setup commands below, then open the
   Vite URL.
2. Read the consent and synthetic-research notice in the study protocol panel
   before continuing. The study uses synthetic applicant profiles only.
3. The server assigns one condition per reviewer/session (`prediction_only` or
   `prediction_lime`) and keeps that condition for the session. Review the
   profile, choose an initial Approve or Decline decision and confidence, then
   reveal the AI output.
4. Complete all 12 assigned cases. After the twelfth saved review, the app
   shows a thank-you screen and does not provide a thirteenth case.
5. Download stored reviews as CSV from
   `GET /api/reviews/export`, or use the dashboard's **Trust dashboard** link
   to inspect condition-level reliance, confidence, and calibration metrics.
   Simulated reviewer rows are excluded by default; enable the dashboard toggle
   to include them.

## Backend setup

The backend is tested with Python 3.13 or another Python version supported by
the pinned scientific Python dependencies.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

The API starts at `http://localhost:8000`.

If the default `python` command selects an unsupported interpreter, use a
compatible Python executable explicitly, for example:

```powershell
py -3.13 -m pip install -r requirements.txt
py -3.13 -m uvicorn app.main:app --reload
```

## Frontend setup

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally
`http://localhost:5173`.

The frontend currently calls the API at
`VITE_API_URL` plus the endpoint path. If `VITE_API_URL` is not set, it falls
back to `http://localhost:8000`. The backend CORS configuration allows the
localhost and loopback Vite origins on port 5173.

For a different backend URL, create `frontend/.env.local`:

```text
VITE_API_URL=http://127.0.0.1:8000
```

To create a production frontend bundle:

```powershell
cd frontend
npm run build
```

## Tests

Backend tests use pytest and cover trust metrics, reason-code mapping,
counterfactual search, and API validation:

```powershell
cd backend
py -3.13 -m pytest
```

The frontend build is the UI validation command:

```powershell
cd frontend
npm run build
```

## API reference

Interactive OpenAPI documentation is available at
`http://localhost:8000/docs` while the backend is running.

### `GET /health`

Returns service status.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /api/options`

Returns the supported categorical values used by the UI.

Example response:

```json
{
  "employment_type": ["Salaried", "Self-employed", "Contract"],
  "home_ownership": ["Own", "Mortgage", "Rent"],
  "loan_purpose": [
    "Home improvement",
    "Education",
    "Vehicle",
    "Debt consolidation"
  ]
}
```

### `POST /api/decide`

Validates an application, predicts an outcome, and returns the local
explanation.

Example request:

```json
{
  "annual_income": 82000,
  "credit_score": 718,
  "debt_to_income": 0.28,
  "employment_years": 6,
  "loan_amount": 24000,
  "savings_balance": 42000,
  "employment_type": "Salaried",
  "home_ownership": "Mortgage",
  "loan_purpose": "Home improvement"
}
```

Example response:

```json
{
  "decision": "Approved",
  "approval_probability": 0.5764,
  "risk_band": "Moderate risk",
  "explanation": [
    {
      "feature": "Employment type",
      "value": "Salaried",
      "contribution": 0.0989,
      "direction": "supports approval",
      "importance": 0.0989
    },
    {
      "feature": "Credit score",
      "value": "718",
      "contribution": 0.0813,
      "direction": "supports approval",
      "importance": 0.0813
    },
    {
      "feature": "Debt-to-income ratio",
      "value": "0.28",
      "contribution": -0.041,
      "direction": "increases risk",
      "importance": 0.041
    }
  ],
  "summary": "The application is more likely to be approved based on the applicant's local feature pattern.",
  "compliance_notes": [
    "Explanation describes local model behavior, not a causal reason for the outcome.",
    "Protected attributes are not used as model features in this demonstration.",
    "Human review is required before using this output for a real credit decision."
  ],
  "model_info": {
    "model": "Sigmoid-calibrated Random Forest classifier",
    "explanation": "LIME local surrogate",
    "training": "Synthetic demonstration data (seed 42)",
    "version": "1.0.0"
  },
  "explanation_reliability": {
    "fidelity": 0.1207,
    "stability": 1.0,
    "top_k": 5,
    "warning": "Explanation reliability is low. Treat these drivers as uncertain and require human review."
  },
  "counterfactuals": [
    {
      "changes": {
        "Credit score": "718 -> 697"
      },
      "decision": "Declined",
      "approval_probability": 0.2977,
      "effort": 0.05
    }
  ],
  "decision_id": "generated-per-request",
  "reason_codes": []
}
```

The actual response includes all ranked explanation items and up to three
counterfactuals. `decision_id` is a new UUID for each request.

Invalid numeric ranges or missing fields result in FastAPI validation errors.
The accepted numeric ranges are defined in `backend/app/main.py`.

## Frontend behavior

The interface has two main panels:

1. **Applicant profile** — collects numeric and categorical application data.
2. **Decision and explanation** — displays the outcome after the API request,
   including:
   - approval likelihood;
   - risk band;
   - summary statement;
   - sorted local feature drivers;
   - positive and negative contribution indicators;
   - responsible-use notes.

The frontend displays a connection error if the API is unavailable. Numeric
form values are converted to JavaScript numbers before being sent to the API.

## Development notes

- The model is trained in memory; no database or model artifact is persisted.
- Restarting the backend retrains the model on the deterministic synthetic
  dataset.
- The UI is designed for local demonstration and does not include
  authentication, authorization, rate limiting, immutable external audit
  storage, or production data handling. The local audit hash chain helps
  detect accidental modification but does not prevent an authorized process
  from rewriting the file.
- The current frontend uses a fixed API base URL. A production deployment
  should move this to environment configuration.
- Production use would require real data governance, monitoring, calibration,
  explainability validation, privacy controls, security review, and human
  decision processes.
- This demo intentionally uses synthetic applicant data and simulated study
  participants only. Do not enter real personal or credit information.

## Dependencies

Backend dependencies are pinned in `backend/requirements.txt`:

- FastAPI and Uvicorn for the HTTP API;
- Pydantic for request and response validation;
- NumPy and pandas for data generation and preparation;
- scikit-learn for preprocessing and classification;
- LIME for local explanations.

Frontend dependencies are declared in `frontend/package.json`:

- React and React DOM;
- Vite and the React Vite plugin;
- Lucide React icons.


## Free deployment (Render API + Vercel frontend)

### Render API service
1. Push this project to a GitHub repository. Do not commit `.venv`, `node_modules`, `frontend/dist`, `reviews.db`, or audit logs.
2. In Render, create a **Web Service** connected to the repository. Set **Root Directory** to `backend`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Set `PYTHON_VERSION` to `3.12.8` if Render asks for a Python version.
6. After deploying, test `https://YOUR-API.onrender.com/health`.

### Vercel frontend
1. Import the same GitHub repository into Vercel.
2. Set **Root Directory** to `frontend`. Framework: Vite. Build command: `npm run build`. Output directory: `dist`.
3. Add environment variable `VITE_API_URL` with the Render API base URL, e.g. `https://YOUR-API.onrender.com` (no trailing slash).
4. Deploy the frontend. Copy its public URL.
5. In Render, add environment variable `FRONTEND_ORIGINS` with that Vercel URL, e.g. `https://your-project.vercel.app`. Redeploy the API.

### Important limitations
- The first model request may take longer because the model is initialized lazily.
- Free hosting may sleep when idle, so the first visit/request can be slow.
- SQLite files on ephemeral hosting may be lost on redeploy/restart. Use synthetic demo data only; do not collect real applicant or participant information. For persistent research data, configure a persistent disk or managed database before collecting responses.
