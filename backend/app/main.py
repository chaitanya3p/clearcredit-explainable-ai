from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import os
import random
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version as package_version
from itertools import combinations
from functools import lru_cache
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from lime.lime_tabular import LimeTabularExplainer
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

NUMERIC_FEATURES = [
    "annual_income",
    "credit_score",
    "debt_to_income",
    "employment_years",
    "loan_amount",
    "savings_balance",
]
CATEGORICAL_FEATURES = ["employment_type", "home_ownership", "loan_purpose"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL_OFFSET = 1.0
MODEL_VERSION = "1.0.0"
DATA_VERSION = "synthetic-training-v1"
DISPLAY_NAMES = {
    "annual_income": "Annual income",
    "credit_score": "Credit score",
    "debt_to_income": "Debt-to-income ratio",
    "employment_years": "Employment history",
    "loan_amount": "Requested loan amount",
    "savings_balance": "Savings balance",
    "employment_type": "Employment type",
    "home_ownership": "Home ownership",
    "loan_purpose": "Loan purpose",
}
STUDY_CASES_PER_SESSION = 12
CASE_POOL_VERSION = "balanced-v3"
_ARTIFACT_CATEGORICAL_NAMES: dict[str, list[str]] = {}


class LoanApplication(BaseModel):
    annual_income: float = Field(..., ge=0, le=2_000_000)
    credit_score: int = Field(..., ge=300, le=850)
    debt_to_income: float = Field(..., ge=0, le=1)
    employment_years: float = Field(..., ge=0, le=60)
    loan_amount: float = Field(..., ge=0, le=2_000_000)
    savings_balance: float = Field(..., ge=0, le=5_000_000)
    employment_type: str = Field(..., min_length=1)
    home_ownership: str = Field(..., min_length=1)
    loan_purpose: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None, exclude=True)
    case_id: str | None = Field(default=None, exclude=True)


class ExplanationItem(BaseModel):
    feature: str
    value: str
    contribution: float
    direction: str
    importance: float


class ExplanationReliability(BaseModel):
    fidelity: float
    stability: float
    rank_correlation: float
    top5_overlap: float
    top_k: int
    warning: str | None = None


class DecisionRecord(BaseModel):
    decision_id: str
    timestamp: str
    applicant: LoanApplication
    model_version: str
    data_version: str = DATA_VERSION
    library_versions: dict[str, str] = Field(default_factory=dict)
    decision: str
    approval_probability: float
    explanation: list[ExplanationItem]
    explanation_reliability: ExplanationReliability | None
    reason_codes: list[str]
    reason_codes_status: Literal["ok", "suppressed_low_fidelity", "not_applicable"] = "not_applicable"
    human_review: dict[str, Any] | None = None


class Counterfactual(BaseModel):
    changes: dict[str, str]
    decision: str
    approval_probability: float
    effort: float


class DecisionResponse(BaseModel):
    decision: str
    approval_probability: float
    risk_band: str
    explanation: list[ExplanationItem]
    summary: str
    compliance_notes: list[str]
    model_info: dict[str, Any]
    explanation_reliability: ExplanationReliability | None
    counterfactuals: list[Counterfactual]
    decision_id: str
    reason_codes: list[str]
    reason_codes_status: Literal["ok", "suppressed_low_fidelity", "not_applicable"]


class ReviewCreate(BaseModel):
    model_config = {"extra": "forbid"}
    session_id: str = Field(..., pattern=r"^reviewer-[A-Za-z0-9]{8}$")
    case_id: str = Field(..., pattern=r"^case-\d{3}$")
    initial_decision: Literal["Approve", "Decline"]
    initial_confidence: float = Field(..., ge=1, le=5)
    final_decision: Literal["Approve", "Decline"]
    final_confidence: float = Field(..., ge=1, le=5)
    initial_seconds: float = Field(..., ge=0, le=3600)
    final_seconds: float = Field(..., ge=0, le=3600)


class ReviewResponse(ReviewCreate):
    timestamp: str
    condition: Literal["prediction_only", "prediction_lime"]
    applicant: LoanApplication
    ai_decision: Literal["Approved", "Declined"]
    ai_probability: float
    ground_truth_label: int | None = None
    decision_id: str
    mode: Literal["study"] = "study"


DATABASE_PATH = "reviews.db"
AUDIT_LOG_PATH = "audit.log.jsonl"
REASON_CODES = {
    "Debt-to-income ratio": "Debt-to-income ratio too high",
    "Credit score": "Credit score below threshold",
    "Requested loan amount": "Requested loan amount is high relative to profile",
    "Savings balance": "Savings balance is low",
    "Annual income": "Annual income is insufficient for this request",
    "Employment history": "Employment history is limited",
}


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _init_database() -> None:
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                condition TEXT NOT NULL,
                applicant_json TEXT NOT NULL,
                initial_decision TEXT NOT NULL,
                initial_confidence REAL NOT NULL,
                ai_decision TEXT NOT NULL,
                ai_probability REAL NOT NULL,
                final_decision TEXT NOT NULL,
                final_confidence REAL NOT NULL,
                ground_truth_label INTEGER,
                mode TEXT NOT NULL DEFAULT 'study',
                case_id TEXT,
                initial_seconds REAL NOT NULL DEFAULT 0,
                final_seconds REAL NOT NULL DEFAULT 0,
                is_simulated INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        try:
            connection.execute("ALTER TABLE reviews ADD COLUMN decision_id TEXT")
        except sqlite3.OperationalError:
            pass
        for column, definition in (
            ("mode", "TEXT NOT NULL DEFAULT 'study'"),
            ("case_id", "TEXT"),
            ("initial_seconds", "REAL NOT NULL DEFAULT 0"),
            ("final_seconds", "REAL NOT NULL DEFAULT 0"),
            ("is_simulated", "INTEGER NOT NULL DEFAULT 0"),
        ):
            try:
                connection.execute(f"ALTER TABLE reviews ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError:
                pass
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_records (
                decision_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                applicant_json TEXT NOT NULL,
                record_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS study_sessions (
                session_id TEXT PRIMARY KEY,
                condition TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS study_cases (
                case_id TEXT PRIMARY KEY,
                applicant_json TEXT NOT NULL,
                ground_truth_label INTEGER NOT NULL,
                ai_wrong INTEGER NOT NULL DEFAULT 0,
                pool_version TEXT NOT NULL DEFAULT 'legacy'
            )
            """
        )
        try:
            connection.execute("ALTER TABLE study_cases ADD COLUMN ai_wrong INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            connection.execute("ALTER TABLE study_cases ADD COLUMN pool_version TEXT NOT NULL DEFAULT 'legacy'")
        except sqlite3.OperationalError:
            pass


def _audit(event: str, path: str, payload: dict[str, Any]) -> None:
    safe_payload = json.loads(json.dumps(payload, default=str))
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "path": path,
        "payload": safe_payload,
    }
    previous_hash = ""
    try:
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as audit_file:
            for line in audit_file:
                if line.strip():
                    previous_hash = json.loads(line).get("entry_hash", "")
    except FileNotFoundError:
        pass
    entry["previous_hash"] = previous_hash
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    entry["entry_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(entry, default=str) + "\n")


def _verify_audit_log() -> dict[str, Any]:
    previous_hash = ""
    entries = 0
    try:
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as audit_file:
            for line_number, line in enumerate(audit_file, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    recorded_hash = entry.pop("entry_hash")
                    if entry.get("previous_hash", "") != previous_hash:
                        return {
                            "valid": False,
                            "entries": entries,
                            "error": f"Previous hash mismatch at line {line_number}.",
                        }
                    canonical = json.dumps(
                        entry,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                    calculated_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    if recorded_hash != calculated_hash:
                        return {
                            "valid": False,
                            "entries": entries,
                            "error": f"Entry hash mismatch at line {line_number}.",
                        }
                    previous_hash = recorded_hash
                    entries += 1
                except (KeyError, json.JSONDecodeError, TypeError):
                    return {
                        "valid": False,
                        "entries": entries,
                        "error": f"Malformed audit entry at line {line_number}.",
                    }
    except FileNotFoundError:
        return {"valid": True, "entries": 0, "error": None}
    return {"valid": True, "entries": entries, "error": None}


def _applicant_hash(application: LoanApplication) -> str:
    canonical = json.dumps(
        application.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _review_from_row(row: sqlite3.Row) -> ReviewResponse:
    return ReviewResponse(
        timestamp=row["timestamp"],
        session_id=row["session_id"],
        case_id=row["case_id"],
        condition=row["condition"],
        applicant=json.loads(row["applicant_json"]),
        initial_decision=row["initial_decision"],
        initial_confidence=row["initial_confidence"],
        ai_decision=row["ai_decision"],
        ai_probability=row["ai_probability"],
        final_decision=row["final_decision"],
        final_confidence=row["final_confidence"],
        initial_seconds=row["initial_seconds"],
        final_seconds=row["final_seconds"],
        ground_truth_label=row["ground_truth_label"],
        decision_id=row["decision_id"],
    )


def _training_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    count = 1600
    data = pd.DataFrame(
        {
            "annual_income": rng.normal(82_000, 32_000, count).clip(18_000, 250_000),
            "credit_score": rng.normal(690, 65, count).clip(300, 850).round(),
            "debt_to_income": rng.beta(2.2, 7, count).clip(0.02, 0.85),
            "employment_years": rng.exponential(6, count).clip(0, 35),
            "loan_amount": rng.normal(28_000, 18_000, count).clip(2_000, 160_000),
            "savings_balance": rng.lognormal(9.5, 1.0, count).clip(0, 350_000),
            "employment_type": rng.choice(["Salaried", "Self-employed", "Contract"], count, p=[0.62, 0.25, 0.13]),
            "home_ownership": rng.choice(["Own", "Mortgage", "Rent"], count, p=[0.28, 0.47, 0.25]),
            "loan_purpose": rng.choice(["Home improvement", "Education", "Vehicle", "Debt consolidation"], count),
        }
    )
    score = (
        0.014 * (data.credit_score - 650)
        - 4.2 * data.debt_to_income
        + 0.000008 * data.annual_income
        + 0.045 * data.employment_years
        - 0.000018 * data.loan_amount
        + 0.000006 * data.savings_balance
        + data.employment_type.map({"Salaried": 0.3, "Self-employed": 0.08, "Contract": -0.12})
        + data.home_ownership.map({"Own": 0.24, "Mortgage": 0.12, "Rent": -0.1})
        + rng.normal(0, 0.65, count)
    )
    probability = 1 / (1 + np.exp(-score + LABEL_OFFSET))
    target = (rng.random(count) < probability).astype(int)
    return data, target


@lru_cache(maxsize=1)
def _artifacts() -> tuple[Pipeline, LimeTabularExplainer, pd.DataFrame, dict[str, Any], list[str], np.ndarray]:
    data, target = _training_data()
    train_data, held_out_data, train_target, held_out_target = train_test_split(
        data[FEATURES],
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=RandomForestClassifier(
                        n_estimators=180,
                        max_depth=8,
                        random_state=42,
                        class_weight="balanced",
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    model.fit(train_data, train_target)
    held_out_probability = model.predict_proba(held_out_data)[:, 1]
    bin_edges = np.linspace(0, 1, 11)
    bins: list[dict[str, float | int | None]] = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        last_bin = upper == 1
        membership = (
            (held_out_probability >= lower)
            & ((held_out_probability <= upper) if last_bin else (held_out_probability < upper))
        )
        count = int(membership.sum())
        bins.append(
            {
                "bin_lower": round(float(lower), 2),
                "bin_upper": round(float(upper), 2),
                "mean_predicted_probability": round(float(held_out_probability[membership].mean()), 4)
                if count
                else None,
                "observed_frequency": round(float(held_out_target[membership].mean()), 4) if count else None,
                "count": count,
            }
        )
    ece = sum(
        (item["count"] / len(held_out_target))
        * abs(item["mean_predicted_probability"] - item["observed_frequency"])
        for item in bins
        if item["count"] and item["mean_predicted_probability"] is not None and item["observed_frequency"] is not None
    )
    calibration = {
        "method": "sigmoid",
        "held_out_count": int(len(held_out_target)),
        "brier_score": round(float(brier_score_loss(held_out_target, held_out_probability)), 4),
        "expected_calibration_error": round(float(ece), 4),
        "bins": bins,
    }
    categorical_values = {
        feature: sorted(data[feature].unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }
    _ARTIFACT_CATEGORICAL_NAMES.update(categorical_values)
    raw_train = train_data.copy()
    raw_training_matrix = raw_train[FEATURES].copy()
    categorical_names: dict[int, list[str]] = {}
    for index, feature in enumerate(CATEGORICAL_FEATURES, start=len(NUMERIC_FEATURES)):
        categorical_names[index] = categorical_values[feature]
        code_map = {value: code for code, value in enumerate(categorical_values[feature])}
        raw_training_matrix[feature] = raw_training_matrix[feature].map(code_map).astype(int)
    raw_training_matrix = raw_training_matrix[FEATURES].to_numpy()
    explainer = LimeTabularExplainer(
        raw_training_matrix,
        feature_names=FEATURES,
        categorical_features=list(categorical_names),
        categorical_names=categorical_names,
        class_names=["Decline", "Approve"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    return model, explainer, data, calibration, FEATURES, raw_training_matrix


def _frame(application: LoanApplication) -> pd.DataFrame:
    return pd.DataFrame([{feature: getattr(application, feature) for feature in FEATURES}], columns=FEATURES)


def _study_case_candidates() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(2025)
    count = 400
    data = pd.DataFrame(
        {
            "annual_income": rng.normal(82_000, 32_000, count).clip(18_000, 250_000),
            "credit_score": rng.normal(690, 65, count).clip(300, 850).round(),
            "debt_to_income": rng.beta(2.2, 7, count).clip(0.02, 0.85),
            "employment_years": rng.exponential(6, count).clip(0, 35),
            "loan_amount": rng.normal(28_000, 18_000, count).clip(2_000, 160_000),
            "savings_balance": rng.lognormal(9.5, 1.0, count).clip(0, 350_000),
            "employment_type": rng.choice(["Salaried", "Self-employed", "Contract"], count, p=[0.62, 0.25, 0.13]),
            "home_ownership": rng.choice(["Own", "Mortgage", "Rent"], count, p=[0.28, 0.47, 0.25]),
            "loan_purpose": rng.choice(["Home improvement", "Education", "Vehicle", "Debt consolidation"], count),
        }
    )
    score = (
        0.014 * (data.credit_score - 650)
        - 4.2 * data.debt_to_income
        + 0.000008 * data.annual_income
        + 0.045 * data.employment_years
        - 0.000018 * data.loan_amount
        + 0.000006 * data.savings_balance
        + data.employment_type.map({"Salaried": 0.3, "Self-employed": 0.08, "Contract": -0.12})
        + data.home_ownership.map({"Own": 0.24, "Mortgage": 0.12, "Rent": -0.1})
        + rng.normal(0, 0.65, count)
    )
    probability = 1 / (1 + np.exp(-score + LABEL_OFFSET))
    return data, (rng.random(count) < probability).astype(int)


def _ensure_study_cases() -> None:
    with _connection() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count, MIN(pool_version) AS version FROM study_cases"
        ).fetchone()
        if row["count"] >= 50 and row["version"] == CASE_POOL_VERSION:
            return
        connection.execute("DELETE FROM study_cases")
    candidates, _ = _study_case_candidates()
    for feature in ("annual_income", "loan_amount", "savings_balance"):
        candidates[feature] = (candidates[feature] / 100).round() * 100
    candidates["debt_to_income"] = candidates["debt_to_income"].round(2)
    candidates["employment_years"] = candidates["employment_years"].round(1)
    candidates["credit_score"] = candidates["credit_score"].round().astype(int)
    model = _artifacts()[0]
    predictions = (model.predict_proba(candidates[FEATURES])[:, 1] >= 0.5).astype(int)
    score = (
        0.014 * (candidates.credit_score - 650)
        - 4.2 * candidates.debt_to_income
        + 0.000008 * candidates.annual_income
        + 0.045 * candidates.employment_years
        - 0.000018 * candidates.loan_amount
        + 0.000006 * candidates.savings_balance
        + candidates.employment_type.map({"Salaried": 0.3, "Self-employed": 0.08, "Contract": -0.12})
        + candidates.home_ownership.map({"Own": 0.24, "Mortgage": 0.12, "Rent": -0.1})
        + np.random.default_rng(2026).normal(0, 0.65, len(candidates))
    )
    probability = 1 / (1 + np.exp(-score + LABEL_OFFSET))
    labels = (np.random.default_rng(2027).random(len(candidates)) < probability).astype(int)
    wrong = np.flatnonzero(predictions != labels)
    right = np.flatnonzero(predictions == labels)
    selected = np.concatenate([wrong[:15], right[:35]])
    if len(selected) < 50:
        raise RuntimeError("Unable to construct the held-out study case pool.")
    with _connection() as connection:
        connection.executemany(
            "INSERT OR IGNORE INTO study_cases (case_id, applicant_json, ground_truth_label, ai_wrong, pool_version) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    f"case-{case_number:03d}",
                    candidates.iloc[index][FEATURES].to_json(),
                    int(labels[index]),
                    int(predictions[index] != labels[index]),
                    CASE_POOL_VERSION,
                )
                for case_number, index in enumerate(selected, start=1)
            ],
        )


_init_database()
_ensure_study_cases()


def _original_feature(feature_name: str) -> str:
    raw_name = feature_name.split("__", 1)[-1]
    return next((feature for feature in FEATURES if raw_name == feature or raw_name.startswith(f"{feature}_")), raw_name)


def _lime_explanation(
    application: LoanApplication,
    model: Pipeline,
    explainer: LimeTabularExplainer,
    feature_names: list[str],
    raw_training_data: np.ndarray,
) -> tuple[list[ExplanationItem], ExplanationReliability]:
    category_names = _ARTIFACT_CATEGORICAL_NAMES
    code_maps = {
        feature: {code: value for code, value in enumerate(category_names[feature])}
        for feature in CATEGORICAL_FEATURES
    }
    raw_row = np.array(
        [
            *[getattr(application, feature) for feature in NUMERIC_FEATURES],
            *[category_names[feature].index(getattr(application, feature)) for feature in CATEGORICAL_FEATURES],
        ],
        dtype=object,
    )

    def predict_raw(rows: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(rows, columns=feature_names)
        for index, feature in enumerate(CATEGORICAL_FEATURES, start=len(NUMERIC_FEATURES)):
            frame[feature] = (
                frame[feature]
                .round()
                .clip(0, len(code_maps[feature]) - 1)
                .astype(int)
                .map(code_maps[feature])
            )
        for feature in NUMERIC_FEATURES:
            frame[feature] = pd.to_numeric(frame[feature])
        return model.predict_proba(frame[FEATURES])

    def run(seed: int) -> tuple[Any, dict[str, float]]:
        local_explainer = copy.deepcopy(explainer)
        local_explainer.random_state = np.random.RandomState(seed)
        result = local_explainer.explain_instance(
            raw_row,
            predict_raw,
            num_features=len(feature_names),
            labels=(1,),
            num_samples=5000 if seed == 42 else 2000,
        )
        grouped: dict[str, float] = {}
        for index, weight in result.as_map()[1]:
            grouped[feature_names[index]] = float(weight)
        return result, grouped

    explanation, grouped = run(42)
    top_k = 5
    ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
    top_names = {name for name, _ in ranked[:top_k]}
    with ThreadPoolExecutor(max_workers=5) as executor:
        runs = [grouped] + [repeated for _, repeated in executor.map(run, range(5))]
    overlap_scores = [
        len(
            {
                name
                for name, _ in sorted(left.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
            }
            & {
                name
                for name, _ in sorted(right.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
            }
        )
        / top_k
        for index, left in enumerate(runs)
        for right in runs[index + 1:]
    ]
    rank_scores = [
        float(
            spearmanr(
                [abs(left.get(name, 0.0)) for name in feature_names],
                [abs(right.get(name, 0.0)) for name in feature_names],
            ).statistic
        )
        for index, left in enumerate(runs)
        for right in runs[index + 1:]
    ]
    fidelity = float(np.clip(getattr(explanation, "score", 0.0), 0.0, 1.0))
    top5_overlap = float(np.mean(overlap_scores))
    rank_correlation = float(np.clip(np.nan_to_num(np.nanmean(rank_scores), nan=0.0), 0.0, 1.0))
    chance_overlap = 0.56
    chance_corrected_overlap = (top5_overlap - chance_overlap) / (1 - chance_overlap)
    stability = float(np.clip(np.mean([rank_correlation, max(0.0, chance_corrected_overlap)]), 0.0, 1.0))
    warning = (
        "Explanation reliability is low. Treat these drivers as uncertain and require human review."
        if fidelity < 0.7 or stability < 0.5
        else None
    )
    items = [
        ExplanationItem(
            feature=DISPLAY_NAMES.get(key, key),
            value=str(getattr(application, key)),
            contribution=round(weight, 4),
            direction="supports approval" if weight > 0 else "increases risk",
            importance=round(abs(weight), 4),
        )
        for key, weight in ranked
    ]
    return items, ExplanationReliability(
        fidelity=round(fidelity, 4),
        stability=round(stability, 4),
        rank_correlation=round(rank_correlation, 4),
        top5_overlap=round(top5_overlap, 4),
        top_k=top_k,
        warning=warning,
    )


def _reason_codes(
    items: list[ExplanationItem],
    decision: str,
    fidelity: float,
) -> tuple[list[str], Literal["ok", "suppressed_low_fidelity", "not_applicable"]]:
    if decision == "Approved":
        return [], "not_applicable"
    if fidelity < 0.5:
        return [], "suppressed_low_fidelity"
    codes = [
        REASON_CODES[item.feature]
        for item in items
        if item.contribution < 0
        and abs(item.contribution) >= 0.01
        and item.feature in REASON_CODES
    ][:3]
    return codes, "ok" if codes else "suppressed_low_fidelity"


def _counterfactuals(application: LoanApplication, model: Pipeline, current_decision: str) -> list[Counterfactual]:
    bounds = {
        "credit_score": (300.0, 850.0),
        "debt_to_income": (0.02, 0.85),
        "loan_amount": (2_000.0, 160_000.0),
        "savings_balance": (0.0, 350_000.0),
    }
    display_features = {
        "credit_score": "Credit score",
        "debt_to_income": "Debt-to-income ratio",
        "loan_amount": "Requested loan amount",
        "savings_balance": "Savings balance",
    }
    helpful_direction = {
        "credit_score": 1,
        "savings_balance": 1,
        "debt_to_income": -1,
        "loan_amount": -1,
    }
    original = {feature: float(getattr(application, feature)) for feature in bounds}
    candidates: list[tuple[float, set[str], dict[str, float]]] = []
    target_is_approved = current_decision == "Approved"

    def flipped(probability: float) -> bool:
        if target_is_approved:
            return probability <= 0.45
        return probability >= 0.55

    def format_value(feature: str, value: float) -> str:
        if feature == "credit_score":
            return str(round(value))
        if feature in {"loan_amount", "savings_balance"}:
            return f"${round(value / 100) * 100:,.0f}"
        return f"{value:.1%}"

    def step_limit(feature: str, current: float, direction: int) -> float:
        if feature == "credit_score":
            return 100.0
        if feature == "savings_balance":
            return max(current, 50_000.0)
        if feature == "loan_amount":
            return current * 0.6 if direction < 0 else max(current, 50_000.0)
        return 0.15

    def evaluate(values: dict[str, float], effort: float) -> None:
        candidates.append((effort, set(values), values))

    feature_values: dict[str, list[float]] = {}
    for feature, (lower, upper) in bounds.items():
        current = original[feature]
        direction = -helpful_direction[feature] if target_is_approved else helpful_direction[feature]
        limit = step_limit(feature, current, direction)
        target = upper if direction > 0 else lower
        target = min(target, current + limit) if direction > 0 else max(target, current - limit)
        feature_values[feature] = []
        for fraction in np.linspace(0.1, 1.0, 10):
            value = current + (target - current) * float(fraction)
            if feature == "credit_score":
                value = round(value)
            feature_values[feature].append(value)
            evaluate({feature: value}, abs(value - current) / limit if limit else 0.0)

    for first_index, first in enumerate(bounds):
        for second in list(bounds)[first_index + 1:]:
            for first_value in feature_values[first][3::3]:
                for second_value in feature_values[second][3::3]:
                    first_limit = step_limit(
                        first, original[first],
                        -helpful_direction[first] if target_is_approved else helpful_direction[first],
                    )
                    second_limit = step_limit(
                        second, original[second],
                        -helpful_direction[second] if target_is_approved else helpful_direction[second],
                    )
                    evaluate(
                        {first: first_value, second: second_value},
                        abs(first_value - original[first]) / first_limit
                        + abs(second_value - original[second]) / second_limit,
                    )

    values_frame = pd.DataFrame(
        [application.model_copy(update=values).model_dump() for _, _, values in candidates],
        columns=FEATURES,
    )
    probabilities = model.predict_proba(values_frame)[:, 1]
    selected: list[tuple[float, set[str], Counterfactual]] = []
    for (effort, features, values), probability in zip(candidates, probabilities):
        if not flipped(float(probability)):
            continue
        changes = {
            display_features[feature]: (
                f"{format_value(feature, original[feature])} -> "
                f"{format_value(feature, values[feature])}"
            )
            for feature in values
            if values[feature] != original[feature]
        }
        counterfactual = Counterfactual(
            changes=changes,
            decision="Approved" if probability >= 0.5 else "Declined",
            approval_probability=round(float(probability), 4),
            effort=round(effort, 4),
        )
        selected.append((effort, features, counterfactual))
    distinct: list[tuple[float, set[str], Counterfactual]] = []
    for candidate in sorted(selected, key=lambda item: item[0]):
        if not any(candidate[1] == existing[1] for existing in distinct):
            distinct.append(candidate)
        if len(distinct) == 3:
            break
    return [candidate[2] for candidate in distinct]


@lru_cache(maxsize=1)
def _library_versions() -> dict[str, str]:
    packages = {
        "fastapi": "fastapi",
        "lime": "lime",
        "numpy": "numpy",
        "pandas": "pandas",
        "pydantic": "pydantic",
        "scikit-learn": "scikit-learn",
        "scipy": "scipy",
    }
    versions: dict[str, str] = {}
    for name, distribution in packages.items():
        try:
            versions[name] = package_version(distribution)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


@lru_cache(maxsize=1)
def _metadata() -> dict[str, Any]:
    return {
        "model": "Sigmoid-calibrated Random Forest classifier",
        "explanation": "LIME local surrogate",
        "training": "Synthetic demonstration data (seed 42)",
        "version": MODEL_VERSION,
        "model_version": MODEL_VERSION,
        "data_version": DATA_VERSION,
        "library_versions": _library_versions(),
    }



app = FastAPI(title="ClearCredit Explainability API", version="1.0.0")

# Set FRONTEND_ORIGINS to deployed frontend URL(s), comma-separated.
_frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys([
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://clearcredit-explainable-ai-om6q.vercel.app",
        *_frontend_origins,
    ])),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def audit_failed_review_calls(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    if request.url.path in {"/api/decide", "/api/reviews"} and response.status_code >= 400:
        _audit(
            "failed_request",
            request.url.path,
            {"method": request.method, "status_code": response.status_code},
        )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/options")
def options() -> dict[str, list[str]]:
    return {
        "employment_type": ["Salaried", "Self-employed", "Contract"],
        "home_ownership": ["Own", "Mortgage", "Rent"],
        "loan_purpose": ["Home improvement", "Education", "Vehicle", "Debt consolidation"],
    }


def _assigned_condition(session_id: str) -> Literal["prediction_only", "prediction_lime"]:
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return "prediction_only" if int.from_bytes(digest[:8], "big") % 2 == 0 else "prediction_lime"


def _session_condition(session_id: str) -> Literal["prediction_only", "prediction_lime"]:
    with _connection() as connection:
        row = connection.execute(
            "SELECT condition FROM study_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is not None:
            return row["condition"]
        assigned = _assigned_condition(session_id)
        connection.execute(
            "INSERT INTO study_sessions (session_id, condition, created_at) VALUES (?, ?, ?)",
            (session_id, assigned, datetime.now(timezone.utc).isoformat()),
        )
        return assigned


@app.get("/api/session/{session_id}/condition")
def session_condition(
    session_id: str,
    condition: Literal["prediction_only", "prediction_lime"] | None = None,
) -> dict[str, str]:
    assigned = "prediction_lime"
    return {"session_id": session_id, "condition": assigned}


@app.get("/api/study/next-case")
def next_study_case(session_id: str) -> dict[str, Any]:
    _session_condition(session_id)
    with _connection() as connection:
        completed = connection.execute(
            "SELECT COUNT(*) FROM reviews WHERE session_id = ? AND mode = 'study'",
            (session_id,),
        ).fetchone()[0]
        if completed >= STUDY_CASES_PER_SESSION:
            return {"done": True, "completed": STUDY_CASES_PER_SESSION}
        wrong_rows = connection.execute(
            """
            SELECT case_id, applicant_json
            FROM study_cases
            WHERE ai_wrong = 1
            """,
        ).fetchall()
        right_rows = connection.execute(
            """
            SELECT case_id, applicant_json
            FROM study_cases
            WHERE ai_wrong = 0
            """,
        ).fetchall()
        seen_case_ids = {
            row[0]
            for row in connection.execute(
                "SELECT case_id FROM reviews WHERE session_id = ? AND mode = 'study' AND case_id IS NOT NULL",
                (session_id,),
            ).fetchall()
        }
    seed = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    selected_rows = rng.sample(wrong_rows, 4) + rng.sample(right_rows, 8)
    rng.shuffle(selected_rows)
    ordered_rows = [
        row for row in selected_rows
        if row["case_id"] not in seen_case_ids
    ]
    if not ordered_rows:
        raise HTTPException(status_code=404, detail="No unseen study cases remain for this session.")
    with _connection() as connection:
        case_number = completed + 1
    return {
        "case_id": ordered_rows[0]["case_id"],
        "case_number": case_number,
        "total_cases": STUDY_CASES_PER_SESSION,
        "applicant": json.loads(ordered_rows[0]["applicant_json"]),
    }


def _study_case_label(case_id: str) -> int:
    with _connection() as connection:
        row = connection.execute(
            "SELECT ground_truth_label FROM study_cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=400, detail="Unknown study case.")
    return int(row["ground_truth_label"])


def _study_case(case_id: str) -> tuple[LoanApplication, int]:
    with _connection() as connection:
        row = connection.execute(
            "SELECT applicant_json, ground_truth_label FROM study_cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=409, detail="Study case does not match the server case pool.")
    return LoanApplication.model_validate(json.loads(row["applicant_json"])), int(row["ground_truth_label"])


@app.post("/api/decide", response_model=DecisionResponse)
def decide(
    application: LoanApplication,
    session_id: str | None = Query(None),
    case_id: str | None = Query(None),
) -> DecisionResponse:
    model, explainer, _, _, feature_names, transformed_training_data = _artifacts()
    row = _frame(application)
    probability = float(model.predict_proba(row)[0, 1])
    decision = "Approved" if probability >= 0.5 else "Declined"
    risk_band = "Low risk" if probability >= 0.7 else "Moderate risk" if probability >= 0.5 else "Higher risk"
    session_id = session_id or application.session_id
    case_id = case_id or application.case_id
    condition = "prediction_lime" if session_id else None
    gated = False
    items, reliability = ([], None) if gated else _lime_explanation(
        application, model, explainer, feature_names, transformed_training_data
    )
    counterfactuals = [] if gated else _counterfactuals(application, model, decision)
    items.sort(key=lambda item: item.importance, reverse=True)
    reason_codes, reason_codes_status = (
        ([], "not_applicable")
        if gated
        else _reason_codes(items, decision, reliability.fidelity)
    )
    decision_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    summary = (
        "The application is more likely to be approved based on the applicant's local feature pattern."
        if decision == "Approved"
        else "The application is more likely to be declined based on the applicant's local feature pattern."
    )
    response = DecisionResponse(
        decision=decision,
        approval_probability=round(probability, 4),
        risk_band=risk_band,
        explanation=items,
        summary=summary,
        compliance_notes=[
            "Explanation describes local model behavior, not a causal reason for the outcome.",
            "Protected attributes are not used as model features in this demonstration.",
            "Human review is required before using this output for a real credit decision.",
        ],
        model_info=_metadata(),
        explanation_reliability=reliability,
        counterfactuals=counterfactuals,
        decision_id=decision_id,
        reason_codes=reason_codes,
        reason_codes_status=reason_codes_status,
    )
    record = DecisionRecord(
        decision_id=decision_id,
        timestamp=timestamp,
        applicant=application,
        model_version=MODEL_VERSION,
        data_version=DATA_VERSION,
        library_versions=_library_versions(),
        decision=decision,
        approval_probability=round(probability, 4),
        explanation=items,
        explanation_reliability=reliability,
        reason_codes=reason_codes,
        reason_codes_status=reason_codes_status,
    )
    with _connection() as connection:
        connection.execute(
            "INSERT INTO decision_records (decision_id, timestamp, applicant_json, record_json) VALUES (?, ?, ?, ?)",
            (decision_id, timestamp, application.model_dump_json(), record.model_dump_json()),
        )
    _audit(
        "decision",
        "/api/decide",
        {
            "decision_id": decision_id,
            "applicant_sha256": _applicant_hash(application),
            "decision": decision,
            "approval_probability": round(probability, 4),
            "risk_band": risk_band,
            "reason_codes_status": reason_codes_status,
            "model_version": MODEL_VERSION,
            "data_version": DATA_VERSION,
        },
    )
    return response


@app.get("/api/decisions/{decision_id}", response_model=DecisionRecord)
def get_decision_record(decision_id: str) -> DecisionRecord:
    with _connection() as connection:
        row = connection.execute(
            "SELECT record_json FROM decision_records WHERE decision_id = ?", (decision_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Decision record not found.")
    return DecisionRecord.model_validate_json(row["record_json"])


@app.get("/api/audit/verify")
def verify_audit() -> dict[str, Any]:
    return _verify_audit_log()


@app.get("/api/model/calibration")
def model_calibration() -> dict[str, Any]:
    return _artifacts()[3]


def _synthetic_group(
    data: pd.DataFrame,
    seed: int = 31415,
    proxy_strength: float = 0.0,
    proxy_feature: str = "credit_score",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probability = np.full(len(data), 0.5)
    if proxy_strength:
        values = pd.to_numeric(data[proxy_feature], errors="coerce").to_numpy(dtype=float)
        normalized = (values - values.mean()) / (values.std() or 1.0)
        probability = np.clip(0.5 + proxy_strength * 0.25 * normalized, 0.02, 0.98)
    return np.where(rng.random(len(data)) < probability, "Group A", "Group B")


def _cramers_v(feature: pd.Series, group: np.ndarray) -> float:
    table = pd.crosstab(feature, group).to_numpy(dtype=float)
    total = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    chi_square = ((table - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    phi2 = chi_square / total
    rows, columns = table.shape
    return float(np.sqrt(max(0.0, phi2 - ((columns - 1) * (rows - 1)) / max(total - 1, 1))
        / max(min(columns - 1, rows - 1), 1)))


def _fairness_report(proxy_strength: float = 0.0) -> dict[str, Any]:
    data, target = _training_data()
    _, held_out_data, _, held_out_target = train_test_split(
        data[FEATURES], target, test_size=0.25, random_state=42, stratify=target
    )
    demographic_group = _synthetic_group(held_out_data, proxy_strength=proxy_strength)
    model = _artifacts()[0]
    predictions = (model.predict_proba(held_out_data[FEATURES])[:, 1] >= 0.5).astype(int)
    groups = sorted(set(demographic_group))
    by_group: dict[str, dict[str, float | int]] = {}
    for group in groups:
        mask = demographic_group == group
        positives = held_out_target[mask] == 1
        negatives = held_out_target[mask] == 0
        by_group[group] = {
            "count": int(mask.sum()),
            "approval_rate": round(float(predictions[mask].mean()), 4),
            "true_positive_rate": round(float(predictions[mask][positives].mean()), 4) if positives.any() else 0.0,
            "false_positive_rate": round(float(predictions[mask][negatives].mean()), 4) if negatives.any() else 0.0,
        }
    approval_rates = [item["approval_rate"] for item in by_group.values()]
    true_positive_rates = [item["true_positive_rate"] for item in by_group.values()]
    proxy_correlations = []
    group_indicator = (demographic_group == groups[0]).astype(float)
    for feature in NUMERIC_FEATURES:
        correlation = float(np.corrcoef(held_out_data[feature].to_numpy(dtype=float), group_indicator)[0, 1])
        proxy_correlations.append({"feature": feature, "type": "pearson", "correlation": round(correlation, 4)})
    for feature in CATEGORICAL_FEATURES:
        proxy_correlations.append({"feature": feature, "type": "cramers_v", "correlation": round(_cramers_v(held_out_data[feature], demographic_group), 4)})
    return {
        "group_column": "synthetic_demographic_group",
        "proxy_strength": proxy_strength,
        "evaluation": "held_out_split",
        "model_features_exclude_group": True,
        "approval_rate_by_group": by_group,
        "demographic_parity_difference": round(max(approval_rates) - min(approval_rates), 4),
        "equal_opportunity_difference": round(max(true_positive_rates) - min(true_positive_rates), 4),
        "proxy_correlations": proxy_correlations,
        "possible_proxy_features": [item for item in proxy_correlations if abs(item["correlation"]) >= 0.2],
        "flags": [
            "Possible demographic disparity detected."
            if max(approval_rates) - min(approval_rates) > 0.1
            else "No demographic parity flag at the 0.10 demo threshold.",
            "Possible proxy features detected."
            if proxy_correlations
            else "No proxy correlation flag at the 0.20 demo threshold.",
        ],
    }


@app.get("/api/fairness")
def fairness(proxy_strength: float = Query(0.0, ge=0.0, le=1.0)) -> dict[str, Any]:
    report = _fairness_report(proxy_strength)
    _audit("fairness_report", "/api/fairness", report)
    return report


@app.post("/api/reviews", response_model=ReviewResponse)
def create_review(review: ReviewCreate) -> ReviewResponse:
    timestamp = datetime.now(timezone.utc).isoformat()
    assigned_condition = _session_condition(review.session_id)
    applicant, ground_truth_label = _study_case(review.case_id)
    with _connection() as connection:
        existing = connection.execute(
            "SELECT 1 FROM reviews WHERE session_id = ? AND case_id = ?",
            (review.session_id, review.case_id),
        ).fetchone()
    if existing is not None:
        raise HTTPException(status_code=409, detail="This study case has already been reviewed in this session.")
    model = _artifacts()[0]
    probability = float(model.predict_proba(_frame(applicant))[0, 1])
    ai_decision = "Approved" if probability >= 0.5 else "Declined"
    decision_id = str(uuid.uuid4())
    with _connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO reviews (
                timestamp, session_id, condition, applicant_json,
                initial_decision, initial_confidence, ai_decision,
                ai_probability, final_decision, final_confidence,
                ground_truth_label, decision_id, mode, case_id, initial_seconds, final_seconds
                , is_simulated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                review.session_id,
                assigned_condition,
                json.dumps(applicant.model_dump()),
                review.initial_decision,
                review.initial_confidence,
                ai_decision,
                probability,
                review.final_decision,
                review.final_confidence,
                ground_truth_label,
                decision_id,
                "study",
                review.case_id,
                review.initial_seconds,
                review.final_seconds,
                0,
            ),
        )
        row = connection.execute("SELECT * FROM reviews WHERE id = ?", (cursor.lastrowid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Review could not be stored.")
    response = _review_from_row(row)
    _audit(
        "review",
        "/api/reviews",
        {
            "decision_id": decision_id,
            "applicant_sha256": _applicant_hash(applicant),
            "session_id": review.session_id,
            "case_id": review.case_id,
            "condition": assigned_condition,
            "ai_decision": ai_decision,
            "ai_probability": round(probability, 4),
            "initial_decision": review.initial_decision,
            "final_decision": review.final_decision,
            "ground_truth_label": ground_truth_label,
            "model_version": MODEL_VERSION,
            "data_version": DATA_VERSION,
        },
    )
    return response


@app.get("/api/reviews/export")
def export_reviews() -> StreamingResponse:
    fields = [
        "timestamp",
        "session_id",
        "condition",
        "applicant",
        "initial_decision",
        "initial_confidence",
        "ai_decision",
        "ai_probability",
        "final_decision",
        "final_confidence",
        "ground_truth_label",
        "mode",
        "case_id",
        "initial_seconds",
        "final_seconds",
        "is_simulated",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    with _connection() as connection:
        rows = connection.execute("SELECT * FROM reviews ORDER BY id").fetchall()
    for row in rows:
        writer.writerow(
            {
                "timestamp": row["timestamp"],
                "session_id": row["session_id"],
                "condition": row["condition"],
                "applicant": row["applicant_json"],
                "initial_decision": row["initial_decision"],
                "initial_confidence": row["initial_confidence"],
                "ai_decision": row["ai_decision"],
                "ai_probability": row["ai_probability"],
                "final_decision": row["final_decision"],
                "final_confidence": row["final_confidence"],
                "ground_truth_label": row["ground_truth_label"],
                "mode": row["mode"],
                "case_id": row["case_id"],
                "initial_seconds": row["initial_seconds"],
                "final_seconds": row["final_seconds"],
                "is_simulated": row["is_simulated"],
            }
        )
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=clearcredit-reviews.csv"},
    )


def _review_metrics(rows: list[sqlite3.Row]) -> dict[str, Any]:
    rows = [
        row
        for row in rows
        if (row["mode"] if isinstance(row, sqlite3.Row) else row.get("mode", "study")) == "study"
    ]
    counted = [row for row in rows if row["ground_truth_label"] is not None]

    def value(row: Any, key: str) -> Any:
        return row[key] if isinstance(row, sqlite3.Row) else row[key]

    def proportion(predicate: Any, source: list[Any] = rows) -> float | None:
        return round(sum(predicate(row) for row in source) / len(source), 4) if source else None

    def decision_matches(value: str, label: int) -> bool:
        return (value == "Approve") == bool(label)

    ai_decision = lambda row: value(row, "ai_decision") == "Approved"
    ai_right = [row for row in counted if ai_decision(row) == bool(value(row, "ground_truth_label"))]
    ai_wrong = [row for row in counted if row not in ai_right]
    final_correct = lambda row: decision_matches(value(row, "final_decision"), value(row, "ground_truth_label"))
    before_correct = lambda row: decision_matches(value(row, "initial_decision"), value(row, "ground_truth_label"))
    over_num = sum(value(row, "final_decision") == ("Approve" if ai_decision(row) else "Decline") for row in ai_wrong)
    under_num = sum(value(row, "final_decision") != ("Approve" if ai_decision(row) else "Decline") for row in ai_right)
    appropriate_num = sum(
        (row in ai_wrong and value(row, "final_decision") != ("Approve" if ai_decision(row) else "Decline"))
        or (row in ai_right and value(row, "final_decision") == ("Approve" if ai_decision(row) else "Decline"))
        for row in counted
    )

    def rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    def bootstrap_interval(fn: Any) -> list[float] | None:
        if not counted:
            return None
        rng = np.random.default_rng(2026)
        samples = [
            result
            for result in (
                fn([counted[index] for index in rng.integers(0, len(counted), len(counted))])
                for _ in range(1000)
            )
            if result is not None
        ]
        if not samples:
            return None
        return [round(float(np.quantile(samples, 0.025)), 4), round(float(np.quantile(samples, 0.975)), 4)]

    def bootstrap_metric(sample: list[Any], metric: str) -> float | None:
        labels = [bool(value(row, "ground_truth_label")) for row in sample]
        ai = [value(row, "ai_decision") == "Approved" for row in sample]
        final = [value(row, "final_decision") == "Approve" for row in sample]
        ai_right_sample = [index for index, item in enumerate(ai) if item == labels[index]]
        ai_wrong_sample = [index for index, item in enumerate(ai) if item != labels[index]]
        if metric == "over_reliance":
            return rate(sum(final[index] == ai[index] for index in ai_wrong_sample), len(ai_wrong_sample))
        if metric == "under_reliance":
            return rate(sum(final[index] != ai[index] for index in ai_right_sample), len(ai_right_sample))
        if metric == "appropriate_reliance_rate":
            return rate(sum((index in ai_wrong_sample and final[index] != ai[index]) or (index in ai_right_sample and final[index] == ai[index]) for index in range(len(sample))), len(sample))
        if metric == "ai_alone_accuracy":
            return rate(len(ai_right_sample), len(sample))
        if metric == "human_accuracy_before":
            return rate(sum((value(row, "initial_decision") == "Approve") == labels[index] for index, row in enumerate(sample)), len(sample))
        return rate(sum(final[index] == labels[index] for index in range(len(sample))), len(sample))

    def confidence_ece(confidence_field: str, decision_field: str) -> float | None:
        if not counted:
            return None
        gaps = []
        for confidence in range(1, 6):
            bucket = [row for row in counted if round(float(value(row, confidence_field))) == confidence]
            if bucket:
                gaps.append(len(bucket) / len(counted) * abs(                0.5 + 0.5 * (confidence - 1) / 4 - np.mean([
                    decision_matches(value(row, decision_field), value(row, "ground_truth_label")) for row in bucket
                ])))
        return round(float(sum(gaps)), 4)

    metric_values = {
        "over_reliance": rate(over_num, len(ai_wrong)),
        "under_reliance": rate(under_num, len(ai_right)),
        "appropriate_reliance_rate": rate(appropriate_num, len(counted)),
        "human_accuracy_before": proportion(before_correct, counted),
        "human_accuracy_after": proportion(final_correct, counted),
        "ai_alone_accuracy": rate(len(ai_right), len(counted)),
        "human_ai_team_accuracy": proportion(final_correct, counted),
        "mean_confidence_correct": round(float(np.mean([value(row, "final_confidence") for row in counted if final_correct(row)])), 4) if any(final_correct(row) for row in counted) else None,
        "mean_confidence_incorrect": round(float(np.mean([value(row, "final_confidence") for row in counted if not final_correct(row)])), 4) if any(not final_correct(row) for row in counted) else None,
        "mean_initial_confidence_correct": round(float(np.mean([value(row, "initial_confidence") for row in counted if before_correct(row)])), 4) if any(before_correct(row) for row in counted) else None,
        "mean_initial_confidence_incorrect": round(float(np.mean([value(row, "initial_confidence") for row in counted if not before_correct(row)])), 4) if any(not before_correct(row) for row in counted) else None,
        "human_confidence_ece_before": confidence_ece("initial_confidence", "initial_decision"),
        "human_confidence_ece_after": confidence_ece("final_confidence", "final_decision"),
    }
    metric_values["over_reliance_numerator"] = over_num
    metric_values["over_reliance_denominator"] = len(ai_wrong)
    metric_values["under_reliance_numerator"] = under_num
    metric_values["under_reliance_denominator"] = len(ai_right)
    ci = {
        key: bootstrap_interval(
            (lambda sample, metric=key: bootstrap_metric(sample, metric))
        )
        for key in ("over_reliance", "under_reliance", "appropriate_reliance_rate", "human_accuracy_before", "human_accuracy_after", "ai_alone_accuracy", "human_ai_team_accuracy")
    }

    return {
        "n": len(counted),
        "review_count": len(rows),
        "ground_truth_count": len(counted),
        "agreement_rate": proportion(lambda row: value(row, "final_decision") == ("Approve" if ai_decision(row) else "Decline")),
        "switch_rate": proportion(lambda row: value(row, "initial_decision") != value(row, "final_decision")),
        **metric_values,
        "confidence_intervals_95": ci,
        "warning": "Fewer than 20 cases; estimates are unstable." if len(counted) < 20 else None,
    }


@app.get("/api/metrics/trust")
def trust_metrics(include_simulated: bool = Query(False)) -> dict[str, Any]:
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM reviews WHERE is_simulated = 0 OR ? ORDER BY id",
            (int(include_simulated),),
        ).fetchall()
    conditions = {
        condition: _review_metrics([row for row in rows if row["condition"] == condition])
        for condition in ("prediction_only", "prediction_lime")
    }
    condition_rows = {
        condition: [row for row in rows if row["condition"] == condition and row["mode"] == "study" and row["ground_truth_label"] is not None]
        for condition in ("prediction_only", "prediction_lime")
    }
    reviewer_groups: dict[str, list[float]] = {}
    for row in condition_rows["prediction_only"] + condition_rows["prediction_lime"]:
        reviewer_groups.setdefault(row["session_id"], []).append(
            float((row["final_decision"] == "Approve") == bool(row["ground_truth_label"]))
        )
    reviewer_accuracy = {
        session_id: float(np.mean(accuracies))
        for session_id, accuracies in reviewer_groups.items()
    }
    reviewer_conditions = {
        session_id: next(row["condition"] for row in condition_rows["prediction_only"] + condition_rows["prediction_lime"] if row["session_id"] == session_id)
        for session_id in reviewer_accuracy
    }
    reviewer_values = {
        condition: np.array(
            [accuracy for session_id, accuracy in reviewer_accuracy.items() if reviewer_conditions[session_id] == condition]
        )
        for condition in ("prediction_only", "prediction_lime")
    }
    observed = (
        float(reviewer_values["prediction_lime"].mean() - reviewer_values["prediction_only"].mean())
        if all(len(values) for values in reviewer_values.values())
        else None
    )
    combined = np.concatenate([reviewer_values["prediction_only"], reviewer_values["prediction_lime"]])
    rng = np.random.default_rng(2027)
    extreme = 0
    if observed is not None:
        first_size = len(reviewer_values["prediction_only"])
        for _ in range(5000):
            shuffled = rng.permutation(combined)
            difference = float(shuffled[first_size:].mean() - shuffled[:first_size].mean())
            extreme += abs(difference) >= abs(observed)
    reviewer_counts = {condition: int(len(reviewer_values[condition])) for condition in reviewer_values}
    return {
        "conditions": conditions,
        "total_reviews": len(rows),
        "include_simulated": include_simulated,
        "simulated_reviews": sum(int(row["is_simulated"]) for row in rows),
        "condition_difference_test": {
            "metric": "human_accuracy_after",
            "difference_prediction_lime_minus_prediction_only": round(observed, 4) if observed is not None else None,
            "permutation_p_value": round((extreme + 1) / 5001, 4) if observed is not None else None,
            "reviewers_per_condition": reviewer_counts,
            "warning": "Fewer than 10 reviewers in one or both conditions; interpret estimates cautiously."
            if any(count < 10 for count in reviewer_counts.values())
            else None,
        },
    }
