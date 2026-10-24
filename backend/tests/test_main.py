from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import json
import time
import app.main as main_module
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from app.main import (
    LABEL_OFFSET,
    REASON_CODES,
    LoanApplication,
    NUMERIC_FEATURES,
    _artifacts,
    _training_data,
    _counterfactuals,
    _connection,
    _lime_explanation,
    _reason_codes,
    _review_metrics,
    _session_condition,
    _study_case_label,
    _fairness_report,
    _audit,
    _verify_audit_log,
    app,
)


class ThresholdModel:
    def predict_proba(self, frame):
        approved = frame["credit_score"].to_numpy() >= 700
        return np.column_stack([np.where(approved, 0.1, 0.9), np.where(approved, 0.9, 0.1)])


def application(**overrides) -> LoanApplication:
    values = {
        "annual_income": 82000,
        "credit_score": 650,
        "debt_to_income": 0.28,
        "employment_years": 6,
        "loan_amount": 24000,
        "savings_balance": 42000,
        "employment_type": "Salaried",
        "home_ownership": "Mortgage",
        "loan_purpose": "Home improvement",
    }
    values.update(overrides)
    return LoanApplication(**values)


def test_review_metrics_calculate_condition_outcomes():
    rows = [
        {
            "ground_truth_label": 1,
            "initial_decision": "Decline",
            "initial_confidence": 2,
            "final_decision": "Approve",
            "final_confidence": 4,
            "ai_decision": "Approved",
        },
        {
            "ground_truth_label": 0,
            "initial_decision": "Approve",
            "initial_confidence": 4,
            "final_decision": "Approve",
            "final_confidence": 5,
            "ai_decision": "Approved",
        },
    ]
    metrics = _review_metrics(rows)
    assert metrics["review_count"] == 2
    assert metrics["agreement_rate"] == 1.0
    assert metrics["switch_rate"] == 0.5
    assert metrics["human_accuracy_before"] == 0.0
    assert metrics["human_accuracy_after"] == 0.5
    assert metrics["over_reliance"] == 1.0
    assert metrics["over_reliance_numerator"] == 1
    assert metrics["over_reliance_denominator"] == 1
    assert metrics["under_reliance"] == 0.0
    assert metrics["under_reliance_denominator"] == 1
    assert metrics["appropriate_reliance_rate"] == 0.5
    assert metrics["ai_alone_accuracy"] == 0.5
    assert metrics["human_ai_team_accuracy"] == 0.5
    assert metrics["human_confidence_ece_after"] is not None


def test_review_metrics_match_hand_computed_reliance_formulas():
    rows = [
        {"ground_truth_label": 1, "initial_decision": "Approve", "initial_confidence": 4,
         "final_decision": "Approve", "final_confidence": 4, "ai_decision": "Approved"},
        {"ground_truth_label": 0, "initial_decision": "Decline", "initial_confidence": 4,
         "final_decision": "Approve", "final_confidence": 3, "ai_decision": "Approved"},
        {"ground_truth_label": 0, "initial_decision": "Decline", "initial_confidence": 3,
         "final_decision": "Approve", "final_confidence": 2, "ai_decision": "Declined"},
        {"ground_truth_label": 1, "initial_decision": "Approve", "initial_confidence": 4,
         "final_decision": "Approve", "final_confidence": 2, "ai_decision": "Declined"},
    ]
    metrics = _review_metrics(rows)
    assert metrics["over_reliance_numerator"] == 1
    assert metrics["over_reliance_denominator"] == 2
    assert metrics["over_reliance"] == 0.5
    assert metrics["under_reliance_numerator"] == 1
    assert metrics["under_reliance_denominator"] == 2
    assert metrics["under_reliance"] == 0.5
    assert metrics["appropriate_reliance_rate"] == 0.5
    assert metrics["human_accuracy_before"] == 1.0
    assert metrics["human_accuracy_after"] == 0.5
    assert metrics["ai_alone_accuracy"] == 0.5


def test_calibration_bins_are_fixed_and_counted():
    calibration = _artifacts()[3]
    assert len(calibration["bins"]) == 10
    assert [item["bin_lower"] for item in calibration["bins"]] == [
        index / 10 for index in range(10)
    ]
    assert sum(item["count"] for item in calibration["bins"]) == calibration["held_out_count"]
    assert all(0 <= item["observed_frequency"] <= 1 for item in calibration["bins"] if item["count"])


def test_session_condition_assignment_is_balanced():
    assignments = [
        _session_condition(f"reviewer-{index:08d}")
        for index in range(1000)
    ]
    only_count = assignments.count("prediction_only")
    lime_count = assignments.count("prediction_lime")
    assert abs(only_count - lime_count) <= 80


def test_reason_codes_use_only_negative_mapped_drivers():
    items = [
        SimpleNamespace(feature="Debt-to-income ratio", contribution=-0.3),
        SimpleNamespace(feature="Credit score", contribution=-0.2),
        SimpleNamespace(feature="Annual income", contribution=0.4),
    ]
    assert _reason_codes(items, "Declined", 0.8) == ([
        REASON_CODES["Debt-to-income ratio"],
        REASON_CODES["Credit score"],
    ], "ok")
    assert _reason_codes(items, "Approved", 0.8) == ([], "not_applicable")


def test_reason_codes_suppress_low_fidelity_and_small_drivers():
    items = [SimpleNamespace(feature="Credit score", contribution=-0.3)]
    assert _reason_codes(items, "Declined", 0.49) == ([], "suppressed_low_fidelity")
    assert _reason_codes(
        [SimpleNamespace(feature="Credit score", contribution=-0.009)],
        "Declined",
        0.8,
    ) == ([], "suppressed_low_fidelity")


def test_counterfactual_search_finds_bounded_credit_score_change():
    candidates = _counterfactuals(application(), ThresholdModel(), "Declined")
    assert candidates
    assert len(candidates) <= 3
    assert any(candidate.decision == "Approved" for candidate in candidates)
    assert any("Credit score" in candidate.changes for candidate in candidates)
    assert candidates[0].approval_probability >= 0.55
    declined_candidates = _counterfactuals(
        application(credit_score=750), ThresholdModel(), "Approved"
    )
    assert declined_candidates
    assert all(candidate.approval_probability <= 0.45 for candidate in declined_candidates)


def test_counterfactuals_follow_helpful_directions_and_training_bounds():
    candidates = _counterfactuals(application(loan_amount=40_000, debt_to_income=0.55), ThresholdModel(), "Declined")
    assert candidates
    assert any(
        "Requested loan amount" in candidate.changes
        or "Debt-to-income ratio" in candidate.changes
        for candidate in candidates
    )
    for candidate in candidates:
        for label, change in candidate.changes.items():
            assert "->" in change
            old, new = change.split(" -> ")
            if label in {"Requested loan amount", "Debt-to-income ratio"}:
                assert not (float(new.strip("%,$").replace(",", "")) > float(old.strip("%,$").replace(",", "")))
        assert all(
            not (label in {"Requested loan amount", "Debt-to-income ratio"} and " -> " in change and
                 float(change.split(" -> ")[1].strip("%,$").replace(",", "")) > float(change.split(" -> ")[0].strip("%,$").replace(",", "")))
            for label, change in candidate.changes.items()
        )


def test_counterfactuals_are_distinct_feature_sets():
    candidates = _counterfactuals(application(), ThresholdModel(), "Declined")
    assert len({tuple(sorted(candidate.changes)) for candidate in candidates}) == len(candidates)


def test_counterfactual_values_stay_within_training_ranges():
    candidates = _counterfactuals(application(), ThresholdModel(), "Declined")
    limits = {
        "Credit score": (300, 850),
        "Debt-to-income ratio": (2, 85),
        "Requested loan amount": (2_000, 160_000),
        "Savings balance": (0, 350_000),
    }
    for candidate in candidates:
        for label, change in candidate.changes.items():
            new_value = change.split(" -> ")[1].replace("$", "").replace(",", "").replace("%", "")
            value = float(new_value)
            lower, upper = limits[label]
            assert lower <= value <= upper


@pytest.fixture
def client():
    return TestClient(app)


def test_decide_validation_rejects_out_of_range_credit_score(client):
    payload = application().model_dump()
    payload["credit_score"] = 299
    response = client.post("/api/decide", json=payload)
    assert response.status_code == 422


def test_review_validation_rejects_invalid_condition(client):
    case = client.get("/api/study/next-case?session_id=reviewer-test0001").json()
    payload = {
        "session_id": "reviewer-test0001",
        "case_id": case["case_id"],
        "condition": "invalid",
        "initial_decision": "Approve",
        "initial_confidence": 3,
        "final_decision": "Approve",
        "final_confidence": 4,
        "initial_seconds": 12,
        "final_seconds": 8,
    }
    response = client.post("/api/reviews", json=payload)
    assert response.status_code == 422


def test_session_condition_is_deterministic_and_supports_researcher_override(client):
    first = client.get("/api/session/reviewer-1/condition").json()
    second = client.get("/api/session/reviewer-1/condition").json()
    assert first == second
    assert first["condition"] in {"prediction_only", "prediction_lime"}
    override = client.get(
        "/api/session/reviewer-1/condition?condition=prediction_lime"
    )
    assert override.status_code == 200
    assert override.json()["condition"] == "prediction_lime"
    assert client.get("/api/session/reviewer-1/condition").json() == first


def test_decide_returns_reason_code_status(client):
    response = client.post("/api/decide", json=application().model_dump())
    assert response.status_code == 200
    body = response.json()
    assert body["reason_codes_status"] in {"ok", "suppressed_low_fidelity"}
    record = client.get(f"/api/decisions/{body['decision_id']}").json()
    assert record["reason_codes_status"] == body["reason_codes_status"]
    assert record["data_version"] == "synthetic-training-v1"
    assert record["library_versions"]["numpy"]


def test_audit_log_hash_chain_verifies_and_detects_tampering(tmp_path, monkeypatch, client):
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(main_module, "AUDIT_LOG_PATH", str(audit_path))
    _audit("decision", "/api/decide", {
        "decision_id": "decision-1",
        "applicant_sha256": "abc",
        "decision": "Approved",
    })
    _audit("review", "/api/reviews", {
        "decision_id": "decision-2",
        "applicant_sha256": "def",
        "final_decision": "Decline",
    })
    assert _verify_audit_log() == {"valid": True, "entries": 2, "error": None}
    verification_response = client.get("/api/audit/verify")
    assert verification_response.status_code == 200
    assert verification_response.json()["valid"] is True
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["final_decision"] = "Approve"
    audit_path.write_text(
        "\n".join([lines[0], json.dumps(tampered)]) + "\n",
        encoding="utf-8",
    )
    result = _verify_audit_log()
    assert result["valid"] is False
    assert "Entry hash mismatch" in result["error"]


def test_review_uses_server_assigned_condition(client):
    session_id = "reviewer-srv00001"
    assigned = client.get(f"/api/session/{session_id}/condition").json()["condition"]
    case_id = client.get(f"/api/study/next-case?session_id={session_id}").json()["case_id"]
    payload = {
        "session_id": session_id,
        "case_id": case_id,
        "initial_decision": "Approve",
        "initial_confidence": 3,
        "final_decision": "Approve",
        "final_confidence": 4,
        "initial_seconds": 3,
        "final_seconds": 4,
    }
    response = client.post("/api/reviews", json=payload)
    assert response.status_code == 200
    assert response.json()["condition"] == assigned


def test_study_case_pool_returns_unlabeled_cases_and_ground_truth_is_server_side(client):
    session_id = "case-pool-session"
    response = client.get(f"/api/study/next-case?session_id={session_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"].startswith("case-")
    assert "ground_truth_label" not in body
    assert set(body["applicant"]) == set(NUMERIC_FEATURES + ["employment_type", "home_ownership", "loan_purpose"])
    assert _study_case_label(body["case_id"]) in {0, 1}


def test_study_case_pool_has_about_thirty_percent_model_errors():
    model = _artifacts()[0]
    with _connection() as connection:
        rows = connection.execute(
            "SELECT applicant_json, ground_truth_label FROM study_cases"
        ).fetchall()
    assert len(rows) == 50
    applicants = [LoanApplication.model_validate(json.loads(row["applicant_json"])) for row in rows]
    predictions = (model.predict_proba(
        pd.DataFrame([item.model_dump() for item in applicants])
    )[:, 1] >= 0.5).astype(int)
    labels = np.array([row["ground_truth_label"] for row in rows])
    error_rate = float((predictions != labels).mean())
    assert 0.20 <= error_rate <= 0.40


def test_study_cases_use_display_precision():
    with _connection() as connection:
        rows = connection.execute("SELECT applicant_json FROM study_cases").fetchall()
    assert len(rows) == 50
    for row in rows:
        applicant = json.loads(row["applicant_json"])
        assert applicant["annual_income"] % 100 == 0
        assert applicant["loan_amount"] % 100 == 0
        assert applicant["savings_balance"] % 100 == 0
        assert applicant["credit_score"] == int(applicant["credit_score"])
        assert round(applicant["debt_to_income"], 2) == applicant["debt_to_income"]
        assert round(applicant["employment_years"], 1) == applicant["employment_years"]


def test_study_case_order_is_session_specific_and_balanced(client):
    session_ids = ["reviewer-ordA0001", "reviewer-ordB0001"]
    orders = []
    for session_id in session_ids:
        session_order = []
        for _ in range(12):
            response = client.get(f"/api/study/next-case?session_id={session_id}")
            assert response.status_code == 200
            body = response.json()
            assert body["total_cases"] == 12
            payload = {
                "session_id": session_id,
                "case_id": body["case_id"],
                "initial_decision": "Approve",
                "initial_confidence": 3,
                "final_decision": "Approve",
                "final_confidence": 3,
                "initial_seconds": 1,
                "final_seconds": 1,
            }
            assert client.post("/api/reviews", json=payload).status_code == 200
            with _connection() as connection:
                ai_wrong = connection.execute(
                    "SELECT ai_wrong FROM study_cases WHERE case_id = ?",
                    (body["case_id"],),
                ).fetchone()[0]
            session_order.append((body["case_id"], ai_wrong))
        orders.append(session_order)
    assert [item[0] for item in orders[0]] != [item[0] for item in orders[1]]
    assert sum(item[1] for item in orders[0]) == 4
    assert sum(item[1] for item in orders[1]) == 4


def test_study_session_ends_after_twelve_reviews(client):
    session_id = "reviewer-end00001"
    for _ in range(12):
        case = client.get(f"/api/study/next-case?session_id={session_id}").json()
        assert "case_id" in case
        payload = {
            "session_id": session_id,
            "case_id": case["case_id"],
            "initial_decision": "Approve",
            "initial_confidence": 3,
            "final_decision": "Approve",
            "final_confidence": 3,
            "initial_seconds": 1,
            "final_seconds": 1,
        }
        assert client.post("/api/reviews", json=payload).status_code == 200
    response = client.get(f"/api/study/next-case?session_id={session_id}")
    assert response.status_code == 200
    assert response.json() == {"done": True, "completed": 12}


def test_decide_condition_gates_explanations(client):
    sessions = {}
    for index in range(100):
        session_id = f"reviewer-gate{index:04d}"
        condition = client.get(f"/api/session/{session_id}/condition").json()["condition"]
        sessions[condition] = session_id
        if len(sessions) == 2:
            break
    decisions = []
    for credit_score in (650, 750):
        gated = client.post(
            f"/api/decide?session_id={sessions['prediction_only']}",
            json=application(credit_score=credit_score).model_dump(),
        ).json()
        decisions.append(gated["decision"])
        assert gated["explanation"] == []
        assert gated["explanation_reliability"] is None
        assert gated["counterfactuals"] == []
        assert gated["reason_codes"] == []
        assert gated["reason_codes_status"] == "not_applicable"
    assert set(decisions) == {"Approved", "Declined"}
    full = client.post(f"/api/decide?session_id={sessions['prediction_lime']}", json=application().model_dump()).json()
    assert full["explanation"]
    assert full["explanation_reliability"] is not None


def test_decide_completes_within_five_seconds(client):
    started = time.perf_counter()
    response = client.post("/api/decide", json=application().model_dump())
    assert response.status_code == 200
    assert time.perf_counter() - started < 5


def test_trust_permutation_uses_reviewer_units(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "DATABASE_PATH", str(tmp_path / "reviews.db"))
    main_module._init_database()
    with main_module._connection() as connection:
        for index in range(10):
            for condition, correct in (("prediction_only", True), ("prediction_lime", False)):
                connection.execute(
                    """
                    INSERT INTO reviews (
                        timestamp, session_id, condition, applicant_json,
                        initial_decision, initial_confidence, ai_decision,
                        ai_probability, final_decision, final_confidence,
                        ground_truth_label, mode, case_id, initial_seconds, final_seconds, is_simulated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "2026-01-01T00:00:00Z",
                        f"reviewer-{'only' if condition == 'prediction_only' else 'lime'}{index:04d}",
                        condition,
                        "{}",
                        "Approve",
                        3,
                        "Approved",
                        0.7,
                        "Approve" if correct else "Decline",
                        3,
                        1,
                        "study",
                        f"case-{index:03d}",
                        1,
                        1,
                        0,
                    ),
                )
    metrics = TestClient(main_module.app).get("/api/metrics/trust").json()
    permutation = metrics["condition_difference_test"]
    assert permutation["reviewers_per_condition"] == {
        "prediction_only": 10,
        "prediction_lime": 10,
    }
    assert permutation["difference_prediction_lime_minus_prediction_only"] == -1.0
    assert permutation["warning"] is None


def test_review_rejects_explore_payload_and_client_model_fields(client):
    payload = {
        "session_id": "reviewer-exp0001",
        "case_id": "case-001",
        "condition": "prediction_only",
        "mode": "explore",
        "applicant": application().model_dump(),
        "initial_decision": "Approve",
        "initial_confidence": 3,
        "ai_decision": "Approved",
        "ai_probability": 0.7,
        "final_decision": "Approve",
        "final_confidence": 4,
    }
    response = client.post("/api/reviews", json=payload)
    assert response.status_code == 422


def test_review_rejects_invalid_session_format(client):
    response = client.post(
        "/api/reviews",
        json={
            "session_id": "not-a-reviewer",
            "case_id": "case-001",
            "initial_decision": "Approve",
            "initial_confidence": 3,
            "final_decision": "Approve",
            "final_confidence": 4,
            "initial_seconds": 2,
            "final_seconds": 3,
        },
    )
    assert response.status_code == 422


def test_review_returns_conflict_for_duplicate_and_case_mismatch(client):
    session_id = "reviewer-dupe0001"
    case_id = client.get(f"/api/study/next-case?session_id={session_id}").json()["case_id"]
    payload = {
        "session_id": session_id,
        "case_id": case_id,
        "initial_decision": "Approve",
        "initial_confidence": 3,
        "final_decision": "Approve",
        "final_confidence": 4,
        "initial_seconds": 2,
        "final_seconds": 3,
    }
    assert client.post("/api/reviews", json=payload).status_code == 200
    assert client.post("/api/reviews", json=payload).status_code == 409
    payload["case_id"] = "case-999"
    assert client.post("/api/reviews", json=payload).status_code == 409


def test_training_approval_rate_is_balanced():
    _, target = _training_data()
    assert 0.30 <= target.mean() <= 0.45
    assert LABEL_OFFSET == 1.0


def test_fairness_uses_held_out_split_and_reports_rates_and_proxy_types():
    report = _fairness_report(proxy_strength=1.0)
    assert report["evaluation"] == "held_out_split"
    assert 0.0 <= report["demographic_parity_difference"] <= 1.0
    assert 0.0 <= report["equal_opportunity_difference"] <= 1.0
    assert all("false_positive_rate" in item for item in report["approval_rate_by_group"].values())
    assert any(item["feature"] == "credit_score" and item["type"] == "pearson" and abs(item["correlation"]) >= 0.2 for item in report["proxy_correlations"])
    assert all(item["type"] == "cramers_v" for item in report["proxy_correlations"] if item["feature"] in {"employment_type", "home_ownership", "loan_purpose"})


def test_sample_applicant_is_approved_in_expected_probability_range(client):
    response = client.post("/api/decide", json=application(credit_score=718).model_dump())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "Approved"
    assert 0.50 <= body["approval_probability"] <= 0.70


def test_weak_applicant_is_declined(client):
    weak = application(
        annual_income=40000,
        credit_score=580,
        debt_to_income=0.55,
        employment_years=1,
        loan_amount=45000,
        savings_balance=2000,
        employment_type="Contract",
        home_ownership="Rent",
        loan_purpose="Vehicle",
    )
    response = client.post("/api/decide", json=weak.model_dump())
    assert response.status_code == 200
    assert response.json()["decision"] == "Declined"


def test_held_out_cases_have_approval_probability_at_least_half():
    model, _, data, _, _, _ = _artifacts()
    _, target = _training_data()
    features = NUMERIC_FEATURES + ["employment_type", "home_ownership", "loan_purpose"]
    _, held_out, _, _ = train_test_split(
        data[features],
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )
    probabilities = model.predict_proba(held_out)[:, 1]
    assert (probabilities >= 0.5).mean() >= 0.15


def test_lime_reliability_is_bounded_and_weak_driver_is_negative():
    weak = application(
        annual_income=40000,
        credit_score=580,
        debt_to_income=0.55,
        employment_years=1,
        loan_amount=45000,
        savings_balance=2000,
        employment_type="Contract",
        home_ownership="Rent",
        loan_purpose="Vehicle",
    )
    model, explainer, _, _, feature_names, raw_training_data = _artifacts()
    items, reliability = _lime_explanation(
        weak, model, explainer, feature_names, raw_training_data
    )
    assert 0 <= reliability.fidelity <= 1
    assert 0 <= reliability.stability <= 1
    assert 0 <= reliability.rank_correlation <= 1
    assert 0 <= reliability.top5_overlap <= 1
    assert items[0].contribution < 0
