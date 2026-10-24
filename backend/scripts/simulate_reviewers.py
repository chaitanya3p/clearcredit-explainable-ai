"""Seed reviews from model-only simulated reviewer behavior.

The simulated reviewers never use the held-out labels to choose a decision or
confidence. Labels are copied into the stored row only through the database
insert so the resulting observations can be scored by the study metrics.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import (  # noqa: E402
    _artifacts,
    _connection,
    _frame,
    _session_condition,
    _init_database,
    _ensure_study_cases,
    LoanApplication,
)


REVIEWER_PARAMETERS = [
    {"skill": 0.45, "reliance_probability": 0.85, "confidence_noise": 0.55},
    {"skill": 0.55, "reliance_probability": 0.75, "confidence_noise": 0.50},
    {"skill": 0.65, "reliance_probability": 0.65, "confidence_noise": 0.40},
    {"skill": 0.75, "reliance_probability": 0.55, "confidence_noise": 0.35},
    {"skill": 0.85, "reliance_probability": 0.45, "confidence_noise": 0.25},
]


def _decision_from_probability(probability: float) -> bool:
    return probability >= 0.5


def _confidence(
    probability: float,
    decision: bool,
    noise: float,
    rng: np.random.Generator,
) -> int:
    certainty = abs(probability - 0.5) * 8 + 2.5
    if not decision:
        certainty = max(1.0, certainty)
    return int(np.clip(round(certainty + rng.normal(0, noise)), 1, 5))


def main() -> None:
    rng = np.random.default_rng(2026)
    _init_database()
    _ensure_study_cases()
    model = _artifacts()[0]

    with _connection() as connection:
        cases = connection.execute(
            "SELECT case_id, applicant_json FROM study_cases ORDER BY case_id"
        ).fetchall()
        connection.execute("DELETE FROM reviews WHERE is_simulated = 1")

    created = 0
    for reviewer_number in range(30):
        parameters = REVIEWER_PARAMETERS[reviewer_number % len(REVIEWER_PARAMETERS)]
        session_id = f"reviewer-sim{reviewer_number + 1:05d}"
        condition = _session_condition(session_id)
        for case in cases:
            application = LoanApplication.model_validate(json.loads(case["applicant_json"]))
            probability = float(model.predict_proba(_frame(application))[0, 1])
            ai_decision = _decision_from_probability(probability)

            # Skill controls how strongly a reviewer forms an independent
            # probability judgment; it never uses the held-out label.
            human_probability = float(
                np.clip(
                    0.5 + parameters["skill"] * (probability - 0.5)
                    + rng.normal(0, 0.18 * (1 - parameters["skill"])),
                    0.05,
                    0.95,
                )
            )
            initial_decision = human_probability >= 0.5
            final_decision = (
                ai_decision
                if rng.random() < parameters["reliance_probability"]
                else initial_decision
            )
            initial_confidence = _confidence(
                human_probability,
                initial_decision,
                parameters["confidence_noise"],
                rng,
            )
            final_confidence = _confidence(
                probability,
                final_decision,
                parameters["confidence_noise"],
                rng,
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            decision_id = str(uuid.uuid4())
            with _connection() as connection:
                connection.execute(
                    """
                    INSERT INTO reviews (
                        timestamp, session_id, condition, applicant_json,
                        initial_decision, initial_confidence, ai_decision,
                        ai_probability, final_decision, final_confidence,
                        ground_truth_label, decision_id, mode, case_id,
                        initial_seconds, final_seconds, is_simulated
                    )
                    SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ground_truth_label,
                        ?, 'study', case_id, ?, ?, 1
                    FROM study_cases
                    WHERE case_id = ?
                    """,
                    (
                        timestamp,
                        session_id,
                        condition,
                        json.dumps(application.model_dump()),
                        "Approve" if initial_decision else "Decline",
                        initial_confidence,
                        "Approved" if ai_decision else "Declined",
                        probability,
                        "Approve" if final_decision else "Decline",
                        final_confidence,
                        decision_id,
                        float(rng.integers(8, 90)),
                        float(rng.integers(8, 90)),
                        case["case_id"],
                    ),
                )
            created += 1

    print(
        f"Created {created} simulated reviews for 30 reviewers "
        f"({len(cases)} held-out cases each)."
    )


if __name__ == "__main__":
    main()
