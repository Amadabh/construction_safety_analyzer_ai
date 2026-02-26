import requests
import boto3
from schemas import RiskAssessment
from utils import initialize_sns, publish_alert
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

# ── Helpers ──────────────────────────────────────────────────────────────────
def _build_message(risk: RiskAssessment) -> tuple[str, str]:
    subject = f"[{risk.alert_level}] Construction Safety Alert - Risk Score: {risk.risk_score}"
    violations_text = "\n".join(
        f"- {v.type} (confidence: {v.confidence:.0%}, duration: {v.duration_seconds}s)"
        for v in risk.violations
    )
    message = f"""
Construction Safety Alert
=========================
Alert Level : {risk.alert_level}
Risk Score  : {risk.risk_score}/100

Violations Detected:
{violations_text if violations_text else 'None'}

Equipment Context: {', '.join(risk.equipment_context) if risk.equipment_context else 'N/A'}
    """.strip()
    return subject, message

# ── Public API ───────────────────────────────────────────────────────────────
def run_alert_agent(risk_assessment: RiskAssessment, report_path: str | None = None) -> list[str]:
    """Entry point called by graph.py to trigger alerts."""
    alerts_sent = []

    try:
        topic_arn = initialize_sns()
    except Exception as e:
        print(f"⚠ SNS initialization failed (check AWS credentials): {e}")
        print("Skipping alert delivery.")
        return alerts_sent

    subject, message = _build_message(risk_assessment)

    if publish_alert(topic_arn, subject, message):
        alerts_sent.append("email")

    print(f"Alerts sent via: {alerts_sent}")
    return alerts_sent