import json
from schemas import Detection, RiskAssessment, AlertLevel, Violation, Regulation

VIOLATION_LABELS = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
MACHINERY_LABELS = {"Excavator", "Wheel Loader", "Machinery", "Dump Truck", "machinery"}

SYSTEM_PROMPT = """You are a construction site safety expert.
Given detected violations, equipment context, and relevant OSHA regulations,
return a JSON risk assessment with this exact structure:
{
  "risk_score": <0-100 integer>,
  "alert_level": <"LOW"|"MEDIUM"|"HIGH"|"CRITICAL">,
  "violations": [
    {
      "type": <violation label>,
      "severity": <"LOW"|"MEDIUM"|"HIGH"|"CRITICAL">,
      "confidence": <float>,
      "reasoning": <one sentence why this is risky>
    }
  ],
  "reasoning": <overall 1-2 sentence explanation>
}
Scoring guide:
- No violations, no equipment: 0-20
- PPE violation, no equipment nearby: 30-50
- PPE violation + heavy machinery present: 60-80
- Multiple PPE violations + machinery + people nearby: 80-100
Return only valid JSON, no other text."""


class RiskAssessor:
    def __init__(self):
        from model import BedrockModel
        self.model = BedrockModel.get_instance()

    def assess(self, detections: list[Detection], regulations: list[Regulation] = None) -> RiskAssessment:
        print("Assessing risk...")
        if not detections:
            return RiskAssessment(risk_score=0, alert_level=AlertLevel.LOW)

        violations_found  = [d for d in detections if d.label in VIOLATION_LABELS]
        equipment_present = [d for d in detections if d.label in MACHINERY_LABELS]
        people_count      = sum(1 for d in detections if d.label == "Person")

        context = {
            "violations": [
                {"label": v.label, "confidence": round(v.confidence, 3)}
                for v in violations_found
            ],
            "heavy_equipment":  [e.label for e in equipment_present],
            "people_nearby":    people_count,
            "relevant_regulations": [
                {"citation": r.citation, "text": r.text[:300]}
                for r in (regulations or [])
            ],
        }

        try:
            data = self.model.invoke_json(SYSTEM_PROMPT, json.dumps(context, indent=2))
            return self._parse_response(data, violations_found, equipment_present)
        except Exception as e:
            print(f"⚠ Risk assessment failed: {e}, falling back to rule-based scoring")
            return self._rule_based_fallback(violations_found, equipment_present)

    def _parse_response(self, data: dict, violations_found: list, equipment_present: list) -> RiskAssessment:
        violations = [
            Violation(
                type=v["type"],
                confidence=v.get("confidence", 0.0),
                severity=AlertLevel(v["severity"]),
                timestamp_start=0.0,
                timestamp_end=0.0,
            )
            for v in data.get("violations", [])
        ]
        return RiskAssessment(
            risk_score=data["risk_score"],
            alert_level=AlertLevel(data["alert_level"]),
            violations=violations,
            equipment_context=[e.label for e in equipment_present],
        )

    def _rule_based_fallback(self, violations_found: list, equipment_present: list) -> RiskAssessment:
        score = min(len(violations_found) * 25, 100)
        level = (
            AlertLevel.CRITICAL if score >= 75 else
            AlertLevel.HIGH     if score >= 50 else
            AlertLevel.MEDIUM   if score >= 25 else
            AlertLevel.LOW
        )
        return RiskAssessment(
            risk_score=score,
            alert_level=level,
            violations=[],
            equipment_context=[e.label for e in equipment_present],
        )