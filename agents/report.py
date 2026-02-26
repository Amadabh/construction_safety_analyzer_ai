import os
import sys
import json
import subprocess
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))
from datetime import datetime
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from schemas import ProcessingResult, AlertLevel, Detection
from config import Config

ALERT_COLORS = {
    AlertLevel.LOW:      ("70AD47", RGBColor(0x70, 0xAD, 0x47)),
    AlertLevel.MEDIUM:   ("FFC000", RGBColor(0xFF, 0xC0, 0x00)),
    AlertLevel.HIGH:     ("FF0000", RGBColor(0xFF, 0x00, 0x00)),
    AlertLevel.CRITICAL: ("7030A0", RGBColor(0x70, 0x30, 0xA0)),
}

VIOLATION_LABELS  = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
MACHINERY_LABELS  = {"Excavator", "Wheel Loader", "Machinery", "Dump Truck", "machinery"}

REPORT_SYSTEM_PROMPT = """You are a senior construction site safety officer writing a formal incident report for legal and compliance purposes.

You will be given JSON data containing:
- risk_score and alert_level
- raw_detections: everything the AI vision system detected, with counts and average confidence per label
- violations: the specific PPE/safety violations (subset of raw_detections)
- equipment_on_site: heavy machinery detected
- workers_detected: number of people observed
- applicable_regulations: OSHA/CAL-OSHA regulations retrieved from the compliance database

Write a detailed, professional safety incident report with EXACTLY these sections using markdown headings:

## 1. Executive Summary
2-3 sentences summarizing the overall safety situation, risk level, and most critical findings. Mention specific numbers (e.g. "6 instances of workers without hard hats were detected across 10 frames").

## 2. Site Conditions Overview
Describe what was observed — number of workers, equipment present, general activity level. Reference the specific detection counts from the data.

## 3. Violations Detected
For EACH unique violation type found, write a dedicated paragraph:
- State exactly how many instances were detected and the average confidence score
- Explain in detail why this specific violation is dangerous — what injuries or fatalities can result
- Describe a realistic accident scenario that could occur if this goes uncorrected
- Reference any equipment present that amplifies the risk (e.g. NO-Hardhat near Excavator = crush/falling object risk)

## 4. Applicable Regulations
For each regulation:
- State the exact citation
- Explain what the regulation requires workers and employers to do
- Explain specifically how the detected violations breach this regulation
- State what enforcement action OSHA can take

## 5. Risk Assessment Analysis
Explain the risk score in detail:
- Why this specific score was assigned
- How the combination of violations + equipment present elevates risk
- What would need to change to reduce the score

## 6. Recommended Corrective Actions
For each violation type, provide specific actionable steps:
- Immediate (stop-work, PPE distribution, supervisor notification)
- Short-term (toolbox talks, training, accountability measures)
- Long-term (procurement policy, daily PPE checks, monitoring)

## 7. Compliance & Legal Implications
- Potential OSHA citation categories (Serious, Willful, Repeat)
- Penalty ranges under Cal/OSHA
- Liability exposure if a worker is injured under these conditions

Use formal, precise language. Be thorough and specific — reference actual detection counts and confidence scores throughout. This report may be used in legal proceedings."""


def set_cell_bg(cell, hex_color: str):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


class ReportGenerator:
    def __init__(self):
        from model import BedrockModel
        self.model = BedrockModel.get_instance()

    def generate_report(self, result: ProcessingResult, detections: list[Detection] = None) -> str:
        print("Generating report...")
        ra = result.risk_assessment
        detections = detections or []

        # ── Build rich detection statistics from raw Roboflow output ──────────
        # Count and average confidence per label — gives LLM precise numbers
        label_stats = {}
        for d in detections:
            if d.label not in label_stats:
                label_stats[d.label] = {"count": 0, "confidence_sum": 0.0}
            label_stats[d.label]["count"]          += 1
            label_stats[d.label]["confidence_sum"] += d.confidence

        raw_detections_summary = [
            {
                "label":              label,
                "count":              stats["count"],
                "avg_confidence":     f"{stats['confidence_sum'] / stats['count']:.1%}",
                "is_violation":       label in VIOLATION_LABELS,
                "is_heavy_equipment": label in MACHINERY_LABELS,
            }
            for label, stats in sorted(label_stats.items())
        ]

        violations_summary = [d for d in raw_detections_summary if d["is_violation"]]
        equipment_summary  = [d for d in raw_detections_summary if d["is_heavy_equipment"]]
        workers_detected   = label_stats.get("Person", {}).get("count", 0)

        context = {
            "video_id":       os.path.basename(result.video_id),
            "analysis_time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_score":     ra.risk_score,
            "alert_level":    ra.alert_level.value,

            # Full raw detection breakdown — LLM uses these exact numbers in the report
            "raw_detections": raw_detections_summary,

            # Violation-only subset with counts + confidence
            "violations": violations_summary,

            # Equipment and people context
            "equipment_on_site": equipment_summary,
            "workers_detected":  workers_detected,
            "total_frames_analyzed": len(set(
                round(d.confidence, 1) for d in detections  # proxy for frame count
            )),

            # LLM risk assessor violations (includes reasoning if available)
            "risk_assessor_violations": [
                {
                    "type":      v.type,
                    "severity":  v.severity.value,
                    "reasoning": getattr(v, "reasoning", ""),
                }
                for v in ra.violations
            ],

            # Full OSHA regulation text
            "applicable_regulations": [
                {
                    "citation":  r.citation,
                    "full_text": r.text,
                    "source":    r.source,
                }
                for r in result.regulations
            ],
        }

        try:
            report_text = self.model.invoke(
                REPORT_SYSTEM_PROMPT,
                json.dumps(context, indent=2)
            )
        except Exception as e:
            print(f"⚠ Report generation failed: {e}")
            report_text = self._fallback_report(result, violations_summary)

        safe_name   = os.path.basename(result.video_id).replace(" ", "_")
        output_path = os.path.join(Config.REPORTS_DIR, f"{safe_name}_report.docx")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._build_docx(result, report_text, raw_detections_summary, output_path)

        pdf_path   = self._convert_to_pdf(output_path)
        final_path = pdf_path if pdf_path else output_path

        print(f"✓ Report saved to {final_path}")
        return final_path

    def _fallback_report(self, result: ProcessingResult, violations_summary: list) -> str:
        ra = result.risk_assessment
        lines = [
            "## 1. Executive Summary",
            f"AI safety analysis detected violations with a risk score of "
            f"{ra.risk_score}/100 (Alert Level: {ra.alert_level.value}).",
            "",
            "## 2. Violations Detected",
        ]
        for v in violations_summary:
            lines.append(f"- {v['label']}: {v['count']} instances detected "
                         f"(avg confidence: {v['avg_confidence']})")
        lines += ["", "## 3. Recommended Actions",
                  "Immediately halt work and ensure all workers are equipped with "
                  "required PPE before resuming operations."]
        return "\n".join(lines)

    @staticmethod
    def _convert_to_pdf(docx_path: str) -> str | None:
        try:
            out_dir = os.path.dirname(docx_path)
            result = subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "pdf",
                 "--outdir", out_dir, docx_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                pdf_path = docx_path.replace(".docx", ".pdf")
                if os.path.exists(pdf_path):
                    print(f"✓ PDF created: {pdf_path}")
                    return pdf_path
            print(f"⚠ LibreOffice conversion failed: {result.stderr.strip()}")
        except FileNotFoundError:
            print("⚠ LibreOffice not found — skipping PDF conversion")
        except subprocess.TimeoutExpired:
            print("⚠ LibreOffice conversion timed out")
        return None

    def _build_docx(self, result: ProcessingResult, report_text: str,
                    raw_detections_summary: list, output_path: str):
        doc = Document()
        ra  = result.risk_assessment

        for section in doc.sections:
            section.top_margin    = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin   = Inches(1)
            section.right_margin  = Inches(1)

        # ── Title ─────────────────────────────────────────────────────────────
        doc.add_heading("Construction Site Safety Incident Report", 0)
        p = doc.add_paragraph()
        p.add_run("Video File: ").bold = True
        p.add_run(os.path.basename(result.video_id))
        p2 = doc.add_paragraph()
        p2.add_run("Report Generated: ").bold = True
        p2.add_run(datetime.now().strftime("%B %d, %Y %H:%M:%S"))
        doc.add_paragraph()

        # ── Risk score banner ─────────────────────────────────────────────────
        hex_color, rgb_color = ALERT_COLORS.get(ra.alert_level, ("000000", RGBColor(0, 0, 0)))
        tbl = doc.add_table(rows=1, cols=2)
        tbl.style = "Table Grid"
        cells = tbl.rows[0].cells
        set_cell_bg(cells[0], "F5F5F5")
        r0 = cells[0].paragraphs[0].add_run(f"Risk Score: {ra.risk_score}/100")
        r0.bold = True; r0.font.size = Pt(16); r0.font.color.rgb = rgb_color
        set_cell_bg(cells[1], hex_color)
        r1 = cells[1].paragraphs[0].add_run(f"Alert Level: {ra.alert_level.value}")
        r1.bold = True; r1.font.size = Pt(16)
        r1.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        doc.add_paragraph()

        # ── Full detections table ─────────────────────────────────────────────
        if raw_detections_summary:
            doc.add_heading("Detection Summary", level=2)
            dtbl = doc.add_table(rows=1, cols=4)
            dtbl.style = "Table Grid"
            for cell, label in zip(dtbl.rows[0].cells,
                                   ["Label", "Count", "Avg Confidence", "Type"]):
                run = cell.paragraphs[0].add_run(label)
                run.bold = True
                set_cell_bg(cell, "2E3A4A")
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

            for d in raw_detections_summary:
                row = dtbl.add_row().cells
                row[0].text = d["label"]
                row[1].text = str(d["count"])
                row[2].text = d["avg_confidence"]
                type_label  = "⚠ VIOLATION" if d["is_violation"] else \
                              "🚧 Equipment" if d["is_heavy_equipment"] else "Person/Object"
                row[3].text = type_label
                # Highlight violation rows red
                if d["is_violation"]:
                    for cell in row:
                        set_cell_bg(cell, "FFE0E0")

            doc.add_paragraph()

        # ── LLM report body ───────────────────────────────────────────────────
        for line in report_text.split("\n"):
            line = line.strip()
            if not line:
                doc.add_paragraph()
                continue
            if line.startswith("## "):
                doc.add_heading(line[3:].strip(), level=2)
            elif line.startswith("# "):
                doc.add_heading(line[2:].strip(), level=1)
            elif line.startswith("- ") or line.startswith("* "):
                p = doc.add_paragraph(style="List Bullet")
                p.add_run(line[2:].strip())
            else:
                doc.add_paragraph(line)

        doc.save(output_path)
