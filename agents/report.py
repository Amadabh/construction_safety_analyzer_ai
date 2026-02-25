import os
import json
from datetime import datetime
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from schemas import ProcessingResult, AlertLevel

ALERT_COLORS = {
    AlertLevel.LOW:      ("70AD47", RGBColor(0x70, 0xAD, 0x47)),
    AlertLevel.MEDIUM:   ("FFC000", RGBColor(0xFF, 0xC0, 0x00)),
    AlertLevel.HIGH:     ("FF0000", RGBColor(0xFF, 0x00, 0x00)),
    AlertLevel.CRITICAL: ("7030A0", RGBColor(0x70, 0x30, 0xA0)),
}

REPORT_SYSTEM_PROMPT = """You are a construction site safety expert writing an incident report.
Based on the safety data provided, write a clear professional report with the following sections:

1. Executive Summary - brief overview of what was found
2. Violations Found - describe each violation and why it is dangerous
3. Applicable Regulations - summarize the relevant OSHA rules that were violated
4. Recommended Actions - what should be done to fix the issues

Use plain professional language. Include headings for each section."""


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

    def generate_report(self, result: ProcessingResult) -> str:
        print("Generating report...")

        # Build context for the LLM
        context = {
            "video_id":    result.video_id,
            "risk_score":  result.risk_assessment.risk_score,
            "alert_level": result.risk_assessment.alert_level.value,
            "violations": [
                {
                    "type":       v.type,
                    "severity":   v.severity.value,
                    "confidence": f"{v.confidence * 100:.1f}%",
                    "duration":   f"{v.duration_seconds:.1f}s",
                }
                for v in result.risk_assessment.violations
            ],
            "regulations": [
                {"citation": r.citation, "text": r.text}
                for r in result.regulations
            ],
        }

        # Let the LLM write the full report content
        try:
            report_text = self.model.invoke(
                REPORT_SYSTEM_PROMPT,
                json.dumps(context, indent=2)
            )
        except Exception as e:
            print(f"⚠ Report generation failed: {e}")
            report_text = f"Safety analysis for {result.video_id} detected {len(result.risk_assessment.violations)} violation(s) with risk score {result.risk_assessment.risk_score}/100."

        # Save to docx
        output_path = f"/home/aditya/home/basic/construction_safety/data/reports/{result.video_id}_report.docx"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._build_docx(result, report_text, output_path)

        print(f"✓ Report saved to {output_path}")
        return output_path

    def _build_docx(self, result: ProcessingResult, report_text: str, output_path: str):
        doc = Document()
        ra  = result.risk_assessment

        for section in doc.sections:
            section.top_margin    = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin   = Inches(1)
            section.right_margin  = Inches(1)

        # Title
        doc.add_heading("Construction Site Safety Report", 0)
        p = doc.add_paragraph()
        p.add_run("Video ID: ").bold = True
        p.add_run(result.video_id)
        p2 = doc.add_paragraph()
        p2.add_run("Generated: ").bold = True
        p2.add_run(datetime.now().strftime("%B %d, %Y %H:%M:%S"))
        doc.add_paragraph()

        # Risk score banner
        hex_color, rgb_color = ALERT_COLORS.get(ra.alert_level, ("000000", RGBColor(0,0,0)))
        tbl = doc.add_table(rows=1, cols=2)
        tbl.style = "Table Grid"
        cells = tbl.rows[0].cells

        set_cell_bg(cells[0], "F5F5F5")
        r0 = cells[0].paragraphs[0].add_run(f"Risk Score: {ra.risk_score}/100")
        r0.bold = True
        r0.font.size = Pt(14)
        r0.font.color.rgb = rgb_color

        set_cell_bg(cells[1], hex_color)
        r1 = cells[1].paragraphs[0].add_run(f"Alert Level: {ra.alert_level.value}")
        r1.bold = True
        r1.font.size = Pt(14)
        r1.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        doc.add_paragraph()

        # LLM generated report body — write each line, detect headings
        for line in report_text.split("\n"):
            line = line.strip()
            if not line:
                doc.add_paragraph()
                continue

            # Detect section headings from LLM output
            if line.startswith("##"):
                doc.add_heading(line.lstrip("#").strip(), level=2)
            elif line.startswith("#"):
                doc.add_heading(line.lstrip("#").strip(), level=1)
            elif line[0].isdigit() and ". " in line[:4]:
                # Numbered heading like "1. Executive Summary"
                doc.add_heading(line, level=2)
            else:
                doc.add_paragraph(line)

        doc.save(output_path)