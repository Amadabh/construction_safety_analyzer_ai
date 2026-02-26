import streamlit as st
import os
from config import Config
from graph import SafetyGraph
from utils import subscribe_email

st.set_page_config(page_title=Config.PROJECT_NAME, layout="wide")
st.title("🚧 Construction Safety AI System")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.text(f"Project: {Config.PROJECT_NAME}")
st.sidebar.text(f"Model: {Config.BEDROCK_MODEL_ID}")

# ── Alert Subscriptions ───────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("🔔 Alert Subscriptions")
st.sidebar.caption("Subscribe to receive safety alerts by email when a video is analysed.")

with st.sidebar.form(key="subscribe_form"):
    email_input = st.text_input("Email address", placeholder="you@example.com")
    submitted = st.form_submit_button("Subscribe")

if submitted:
    email_input = email_input.strip()
    if not email_input or "@" not in email_input:
        st.sidebar.warning("⚠️ Please enter a valid email address.")
    else:
        with st.sidebar.spinner("Subscribing..."):
            result = subscribe_email(email_input)
        if not result["ok"]:
            st.sidebar.error(f"❌ {result['status']}\n\n`{result.get('error', '')}`")
        elif result["pending"]:
            st.sidebar.info(f"📧 {result['status']}")
        else:
            st.sidebar.success(f"✅ {result['status']}")

@st.cache_resource(show_spinner=False)
def ensure_osha_ingested():
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        if client.collection_exists(Config.QDRANT_COLLECTION):
            count = client.count(Config.QDRANT_COLLECTION).count
            if count > 0:
                return f"✅ OSHA knowledge base ready ({count} chunks)"
        pdf_path = os.path.join(Config.DATA_DIR, "docs", "CAL_OSHA.pdf")
        if not os.path.exists(pdf_path):
            return "⚠️ CAL_OSHA.pdf not found in data/docs/ — skipping ingestion"
        from ingestion import ingest
        ingest(pdf_path)
        count = client.count(Config.QDRANT_COLLECTION).count
        return f"✅ OSHA knowledge base ingested ({count} chunks)"
    except Exception as e:
        return f"⚠️ Qdrant not reachable — RAG disabled ({e})"

with st.spinner("Checking OSHA knowledge base..."):
    qdrant_status = ensure_osha_ingested()
st.sidebar.caption(qdrant_status)

uploaded_file = st.file_uploader("Upload Construction Site Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_path = os.path.join(Config.INPUT_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Video uploaded: {uploaded_file.name}")
    
    # Show video preview
    st.video(video_path)

    if st.button("🔍 Analyze Safety"):
        
        # ── Live Pipeline Progress ────────────────────────────────────────
        st.header("⚙️ Pipeline Execution")

        NODE_LABELS = {
            "process_video":       "🎬 **Agent 1:** Extracting frames from video...",
            "detect_objects":      "👁️ **Agent 2:** Running object detection on frames...",
            "retrieve_regulations":"📚 **Agent 3:** Retrieving relevant OSHA regulations from Qdrant...",
            "assess_risk":         "🧠 **Agent 4:** Assessing risk with AWS Bedrock (Claude)...",
            "generate_report":     "📝 **Agent 5:** Generating DOCX report...",
            "send_alerts":         "🔔 **Agent 6:** Routing alerts via AWS SNS...",
        }

        with st.status("Running Safety Analysis Pipeline...", expanded=True) as status:
            graph = SafetyGraph()
            result = {}

            for node_name, partial_state in graph.stream(video_path):
                st.write(NODE_LABELS.get(node_name, f"⚙️ **{node_name}**"))

                if node_name == "process_video":
                    st.write(f"   ↳ Extracted {len(partial_state.get('frames', []))} frames")
                elif node_name == "detect_objects":
                    st.write(f"   ↳ {len(partial_state.get('detections', []))} detections across all frames")
                elif node_name == "retrieve_regulations":
                    st.write(f"   ↳ Retrieved {len(partial_state.get('regulations', []))} regulations")
                elif node_name == "assess_risk":
                    ra = partial_state.get("risk_assessment")
                    if ra:
                        st.write(f"   ↳ Risk Score: {ra.risk_score} | Level: {ra.alert_level.value}")
                elif node_name == "generate_report":
                    st.write("   ↳ Report ready for download")
                elif node_name == "send_alerts":
                    st.write(f"   ↳ {len(partial_state.get('alerts_sent', []))} alert(s) dispatched")

                result.update(partial_state)

            status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)

        # ── Key Metrics ───────────────────────────────────────────────────
        st.header("📊 Analysis Results")
        
        risk_score = result["risk_assessment"].risk_score
        alert_level = result["risk_assessment"].alert_level
        violations = result["risk_assessment"].violations

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Risk Score", f"{risk_score}/100")
        col2.metric("Alert Level", alert_level.value)
        col3.metric("Violations", len(violations))
        col4.metric("Regulations Found", len(result.get("regulations", [])))

        # Color code alert level
        alert_colors = {
            "LOW": "✅ LOW", 
            "MEDIUM": "🟡 MEDIUM", 
            "HIGH": "🟠 HIGH", 
            "CRITICAL": "🔴 CRITICAL"
        }
        st.subheader(f"Alert Level: {alert_colors.get(alert_level.value, alert_level.value)}")

        # ── Risk Score Visual ─────────────────────────────────────────────
        st.subheader("🎯 Risk Score Breakdown")
        st.progress(risk_score / 100)
        
        # ── Violations ────────────────────────────────────────────────────
        if violations:
            st.subheader("⚠️ Detected Violations")
            for i, v in enumerate(violations):
                with st.expander(f"Violation {i+1}: {v.type} — {v.severity.value}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Confidence", f"{v.confidence:.0%}")
                    col2.metric("Duration", f"{v.duration_seconds}s")
                    col3.metric("Severity", v.severity.value)
                    st.caption(f"⏱️ Detected from {v.timestamp_start}s to {v.timestamp_end}s")
        else:
            st.success("✅ No violations detected!")

        # ── OSHA Regulations ──────────────────────────────────────────────
        if result.get("regulations"):
            st.subheader("📋 Relevant OSHA Regulations")
            for reg in result["regulations"]:
                with st.expander(f"📌 {reg.citation}"):
                    st.markdown(reg.text)
                    st.caption(f"Source: {reg.source}")

        # ── Alerts Sent ───────────────────────────────────────────────────
        if result.get("alerts_sent"):
            st.subheader("🔔 Alerts Dispatched")
            for alert in result["alerts_sent"]:
                st.info(f"✉️ Alert sent via: **{alert}**")

        # ── Report Download ───────────────────────────────────────────────
        if result.get("final_report"):
            st.subheader("📄 Safety Report")
            report_path = result.get("final_report")
            if report_path and os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download DOCX Report",
                        data=f,
                        file_name=os.path.basename(report_path),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

        st.success("🎉 Analysis Complete!")