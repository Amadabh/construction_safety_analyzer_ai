import streamlit as st
import os
from config import Config
from graph import SafetyGraph

st.set_page_config(page_title=Config.PROJECT_NAME, layout="wide")

st.title("🚧 Construction Safety AI System")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.text(f"Project: {Config.PROJECT_NAME}")
st.sidebar.text(f"Model: {Config.BEDROCK_MODEL_ID}")

# Main content
uploaded_file = st.file_uploader("Upload Construction Site Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file
    video_path = os.path.join(Config.INPUT_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Video uploaded: {uploaded_file.name}")
    
    if st.button("Analyze Safety"):
        with st.spinner("Initializing Agents..."):
            graph = SafetyGraph()
        
        with st.spinner("Processing Video..."):
            # Run the graph
            result = graph.run(video_path)
            
            # Display Results
            st.header("Analysis Results")
            
            # Key Metrics
            col1, col2, col3 = st.columns(3)
            risk_score = result["risk_assessment"].risk_score
            alert_level = result["risk_assessment"].alert_level
            
            col1.metric("Risk Score", f"{risk_score}/100")
            col2.metric("Alert Level", alert_level.value)
            col3.metric("Violations Detected", len(result["risk_assessment"].violations))
            
            # Detailed Violations
            if result["risk_assessment"].violations:
                st.subheader("Detected Violations")
                for v in result["risk_assessment"].violations:
                    st.error(f"{v.type} ({v.severity.value}) - {v.duration_seconds}s")
            
            # Regulations
            if result["regulations"]:
                st.subheader("Relevant OSHA Regulations")
                for reg in result["regulations"]:
                    with st.expander(f"{reg.citation}"):
                        st.markdown(reg.text)
                        
            st.success("Analysis Complete!")
