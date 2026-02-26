from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from schemas import ProcessingResult, Frame, Detection, RiskAssessment, Regulation, AlertLevel
from agents.video import VideoProcessor
from agents.vision import VisionDetector
from agents.risk import RiskAssessor
from agents.rag import RAGRetriever
from agents.alert import run_alert_agent
from agents.report import ReportGenerator


class GraphState(TypedDict):
    video_path: str
    frames: Annotated[list[Frame], operator.add]
    detections: Annotated[list[Detection], operator.add]
    regulations: Annotated[list[Regulation], operator.add]
    risk_assessment: RiskAssessment
    final_report: str
    alerts_sent: Annotated[list[str], operator.add]


class SafetyGraph:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.vision_detector = VisionDetector()
        self.risk_assessor   = RiskAssessor()
        self.rag_retriever   = RAGRetriever()
        self.report_generator = ReportGenerator()
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("process_video",        self.process_video)
        workflow.add_node("detect_objects",        self.detect_objects)
        workflow.add_node("retrieve_regulations",  self.retrieve_regulations)
        workflow.add_node("assess_risk",           self.assess_risk)
        workflow.add_node("generate_report",       self.generate_report)
        workflow.add_node("send_alerts",           self.send_alerts)

        workflow.set_entry_point("process_video")
        workflow.add_edge("process_video",       "detect_objects")
        workflow.add_edge("detect_objects",      "retrieve_regulations")
        workflow.add_edge("retrieve_regulations","assess_risk")
        workflow.add_edge("assess_risk",         "generate_report")
        workflow.add_edge("generate_report",     "send_alerts")
        workflow.add_edge("send_alerts",         END)

        return workflow.compile()

    def process_video(self, state: GraphState):
        frames = self.video_processor.process(state["video_path"])
        print(f"[process_video] Extracted {len(frames)} frames")
        return {"frames": frames}

    def detect_objects(self, state: GraphState):
        detections = self.vision_detector.detect(state["frames"])
        print(f"[detect_objects] {len(detections)} total detections")
        return {"detections": detections}

    def retrieve_regulations(self, state: GraphState):
        regs = self.rag_retriever.retrieve_regulations(state["detections"])
        print(f"[retrieve_regulations] {len(regs)} regulations retrieved")
        return {"regulations": regs}

    def assess_risk(self, state: GraphState):
        risk = self.risk_assessor.assess(
            state["detections"],
            regulations=state.get("regulations", [])
        )
        print(f"[assess_risk] Score: {risk.risk_score} | Level: {risk.alert_level}")
        return {"risk_assessment": risk}

    def generate_report(self, state: GraphState):
        result = ProcessingResult(
            video_id=state["video_path"],
            risk_assessment=state["risk_assessment"],
            regulations=state.get("regulations", []),
            report_path=None
        )
        # Pass raw detections so report has full per-label counts + confidence
        report = self.report_generator.generate_report(
            result,
            detections=state.get("detections", [])   # <-- key addition
        )
        return {"final_report": report}

    def send_alerts(self, state: GraphState):
        alerts_sent = run_alert_agent(
            state["risk_assessment"],
            report_path=state.get("final_report")
        )
        return {"alerts_sent": alerts_sent}

    def run(self, video_path: str):
        initial_state = GraphState(
            video_path=video_path,
            frames=[],
            detections=[],
            regulations=[],
            risk_assessment=RiskAssessment(
                risk_score=0,
                alert_level=AlertLevel.LOW,
                violations=[]
            ),
            final_report="",
            alerts_sent=[]
        )
        return self.workflow.invoke(initial_state)
