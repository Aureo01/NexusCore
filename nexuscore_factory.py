import os 
import json
import time 
import asyncio
import sqlite3
import logging
import numpy as np 
import torch
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager
import queue
import threading

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

WORKSPACE = Path("./nexuscore_factory")
WORKSPACE.mkdir(exist_ok=True)
BD_PATH = WORKSPACE / "multimodal_memory.db"
LOG_FILE = WORKSPACE / "nexuscore.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)]
)
logger = logging.getLogger("nexuscore_ia")

# Demo-safe fallback
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-demo")

class TemporalAligner:
    
    def __init__(self, window_ms: float = 500.0, device: str = "cpu"):
        self.window_ms = window_ms
        self.device = device
        self.buffers = {
            "sensors": queue.Queue(maxsize=100),
            "video": queue.Queue(maxsize=30),
            "audio": queue.Queue(maxsize=50),
            "text": queue.Queue(maxsize=50)
        }
        self.last_sync_ts = 0.0
    def push(self, modality: str, data : Any, timestamp: float):
        self.buffers[modality].put({"data": data, "ts":timestamp})

    def get_aligned_window(self):
        """Returns modalities aligned within window_ms"""
        now = time.time()
        aligned = {}
        for mod, q in self.buffers.items():
            items =  []
            while not q.empty():
                item = q.get()
                if abs(item["ts"] - now) * 1000 <= self.window_ms:
                    items.append(item)
            aligned[mod] = items
        self.last_sync_ts = now
        return aligned

aligner = TemporalAligner(window_ms=300.0)    

# Edge-optimized modality processor placeholders
class ModalityProcessors:
    """Real-world replacement points for actual models. Returns structured features."""

    @staticmethod
    def process_sensors(raw: Dict):
        vibration = raw.get("vibration", 0.0)
        temp = raw.get("temperature", 0.0)
        pressure = raw.get("pressure", 0.0)
        anomaly_score = min(1.0, (vibration/4.5)*0.4 + (temp/85)*0.3 + (pressure/8.0)*0.3)
        return{"features": np.array([vibration, temp, pressure]), "anomaly_score": round(anomaly_score, 3), "status": "normal" if anomaly_score < 0.6 else "alert"}
            
    @staticmethod
    def process_video(frame: np.ndarray) -> Dict:
        # Replace with: YOLOv8/RT-DETR for defects + SigLIP for semantic tags
        # Simulated output
        has_leak = np.random.rand() < 0.1
        person_detected = np.random.rand() < 0.05
        return {
            "objects": ["leak"] if has_leak else[],
            "personnel": ["operator"] if person_detected else [],
            "scene_embedding": np.random.randn(512).tolist(), #CLIP/SigLIP embedding
            "confidence": 0.92
        }
    
    @staticmethod
    def process_audio(waveform: np.ndarray):
        # Replace with: Whisper (STT) + YAMNet (sound classification) + VAD
        # Simulated output
        stt_text = "Pressure valve hissing detected" if np.random.rand() < 0.15 else ""
        sound_class = "hiss" if stt_text else "ambient"        
        return {"transcript": stt_text, "sound_class": sound_class, "vad_activate": len(waveform) > 100}

    @staticmethod
    def process_documents(text_chunks: List[str]):
        # Replace with: RAG pipeline + Cross-Encoder reranker
        return {"retrieved_context": text_chunks[:3], "doc_ids": ["SOP-042", "MAN-118", "SAFE-07"]}
        
processors = ModalityProcessors()

class MultimodalFuser:
    """Cross-attention fusion + decision logic for industrial reasoning."""
    def __init__(self):
        # Future upgrade: replace heuristic fusion with a multimodal transformer
        self.fusion_weights = {"sensors": 0.35, "video": 0.30, "audio": 0.15, "docs": 0.20}

    def fuse(self, aligned: Dict[str, List[Dict]]):
        # 1. Extract features per modality

        sensor_feat = aligned["sensors"][-1]["data"]["features"] if aligned["sensors"] else np.zeros(3)
        video_feat = np.array(aligned["video"][-1]["data"]["scene_embedding"]) if aligned["video"] else np.zeros(512)
        audio_txt = " ".join([d["data"]["transcript"] for d in aligned["audio"] if d["data"]["transcript"]])
        doc_ctx = " ".join(aligned["docs"][-1]["data"]["retrieved_context"]) if aligned["docs"] else ""

        # 2. Cross-modal reasoning rules (production: replace with VLM/LLM fusion)
        alerts = []
        confidence = 0.0

        if sensor_feat[2] > 7.5 and "hiss" in audio_txt.lower():
            alerts.append("HIGH PRESSURE LEAK CONFIRMED BY SENSORS + AUDIO")
            confidence += 0.4 
        if any("leak" in v["data"]["objects"] for v in aligned["video"]):
            alerts.append("VISUAL LEAK DETECTED")
            confidence += 0.25
        
        # 3. Decision synthesis
        severity = "CRITICAL" if confidence > 0.8 else "HIGH" if confidence > 0.6 else "MEDIUM" if confidence > 0.3 else "LOW"
        action = "IMMEDIATE_SHUTDOWN" if severity == "CRITICAL" else "INSPECT_AND_LOG" if severity == "HIGH" else "MONITOR"
        return {
         
            "fused_state": {
                "sensor_status": "normal" if sensor_feat.max() < 4.0 else "degraded",
                "visual_scene": "leak_detected" if any("leak" in v["data"]["objects"] for v in aligned["video"]) else "clear",
                "audio_context": audio_txt or "ambient",
                "doc_reference": doc_ctx[:100] + "..." if doc_ctx else "none"
        },
            "alerts": alerts,
            "confidence": round(confidence, 2),
            "severity": severity,
            "recommended_action": action,
            "requires_human": severity in ["CRITICAL", "HIGH"]
        }

fuser = MultimodalFuser()

class FactoryMultimodalState(TypedDict):
    aligned_data: Dict
    fused_reasoning: Dict
    llm_response: str
    tool_actions: List[str]
    safety_gate: bool
    timestamp: float

class NexusCoreOrchestrator:
    """LangGraph brain that routes, fuses, reasons, and acts across modalities."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.graph = self._build_graph()
        
    def _build_graph(self):
        builder = StateGraph(FactoryMultimodalState)
        
        builder.add_node("fuse_modalities", self.fuse_node)
        builder.add_node("llm_reasoning", self.reason_node)
        builder.add_node("execute_actions", self.action_node)
        builder.add_node("safety_check", self.safety_node)
        
        builder.set_entry_point("fuse_modalities")
        builder.add_edge("fuse_modalities", "llm_reasoning")
        builder.add_edge("llm_reasoning", "safety_check")
        builder.add_conditional_edges(
            "safety_check",
            lambda state: "execute_actions" if state["safety_gate"] else END,
            {"execute_actions": "execute_actions", END: END}
        )
        builder.add_edge("execute_actions", END)
        
        return builder.compile(checkpointer=MemorySaver())

    def fuse_node(self, state: FactoryMultimodalState):
        fused = fuser.fuse(state["aligned_data"])
        return {**state, "fused_reasoning": fused, "timestamp": time.time()}
    
    def reason_node(self, state: FactoryMultimodalState) -> FactoryMultimodalState:
        ctx = state["fused_reasoning"]
        prompt = f"""
You are NexusCore, an industrial multimodal AI brain. Synthesize this fused state into a concise operational directive.
Fused State: {json.dumps(ctx['fused_state'], indent=2)}
Alerts: {ctx['alerts']}
Confidence: {ctx['confidence']} | Severity: {ctx['severity']}
Action: {ctx['recommended_action']}

Rules:
1. Be precise. Cite modality sources.
2. If severity >= HIGH, mandate human verification.
3. Output ONLY a 2-3 sentence operational directive.
"""
        response = self.llm.invoke([SystemMessage(content="Industrial AI Operator."), HumanMessage(content=prompt)])
        return {**state, "llm_response": response.content}

    def safety_node(self, state: FactoryMultimodalState):
        # Safety gate: block autonomous critical actions
        gated = state["fused_reasoning"]["severity"] != "CRITICAL" or state["fused_reasoning"]["requires_human"]
        return {**state, "safety_gate": gated}
    
    def action_node(self, state: FactoryMultimodalState):
        actions = []
        severity = state["fused_reasoning"]["severity"]

        if severity in ["HIGH", "CRITICAL"]:
            actions.append("ALERT DISPATCHED TO CONTROL ROOM")
            actions.append("CAMARA ZOOM & RECORDING ENABLED")
            actions.append("PA SYSTEM: 'SELECTION B EVACUATE'")
        if severity == "MEDIUM":
            actions.append("LOGGED TO MES/ERP")
            actions.append("MAINTENANCE TICKET AUTO-CREATED")

        return {**state, "tool_actions": actions}

    async def process_stream_cycle(self, cycle_id: int):
        aligner.push("sensors", processors.process_sensors({
            "vibration": np.random.uniform(1.0, 5.5),
            "temperature": np.random.uniform(60, 92),
            "pressure": np.random.uniform(3.0, 8.8)
        }), time.time())
        await asyncio.sleep(0.2)

        aligned = aligner.get_aligned_window()
        state: FactoryMultimodalState = {
            "aligned_data": aligned,
            "fused_reasoning": {},
            "llm_response": "",
            "tool_actions": [],
            "safety_gate": False,
            "timestamp": time.time()
        }
        config = {"configurable": {"thread_id": str(cycle_id)}}
        result = await self.graph.ainvoke(state, config=config)
        return result    

orchestrator = NexusCoreOrchestrator()
            
async def video_stream():
    while True:
        aligner.push("video", processors.process_video(np.random.rand(224,224,3)), time.time())
        await asyncio.sleep(0.33)  # ~30 FPS

async def audio_stream():
    while True:
        aligner.push("audio", processors.process_audio(np.random.randn(1600)), time.time())
        await asyncio.sleep(0.5)

async def doc_stream():
    while True:
        aligner.push("docs", processors.process_documents(["SOP-042: Pressure valve hissing requires immediate isolation.", "MAN-118: Vibration >4.5 mm/s triggers maintenance ticket."]), time.time())
        await asyncio.sleep(5.0)

# CLI DASHBOARD & CONTROL 
async def run_factory_brain():
    print("""
o=============================================================
     |- - NexusCore v1.0 — Multimodal Factory AI     
  Unified Brain: Sensors + Video + Audio + Docs + LLM - -|     
=============================================================o-

Starting real-time multimodal fusion loop...
Type /exit to shutdown.
    """)                

    # Start streams in background
    asyncio.create_task(orchestrator.process_stream_cycle(0))
    asyncio.create_task(video_stream())
    asyncio.create_task(audio_stream())
    asyncio.create_task(doc_stream())
    
    cycle = 0
    while True:
        try:

            loop = asyncio.get_event_loop()
            cmd = (await loop.run_in_executor(None, input, "\n NexusCore> ")).strip()
            if cmd.lower() == "/exit":
                print("NexusCore shutting down. Streams halted.")
                break
                
            cycle += 1
            result = await orchestrator.process_stream_cycle(cycle)
            
            ctx = result["fused_reasoning"].get("fused_state", {})
            alerts = result["fused_reasoning"].get("alerts", [])
            severity = result["fused_reasoning"].get("severity", "LOW")
            
            print(f"\n Cycle {cycle} | Severity: {severity} | Confidence: {result['fused_reasoning'].get('confidence', 0)}")
            print(f" Visual: {ctx.get('visual_scene', 'N/A')} | Audio: {ctx.get('audio_context', 'N/A')}")
            print(f" Sensors: {ctx.get('sensor_status', 'N/A')} |  Docs: {ctx.get('doc_reference', 'N/A')}")
            if alerts: print(f" Alerts: {'; '.join(alerts)}")
            if result["llm_response"]: print(f" Directive: {result['llm_response']}")
            if result["tool_actions"]: print(f" Actions: {' | '.join(result['tool_actions'])}")
            print("-" * 60)

            await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("\n\n Goodbye, Operator.")
            break
        except Exception as e:
            logger.error(f"Brain Error: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_factory_brain())
