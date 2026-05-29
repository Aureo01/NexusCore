# NexusCore
Real-time multimodal AI brain for industrial environments. Fuses sensor telemetry, video, audio, and document context into a unified reasoning pipeline using LangGraph, enabling autonomous anomaly detection, severity classification, and operational directives with a built-in safety gate for human-in-the-loop control.

### *Multimodal AI brain for industrial environments*

> Fuses sensor telemetry, video, audio, and documents into real-time decisions — with a built-in safety layer that keeps humans in control.

---

## What is this?

Most industrial monitoring systems look at **one thing at a time** — a sensor spike here, a camera alert there. NexusCore looks at everything **at once**, the same way a human operator would.

It listens to the machines. It watches the cameras. It reads the manuals. And it reasons across all of it simultaneously to decide what's happening and what to do about it.

---

## How it works

```
Sensors ──┐
Video  ──┤──► Temporal Aligner ──► Multimodal Fuser ──► LangGraph Brain ──► Action
Audio  ──┤                                                      │
Docs   ──┘                                               Safety Gate
                                                               │
                                                     Auto-act  OR   Alert human
```

**Step by step:**

1. **Four data streams run in parallel** — sensor readings (vibration, temperature, pressure), video frames, audio waveforms, and operational documents.
2. **The Temporal Aligner** syncs them into a 300ms window so everything is compared at the same moment in time.
3. **The Multimodal Fuser** cross-references the streams and detects anomalies — for example, high pressure *confirmed by* a hissing sound *confirmed by* a visual leak.
4. **LangGraph orchestrates the reasoning** — it fuses the evidence, scores severity, and generates a concise operational directive.
5. **The Safety Gate** decides: can the system act autonomously, or does a human need to verify first?

---

## Severity levels

| Level | Example trigger | Action |
|---|---|---|
| 🟢 LOW | Normal readings | Monitor |
| 🟡 MEDIUM | Mild vibration spike | Log + create maintenance ticket |
| 🟠 HIGH | Sensor + audio anomaly | Alert control room + enable recording |
| 🔴 CRITICAL | Confirmed multi-modal leak | Immediate shutdown + human required |

---

## Tech stack

| Component | Technology |
|---|---|
| AI Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Language Model | GPT-4o-mini via LangChain |
| Sensor processing | NumPy (swap with real edge models) |
| Vision | Placeholder → YOLOv8 / RT-DETR |
| Audio | Placeholder → Whisper + YAMNet |
| RAG / Docs | Placeholder → Cross-Encoder reranker |
| Async runtime | Python asyncio |
| Memory | LangGraph MemorySaver |

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/Aureo01/NexusCore.git
cd nexuscore
```

**2. Install dependencies**
```bash
pip install langgraph langchain langchain-openai numpy torch
```

**3. Set your OpenAI API key**
```bash
export OPENAI_API_KEY=your-key-here
```

**4. Run**
```bash
python nexuscore_factory.py
```

You'll see the CLI dashboard start up. Press Enter to trigger a reasoning cycle, type `/exit` to shut down.

---

## Project structure

```
nexuscore_factory.py     ← Main file (all logic lives here)
nexuscore_factory/
  ├── multimodal_memory.db   ← Auto-created on first run
  └── nexuscore.log          ← Runtime logs
```

---

## Design decisions

**Why one file?**
Intentional — this is a proof-of-concept and portfolio piece. The architecture is modular by design (each class is a clean swap point), but kept consolidated for readability.

**Why placeholder processors?**
The vision, audio, and document processors are clearly marked as swap points. In production you'd drop in YOLOv8 for defect detection, Whisper for speech-to-text, and a RAG pipeline for document retrieval — the interface doesn't change.

**Why a safety gate?**
CRITICAL severity events are never acted on autonomously. The system can log, alert, and record — but shutdowns always require human confirmation. This is intentional and non-negotiable.

---

## Roadmap ideas

- [ ] Replace placeholder processors with real edge models
- [ ] Add SQLite persistence for event history
- [ ] Web dashboard (FastAPI + WebSockets)
- [ ] MQTT integration for real PLC/sensor data
- [ ] Multi-factory support with isolated graph threads

---

## Author

Built by Aureo01 — feel free to reach out or open an issue.

---

*NexusCore is a portfolio project demonstrating multimodal AI orchestration for industrial use cases.*
