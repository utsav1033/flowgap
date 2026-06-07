"""
FastAPI: serves analysis.json for the frontend.
Run with: uvicorn api.main:app --reload
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).parent.parent
ANALYSIS_PATH = ROOT / "data" / "analysis.json"
EVAL_PATH = ROOT / "data" / "eval_results.json"
INTENDED_FLOW_PATH = ROOT / "gen" / "intended_flow.yaml"

app = FastAPI(title="FlowGap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found — run the pipeline first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_intended_flow() -> dict:
    import yaml
    if not INTENDED_FLOW_PATH.exists():
        return {}
    with open(INTENDED_FLOW_PATH) as f:
        return yaml.safe_load(f)


@app.get("/api/analysis")
def get_analysis():
    """Full analysis: metric, graph, gaps with node specs."""
    return _load_json(ANALYSIS_PATH)


@app.get("/api/metric")
def get_metric():
    """Headline metric only."""
    analysis = _load_json(ANALYSIS_PATH)
    return analysis["metric"]


@app.get("/api/graph")
def get_graph():
    """Transition graph nodes and edges."""
    analysis = _load_json(ANALYSIS_PATH)
    return analysis["graph"]


@app.get("/api/gaps")
def get_gaps():
    """Detected gaps with node specs."""
    analysis = _load_json(ANALYSIS_PATH)
    return {"gaps": analysis["gaps"]}


@app.get("/api/eval")
def get_eval():
    """Precision/recall eval results."""
    return _load_json(EVAL_PATH)


@app.get("/api/intended_flow")
def get_intended_flow():
    """The intended flow YAML as JSON for graph rendering."""
    return _load_intended_flow()


@app.get("/health")
def health():
    return {"status": "ok"}
