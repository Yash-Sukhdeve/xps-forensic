#!/usr/bin/env python3
"""Live monitoring server for XPS-Forensic experiments.

Serves a JSON API + static dashboard. Parses E1 log, SAL training log,
GPU stats, and system metrics in real-time.

Usage:
    python scripts/live_monitor.py              # default port 8060
    python scripts/live_monitor.py --port 8080  # custom port
"""
import json
import os
import re
import subprocess
import time
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
E1_LOG = ROOT / "results" / "e1_run.log"
SAL_LOG = ROOT / "results" / "sal_training.log"
DASHBOARD = ROOT / "results" / "monitor.html"
DATA_FILE = ROOT / "results" / "live_data.json"

def parse_e1_log():
    """Parse E1 experiment log for live progress."""
    data = {"detectors": {}, "current_detector": None, "running": False,
            "errors": [], "completed_detectors": []}
    if not E1_LOG.exists():
        return data

    text = E1_LOG.read_text(errors="replace")
    current_det = None

    for line in text.split("\n"):
        m = re.search(r"Detector: (\w+)", line)
        if m:
            current_det = m.group(1).lower()
            data["current_detector"] = current_det
            if current_det not in data["detectors"]:
                data["detectors"][current_det] = {
                    "status": "running", "progress": 0, "total": 71239,
                    "rate": 0, "errors": 0, "elapsed": 0, "results": {}
                }

        if "SKIP" in line and current_det:
            data["detectors"].setdefault(current_det, {})["status"] = "skipped"

        if "failed to load" in line and current_det:
            data["detectors"].setdefault(current_det, {})["status"] = "failed"
            m2 = re.search(r"failed to load model .* (.*)\. Skipping", line)
            if m2:
                data["errors"].append({"detector": current_det, "error": m2.group(1)})

        m = re.search(r"Processed (\d+)/(\d+) \(([\d.]+)s, ([\d.]+) utt/s, (\d+) errors\)", line)
        if m and current_det:
            d = data["detectors"].setdefault(current_det, {})
            d["progress"] = int(m.group(1))
            d["total"] = int(m.group(2))
            d["elapsed"] = float(m.group(3))
            d["rate"] = float(m.group(4))
            d["errors"] = int(m.group(5))
            d["status"] = "running"
            remaining = (d["total"] - d["progress"]) / max(d["rate"], 0.1)
            d["eta_seconds"] = remaining

        if "Inference complete" in line and current_det:
            d = data["detectors"].setdefault(current_det, {})
            d["status"] = "computing_metrics"
            m2 = re.search(r"(\d+)/(\d+) utterances in ([\d.]+)s \((\d+) errors\)", line)
            if m2:
                d["progress"] = int(m2.group(1))
                d["elapsed"] = float(m2.group(3))
                d["errors"] = int(m2.group(4))

        m = re.search(r"Utt-EER: ([\d.]+)", line)
        if m and current_det:
            d = data["detectors"].setdefault(current_det, {})
            d.setdefault("results", {})["utt_eer"] = float(m.group(1))
            d["status"] = "complete"
            if current_det not in data["completed_detectors"]:
                data["completed_detectors"].append(current_det)

        m = re.search(r"Seg-EER@(\d+)ms: ([\d.]+)", line)
        if m and current_det:
            d = data["detectors"].setdefault(current_det, {})
            d.setdefault("results", {})[f"seg_eer_{m.group(1)}ms"] = float(m.group(2))

        m = re.search(r"Seg-F1@(\d+)ms.*: ([\d.]+)", line)
        if m and current_det:
            d = data["detectors"].setdefault(current_det, {})
            d.setdefault("results", {})[f"seg_f1"] = float(m.group(2))

        m = re.search(r"Frame-level EER: ([\d.]+), threshold: ([\d.]+)", line)
        if m and current_det:
            d = data["detectors"].setdefault(current_det, {})
            d.setdefault("results", {})["frame_eer"] = float(m.group(1))
            d.setdefault("results", {})["frame_thresh"] = float(m.group(2))

        if "Results saved" in line:
            data["e1_complete"] = True

    # Check if process is alive
    try:
        result = subprocess.run(["pgrep", "-f", "run_e1_baseline"], capture_output=True)
        data["running"] = result.returncode == 0
    except Exception:
        pass

    return data


def parse_sal_log():
    """Parse SAL training log."""
    data = {"status": "unknown", "epoch": 0, "max_epochs": 60, "waiting": False}
    if not SAL_LOG.exists():
        data["status"] = "not_started"
        return data

    text = SAL_LOG.read_text(errors="replace")
    if "Waiting for E1" in text:
        data["waiting"] = True
        data["status"] = "waiting_for_e1"
    if "Starting SAL training" in text:
        data["status"] = "training"
        data["waiting"] = False
    if "ModuleNotFoundError" in text or "Error" in text.split("\n")[-1]:
        data["status"] = "error"
        data["error"] = text.split("\n")[-1][:200]
    if "SAL Training Complete" in text:
        data["status"] = "complete"

    # Parse epoch progress from Lightning
    for m in re.finditer(r"Epoch (\d+)", text):
        data["epoch"] = max(data["epoch"], int(m.group(1)))

    return data


def get_gpu_stats():
    """Get GPU utilization and memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "utilization": int(parts[0].strip()),
                "memory_used_mb": int(parts[1].strip()),
                "memory_total_mb": int(parts[2].strip()),
                "temperature_c": int(parts[3].strip()),
                "power_w": float(parts[4].strip()),
            }
    except Exception:
        pass
    return {"utilization": 0, "memory_used_mb": 0, "memory_total_mb": 16376, "temperature_c": 0, "power_w": 0}


def get_system_stats():
    """Get CPU and RAM usage."""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_used_gb": psutil.virtual_memory().used / 1e9,
            "ram_total_gb": psutil.virtual_memory().total / 1e9,
        }
    except ImportError:
        return {"cpu_percent": 0, "ram_used_gb": 0, "ram_total_gb": 64}


def get_plan_status():
    """Get project plan progress."""
    return {
        "phases": [
            {"name": "Literature Review", "status": "complete", "pct": 100, "dates": "Jan-Feb 2026"},
            {"name": "XPS Core Implementation", "status": "complete", "pct": 100, "dates": "Feb-Mar 2026",
             "detail": "4,516 LOC core + 1,714 LOC tests = 136/136 pass"},
            {"name": "BAM Baseline (0.16s)", "status": "complete", "pct": 100, "dates": "Mar 9-11",
             "detail": "EER 8.33% (published 8.43%) — PASSED"},
            {"name": "FARA Exploratory", "status": "archived", "pct": 80, "dates": "Mar 24-31",
             "detail": "Best EER 9.19% at epoch 7. Paused/archived."},
            {"name": "Datasets Acquired", "status": "complete", "pct": 100, "dates": "Apr 1",
             "detail": "PartialSpoof (121K), LlamaPS (140K), PartialEdit (176K), HQ-MPSD (needs extract)"},
            {"name": "E1 Baseline (3 detectors)", "status": "running", "pct": 40, "dates": "Apr 2",
             "detail": "BAM + CFPRF + MRM on PartialSpoof eval"},
            {"name": "E2 Calibration", "status": "pending", "pct": 0, "detail": "Code ready"},
            {"name": "E3 CPSL Coverage", "status": "pending", "pct": 0, "detail": "Code ready"},
            {"name": "E4 PDSM-PS", "status": "blocked", "pct": 0, "detail": "Bug #5 partially fixed"},
            {"name": "E5 Cross-Dataset", "status": "pending", "pct": 0, "detail": "3/4 loaders ready"},
            {"name": "E6-E8 Extended", "status": "scaffold", "pct": 0, "detail": "Scaffolds exist"},
            {"name": "Manuscript", "status": "pending", "pct": 0, "dates": "May 1-21"},
            {"name": "Submission", "status": "pending", "pct": 0, "dates": "Jun 15"},
        ],
        "gates": {
            "G1": {"name": "Survey submitted", "status": "pending", "owner": "YS"},
            "G2": {"name": "Data acquired", "status": "pass", "detail": "4/4 datasets"},
            "G3": {"name": "Baselines reproduced", "status": "running", "detail": "E1 in progress"},
            "G4": {"name": "Calibration validated", "status": "pending"},
            "G5": {"name": "CPSL verified", "status": "pending"},
            "G6": {"name": "PDSM-PS integrated", "status": "blocked", "detail": "Bug #5"},
            "G7": {"name": "Manuscript draft", "status": "pending"},
            "G8": {"name": "CITeR report", "status": "pending"},
        },
        "bugs": {
            "open": [
                {"id": 5, "sev": "high", "desc": "PDSM-PS Captum integration (perturbation.py added)"},
                {"id": 6, "sev": "med", "desc": "E6/E7/E8 scaffolds need implementation"},
                {"id": 7, "sev": "med", "desc": "Evidence schema disclosure"},
            ],
            "fixed": [
                {"id": "BAM", "desc": "Score inversion (probs[:,0] not [:,1])"},
                {"id": "CFPRF", "desc": "weights_only=False + path.resolve()"},
                {"id": "E1-A2", "desc": "Pooled segment EER (was per-utt averaged)"},
                {"id": "E1-B2", "desc": "NaN score guard"},
                {"id": "PE", "desc": "PartialEdit CSV loader"},
                {"id": 1, "desc": "Calibration ranking orientation"},
                {"id": 2, "desc": "Frame resolution mismatch"},
                {"id": 3, "desc": "Stage-1 APS naming"},
                {"id": 4, "desc": "Composed guarantee (Bonferroni)"},
            ],
        },
        "codebase": {"loc": 8038, "tests": 136, "files": 35},
        "detectors": {
            "bam": {"status": "ready", "backbone": "WavLM-Large", "res": "160ms"},
            "sal": {"status": "training", "backbone": "WavLM/XLSR", "res": "160ms"},
            "cfprf": {"status": "ready", "backbone": "XLSR-300M", "res": "20ms"},
            "mrm": {"status": "ready", "backbone": "wav2vec 2.0", "res": "20ms"},
        },
        "contributions": {
            "cpsl": {"status": "complete", "loc": 686, "tests": 13},
            "calibration": {"status": "complete", "loc": 239, "tests": 12},
            "pdsm_ps": {"status": "partial", "loc": 645, "tests": 12, "bug": 5},
        },
    }


def get_ruflow_status():
    """Get ruflow/claude-flow system status via CLI."""
    data = {
        "agents": [], "memory": {}, "hooks": [],
        "intelligence": {}, "swarm": {}, "available": False,
    }
    try:
        # Try to read from the MCP tools output if available
        result = subprocess.run(
            ["npx", "@claude-flow/cli@latest", "system", "status"],
            capture_output=True, text=True, timeout=5, cwd=str(ROOT.parent)
        )
        if result.returncode == 0:
            data["available"] = True
            data["raw_status"] = result.stdout[:500]
    except Exception:
        pass

    # Static info about what's configured
    data["configured"] = {
        "hooks_active": 26,
        "hook_types": ["PreToolUse", "PostToolUse", "SessionStart", "Intelligence", "Analytics"],
        "moe_experts": ["coder", "tester", "reviewer", "architect", "security", "performance", "researcher", "coordinator"],
        "intelligence_modules": ["SONA", "EWC++", "MoE Router", "Flash Attention", "LoRA", "HNSW"],
        "memory_backend": "sql.js + HNSW",
        "topology": "mesh",
        "max_agents": 10,
    }
    data["session_agents"] = [
        {"name": "detector-explainer", "type": "model-deep-dive", "status": "completed"},
        {"name": "cpsl-explainer", "type": "senior-research-scientist", "status": "completed"},
        {"name": "experiment-explainer", "type": "senior-research-scientist", "status": "completed"},
        {"name": "bam-debugger", "type": "coder", "status": "completed", "finding": "Score inversion fixed"},
        {"name": "cfprf-debugger", "type": "coder", "status": "completed", "finding": "weights_only=False"},
        {"name": "mrm-verifier", "type": "senior-research-scientist", "status": "completed", "finding": "Results verified"},
        {"name": "e1-reviewer", "type": "reviewer", "status": "completed", "finding": "3 HIGH issues fixed"},
        {"name": "dataloader-validator", "type": "tester", "status": "completed"},
        {"name": "science-validator", "type": "researcher", "status": "completed", "finding": "10 eqs verified, 1 citation removed"},
        {"name": "nan-debugger", "type": "researcher", "status": "completed", "finding": "torch.cdist float16 overflow"},
        {"name": "sparc-auditor", "type": "researcher", "status": "completed", "finding": "25% plan alignment"},
    ]
    return data


def build_live_data():
    """Assemble all live monitoring data."""
    return {
        "timestamp": datetime.now().isoformat(),
        "e1": parse_e1_log(),
        "sal": parse_sal_log(),
        "gpu": get_gpu_stats(),
        "system": get_system_stats(),
        "tests": {"total": 136, "passing": 136},
        "git_branch": "feat/xps-forensic-implementation",
        "git_commit": "9ca6948",
        "plan": get_plan_status(),
        "ruflow": get_ruflow_status(),
    }


def update_loop():
    """Background thread: update live_data.json every 5 seconds."""
    while True:
        try:
            data = build_live_data()
            DATA_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[monitor] Error: {e}")
        time.sleep(5)


class MonitorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "results"), **kwargs)

    def do_GET(self):
        if self.path == "/api/live":
            data = build_live_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/monitor.html"
            super().do_GET()
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress request logs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8060)
    args = parser.parse_args()

    # Start background updater
    updater = Thread(target=update_loop, daemon=True)
    updater.start()

    # Initial data write
    DATA_FILE.write_text(json.dumps(build_live_data(), indent=2))

    server = HTTPServer(("0.0.0.0", args.port), MonitorHandler)
    print(f"\n  XPS-Forensic Live Monitor")
    print(f"  http://localhost:{args.port}")
    print(f"  API: http://localhost:{args.port}/api/live\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
