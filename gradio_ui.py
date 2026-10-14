import os
import sys
import time
import csv
import json
import base64
import datetime
import tempfile
import logging
import collections
import asyncio
import cv2
import requests
import zipfile
import numpy as np
import pandas as pd
import gradio as gr
from typing import Dict, Any, List, Optional, Union, Tuple

# Silence WinError 10054 ConnectionResetError noise on Windows asyncio event loops
if sys.platform == "win32":
    try:
        import asyncio.proactor_events as _asyncio_pe
        _orig_call_conn_lost = _asyncio_pe._ProactorBasePipeTransport._call_connection_lost
        def _safe_call_connection_lost(self, exc):
            try:
                _orig_call_conn_lost(self, exc)
            except ConnectionResetError:
                pass
            except OSError as e:
                if getattr(e, "winerror", None) == 10054:
                    pass
                else:
                    raise
        _asyncio_pe._ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost
    except Exception:
        pass

# ── Windows Gradio Upload File-Locking Self-Healing Patch ──
# Prevents PermissionError: [Errno 13] when uploading or re-uploading videos on Windows
try:
    import gradio.route_utils as _gr_route_utils
    import gradio.routes as _gr_routes
    import shutil as _shutil

    def _safe_move_uploaded_files_to_cache(files, destinations):
        for file, dest in zip(files, destinations, strict=False):
            try:
                if not os.path.exists(file):
                    continue
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    try:
                        os.unlink(file)
                    except OSError:
                        pass
                    continue
                try:
                    os.replace(file, dest)
                    continue
                except OSError:
                    pass
                _shutil.move(file, dest)
            except PermissionError:
                if os.path.exists(dest):
                    try:
                        if os.path.exists(file):
                            os.unlink(file)
                    except OSError:
                        pass
                else:
                    raise
            except Exception:
                pass

    _orig_upload_fn = _gr_route_utils.upload_fn

    async def _safe_upload_fn(request, upload_dir, max_file_size, upload_id, force_move=True, upload_progress=None):
        out_files, to_copy, locs = await _orig_upload_fn(
            request, upload_dir, max_file_size, upload_id, force_move=force_move, upload_progress=upload_progress
        )
        filtered_to_copy = []
        filtered_locs = []
        for src_f, dst_f in zip(to_copy, locs):
            if os.path.exists(dst_f) and os.path.getsize(dst_f) > 0:
                try:
                    if os.path.exists(src_f):
                        os.unlink(src_f)
                except OSError:
                    pass
            else:
                filtered_to_copy.append(src_f)
                filtered_locs.append(dst_f)
        return out_files, filtered_to_copy, filtered_locs

    _gr_route_utils.move_uploaded_files_to_cache = _safe_move_uploaded_files_to_cache
    _gr_routes.move_uploaded_files_to_cache = _safe_move_uploaded_files_to_cache
    _gr_route_utils.upload_fn = _safe_upload_fn
    _gr_routes.upload_fn = _safe_upload_fn
except Exception as _patch_err:
    print(f"[Warning] Could not initialize Windows Gradio upload patch: {_patch_err}")


# Try importing psutil for CPU & RAM system hardware monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Try importing torch for CUDA GPU hardware monitoring
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add project root to path for rdd_dedup imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import the dedup pipeline
try:
    from rdd_dedup.pipeline import RDDPipeline
    from rdd_dedup.config import PipelineConfig
    from rdd_dedup.tracking.defect_track import FinalizedDefect, PipelineResult
    HAS_DEDUP_PIPELINE = True
except ImportError as e:
    print(f"[Warning] rdd_dedup pipeline not available: {e}")
    HAS_DEDUP_PIPELINE = False

# Import the visualizer utility
try:
    from rdd_dedup.utils.visualizer import PipelineVisualizer, generate_visualizations_from_directory
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

# Import the dataset collection system
try:
    from rdd_dedup.dataset import (
        ClassConfig, DatasetManager, AnnotationManager, CropSelector, CandidateAnnotation
    )
    HAS_DATASET_SYSTEM = True
except ImportError as e:
    print(f"[Warning] rdd_dedup dataset system not available: {e}")
    HAS_DATASET_SYSTEM = False

# Import the GIS enrichment backend
try:
    from backend.geo_service import enrich_inspection_with_gis
    HAS_GIS_BACKEND = True
except ImportError as e:
    print(f"[Warning] GIS backend not available: {e}")
    HAS_GIS_BACKEND = False

# Try importing ultralytics for YOLO support, with fallback handling
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

# Try importing imageio for HTML5 H.264 video encoding compatibility
try:
    import imageio
    import imageio_ffmpeg
    HAS_IMAGEIO_FFMPEG = True
except ImportError:
    HAS_IMAGEIO_FFMPEG = False


# Default RDD Class mapping and color palette (BGR format for OpenCV)
DEFAULT_RDD_CLASSES = {
    0: {"name": "D00: Longitudinal Crack", "color": (255, 165, 0)},      # Orange
    1: {"name": "D10: Transverse Crack", "color": (255, 255, 0)},        # Yellow
    2: {"name": "D20: Alligator Crack", "color": (255, 0, 255)},         # Magenta
    3: {"name": "D40: Pothole", "color": (0, 0, 255)},                   # Red
    4: {"name": "Repair / Patch", "color": (0, 255, 0)},                 # Green
}

# Color palette array for dynamic class assignment
CLASS_COLOR_PALETTE = [
    (0, 0, 255),      # Red
    (255, 165, 0),    # Orange
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 0),      # Green
    (255, 192, 203),  # Pink
    (0, 255, 255),    # Cyan
    (128, 0, 128),    # Purple
]

# Default directories (BASE_DIR already set above for imports)
MODELS_DIR = os.path.join(BASE_DIR, "models")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "RDD_Ahmedabad")
CLASS_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "rdd_ahmedabad_classes.yaml")

# ═══════════════════════════════════════════════════════════════════════════════
# RDD_Ahmedabad Dataset Collection — Global State
# ═══════════════════════════════════════════════════════════════════════════════

# Load class config and initialize dataset managers
RDD_CLASS_CONFIG = None
RDD_DATASET_MANAGER = None
RDD_ANNOTATION_MANAGER = None
RDD_CROP_SELECTOR = None

if HAS_DATASET_SYSTEM and os.path.exists(CLASS_CONFIG_PATH):
    try:
        RDD_CLASS_CONFIG = ClassConfig.load(CLASS_CONFIG_PATH)
        RDD_DATASET_MANAGER = DatasetManager(DATASET_DIR, RDD_CLASS_CONFIG)
        RDD_DATASET_MANAGER.ensure_structure()
        RDD_ANNOTATION_MANAGER = AnnotationManager(os.path.join(DATASET_DIR, "metadata"))
        RDD_CROP_SELECTOR = CropSelector(strategy="highest_confidence")
        print(f"[RDD Dataset] Loaded {RDD_CLASS_CONFIG.get_num_classes()} classes: {RDD_CLASS_CONFIG.get_class_names()}")
    except Exception as e:
        print(f"[Warning] Failed to initialize dataset system: {e}")
        HAS_DATASET_SYSTEM = False

# Global state for annotation review session
_ANNOTATION_STATE = {
    "candidates": [],           # List[CandidateAnnotation]
    "current_index": 0,
    "staged_defects": [],       # List[dict]: staged annotations before permanent commit
    "next_staged_id": 1,        # Monotonic ID counter for staged defects
    "editing_staged_id": None,  # ID of staged defect currently being edited
    "accepted_count": 0,
    "skipped_count": 0,
    "changed_count": 0,
    "last_pipeline_result": None,
    "last_video_path": None,
    "last_inspection_mode": "dedup",
}

# Global state for GIS enrichment session
_GIS_STATE = {
    "last_video_path": None,
    "last_output_dir": None,
    "last_summary_path": None,
    "last_pipeline_result": None,
}

# Global flag to signal cancellation / revocation of ongoing video processing
_CANCEL_FLAG = {"requested": False}


def revoke_processing():
    """Sets the revocation request flag to cancel active backend video processing."""
    _CANCEL_FLAG["requested"] = True
    return "⚠️ **Process Revocation Requested**: Terminating active backend video processing task..."



# ═══════════════════════════════════════════════════════════════════════════════
# Hardware & System Performance Monitor
# ═══════════════════════════════════════════════════════════════════════════════

class SystemMonitor:
    """
    Collects real-time CPU, RAM, and GPU hardware statistics.
    Detects Intel(R) Iris(R) Xe Graphics and CUDA GPUs.
    Maintains a sliding history buffer for performance spike analysis.
    """
    def __init__(self, max_history=60):
        self.max_history = max_history
        self.history = collections.deque(maxlen=max_history)
        self.start_time = time.time()
        self._cached_gpu_info = None
        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None)

    def _detect_gpu(self):
        if self._cached_gpu_info is not None:
            return self._cached_gpu_info

        # 1. Check CUDA GPU
        if HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    self._cached_gpu_info = (gpu_name, True, "CUDA Active")
                    return self._cached_gpu_info
            except Exception:
                pass

        # 2. Check Windows Video Controller (e.g. Intel Iris Xe Graphics)
        try:
            import subprocess
            out = subprocess.check_output(
                ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                text=True, timeout=2
            ).strip()
            if out:
                first_gpu = out.split("\n")[0].strip()
                self._cached_gpu_info = (first_gpu, False, "Integrated Shared Memory")
                return self._cached_gpu_info
        except Exception:
            pass

        self._cached_gpu_info = ("Intel(R) Iris(R) Xe Graphics", False, "Shared Memory")
        return self._cached_gpu_info

    def get_gpu_stats(self):
        gpu_name, is_cuda, gpu_type = self._detect_gpu()
        gpu_info = {
            "available": is_cuda,
            "name": gpu_name,
            "vram_used_mb": 0.0,
            "vram_total_mb": 0.0,
            "vram_percent": 0.0,
            "util_percent": 0.0,
            "details": f"{gpu_name} ({gpu_type})"
        }
        if is_cuda and HAS_TORCH:
            try:
                allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
                total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                gpu_info["vram_used_mb"] = round(reserved_mb, 1)
                gpu_info["vram_total_mb"] = round(total_mb, 1)
                gpu_info["vram_percent"] = round((reserved_mb / max(1.0, total_mb)) * 100, 1)
                gpu_info["util_percent"] = round((allocated_mb / max(1.0, total_mb)) * 100, 1)
                gpu_info["details"] = f"{gpu_name} | VRAM: {gpu_info['vram_used_mb']}MB / {gpu_info['vram_total_mb']}MB ({gpu_info['vram_percent']}%)"
            except Exception:
                pass
        return gpu_info

    def get_cpu_stats(self):
        if not HAS_PSUTIL:
            return {"percent": 0.0, "cores": 1, "threads": 1, "freq_ghz": 0.0}

        cpu_pct = psutil.cpu_percent(interval=None)
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_threads = psutil.cpu_count(logical=True) or 1
        
        freq_ghz = 0.0
        try:
            freq_info = psutil.cpu_freq()
            if freq_info and freq_info.current:
                freq_ghz = round(freq_info.current / 1000.0, 2)
        except Exception:
            pass

        return {
            "percent": cpu_pct,
            "cores": physical_cores,
            "threads": logical_threads,
            "freq_ghz": freq_ghz,
        }

    def get_ram_stats(self):
        if not HAS_PSUTIL:
            return {"used_gb": 0.0, "total_gb": 0.0, "available_gb": 0.0, "percent": 0.0}

        mem = psutil.virtual_memory()
        return {
            "used_gb": round(mem.used / (1024 ** 3), 2),
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "percent": mem.percent,
        }

    def sample(self):
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        elapsed_sec = int(time.time() - self.start_time)

        cpu = self.get_cpu_stats()
        ram = self.get_ram_stats()
        gpu = self.get_gpu_stats()

        sample_dict = {
            "Time": now_str,
            "Elapsed_s": elapsed_sec,
            "CPU_Percent": cpu["percent"],
            "RAM_Percent": ram["percent"],
            "RAM_Used_GB": ram["used_gb"],
            "RAM_Total_GB": ram["total_gb"],
            "GPU_VRAM_Percent": gpu["vram_percent"],
            "GPU_VRAM_Used_MB": gpu["vram_used_mb"],
            "GPU_Name": gpu["name"],
        }
        self.history.append(sample_dict)
        return sample_dict, cpu, ram, gpu

    def get_melted_df(self):
        if not self.history:
            self.sample()
        
        records = []
        for s in self.history:
            records.append({"Time": s["Time"], "Metric": "CPU Utilization (%)", "Value": s["CPU_Percent"]})
            records.append({"Time": s["Time"], "Metric": "RAM Usage (%)", "Value": s["RAM_Percent"]})
            if s["GPU_VRAM_Percent"] > 0:
                records.append({"Time": s["Time"], "Metric": "GPU VRAM (%)", "Value": s["GPU_VRAM_Percent"]})
        
        return pd.DataFrame(records)

    def get_summary_df(self):
        if not self.history:
            self.sample()
        return pd.DataFrame(list(self.history))


# Instantiate global hardware monitor
SYSTEM_MONITOR = SystemMonitor(max_history=60)


def update_system_dashboard():
    """Gradio handler function to update compact telemetry status bar."""
    sample, cpu, ram, gpu = SYSTEM_MONITOR.sample()
    df_melted = SYSTEM_MONITOR.get_melted_df()
    df_raw = SYSTEM_MONITOR.get_summary_df()
    
    elapsed_total = int(time.time() - SYSTEM_MONITOR.start_time)
    uptime_min = elapsed_total // 60
    uptime_sec = elapsed_total % 60
    
    cpu_color = "#ef4444" if cpu["percent"] > 85 else "#f97316" if cpu["percent"] > 60 else "#10b981"
    ram_color = "#ef4444" if ram["percent"] > 85 else "#f97316" if ram["percent"] > 60 else "#10b981"
    
    cards_html = f"""
    <div class="sys-bar">
        <div class="sys-metric">
            <span class="label">⚡ CPU</span>
            <span class="value" style="color: {cpu_color};">{cpu['percent']}%</span>
            <div class="progress-bar"><div class="progress-fill" style="width: {min(100, cpu['percent'])}%; background: {cpu_color};"></div></div>
            <span class="detail">{cpu['cores']}C / {cpu['threads']}T</span>
        </div>
        <div class="sys-metric">
            <span class="label">🧠 RAM</span>
            <span class="value" style="color: {ram_color};">{ram['percent']}%</span>
            <div class="progress-bar"><div class="progress-fill" style="width: {min(100, ram['percent'])}%; background: {ram_color};"></div></div>
            <span class="detail">{ram['used_gb']} / {ram['total_gb']} GB</span>
        </div>
        <div class="sys-metric">
            <span class="label">🎮 GPU</span>
            <span class="value" style="color: #38bdf8; font-size: 0.8rem;">{gpu['name']}</span>
        </div>
        <div class="sys-metric">
            <span class="detail">⏱️ {uptime_min}m {uptime_sec}s</span>
        </div>
    </div>
    """
    
    return cards_html, df_melted, df_raw


def get_available_models(use_openvino=False):
    """Scans the models directory for available models.
    If use_openvino is True, scans for OpenVINO model directories (containing .xml).
    If use_openvino is False, scans for PyTorch (.pt) weights."""
    model_entries = []
    
    if os.path.exists(MODELS_DIR):
        if use_openvino:
            for root, dirs, files in os.walk(MODELS_DIR):
                if any(f.endswith(".xml") for f in files) or os.path.basename(root).endswith("_openvino_model"):
                    rel_path = os.path.relpath(root, MODELS_DIR).replace("\\", "/")
                    if rel_path != "." and rel_path not in model_entries:
                        model_entries.append(rel_path)
                    dirs.clear()
            if not model_entries:
                model_entries = ["No OpenVINO models found"]
        else:
            for root, _, files in os.walk(MODELS_DIR):
                for file in files:
                    if file.endswith(".pt"):
                        rel_path = os.path.relpath(os.path.join(root, file), MODELS_DIR)
                        rel_path = rel_path.replace("\\", "/")
                        model_entries.append(rel_path)
                        
            if not model_entries:
                model_entries = ["rdd_yolo.pt (Default)", "yolov8n.pt (Baseline)", "Demo / Simulation Mode"]
            else:
                model_entries.append("yolov8n.pt (Baseline)")
                model_entries.append("Demo / Simulation Mode")

    return model_entries


def get_example_videos():
    """Finds available sample videos in the inputs directory."""
    examples = []
    if os.path.exists(INPUTS_DIR):
        for file in sorted(os.listdir(INPUTS_DIR)):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                examples.append(os.path.join(INPUTS_DIR, file))
    return examples


def get_example_gpx_files():
    """Finds available sample GPX files in inputs/ and subdirectories."""
    gpx_files = []
    if os.path.exists(INPUTS_DIR):
        for root, _, files in os.walk(INPUTS_DIR):
            for file in sorted(files):
                if file.lower().endswith(".gpx"):
                    gpx_files.append(os.path.join(root, file))
    return gpx_files


MAX_CLASS_SLIDERS = 16
_MODEL_CLASSES_CACHE: Dict[str, Dict[int, str]] = {}


def get_model_class_dict(model_name: str) -> Dict[int, str]:
    """
    Extracts class names for any model (.pt, OpenVINO folder, or Demo).
    Uses caching and lightweight metadata inspection for instant UI updates.
    """
    if not model_name:
        return {cid: info["name"] for cid, info in DEFAULT_RDD_CLASSES.items()}

    clean_name = model_name.split(" ")[0].strip()
    if clean_name in _MODEL_CLASSES_CACHE:
        return _MODEL_CLASSES_CACHE[clean_name]

    if "Demo" in model_name:
        return {cid: info["name"] for cid, info in DEFAULT_RDD_CLASSES.items()}

    # 1. Check OpenVINO folder
    full_path = os.path.join(MODELS_DIR, clean_name)
    if os.path.isdir(full_path):
        meta_yaml = os.path.join(full_path, "metadata.yaml")
        if os.path.exists(meta_yaml):
            try:
                import yaml
                with open(meta_yaml, "r", encoding="utf-8") as f:
                    ydata = yaml.safe_load(f)
                    if isinstance(ydata, dict) and "names" in ydata and isinstance(ydata["names"], dict):
                        res = {int(k): str(v).title() for k, v in ydata["names"].items()}
                        _MODEL_CLASSES_CACHE[clean_name] = res
                        return res
            except Exception as e:
                print(f"[Warning] Failed to read OpenVINO metadata.yaml: {e}")

    # 2. Check PyTorch .pt weights file (fast checkpoint inspect without loading full model)
    pt_path = full_path if os.path.isfile(full_path) else os.path.join(MODELS_DIR, clean_name)
    if not os.path.isfile(pt_path) and os.path.isfile(clean_name):
        pt_path = clean_name

    if os.path.isfile(pt_path) and pt_path.endswith(".pt"):
        try:
            import torch
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            m = ckpt.get("model")
            names = getattr(m, "names", None)
            if isinstance(names, dict) and names:
                res = {int(k): str(v).title() for k, v in names.items()}
                _MODEL_CLASSES_CACHE[clean_name] = res
                return res
            elif isinstance(names, list) and names:
                res = {i: str(v).title() for i, v in enumerate(names)}
                _MODEL_CLASSES_CACHE[clean_name] = res
                return res
        except Exception:
            pass

    # 3. Fallback: load via load_yolo_model
    try:
        _, cmap = load_yolo_model(clean_name)
        if cmap:
            res = {cid: data["name"] for cid, data in cmap.items()}
            _MODEL_CLASSES_CACHE[clean_name] = res
            return res
    except Exception:
        pass

    # 4. Ultimate fallback
    return {cid: info["name"] for cid, info in DEFAULT_RDD_CLASSES.items()}


def on_model_selection_change(model_name, master_conf=0.25):
    """
    Updates the per-class confidence threshold sliders when a new model is selected.
    Dynamically shows sliders for the exact classes present in the model and hides unused sliders.
    """
    classes_dict = get_model_class_dict(model_name)
    class_items = list(classes_dict.items())
    num_classes = len(class_items)
    val = float(master_conf) if master_conf else 0.25

    updates = []
    for i in range(MAX_CLASS_SLIDERS):
        if i < num_classes:
            cid, cname = class_items[i]
            updates.append(gr.update(label=f"[{cid}] {cname}", value=val, visible=True))
        else:
            updates.append(gr.update(visible=False))

    clean_name = os.path.basename(model_name) if model_name else "Selected Model"
    info_text = f"🎯 **{num_classes} classes detected** for `{clean_name}`. Set independent confidence thresholds per class below:"
    return [info_text] + updates


def sync_all_class_sliders(model_name, target_conf=0.25):
    """Syncs all active class confidence sliders to a given target confidence value."""
    classes_dict = get_model_class_dict(model_name)
    num_classes = len(classes_dict)
    val = float(target_conf) if target_conf is not None else 0.25

    updates = []
    for i in range(MAX_CLASS_SLIDERS):
        if i < num_classes:
            updates.append(gr.update(value=val))
        else:
            updates.append(gr.update())
    return updates


def on_generate_geo(gpx_upload, gpx_dropdown, offset_sec, custom_output_dir=None):
    """Event handler for generating GIS / GeoJSON enrichment."""
    if not HAS_GIS_BACKEND:
        return (
            "❌ **GIS Module Unavailable**: `backend.geo_service` could not be loaded.",
            "",
            gr.update(visible=False)
        )

    gpx_path = None
    if gpx_upload is not None:
        if isinstance(gpx_upload, dict):
            gpx_path = gpx_upload.get("path") or gpx_upload.get("name")
        elif hasattr(gpx_upload, "name"):
            gpx_path = gpx_upload.name
        elif hasattr(gpx_upload, "path"):
            gpx_path = gpx_upload.path
        elif isinstance(gpx_upload, str):
            gpx_path = gpx_upload

    if not gpx_path and gpx_dropdown:
        gpx_path = gpx_dropdown

    if not gpx_path or not os.path.exists(gpx_path):
        return (
            "⚠️ **Missing GPX File**: Please upload a `.gpx` file or select one from the dropdown.",
            "",
            gr.update(visible=False)
        )

    if custom_output_dir and os.path.exists(custom_output_dir):
        output_dir = custom_output_dir
        summary_path = os.path.join(output_dir, "summary.json")
        mp4_files = [f for f in os.listdir(output_dir) if f.startswith("output_video_") and f.endswith(".mp4")]
        if not mp4_files:
            mp4_files = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
        video_path = os.path.join(output_dir, mp4_files[0]) if mp4_files else _GIS_STATE.get("last_video_path")
    else:
        video_path = _GIS_STATE.get("last_video_path")
        output_dir = _GIS_STATE.get("last_output_dir")
        summary_path = _GIS_STATE.get("last_summary_path")

        # Smart fallback: if page was refreshed, auto-detect most recent inspection run from outputs/
        if not output_dir or not os.path.exists(output_dir):
            if os.path.exists(OUTPUTS_DIR):
                recent_runs = [
                    os.path.join(OUTPUTS_DIR, d)
                    for d in os.listdir(OUTPUTS_DIR)
                    if os.path.isdir(os.path.join(OUTPUTS_DIR, d)) and os.path.exists(os.path.join(OUTPUTS_DIR, d, "summary.json"))
                ]
                if recent_runs:
                    recent_runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    output_dir = recent_runs[0]
                    summary_path = os.path.join(output_dir, "summary.json")

    if not output_dir or not os.path.exists(output_dir):
        return (
            "⚠️ **No Completed Inspection Run**: Please run a video inspection first or select a completed run from history.",
            "",
            gr.update(visible=False)
        )

    try:
        res = enrich_inspection_with_gis(
            video_path=video_path,
            gpx_path=gpx_path,
            summary_source=summary_path or output_dir,
            output_dir=output_dir,
            time_offset_seconds=float(offset_sec or 0.0),
            embed_images_as_base64=True,
        )

        status_md = f"""
### 🗺️ GIS Geolocation Enrichment Complete!
- **Unique Road Defects Mapped**: `{res['total_defects']}`
- **Survey Distance**: `{res['survey_distance_meters']:.1f} m` | **Duration**: `{res['survey_duration_seconds']:.1f}s`
- **Video Stream**: `{res['video_fps']} FPS` (`{res['total_video_frames']} frames`)
- **Synchronization**: {res['sync_info']}
- **GeoJSON File**: `{os.path.basename(res['geojson_path'])}`
"""
        if res.get("warnings"):
            status_md += "\\n**Notices:**\\n" + "\\n".join(f"- {w}" for w in res["warnings"][:3])

        return (
            status_md,
            res["html_map"],
            gr.update(value=res["geojson_path"], visible=True)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            f"❌ **GIS Processing Error**: {e}",
            "",
            gr.update(visible=False)
        )


def on_generate_geo_history(selected_run, gpx_upload, gpx_dropdown, offset_sec):
    """Event handler for generating GIS / GeoJSON enrichment for a historical run."""
    if not selected_run:
        return (
            "⚠️ **No Past Run Selected**: Please select a run from the dropdown above.",
            "",
            gr.update(visible=False)
        )
    run_dir = os.path.join(OUTPUTS_DIR, selected_run)
    return on_generate_geo(gpx_upload, gpx_dropdown, offset_sec, custom_output_dir=run_dir)


def get_history_runs():
    """Returns a list of available previous analysis run folder names sorted newest first."""
    runs = []
    if os.path.exists(OUTPUTS_DIR):
        for entry in sorted(os.listdir(OUTPUTS_DIR), reverse=True):
            entry_path = os.path.join(OUTPUTS_DIR, entry)
            if os.path.isdir(entry_path):
                runs.append(entry)
    return runs


def load_history_details(selected_run_folder):
    """Loads past analysis details (video, csv, crops gallery, summary markdown, and GIS map) for selected run."""
    empty_gis_download = gr.update(visible=False)
    if not selected_run_folder:
        return None, None, [], "Please select an analysis run from the dropdown.", [], [], "", "", empty_gis_download

    run_dir = os.path.join(OUTPUTS_DIR, selected_run_folder)
    if not os.path.exists(run_dir):
        return None, None, [], f"❌ Error: Run directory '{selected_run_folder}' not found.", [], [], "", "", empty_gis_download

    # Locate output mp4 video
    mp4_files = [f for f in os.listdir(run_dir) if f.startswith("output_video_") and f.endswith(".mp4")]
    if not mp4_files:
        mp4_files = [f for f in os.listdir(run_dir) if f.endswith(".mp4")]
    video_path = os.path.join(run_dir, mp4_files[0]) if mp4_files else None

    # Locate csv log
    csv_files = [f for f in os.listdir(run_dir) if f.startswith("detections_") and f.endswith(".csv")]
    if not csv_files:
        csv_files = [f for f in os.listdir(run_dir) if f.endswith(".csv")]
    csv_path = os.path.join(run_dir, csv_files[0]) if csv_files else None

    # Locate crops (limit to 60 for fast loading)
    crop_gallery = []
    crops_dir = os.path.join(run_dir, "crops")
    if os.path.exists(crops_dir):
        valid_crops = [f for f in sorted(os.listdir(crops_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in valid_crops[:60]:
            crop_path = os.path.join(crops_dir, f)
            crop_gallery.append((crop_path, f))
            
    # Locate segmentation masks (limit to 60 for fast loading)
    segmentation_gallery = []
    masks_dir = os.path.join(run_dir, "masks")
    if os.path.exists(masks_dir):
        valid_masks = [f for f in sorted(os.listdir(masks_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in valid_masks[:60]:
            mask_path = os.path.join(masks_dir, f)
            segmentation_gallery.append((mask_path, f))

    # Read summary.json or build summary markdown
    json_path = os.path.join(run_dir, "summary.json")
    _GIS_STATE["last_output_dir"] = run_dir
    _GIS_STATE["last_summary_path"] = json_path if os.path.exists(json_path) else None
    if video_path and os.path.exists(video_path):
        _GIS_STATE["last_video_path"] = video_path
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            summary_md = f"""
### 📜 Previous Inspection Report (`{selected_run_folder}`)

- **Run Timestamp Folder**: `outputs/{data.get("timestamp_folder", selected_run_folder)}`
- **Input Video File**: `{data.get("video_input", "N/A")}`
- **Model Applied**: `{data.get("model_used", "N/A")}`
- **Frames Processed**: `{data.get("processed_frames", "N/A")}` / `{data.get("total_frames", "N/A")}`
- **Processing Time**: `{data.get("processing_time_sec", "N/A")}s` (`{data.get("fps", "N/A")} FPS`)
- **Total Defects Detected**: **{data.get("total_defects_detected", 0)}**
- **Confidence Threshold**: `{data.get("confidence_threshold", "N/A")}` | **IoU**: `{data.get("iou_threshold", "N/A")}`

#### 🛠️ Defect Category Breakdown:
"""
            defect_counts = data.get("defect_counts", {})
            for cls_name, count in defect_counts.items():
                summary_md += f"- **{cls_name}**: `{count}` detected\n"

            # Defect Area Measurements (Pothole, Manhole, Patch only)
            past_defects = data.get("finalized_defects", [])
            past_areas = [d for d in past_defects if (d.get("real_area_m2") or (isinstance(d.get("measurement_data"), dict) and d["measurement_data"].get("real_area_m2")))]
            if past_areas:
                summary_md += "\n#### 📐 Measured Defect Surface Areas (Pothole / Manhole / Patch):\n"
                for d in past_areas[:15]:
                    area_val = d.get("real_area_m2") or d.get("measurement_data", {}).get("real_area_m2")
                    dtype = d.get("defect_type") or d.get("class_name") or "Defect"
                    did = str(d.get("defect_id", ""))
                    did_str = f" (`#{did[:8]}`)" if did else ""
                    summary_md += f"- **{dtype}**{did_str}: **{float(area_val):.3f} m²**\n"
                if len(past_areas) > 15:
                    summary_md += f"- *...and {len(past_areas) - 15} more (see CSV for full defect logs)*\n"
        except Exception:
            summary_md = f"### 📜 Inspection Run: `{selected_run_folder}`\n\n*Loaded from `{run_dir}`*"
    else:
        summary_md = f"### 📜 Inspection Run: `{selected_run_folder}`\n\n*Loaded from `{run_dir}`*"

    # Load or generate visualization charts for history
    charts_gallery = []
    if HAS_VISUALIZER:
        try:
            chart_data = generate_visualizations_from_directory(run_dir)
            for chart_path, caption in chart_data:
                if os.path.exists(chart_path):
                    charts_gallery.append((chart_path, caption))
        except Exception as e:
            print(f"[Warning] Failed to load history visualization charts: {e}")

    # Check if GIS GeoJSON already exists for this past run
    hist_map_html = ""
    hist_gis_status = "Upload or select a `.gpx` survey track below and click **📍 Generate GEO Information** to map this past run."
    hist_gis_download = gr.update(visible=False)

    existing_geojson = os.path.join(run_dir, "defects.geojson")
    if os.path.exists(existing_geojson) and HAS_GIS_BACKEND:
        try:
            with open(existing_geojson, "r", encoding="utf-8") as gf:
                geo_data = json.load(gf)
            from backend.geo_service import generate_leaflet_map_html
            hist_map_html = generate_leaflet_map_html(geo_data)
            num_pts = len([f for f in geo_data.get("features", []) if f.get("geometry", {}).get("type") == "Point"])
            hist_gis_status = f"✅ **Loaded Existing GIS Map**: Found `{num_pts}` geolocated road defects from `{os.path.basename(existing_geojson)}`."
            hist_gis_download = gr.update(value=existing_geojson, visible=True)
        except Exception as e:
            hist_map_html = ""
            hist_gis_status = f"⚠️ Could not load existing `defects.geojson`: {e}"
            hist_gis_download = gr.update(visible=False)

    return (
        video_path, csv_path, crop_gallery, summary_md, charts_gallery, segmentation_gallery,
        hist_gis_status, hist_map_html, hist_gis_download
    )


def load_yolo_model(model_name):
    """Loads specified YOLO model from models/ directory or HuggingFace."""
    if not HAS_ULTRALYTICS or "Demo" in model_name:
        return None, {}

    try:
        clean_model_name = model_name.split(" ")[0].strip()
        full_path = os.path.join(MODELS_DIR, clean_model_name)
        if os.path.exists(full_path):
            print(f"[RDD UI] Loading local model from: {full_path}")
            model = YOLO(full_path)
        elif os.path.exists(clean_model_name):
            print(f"[RDD UI] Loading model from path: {clean_model_name}")
            model = YOLO(clean_model_name)
        elif os.path.exists(os.path.join(MODELS_DIR, "rdd_yolo.pt")):
            print(f"[RDD UI] Loading default RDD-YOLO from models/rdd_yolo.pt")
            model = YOLO(os.path.join(MODELS_DIR, "rdd_yolo.pt"))
        elif clean_model_name.startswith("yolov8n"):
            print(f"[RDD UI] Loading baseline model yolov8n.pt")
            model = YOLO("yolov8n.pt")
        else:
            print(f"[RDD UI] Attempting to load model '{clean_model_name}'...")
            model = YOLO(clean_model_name)

        # Extract class names dynamically from model
        class_map = {}
        if hasattr(model, 'names') and model.names:
            for cid, cname in model.names.items():
                color = CLASS_COLOR_PALETTE[int(cid) % len(CLASS_COLOR_PALETTE)]
                class_map[int(cid)] = {"name": str(cname).title(), "color": color}
        else:
            class_map = DEFAULT_RDD_CLASSES

        return model, class_map

    except Exception as e:
        print(f"[Warning] Failed to load model '{model_name}': {e}. Falling back to demo mode.")
        return None, DEFAULT_RDD_CLASSES


def estimate_video_info(video_path, frame_skip=1):
    """Calculates estimated processing time and video metadata before starting inference."""
    if not video_path:
        return ""
    if isinstance(video_path, dict):
        video_path = video_path.get("path") or video_path.get("name") or video_path.get("video", {}).get("path")
    elif hasattr(video_path, "path"):
        video_path = video_path.path
    if not video_path or not os.path.exists(video_path):
        return ""

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            return ""

        duration_sec = round(total_frames / fps, 1)
        frames_to_process = total_frames // max(1, int(frame_skip))

        # Baseline average processing speed on host CPU (approx 12 FPS)
        avg_proc_fps = 12.0
        est_proc_sec = frames_to_process / avg_proc_fps

        est_finish = datetime.datetime.now() + datetime.timedelta(seconds=est_proc_sec)
        finish_str = est_finish.strftime("%I:%M %p")

        if est_proc_sec >= 60:
            est_min = int(est_proc_sec // 60)
            est_s = int(est_proc_sec % 60)
            time_display = f"~{est_min} min {est_s} sec"
        else:
            time_display = f"~{int(est_proc_sec)} sec"

        return (
            f"⏱️ **Estimated Completion Time:** `{time_display}` *(Finish around {finish_str})* | "
            f"Video: `{duration_sec}s` ({total_frames} frames, {width}x{height} @ {fps:.1f} FPS)"
        )
    except Exception:
        return ""
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def compute_live_eta(fraction, start_time):
    """Computes dynamic time remaining (ETA) and estimated finish timestamp."""
    if fraction <= 0.01:
        return "", ""
    
    elapsed = time.time() - start_time
    total_est = elapsed / fraction
    remaining_sec = max(0.0, total_est - elapsed)
    
    finish_dt = datetime.datetime.now() + datetime.timedelta(seconds=remaining_sec)
    finish_str = finish_dt.strftime("%I:%M %p")
    
    if remaining_sec >= 60:
        rem_min = int(remaining_sec // 60)
        rem_sec = int(remaining_sec % 60)
        time_str = f"~{rem_min}m {rem_sec}s remaining"
    else:
        time_str = f"~{int(remaining_sec)}s remaining"
        
    return time_str, finish_str


def reencode_to_h264(input_path, output_path):
    """Re-encodes video to H.264 format using imageio-ffmpeg for browser compatibility."""
    if HAS_IMAGEIO_FFMPEG:
        reader = None
        writer = None
        try:
            reader = imageio.get_reader(input_path)
            fps = reader.get_meta_data().get('fps', 30)
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
            for frame in reader:
                writer.append_data(frame)
            return output_path
        except Exception as e:
            print(f"[Warning] H.264 re-encoding failed: {e}")
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
    return input_path


def on_toggle_dedup(enabled):
    """Event handler when the user toggles duplicate removal ON/OFF."""
    if enabled:
        status_html = '<div class="dedup-active"><span class="dedup-dot dedup-dot-on"></span> Dedup Pipeline — ACTIVE <span style="font-weight:400; color:#94a3b8; font-size:0.8rem;">· BoT-SORT + Multi-Feature Verification</span></div>'
        btn_label = "🚀 Run RDD-YOLO Detection + Dedup"
        row_visibility = gr.update(visible=True)
    else:
        status_html = '<div class="dedup-inactive"><span class="dedup-dot dedup-dot-off"></span> Dedup Pipeline — OFF <span style="font-weight:400; color:#94a3b8; font-size:0.8rem;">· Raw Frame-by-Frame Detection</span></div>'
        btn_label = "🚀 Run RDD-YOLO Detection (Raw Mode)"
        row_visibility = gr.update(visible=False)
    return status_html, btn_label, row_visibility


def on_toggle_openvino(is_openvino):
    """Event handler when the user toggles OpenVINO model selection ON/OFF."""
    choices = get_available_models(use_openvino=is_openvino)
    val = choices[0] if choices else None
    if is_openvino:
        return gr.Dropdown(
            choices=choices,
            value=val,
            label="OpenVINO Model Folder",
            info="Select an OpenVINO model folder from models/"
        )
    else:
        return gr.Dropdown(
            choices=choices,
            value=val,
            label="Model Weights (.pt)",
            info="Select a PyTorch .pt model file from models/"
        )


def process_video(video_path, model_name, conf_threshold, iou_threshold, frame_skip,
                  enable_dedup=True, merge_threshold=0.65, track_buffer=45,
                  enable_segmentation=True, cam_height=1.2, cam_tilt=15.0, cam_focal=1200.0,
                  *slider_and_other_args, progress=gr.Progress()):
    """
    Processes video for Road Damage Detection with integrated duplicate removal
    and real-world defect area calculation (Pothole, Manhole, Patch only).
    Supports dynamic per-class custom confidence thresholds.

    When dedup is enabled, runs the full pipeline:
      YOLO + BoT-SORT tracking → TrackManager → DuplicateVerifier → DefectDatabase

    When dedup is disabled, falls back to the original frame-by-frame detection.
    """
    if isinstance(video_path, dict):
        video_path = video_path.get("path") or video_path.get("name") or video_path.get("video", {}).get("path")
    elif hasattr(video_path, "path"):
        video_path = video_path.path
    if not video_path or not os.path.exists(video_path):
        return None, None, [], "❌ Error: Invalid or missing video input file.", [], []

    # Reset revocation flag for new processing session
    _CANCEL_FLAG["requested"] = False

    # Extract dynamic per-class confidence thresholds
    slider_values = list(slider_and_other_args[:MAX_CLASS_SLIDERS])
    classes_dict = get_model_class_dict(model_name)
    class_items = list(classes_dict.items())
    custom_class_thresholds = {}
    for i, (cid, cname) in enumerate(class_items):
        if i < len(slider_values) and slider_values[i] is not None:
            val = float(slider_values[i])
        else:
            val = float(conf_threshold or 0.25)
        custom_class_thresholds[cname] = val
        custom_class_thresholds[str(cname).upper()] = val
        custom_class_thresholds[cid] = val

    # ── Dedup Pipeline Path ──
    if enable_dedup and HAS_DEDUP_PIPELINE and "Demo" not in model_name:
        return _process_video_with_dedup(
            video_path, model_name, conf_threshold, iou_threshold,
            frame_skip, merge_threshold, track_buffer, enable_segmentation,
            cam_height, cam_tilt, cam_focal,
            custom_class_thresholds, progress
        )

    # ── Fallback: Original frame-by-frame detection (no tracking/dedup) ──
    return _process_video_legacy(
        video_path, model_name, conf_threshold, iou_threshold,
        frame_skip, progress, custom_class_thresholds,
        cam_height, cam_tilt, cam_focal
    )


def _process_video_with_dedup(video_path, model_name, conf_threshold, iou_threshold,
                              frame_skip, merge_threshold, track_buffer, enable_segmentation,
                              cam_height, cam_tilt, cam_focal,
                              custom_class_thresholds, progress):
    """
    Full dedup pipeline: YOLO + BoT-SORT + TrackManager + DuplicateVerifier.
    Applies custom per-class confidence filtering dynamically.
    Estimates real-world ground area (m²) for Pothole, Manhole, and Patch.
    Returns the same 6 outputs as the original process_video for Gradio compatibility.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # Clean display suffixes from dropdown choices e.g. "rdd_yolo.pt (Default)" -> "rdd_yolo.pt"
    clean_model_name = model_name.split(" ")[0].strip()

    model_path = clean_model_name
    full_model_path = os.path.join(MODELS_DIR, clean_model_name)
    if os.path.exists(full_model_path):
        model_path = os.path.join("models", clean_model_name)
    elif os.path.exists(clean_model_name):
        model_path = clean_model_name
    elif os.path.exists(os.path.join(MODELS_DIR, "rdd_yolo.pt")):
        model_path = "models/rdd_yolo.pt"

    # Determine base confidence threshold for YOLO detector
    valid_thresholds = [v for v in custom_class_thresholds.values() if isinstance(v, (int, float))] if custom_class_thresholds else []
    base_conf = min(valid_thresholds) if valid_thresholds else float(conf_threshold)
    base_conf = max(0.05, min(base_conf, float(conf_threshold)))

    config = PipelineConfig(
        model_path=model_path,
        confidence_threshold=base_conf,
        iou_threshold=iou_threshold,
        frame_skip=int(frame_skip),
        merge_threshold=merge_threshold,
        track_buffer_frames=int(track_buffer),
        enable_segmentation=enable_segmentation,
        camera_height_m=float(cam_height or 1.2),
        camera_tilt_deg=float(cam_tilt or 15.0),
        camera_focal_length_px=float(cam_focal) if cam_focal else None,
        save_annotated_video=True,
        save_crops=True,
        database_type="sqlite",
        _base_dir=BASE_DIR,
    )

    # Create pipeline and run
    pipeline = RDDPipeline(config=config)

    # Wrap pipeline.detector.detect_and_track with per-class threshold filter
    orig_detect_and_track = pipeline.detector.detect_and_track
    def filtered_detect_and_track(frame, frame_idx):
        dets = orig_detect_and_track(frame, frame_idx)
        if not custom_class_thresholds:
            return dets
        filtered = []
        for d in dets:
            thresh = custom_class_thresholds.get(
                d.class_name,
                custom_class_thresholds.get(
                    str(d.class_name).upper(),
                    custom_class_thresholds.get(d.class_id, float(conf_threshold))
                )
            )
            if d.confidence >= thresh:
                filtered.append(d)
        return filtered
    pipeline.detector.detect_and_track = filtered_detect_and_track

    start_time = time.time()

    def gradio_progress_callback(fraction, message):
        time_str, finish_str = compute_live_eta(fraction, start_time)
        if time_str:
            progress(fraction, desc=f"⏳ {time_str} (Est. finish: {finish_str}) | {message}")
        else:
            progress(fraction, desc=message)

    progress(0, desc=f"Loading model & initializing dedup pipeline...")

    try:
        result = pipeline.process_video(
            video_path,
            output_dir=OUTPUTS_DIR,
            progress_callback=gradio_progress_callback,
            cancel_check=lambda: _CANCEL_FLAG["requested"],
        )
    except InterruptedError as e:
        return None, None, [], f"⚠️ **Process Revoked**: {e}", [], []
    except Exception as e:
        if _CANCEL_FLAG["requested"]:
            return None, None, [], "⚠️ **Process Revoked**: Video processing was cancelled by user revocation request.", [], []
        import traceback
        traceback.print_exc()
        return None, None, [], f"❌ Pipeline error: {e}", [], []

    # Store result for dataset builder
    _ANNOTATION_STATE["last_pipeline_result"] = result
    _ANNOTATION_STATE["last_video_path"] = video_path
    _ANNOTATION_STATE["last_inspection_mode"] = "dedup"

    # Store result for GIS enrichment
    _GIS_STATE["last_video_path"] = video_path
    _GIS_STATE["last_output_dir"] = result.output_dir
    _GIS_STATE["last_summary_path"] = os.path.join(result.output_dir, "summary.json")
    _GIS_STATE["last_pipeline_result"] = result

    # ── Convert outputs to Gradio format ──

    # 1. Annotated video — re-encode to H.264 for browser compatibility
    final_video = result.annotated_video_path
    if final_video and os.path.exists(final_video):
        h264_path = final_video.replace(".mp4", "_h264.mp4")
        final_video = reencode_to_h264(final_video, h264_path)
    else:
        final_video = None

    # 2. CSV file
    csv_path = None
    if result.output_dir:
        csv_files = [
            f for f in os.listdir(result.output_dir)
            if f.startswith("unique_defects_") and f.endswith(".csv")
        ]
        if csv_files:
            csv_path = os.path.join(result.output_dir, csv_files[0])

    # 3. Crop gallery — load saved crops from the dedup pipeline
    crop_gallery_images = []
    crops_dir = os.path.join(result.output_dir, "crops")
    if os.path.exists(crops_dir):
        crop_files = sorted([
            f for f in os.listdir(crops_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        for crop_file in crop_files[:60]:  # Limit for UI performance
            crop_path = os.path.join(crops_dir, crop_file)
            try:
                crop_img = cv2.imread(crop_path)
                if crop_img is not None:
                    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    # Parse label from filename: defect_track0001_frame000042_cls3_conf0.89.jpg
                    parts = crop_file.replace(".jpg", "").split("_")
                    label = crop_file
                    for p in parts:
                        if p.startswith("conf"):
                            label = p.replace("conf", "Conf: ")
                        if p.startswith("cls"):
                            label = f"Class {p.replace('cls', '')} | {label}"
                    crop_gallery_images.append((crop_rgb, label))
            except Exception:
                continue

    # 3.5 Segmentation overlay gallery
    segmentation_gallery_images = []
    if hasattr(result, 'finalized_defects'):
        for d in result.finalized_defects:
            if d.measurement_data and d.measurement_data.get("overlay_path"):
                overlay_path = d.measurement_data.get("overlay_path")
                if os.path.exists(overlay_path):
                    area = d.measurement_data.get("pixel_area", 0)
                    quality = d.measurement_data.get("mask_quality_score", 0.0)
                    caption = f"{d.defect_type} - {area}px (quality {quality:.2f})"
                    segmentation_gallery_images.append((overlay_path, caption))

    # 4. Summary markdown with dedup statistics
    reduction_pct = 0.0
    if result.total_raw_detections > 0:
        reduction_pct = (1 - result.total_unique_defects / result.total_raw_detections) * 100

    summary_md = f"""
### 📊 RDD-YOLO Inspection Summary — With Duplicate Removal

- **Output Folder**: `{os.path.basename(result.output_dir)}`
- **Model Used**: `{config.model_path}`
- **Frames Processed**: `{result.processed_frames}` / `{result.total_frames}`
- **Processing Time**: `{result.processing_time_sec:.1f}s` (`{result.fps:.1f} FPS`)

---

#### 🔍 Detection & Deduplication Results

| Metric | Count |
|---|---|
| Raw Frame Detections | **{result.total_raw_detections}** |
| BoT-SORT Tracks Created | **{result.tracks_created}** |
| Tracks Merged (Duplicates) | **{result.tracks_merged}** |
| **Unique Physical Defects** | **🎯 {result.total_unique_defects}** |
| **Duplicate Reduction** | **📉 {reduction_pct:.1f}%** |

#### 🛠️ Unique Defect Breakdown:
"""
    for cls_name, count in sorted(result.defect_counts.items()):
        summary_md += f"- **{cls_name}**: `{count}` unique defects\n"

    # Defect Area Measurements (Pothole, Manhole, Patch only)
    area_entries = []
    if hasattr(result, "finalized_defects"):
        for d in result.finalized_defects:
            if getattr(d, "real_area_m2", None) is not None and d.real_area_m2 > 0:
                area_entries.append(f"- **{d.defect_type}** (`#{d.defect_id[:8]}`): **{d.real_area_m2:.3f} m²** (observed across {d.frames_observed} frames)")
    if area_entries:
        summary_md += "\n#### 📐 Real-World Defect Area Measurements (Pothole / Manhole / Patch):\n"
        summary_md += f"*Camera Height: `{cam_height}m` · Downward Tilt: `{cam_tilt}°` · Focal Length: `{cam_focal}px`*\n\n"
        for line in area_entries[:15]:
            summary_md += f"{line}\n"
        if len(area_entries) > 15:
            summary_md += f"- *...and {len(area_entries) - 15} more defects (see CSV / GeoJSON for full list)*\n"

    # Build per-class threshold display string
    c_th_items = []
    for cname, th in custom_class_thresholds.items():
        if isinstance(cname, str) and not cname.isupper() and isinstance(th, (int, float)):
            c_th_items.append(f"`{cname}`: **{th:.2f}**")
    c_th_str = " · ".join(c_th_items) if c_th_items else f"`{conf_threshold}`"

    summary_md += f"""
---
#### ⚙️ Pipeline Configuration
- Global Master Confidence: `{conf_threshold}`
- Per-Class Thresholds: {c_th_str}
- IoU Threshold: `{iou_threshold}`
- Merge Threshold: `{merge_threshold}`
- Track Buffer: `{int(track_buffer)} frames`
- Motion Compensation: `{'ON' if config.enable_motion_compensation else 'OFF'}`

📁 *All files saved to: `{result.output_dir}`*
📊 *Merge audit log: `merge_audit_log.json`*
🗄️ *Defect database: `defects.db`*
"""

    # 5. Visualization charts
    charts_gallery = []
    if HAS_VISUALIZER and result.output_dir:
        try:
            chart_data = generate_visualizations_from_directory(result.output_dir)
            for chart_path, caption in chart_data:
                if os.path.exists(chart_path):
                    charts_gallery.append((chart_path, caption))
        except Exception as e:
            print(f"[Warning] Failed to generate visualization charts: {e}")

    return final_video, csv_path, crop_gallery_images, summary_md, charts_gallery, segmentation_gallery_images


# ═══════════════════════════════════════════════════════════════════════════════
# RDD_Ahmedabad Dataset Builder — Backend Functions (Staged Workflow)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_annotation_candidates():
    """
    Extract annotation candidates from the last video inspection run (supports Dedup and Raw Mode).
    Returns: (status_message, total_count)
    """
    result = _ANNOTATION_STATE.get("last_pipeline_result")
    video_path = _ANNOTATION_STATE.get("last_video_path")
    mode = _ANNOTATION_STATE.get("last_inspection_mode", "dedup")

    if result is None or video_path is None:
        return "❌ No inspection result available. Run a video inspection first on the '🚀 Run New Inspection' tab.", 0

    if not HAS_DATASET_SYSTEM or RDD_CROP_SELECTOR is None:
        return "❌ Dataset system not available. Check configs/rdd_ahmedabad_classes.yaml.", 0

    candidates = RDD_CROP_SELECTOR.select_from_pipeline_result(result, video_path)

    if not candidates:
        return "⚠️ No annotation candidates found. The inspection did not detect any defects.", 0

    _ANNOTATION_STATE["candidates"] = candidates
    _ANNOTATION_STATE["current_index"] = 0
    _ANNOTATION_STATE["staged_defects"] = []
    _ANNOTATION_STATE["next_staged_id"] = 1
    _ANNOTATION_STATE["editing_staged_id"] = None
    _ANNOTATION_STATE["accepted_count"] = 0
    _ANNOTATION_STATE["skipped_count"] = 0
    _ANNOTATION_STATE["changed_count"] = 0

    mode_label = "Deduplicated" if mode == "dedup" else "Raw Detections (No Dedup)"
    return f"✅ {len(candidates)} defect crops ready for review ({mode_label}).", len(candidates)


def prepare_image_annotation_candidates(image_files):
    """
    Load one or more uploaded image files into Stage 1 of Dataset Builder for manual crop and annotation.
    Allows selecting multiple images at once, stepping through them sequentially, cropping each defect,
    assigning defect classes, and staging them for batch commit.
    Returns: (status_message, total_count)
    """
    if image_files is None:
        return "❌ No image files provided.", 0

    if not HAS_DATASET_SYSTEM:
        return "❌ Dataset system not available. Check configs/rdd_ahmedabad_classes.yaml.", 0

    # Normalize to list
    if not isinstance(image_files, (list, tuple)):
        file_list = [image_files]
    else:
        file_list = list(image_files)

    if not file_list:
        return "❌ No images provided in upload.", 0

    default_class = "POTHOLE"
    default_cid = 0
    if RDD_CLASS_CONFIG:
        valid_names = RDD_CLASS_CONFIG.get_class_names()
        if valid_names:
            default_class = valid_names[0]
            default_cid = RDD_CLASS_CONFIG.get_class_id(default_class)

    candidates = []
    loaded_filenames = []

    for idx, item in enumerate(file_list):
        image_path = None
        if isinstance(item, str):
            image_path = item
        elif hasattr(item, "name"):
            image_path = item.name
        elif isinstance(item, dict):
            image_path = item.get("path") or item.get("name")

        if not image_path or not os.path.exists(image_path):
            continue

        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            try:
                with open(image_path, "rb") as f:
                    buf = np.frombuffer(f.read(), dtype=np.uint8)
                    frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                pass

        if frame_bgr is None or frame_bgr.size == 0:
            continue

        h, w = frame_bgr.shape[:2]
        image_filename = os.path.basename(image_path)
        image_stem = os.path.splitext(image_filename)[0]
        loaded_filenames.append(image_filename)

        bw = max(20, int(w * 0.40))
        bh = max(20, int(h * 0.40))
        cx, cy = w // 2, h // 2
        bx1 = max(0, cx - bw // 2)
        by1 = max(0, cy - bh // 2)
        bx2 = min(w, bx1 + bw)
        by2 = min(h, by1 + bh)

        crop = frame_bgr[by1:by2, bx1:bx2].copy()
        cand = CandidateAnnotation(
            crop_image=crop,
            original_frame=frame_bgr.copy(),
            bbox=(bx1, by1, bx2, by2),
            initial_bbox=(bx1, by1, bx2, by2),
            class_id=default_cid,
            class_name=default_class,
            predicted_class="Manual Crop",
            confidence=1.0,
            frame_number=1,
            track_id=idx + 1,
            source_video=image_filename,
            defect_id=f"img_{image_stem}_{idx+1}"
        )
        candidates.append(cand)

    if not candidates:
        return "❌ Failed to decode the uploaded image(s). Please upload valid JPG or PNG files.", 0

    _ANNOTATION_STATE["candidates"] = candidates
    _ANNOTATION_STATE["current_index"] = 0
    _ANNOTATION_STATE["editing_staged_id"] = None
    _ANNOTATION_STATE["last_video_path"] = loaded_filenames[0]
    _ANNOTATION_STATE["last_inspection_mode"] = "batch_image" if len(candidates) > 1 else "single_image"

    if len(candidates) == 1:
        status_msg = f"🖼️ Image `{loaded_filenames[0]}` loaded. Adjust the crop box on the canvas, assign class, and click **Accept Defect**."
    else:
        status_msg = f"🖼️ **{len(candidates)} images loaded for defect cropping!** Crop Image #1 on the canvas and click **Accept Defect** (or **Add Another in Frame** for multiple defects per image)."

    return status_msg, len(candidates)


def get_current_candidate_display():
    """
    Return display data for the current annotation candidate.
    Returns: (crop_image, info_markdown, suggested_class, progress_text, is_done)
    """
    candidates = _ANNOTATION_STATE.get("candidates", [])
    idx = _ANNOTATION_STATE.get("current_index", 0)
    staged = _ANNOTATION_STATE.get("staged_defects", [])
    skipped = _ANNOTATION_STATE.get("skipped_count", 0)
    mode = _ANNOTATION_STATE.get("last_inspection_mode", "")

    if not candidates or idx >= len(candidates):
        return None, "All candidates reviewed.", "", f"Review Complete — Staged: {len(staged)} | Skipped: {skipped}", True

    c = candidates[idx]

    # Auto-suggest class using model mapping
    suggested = c.class_name
    if RDD_CLASS_CONFIG:
        suggested = RDD_CLASS_CONFIG.map_model_class(c.predicted_class)

    valid_classes = RDD_CLASS_CONFIG.get_class_names() if RDD_CLASS_CONFIG else []

    # Normalize suggested class name
    norm_suggested = suggested.strip().upper().replace(" ", "_").replace("-", "_") if isinstance(suggested, str) else ""

    resolved_class = None
    if valid_classes and norm_suggested in valid_classes:
        resolved_class = norm_suggested
    elif valid_classes and norm_suggested.endswith("S") and norm_suggested[:-1] in valid_classes:
        resolved_class = norm_suggested[:-1]
    elif valid_classes and (norm_suggested + "S") in valid_classes:
        resolved_class = norm_suggested + "S"
    elif valid_classes and c.class_name:
        cand_norm = c.class_name.strip().upper().replace(" ", "_").replace("-", "_")
        if cand_norm in valid_classes:
            resolved_class = cand_norm

    is_manual = (c.predicted_class in ("Manual", "Manual Crop") or mode in ("single_image", "batch_image"))
    is_unresolved = (resolved_class is None and not is_manual)

    if is_unresolved:
        dropdown_val = None
        suggested_display = f"<span style='color:#ef4444;font-weight:bold;'>⚠️ UNRESOLVED (`{c.predicted_class}`)</span>"
        warning_banner = f"""
> [!WARNING]
> ⚠️ **UNRESOLVED DEFECT CLASS — ACTION REQUIRED**  
> The model prediction `{c.predicted_class}` does not match any known RDD_Ahmedabad class.  
> **You MUST explicitly select a valid class from the 'Assign Class' dropdown before accepting this defect.**
"""
        progress_suffix = " ⚠️ UNRESOLVED CLASS"
    else:
        dropdown_val = resolved_class or valid_classes[0] if valid_classes else "POTHOLE"
        suggested_display = f"`{dropdown_val}`"
        warning_banner = ""
        progress_suffix = ""

    header_label = f"Image #{idx + 1} of {len(candidates)}" if is_manual else f"Detection #{idx + 1} / {len(candidates)}"
    conf_pill = f'<span class="ds-conf-pill" style="background:rgba(56,189,248,0.15);color:#38bdf8;">Manual Crop</span>' if is_manual else f'<span class="ds-conf-pill">{c.confidence:.0%} conf</span>'

    info_md = f"""
{warning_banner}
<div class="ds-candidate-card">
  <div class="ds-card-row">
    <span class="ds-detection-num">{header_label}</span>
    {conf_pill}
  </div>
  <div class="ds-meta-grid">
    <div class="ds-meta-item"><span class="ds-meta-lbl">Type:</span> <span class="ds-meta-val"><code>{c.predicted_class}</code></span></div>
    <div class="ds-meta-item"><span class="ds-meta-lbl">Assigned:</span> <span class="ds-meta-val">{suggested_display}</span></div>
  </div>
  <details class="ds-details-accordion">
    <summary>📋 Details (Image, Dimensions, Bbox)</summary>
    <div class="ds-details-content">
      <div style="grid-column: span 2;"><b>Source:</b> <code>{c.source_video}</code></div>
      <div><b>Bbox Coords:</b> <code>{c.bbox}</code></div>
      <div><b>Size:</b> <code>{c.bbox[2]-c.bbox[0]}x{c.bbox[3]-c.bbox[1]} px</code></div>
    </div>
  </details>
</div>
"""
    if is_manual:
        progress_text = f"Image {idx + 1} of {len(candidates)} — Staged: {len(staged)} | Skipped: {skipped}{progress_suffix}"
    else:
        progress_text = f"Candidate {idx + 1} / {len(candidates)}{progress_suffix} | Staged: {len(staged)} | Skipped: {skipped}"

    # Convert BGR crop to RGB for Gradio display
    crop_rgb = None
    if c.crop_image is not None and c.crop_image.size > 0:
        crop_rgb = cv2.cvtColor(c.crop_image, cv2.COLOR_BGR2RGB)

    return crop_rgb, info_md, dropdown_val, progress_text, False


def _resolve_bbox(candidate_or_defect, explicit_bbox=None):
    """
    Resolve and validate a bounding box for a candidate annotation or staged defect dict.
    - If explicit_bbox is provided: validate against frame dimensions (clamp to [0, width] / [0, height]),
      reject if x2 <= x1 or y2 <= y1, and return (int(x1), int(y1), int(x2), int(y2)).
    - Otherwise: fall back to initial_bbox (never live-mutated bbox).
    """
    frame = None
    initial_bbox = None
    fallback_bbox = None

    if isinstance(candidate_or_defect, dict):
        frame = candidate_or_defect.get("original_frame")
        initial_bbox = candidate_or_defect.get("initial_bbox")
        fallback_bbox = candidate_or_defect.get("bbox")
    elif candidate_or_defect is not None:
        frame = getattr(candidate_or_defect, "original_frame", None)
        initial_bbox = getattr(candidate_or_defect, "initial_bbox", None)
        fallback_bbox = getattr(candidate_or_defect, "bbox", None)

    if explicit_bbox is not None:
        try:
            bx1, by1, bx2, by2 = [float(v) for v in explicit_bbox]
            if frame is not None and hasattr(frame, "shape"):
                h, w = frame.shape[:2]
                bx1 = max(0.0, min(float(w), bx1))
                by1 = max(0.0, min(float(h), by1))
                bx2 = max(0.0, min(float(w), bx2))
                by2 = max(0.0, min(float(h), by2))

            ix1, iy1, ix2, iy2 = int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))
            if ix2 > ix1 and iy2 > iy1:
                return (ix1, iy1, ix2, iy2)
        except Exception:
            pass

    # Fall back to initial_bbox (never live-mutated bbox if initial_bbox exists)
    if initial_bbox is not None:
        return tuple(int(round(v)) for v in initial_bbox)
    if fallback_bbox is not None:
        return tuple(int(round(v)) for v in fallback_bbox)
    return (0, 0, 0, 0)


def stage_candidate_annotation(chosen_class, bbox=None):
    """
    Stage the current candidate in memory (does NOT write to dataset yet).
    Advances to the next candidate.
    Returns: next candidate display tuple.
    """
    candidates = _ANNOTATION_STATE.get("candidates", [])
    idx = _ANNOTATION_STATE.get("current_index", 0)

    if not candidates or idx >= len(candidates):
        return get_current_candidate_display()

    c = candidates[idx]
    final_bbox = _resolve_bbox(c, explicit_bbox=bbox)

    # Resolve class
    raw_choice = chosen_class.strip() if chosen_class else ""
    mapped_choice = RDD_CLASS_CONFIG.map_model_class(raw_choice) if (RDD_CLASS_CONFIG and raw_choice) else raw_choice
    class_id = RDD_CLASS_CONFIG.get_class_id(mapped_choice) if (RDD_CLASS_CONFIG and mapped_choice) else -1
    if class_id >= 0 and RDD_CLASS_CONFIG:
        class_name = RDD_CLASS_CONFIG.get_class_name(class_id)
    else:
        cand_raw = c.class_name or c.predicted_class or ""
        cand_mapped = RDD_CLASS_CONFIG.map_model_class(cand_raw) if (RDD_CLASS_CONFIG and cand_raw) else cand_raw
        cand_id = RDD_CLASS_CONFIG.get_class_id(cand_mapped) if (RDD_CLASS_CONFIG and cand_mapped and not raw_choice) else -1
        if cand_id >= 0 and RDD_CLASS_CONFIG and not raw_choice:
            class_id = cand_id
            class_name = RDD_CLASS_CONFIG.get_class_name(class_id)
        else:
            # DO NOT DEFAULT TO POTHOLE! Keep class_id = -1 and original unresolved string
            class_id = -1
            class_name = raw_choice.upper() if raw_choice else (c.predicted_class or c.class_name or "UNRESOLVED")

    # Track changed class
    if RDD_CLASS_CONFIG:
        original_suggested = RDD_CLASS_CONFIG.map_model_class(c.predicted_class)
        if class_name != original_suggested:
            _ANNOTATION_STATE["changed_count"] = _ANNOTATION_STATE.get("changed_count", 0) + 1

    # Extract clean crop copy using final_bbox
    crop_copy = None
    if c.original_frame is not None:
        x1, y1, x2, y2 = final_bbox
        h, w = c.original_frame.shape[:2]
        crop_copy = c.original_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
    if crop_copy is None or crop_copy.size == 0:
        crop_copy = c.crop_image.copy() if c.crop_image is not None and c.crop_image.size > 0 else None

    staged_id = _ANNOTATION_STATE["next_staged_id"]
    _ANNOTATION_STATE["next_staged_id"] += 1

    staged_item = {
        "id": staged_id,
        "crop_image": crop_copy,
        "original_frame": c.original_frame.copy() if c.original_frame is not None else None,
        "bbox": final_bbox,
        "initial_bbox": tuple(c.initial_bbox),
        "class_id": class_id,
        "class_name": class_name,
        "predicted_class": c.predicted_class,
        "confidence": float(c.confidence),
        "source_video": c.source_video,
        "frame_number": c.frame_number,
        "track_id": c.track_id,
    }

    _ANNOTATION_STATE["staged_defects"].append(staged_item)
    _ANNOTATION_STATE["accepted_count"] = len(_ANNOTATION_STATE["staged_defects"])
    _ANNOTATION_STATE["current_index"] = idx + 1

    return get_current_candidate_display()


def stage_another_defect_annotation(chosen_class, bbox=None):
    """
    Stage the current candidate's bounding box and class in memory,
    but DO NOT advance to next candidate frame. Allows staging multiple
    defects in the same frame.
    """
    candidates = _ANNOTATION_STATE.get("candidates", [])
    idx = _ANNOTATION_STATE.get("current_index", 0)

    if not candidates or idx >= len(candidates):
        return get_current_candidate_display()

    c = candidates[idx]
    final_bbox = _resolve_bbox(c, explicit_bbox=bbox)

    # Resolve class
    raw_choice = chosen_class.strip() if chosen_class else ""
    mapped_choice = RDD_CLASS_CONFIG.map_model_class(raw_choice) if (RDD_CLASS_CONFIG and raw_choice) else raw_choice
    class_id = RDD_CLASS_CONFIG.get_class_id(mapped_choice) if (RDD_CLASS_CONFIG and mapped_choice) else -1
    if class_id >= 0 and RDD_CLASS_CONFIG:
        class_name = RDD_CLASS_CONFIG.get_class_name(class_id)
    else:
        cand_raw = c.class_name or c.predicted_class or ""
        cand_mapped = RDD_CLASS_CONFIG.map_model_class(cand_raw) if (RDD_CLASS_CONFIG and cand_raw) else cand_raw
        cand_id = RDD_CLASS_CONFIG.get_class_id(cand_mapped) if (RDD_CLASS_CONFIG and cand_mapped and not raw_choice) else -1
        if cand_id >= 0 and RDD_CLASS_CONFIG and not raw_choice:
            class_id = cand_id
            class_name = RDD_CLASS_CONFIG.get_class_name(class_id)
        else:
            # DO NOT DEFAULT TO POTHOLE! Keep class_id = -1 and original unresolved string
            class_id = -1
            class_name = raw_choice.upper() if raw_choice else (c.predicted_class or c.class_name or "UNRESOLVED")

    # Extract clean crop copy using final_bbox
    crop_copy = None
    if c.original_frame is not None:
        x1, y1, x2, y2 = final_bbox
        h, w = c.original_frame.shape[:2]
        crop_copy = c.original_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
    if crop_copy is None or crop_copy.size == 0:
        crop_copy = c.crop_image.copy() if c.crop_image is not None and c.crop_image.size > 0 else None

    staged_id = _ANNOTATION_STATE["next_staged_id"]
    _ANNOTATION_STATE["next_staged_id"] += 1

    staged_item = {
        "id": staged_id,
        "crop_image": crop_copy,
        "original_frame": c.original_frame.copy() if c.original_frame is not None else None,
        "bbox": final_bbox,
        "initial_bbox": final_bbox,
        "class_id": class_id,
        "class_name": class_name,
        "predicted_class": c.predicted_class,
        "confidence": 1.0,  # Manual addition
        "source_video": c.source_video,
        "frame_number": c.frame_number,
        "track_id": c.track_id,
    }

    _ANNOTATION_STATE["staged_defects"].append(staged_item)
    _ANNOTATION_STATE["accepted_count"] = len(_ANNOTATION_STATE["staged_defects"])

    # Reset active candidate's c.bbox back to c.initial_bbox so primary candidate stays intact
    c.bbox = tuple(c.initial_bbox)
    if c.original_frame is not None:
        ix1, iy1, ix2, iy2 = [int(v) for v in c.initial_bbox]
        c.crop_image = c.original_frame[iy1:iy2, ix1:ix2].copy()

    # Return same candidate display without advancing
    return get_current_candidate_display()


# Backwards compatibility aliases
accept_annotation = stage_candidate_annotation
add_another_defect_annotation = stage_another_defect_annotation


def skip_annotation():
    """Skip the current candidate and advance to the next."""
    _ANNOTATION_STATE["skipped_count"] = _ANNOTATION_STATE.get("skipped_count", 0) + 1
    _ANNOTATION_STATE["current_index"] = _ANNOTATION_STATE.get("current_index", 0) + 1
    return get_current_candidate_display()


def previous_annotation():
    """Step back to the previous candidate if available."""
    idx = _ANNOTATION_STATE.get("current_index", 0)
    if idx > 0:
        _ANNOTATION_STATE["current_index"] = idx - 1
    return get_current_candidate_display()


def get_staged_review_data():
    """
    Generate review data, confirmation prompt, distribution table, gallery items, and dropdown choices.
    """
    staged = _ANNOTATION_STATE.get("staged_defects", [])
    candidates = _ANNOTATION_STATE.get("candidates", [])
    skipped = _ANNOTATION_STATE.get("skipped_count", 0)

    if not staged:
        prompt_md = """### ⚠️ No Defects Staged
No defect crops were accepted during this review session.
Click **↩️ Return to Candidate Review** to review candidates and accept defects, or load another inspection run."""
        dist_md = "<div style='color:#94a3b8;padding:12px;'>0 defects staged.</div>"
        gallery_items = []
        choices = []
        return prompt_md, dist_md, gallery_items, choices

    unresolved_count = sum(1 for d in staged if d.get("class_id", -1) < 0)
    unresolved_warning = ""
    if unresolved_count > 0:
        unresolved_warning = f"""
> [!CAUTION]
> ⚠️ **ACTION REQUIRED: {unresolved_count} staged defect(s) have an UNRESOLVED class!**  
> Staged defects with an unresolved class are marked with ⚠️. You must expand the **Defect Editor** below to select and assign a valid class for each one. Unresolved defects will be skipped and NOT written to the dataset.
"""

    prompt_md = f"""### 📋 Review & Confirmation
{unresolved_warning}
You have finished reviewing candidate detections from this inspection.

> ❓ **Do you want to edit what you have done?**
> - **Total Staged Defects**: **{len(staged)}**
> - **Total Skipped**: **{skipped}**
> 
> Inspect the selected defect crops and their class distribution below. If you want to **edit a specific defect image** (adjust bounding box margins/coordinates or change class) or **remove it**, expand the **Defect Editor** below.
>
> When you are satisfied with all defects, click **💾 Confirm & Add to RDD_Ahmedabad Dataset** to permanently commit them!
"""

    class_counts = collections.Counter(d["class_name"] for d in staged)
    dist_rows = "\n".join([f"| **{c_name}** | `{count}` crops |" for c_name, count in sorted(class_counts.items())])
    dist_md = f"""#### 📊 Staged Defect Distribution ({len(staged)} Total)

| Defect Class | Staged Count |
|---|---|
{dist_rows}
"""

    gallery_items = []
    choices = []
    for d in staged:
        crop_rgb = None
        if d.get("crop_image") is not None and d["crop_image"].size > 0:
            crop_rgb = cv2.cvtColor(d["crop_image"], cv2.COLOR_BGR2RGB)
        is_unres = d.get("class_id", -1) < 0
        tag = "⚠️ UNRESOLVED: " if is_unres else ""
        caption = f"#{d['id']}: {tag}{d['class_name']} (Frame {d['frame_number']}, Track {d['track_id']}, Conf: {d['confidence']:.2f})"
        if crop_rgb is not None:
            gallery_items.append((crop_rgb, caption))
        choices.append(f"#{d['id']} — {tag}{d['class_name']} (Frame {d['frame_number']}, Track {d['track_id']})")

    return prompt_md, dist_md, gallery_items, choices


def _get_staged_defect_by_id(staged_id):
    staged = _ANNOTATION_STATE.get("staged_defects", [])
    for d in staged:
        if d["id"] == staged_id:
            return d
    return None


def _get_staged_id_from_choice_label(label):
    if not label or not str(label).startswith("#"):
        return None
    try:
        id_str = str(label).split("—")[0].replace("#", "").strip()
        return int(id_str)
    except Exception:
        return None


def _get_staged_overlays_for_frame(source_video, frame_number, exclude_id=None):
    """
    Find already-staged defect bounding boxes matching BOTH source_video AND frame_number.
    """
    staged = _ANNOTATION_STATE.get("staged_defects", [])
    overlays = []
    for d in staged:
        if exclude_id is not None and d.get("id") == exclude_id:
            continue
        if d.get("source_video") == source_video and d.get("frame_number") == frame_number:
            overlays.append({
                "bbox": [int(v) for v in d["bbox"]],
                "label": f"#{d['id']} {d['class_name']}"
            })
    return overlays


def select_staged_defect_for_edit(selected_label):
    """
    Loads the chosen staged defect into the editor canvas and controls.
    """
    staged_id = _get_staged_id_from_choice_label(selected_label)
    if staged_id is None:
        placeholder = render_bbox_editor_html(None, (0, 0, 0, 0), elem_prefix="ds_edit_bbox_")
        return None, placeholder, gr.Dropdown(), 0, 0, 0, 0, 0, "Select a staged defect above to edit."

    d = _get_staged_defect_by_id(staged_id)
    if d is None:
        placeholder = render_bbox_editor_html(None, (0, 0, 0, 0), elem_prefix="ds_edit_bbox_")
        return None, placeholder, gr.Dropdown(), 0, 0, 0, 0, 0, "Defect not found."

    _ANNOTATION_STATE["editing_staged_id"] = staged_id

    frame_bgr = d.get("original_frame")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr is not None else None
    x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
    overlays = _get_staged_overlays_for_frame(d.get("source_video"), d.get("frame_number"), exclude_id=staged_id)
    html = render_bbox_editor_html(frame_rgb, d["bbox"], canvas_id="edit_bbox_canvas", elem_prefix="ds_edit_bbox_", staged_overlays=overlays)

    crop_rgb = None
    if d.get("crop_image") is not None and d["crop_image"].size > 0:
        crop_rgb = cv2.cvtColor(d["crop_image"], cv2.COLOR_BGR2RGB)

    valid_classes = RDD_CLASS_CONFIG.get_class_names() if RDD_CLASS_CONFIG else []
    dropdown_val = d["class_name"] if d["class_name"] in valid_classes else None
    if d.get("class_id", -1) < 0 or d["class_name"] not in valid_classes:
        status_msg = f"⚠️ Editing **Defect #{d['id']}**: UNRESOLVED (`{d['class_name']}`) — Please select a valid class from the dropdown below!"
    else:
        status_msg = f"Editing **Defect #{d['id']}**: `{d['class_name']}` (Frame {d['frame_number']}, Track {d['track_id']})"
    return crop_rgb, html, gr.Dropdown(value=dropdown_val), 0, x1, y1, x2, y2, status_msg


def adjust_staged_bbox_padding(padding_px):
    """
    Expand/contract the bounding box of the staged defect currently being edited.
    """
    staged_id = _ANNOTATION_STATE.get("editing_staged_id")
    d = _get_staged_defect_by_id(staged_id)
    if d is None or d.get("original_frame") is None:
        return None, render_bbox_editor_html(None, (0, 0, 0, 0), elem_prefix="ds_edit_bbox_"), 0, 0, 0, 0

    frame_bgr = d["original_frame"]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    ix1, iy1, ix2, iy2 = d["initial_bbox"]
    pad = int(padding_px)

    nx1 = max(0, ix1 - pad)
    ny1 = max(0, iy1 - pad)
    nx2 = min(w, ix2 + pad)
    ny2 = min(h, iy2 + pad)

    if nx2 <= nx1 or ny2 <= ny1:
        return None, render_bbox_editor_html(None, (0, 0, 0, 0), elem_prefix="ds_edit_bbox_"), 0, 0, 0, 0

    d["bbox"] = (nx1, ny1, nx2, ny2)
    d["crop_image"] = frame_bgr[ny1:ny2, nx1:nx2].copy()

    crop_rgb = cv2.cvtColor(d["crop_image"], cv2.COLOR_BGR2RGB)
    overlays = _get_staged_overlays_for_frame(d.get("source_video"), d.get("frame_number"), exclude_id=staged_id)
    html = render_bbox_editor_html(frame_rgb, d["bbox"], canvas_id="edit_bbox_canvas", elem_prefix="ds_edit_bbox_", staged_overlays=overlays)
    return crop_rgb, html, nx1, ny1, nx2, ny2


def apply_staged_bbox_coords(x1, y1, x2, y2):
    """
    Apply fine-tuned bounding box coordinates to the staged defect currently being edited.
    """
    staged_id = _ANNOTATION_STATE.get("editing_staged_id")
    d = _get_staged_defect_by_id(staged_id)
    if d is None or d.get("original_frame") is None:
        return None, render_bbox_editor_html(None, (0, 0, 0, 0), elem_prefix="ds_edit_bbox_")

    frame_bgr = d["original_frame"]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    nx1 = max(0, int(x1))
    ny1 = max(0, int(y1))
    nx2 = min(w, int(x2))
    ny2 = min(h, int(y2))

    if nx2 <= nx1 or ny2 <= ny1:
        return None, render_bbox_editor_html(None, (0, 0, 0, 0), elem_prefix="ds_edit_bbox_")

    d["bbox"] = (nx1, ny1, nx2, ny2)
    d["crop_image"] = frame_bgr[ny1:ny2, nx1:nx2].copy()

    crop_rgb = cv2.cvtColor(d["crop_image"], cv2.COLOR_BGR2RGB)
    overlays = _get_staged_overlays_for_frame(d.get("source_video"), d.get("frame_number"), exclude_id=staged_id)
    html = render_bbox_editor_html(frame_rgb, d["bbox"], canvas_id="edit_bbox_canvas", elem_prefix="ds_edit_bbox_", staged_overlays=overlays)
    return crop_rgb, html


def reset_staged_bbox_to_original():
    """
    Reset bounding box for the staged defect currently being edited.
    """
    staged_id = _ANNOTATION_STATE.get("editing_staged_id")
    d = _get_staged_defect_by_id(staged_id)
    if d is None or d.get("original_frame") is None:
        return None, render_bbox_editor_html(None, (0, 0, 0, 0), elem_prefix="ds_edit_bbox_"), 0, 0, 0, 0, 0

    frame_bgr = d["original_frame"]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ix1, iy1, ix2, iy2 = d["initial_bbox"]
    d["bbox"] = d["initial_bbox"]
    d["crop_image"] = frame_bgr[iy1:iy2, ix1:ix2].copy()

    crop_rgb = cv2.cvtColor(d["crop_image"], cv2.COLOR_BGR2RGB)
    overlays = _get_staged_overlays_for_frame(d.get("source_video"), d.get("frame_number"), exclude_id=staged_id)
    html = render_bbox_editor_html(frame_rgb, d["bbox"], canvas_id="edit_bbox_canvas", elem_prefix="ds_edit_bbox_", staged_overlays=overlays)
    return crop_rgb, html, ix1, iy1, ix2, iy2, 0


def save_staged_defect_changes(chosen_class, bbox=None):
    """
    Save class and crop modifications for the currently edited staged defect.
    Updates the review prompt, distribution breakdown, gallery, and choices.
    """
    staged_id = _ANNOTATION_STATE.get("editing_staged_id")
    d = _get_staged_defect_by_id(staged_id)
    if d is None:
        prompt_md, dist_md, gallery_items, choices = get_staged_review_data()
        return prompt_md, dist_md, gallery_items, gr.Dropdown(choices=choices), "⚠️ No defect selected to save."

    final_bbox = _resolve_bbox(d, explicit_bbox=bbox)
    d["bbox"] = final_bbox

    if d.get("original_frame") is not None:
        x1, y1, x2, y2 = final_bbox
        h, w = d["original_frame"].shape[:2]
        d["crop_image"] = d["original_frame"][max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()

    raw_choice = chosen_class.strip() if chosen_class else ""
    mapped_choice = RDD_CLASS_CONFIG.map_model_class(raw_choice) if (RDD_CLASS_CONFIG and raw_choice) else raw_choice
    class_id = RDD_CLASS_CONFIG.get_class_id(mapped_choice) if (RDD_CLASS_CONFIG and mapped_choice) else -1
    if class_id >= 0 and RDD_CLASS_CONFIG:
        class_name = RDD_CLASS_CONFIG.get_class_name(class_id)
    else:
        # DO NOT DEFAULT TO POTHOLE! Keep class_id = -1 and raw unresolved string
        class_id = -1
        class_name = raw_choice.upper() if raw_choice else (d.get("class_name") or "UNRESOLVED")

    d["class_name"] = class_name
    d["class_id"] = class_id

    prompt_md, dist_md, gallery_items, choices = get_staged_review_data()
    cur_label = next((c for c in choices if c.startswith(f"#{d['id']} ")), choices[0] if choices else None)
    if class_id < 0:
        return prompt_md, dist_md, gallery_items, gr.Dropdown(choices=choices, value=cur_label), f"⚠️ Saved changes for Defect #{d['id']}, but class `{d['class_name']}` is still UNRESOLVED! Please select a valid class from the dropdown."
    return prompt_md, dist_md, gallery_items, gr.Dropdown(choices=choices, value=cur_label), f"✅ Saved changes for Defect #{d['id']} (`{d['class_name']}`)!"


def delete_staged_defect():
    """
    Remove the currently edited defect from the staged batch.
    """
    staged_id = _ANNOTATION_STATE.get("editing_staged_id")
    staged = _ANNOTATION_STATE.get("staged_defects", [])
    _ANNOTATION_STATE["staged_defects"] = [d for d in staged if d["id"] != staged_id]
    _ANNOTATION_STATE["editing_staged_id"] = None
    _ANNOTATION_STATE["accepted_count"] = len(_ANNOTATION_STATE["staged_defects"])

    prompt_md, dist_md, gallery_items, choices = get_staged_review_data()
    placeholder_html = render_bbox_editor_html(None, (0, 0, 0, 0), elem_prefix="ds_edit_bbox_")
    new_val = choices[0] if choices else None
    return (
        prompt_md, dist_md, gallery_items, gr.Dropdown(choices=choices, value=new_val),
        None, placeholder_html, gr.Dropdown(value=None), 0, 0, 0, 0, 0,
        f"🗑️ Removed Defect #{staged_id} from batch."
    )


def commit_staged_defects_to_dataset():
    """
    Permanently commit all staged defects into the RDD_Ahmedabad dataset.
    This writes original frames, labels, crops, data.yaml, and metadata.
    """
    staged = _ANNOTATION_STATE.get("staged_defects", [])
    if not staged:
        return "⚠️ No staged defects to save. Please review candidates and accept defects first.", get_dataset_stats_md(), gr.update(visible=True)

    if not RDD_CLASS_CONFIG or not RDD_DATASET_MANAGER or not RDD_ANNOTATION_MANAGER:
        return "❌ Dataset manager system not initialized.", get_dataset_stats_md(), gr.update(visible=True)

    saved_count = 0
    class_saved_counts = collections.Counter()
    split_saved_counts = collections.Counter()
    skipped_unresolved = []

    valid_classes = set(RDD_CLASS_CONFIG.get_class_names())

    for d in staged:
        cid = d.get("class_id", -1)
        cname = d.get("class_name", "")

        # Guard: skip defects with class_id == -1 or unmapped class name
        if cid < 0 or cname not in valid_classes:
            skipped_unresolved.append(f"Defect #{d.get('id', '?')} (`{cname}`)")
            continue

        try:
            saved = RDD_DATASET_MANAGER.add_annotation(
                frame=d["original_frame"],
                bbox=d["bbox"],
                class_id=d["class_id"],
                class_name=d["class_name"],
                source_video=d["source_video"],
                frame_number=d["frame_number"],
                track_id=d["track_id"],
                confidence=d["confidence"],
                crop_image=d["crop_image"],
            )
            x1, y1, x2, y2 = d["bbox"]
            RDD_ANNOTATION_MANAGER.add_record(
                image_name=saved.get("image_filename", ""),
                crop_name=saved.get("crop_filename", ""),
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                source_video=d["source_video"],
                frame_number=d["frame_number"],
                track_id=d["track_id"],
                bbox_x=x1,
                bbox_y=y1,
                bbox_width=x2 - x1,
                bbox_height=y2 - y1,
                split=saved.get("split", "train"),
            )
            saved_count += 1
            class_saved_counts[d["class_name"]] += 1
            split_saved_counts[saved.get("split", "train")] += 1
        except Exception as e:
            print(f"[Dataset Commit Error] Failed to save defect #{d['id']}: {e}")

    # Auto-update data.yaml if any defects were saved
    if saved_count > 0:
        try:
            RDD_DATASET_MANAGER.update_data_yaml()
        except Exception as e:
            print(f"[Dataset Commit Error] update_data_yaml failed: {e}")

    # Clear staged defects and candidates
    _ANNOTATION_STATE["staged_defects"] = []
    _ANNOTATION_STATE["editing_staged_id"] = None
    _ANNOTATION_STATE["candidates"] = []

    # Get updated dataset statistics
    stats_md = get_dataset_stats_md()

    skipped_md = ""
    if skipped_unresolved:
        skipped_md = f"""
> [!WARNING]
> ⚠️ **{len(skipped_unresolved)} Staged Defect(s) Skipped (Unresolved Class):**  
> {', '.join(skipped_unresolved)}  
> These defects were not saved to disk because no valid RDD_Ahmedabad class was assigned.
"""

    if saved_count == 0:
        fail_md = f"""### ⚠️ No Defects Added to RDD_Ahmedabad Dataset

{skipped_md}

All staged defect(s) were skipped because they had unresolved classes. Please load detections again, assign valid classes, and commit.
"""
        return fail_md, stats_md, gr.update(visible=False)

    class_breakdown = "\n".join([f"- **{c}**: `{cnt}` crops" for c, cnt in sorted(class_saved_counts.items())])
    split_breakdown = ", ".join([f"**{sp}**: {cnt}" for sp, cnt in sorted(split_saved_counts.items())])

    success_md = f"""### 🎉 Successfully Added {saved_count} Defects to RDD_Ahmedabad Dataset!
{skipped_md}
All annotations, original frames, crop images, and YOLO labels have been confirmed and permanently saved.

| Metric | Detail |
|---|---|
| **Defects Committed** | **{saved_count}** |
| **Splits Assigned** | {split_breakdown} |

#### Saved Defect Breakdown:
{class_breakdown}

---
💾 Stored in `datasets/RDD_Ahmedabad/` (`images/`, `labels/`, `crops/`, `metadata/`).
"""
    return success_md, stats_md, gr.update(visible=False)


def discard_staged_batch():
    """Discard all staged defects without saving to dataset."""
    count = len(_ANNOTATION_STATE.get("staged_defects", []))
    _ANNOTATION_STATE["staged_defects"] = []
    _ANNOTATION_STATE["editing_staged_id"] = None
    _ANNOTATION_STATE["candidates"] = []
    _ANNOTATION_STATE["current_index"] = 0
    _ANNOTATION_STATE["skipped_count"] = 0
    return f"🗑️ Discarded {count} staged defects. No changes were made to the RDD_Ahmedabad dataset."


def get_dataset_stats_md():
    """Return dataset statistics as markdown."""
    if RDD_DATASET_MANAGER:
        return RDD_DATASET_MANAGER.get_stats_markdown()
    return "Dataset system not initialized."


# ═══════════════════════════════════════════════════════════════════════════════
# RDD_Ahmedabad Dataset Curation & Revival — State & Handlers
# ═══════════════════════════════════════════════════════════════════════════════

_CURATE_STATE = {
    "df": None,
    "current_pos": 0,
    "current_row": None,
    "current_frame_bgr": None,
    "current_frame_rgb": None,
    "current_bbox": (0, 0, 0, 0),
    "initial_bbox": (0, 0, 0, 0),
    "active_image_name": "",
    "active_split": "train",
    "active_class_name": "",
    "class_filter": "ALL",
    "split_filter": "ALL",
    "search_term": "",
    "status_msg": "",
}


def get_curate_class_choices():
    """Return dropdown choices for the class filter with counts from annotations.csv."""
    default_choices = [("ALL (All Classes)", "ALL")]
    if not RDD_ANNOTATION_MANAGER:
        return default_choices
    try:
        df = RDD_ANNOTATION_MANAGER.load_annotations()
        if df.empty or "class_name" not in df.columns:
            return default_choices
        counts = df["class_name"].value_counts()
        choices = [("ALL (All Classes)", "ALL")]
        for cname, cnt in counts.items():
            choices.append((f"{cname} ({cnt})", str(cname)))
        return choices
    except Exception:
        return default_choices


def load_curate_dataset(class_filter="ALL", split_filter="ALL", search_term=""):
    """
    Load annotations from annotations.csv and filter by class, split, and search keyword.
    Updates _CURATE_STATE and returns dropdown choices for jump selector.
    """
    if not RDD_ANNOTATION_MANAGER:
        _CURATE_STATE["df"] = pd.DataFrame()
        _CURATE_STATE["current_pos"] = 0
        return []

    df = RDD_ANNOTATION_MANAGER.load_annotations()
    if df.empty:
        _CURATE_STATE["df"] = df
        _CURATE_STATE["current_pos"] = 0
        return []

    if class_filter and class_filter != "ALL":
        df = df[df["class_name"] == class_filter]

    if split_filter and split_filter != "ALL":
        df = df[df["split"] == split_filter]

    if search_term and search_term.strip():
        term = search_term.strip().lower()
        mask = (
            df["image_name"].astype(str).str.lower().str.contains(term, na=False) |
            df["source_video"].astype(str).str.lower().str.contains(term, na=False) |
            df["class_name"].astype(str).str.lower().str.contains(term, na=False)
        )
        df = df[mask]

    df = df.reset_index(drop=True)
    _CURATE_STATE["df"] = df
    _CURATE_STATE["current_pos"] = 0
    _CURATE_STATE["class_filter"] = class_filter
    _CURATE_STATE["split_filter"] = split_filter
    _CURATE_STATE["search_term"] = search_term

    choices = []
    for i, r in df.iterrows():
        cname = r.get("class_name", "")
        img_name = str(r.get("image_name", ""))
        short_img = img_name[:24] + "..." if len(img_name) > 24 else img_name
        split = r.get("split", "")
        label = f"#{i+1}: [{cname}] {short_img} ({split})"
        choices.append((label, i))
    return choices


def get_curate_defect_display(pos=None):
    """
    Load and return UI display elements for the defect at position pos in the filtered dataset.
    Returns:
    (crop_rgb, canvas_html, info_md, chosen_class, x1, y1, x2, y2, padding, pos_badge, jump_val, status_md)
    """
    df = _CURATE_STATE.get("df")
    if df is None:
        load_curate_dataset()
        df = _CURATE_STATE.get("df")

    placeholder_html = render_bbox_editor_html(None, (0, 0, 0, 0), canvas_id="curate_canvas", elem_prefix="ds_curate_bbox_")
    rdd_classes = RDD_CLASS_CONFIG.get_class_names() if RDD_CLASS_CONFIG else ["POTHOLE"]
    default_class = rdd_classes[0] if rdd_classes else "POTHOLE"

    if df is None or df.empty:
        empty_info = """
<div class="ds-candidate-card" style="padding:20px;text-align:center;color:#94a3b8;">
  No defect records matched the current filter/search criteria.
</div>
"""
        return None, placeholder_html, empty_info, default_class, 0, 0, 0, 0, 0, "No defects available", None, ""

    if pos is not None:
        try:
            pos = int(pos)
        except (ValueError, TypeError):
            pos = 0
    else:
        pos = _CURATE_STATE.get("current_pos", 0)

    pos = max(0, min(len(df) - 1, pos))
    _CURATE_STATE["current_pos"] = pos

    row = df.iloc[pos]
    _CURATE_STATE["current_row"] = row.to_dict()

    image_name = str(row.get("image_name", ""))
    split = str(row.get("split", "train"))
    class_name = str(row.get("class_name", default_class))
    crop_name = str(row.get("crop_name", f"{os.path.splitext(image_name)[0]}_crop.jpg"))

    try:
        x = int(row.get("bbox_x", 0))
        y = int(row.get("bbox_y", 0))
        w = int(row.get("bbox_width", 0))
        h = int(row.get("bbox_height", 0))
    except (ValueError, TypeError):
        x, y, w, h = 0, 0, 100, 100

    bbox = (x, y, x + w, y + h)
    _CURATE_STATE["current_bbox"] = bbox
    _CURATE_STATE["initial_bbox"] = bbox
    _CURATE_STATE["active_image_name"] = image_name
    _CURATE_STATE["active_split"] = split
    _CURATE_STATE["active_class_name"] = class_name

    # Load original frame
    image_path = os.path.join(DATASET_DIR, "images", split, image_name)
    frame_bgr = None
    if os.path.exists(image_path):
        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            try:
                with open(image_path, "rb") as f:
                    buf = np.frombuffer(f.read(), dtype=np.uint8)
                    frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                pass

    crop_rgb = None
    canvas_html = placeholder_html

    if frame_bgr is not None and frame_bgr.size > 0:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _CURATE_STATE["current_frame_bgr"] = frame_bgr
        _CURATE_STATE["current_frame_rgb"] = frame_rgb

        img_h, img_w = frame_bgr.shape[:2]
        cx1 = max(0, min(img_w, x))
        cy1 = max(0, min(img_h, y))
        cx2 = max(0, min(img_w, x + w))
        cy2 = max(0, min(img_h, y + h))

        if cx2 > cx1 and cy2 > cy1:
            crop_bgr = frame_bgr[cy1:cy2, cx1:cx2].copy()
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        canvas_html = render_bbox_editor_html(
            frame_rgb,
            bbox,
            canvas_id="curate_canvas",
            elem_prefix="ds_curate_bbox_"
        )
    else:
        # Fallback to existing saved crop if original frame is missing
        crop_path = os.path.join(DATASET_DIR, "crops", class_name, crop_name)
        if os.path.exists(crop_path):
            existing_crop = cv2.imread(crop_path)
            if existing_crop is not None and existing_crop.size > 0:
                crop_rgb = cv2.cvtColor(existing_crop, cv2.COLOR_BGR2RGB)

    conf_val = row.get("confidence")
    conf_str = f"{float(conf_val):.1%}" if pd.notnull(conf_val) and str(conf_val).strip() != "" else "N/A"
    frame_num = str(row.get("frame_number", "N/A"))
    track_id = str(row.get("track_id", "N/A"))
    source_video = str(row.get("source_video", "N/A"))

    info_md = f"""
<div class="ds-candidate-card" style="border-left: 3px solid #a855f7;">
  <div class="ds-card-row">
    <span style="font-weight:700;color:#f1f5f9;font-size:0.92rem;">Defect #{pos + 1} of {len(df)}</span>
    <span class="pill pill-active" style="padding:2px 8px;font-size:0.75rem;">{split.upper()}</span>
  </div>
  <div class="ds-meta-grid" style="margin-top:6px;">
    <div class="ds-meta-item"><span class="ds-meta-lbl">Class:</span> <span class="ds-meta-val"><code style="color:#c084fc;font-weight:700;">{class_name}</code></span></div>
    <div class="ds-meta-item"><span class="ds-meta-lbl">Confidence:</span> <span class="ds-meta-val">{conf_str}</span></div>
    <div class="ds-meta-item"><span class="ds-meta-lbl">Source Video:</span> <span class="ds-meta-val" title="{source_video}"><code>{source_video[:22]}</code></span></div>
    <div class="ds-meta-item"><span class="ds-meta-lbl">Dimensions:</span> <span class="ds-meta-val"><code>{w}x{h} px</code></span></div>
  </div>
  <details class="ds-details-accordion" style="margin-top:6px;">
    <summary>📋 Provenance & File Names</summary>
    <div class="ds-details-content">
      <div style="grid-column: span 2;"><b>Image:</b> <code>{image_name}</code></div>
      <div><b>Frame:</b> <code>{frame_num}</code></div>
      <div><b>Track:</b> <code>{track_id}</code></div>
      <div style="grid-column: span 2;"><b>Coordinates:</b> <code>[{x}, {y}, {x+w}, {y+h}]</code></div>
    </div>
  </details>
</div>
"""
    pos_badge = f"📍 Defect **{pos + 1}** of **{len(df)}** · Filtered Total: **{len(df)}**"
    status_md = _CURATE_STATE.get("status_msg", "")
    _CURATE_STATE["status_msg"] = ""

    return (
        crop_rgb, canvas_html, info_md, class_name,
        x, y, x + w, y + h, 0, pos_badge, pos, status_md
    )


def curate_adjust_padding(padding_px):
    """Expand/shrink the current defect bbox on the canvas by padding_px pixels."""
    frame_bgr = _CURATE_STATE.get("current_frame_bgr")
    frame_rgb = _CURATE_STATE.get("current_frame_rgb")
    if frame_bgr is None or frame_rgb is None:
        return None, render_bbox_editor_html(None, (0, 0, 0, 0)), 0, 0, 0, 0

    h, w = frame_bgr.shape[:2]
    ix1, iy1, ix2, iy2 = _CURATE_STATE.get("initial_bbox", (0, 0, 0, 0))
    pad = int(padding_px)

    nx1 = max(0, ix1 - pad)
    ny1 = max(0, iy1 - pad)
    nx2 = min(w, ix2 + pad)
    ny2 = min(h, iy2 + pad)

    if nx2 <= nx1 or ny2 <= ny1:
        return None, render_bbox_editor_html(frame_rgb, (ix1, iy1, ix2, iy2), canvas_id="curate_canvas", elem_prefix="ds_curate_bbox_"), ix1, iy1, ix2, iy2

    new_bbox = (nx1, ny1, nx2, ny2)
    _CURATE_STATE["current_bbox"] = new_bbox

    crop_bgr = frame_bgr[ny1:ny2, nx1:nx2].copy()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr.size > 0 else None
    canvas_html = render_bbox_editor_html(frame_rgb, new_bbox, canvas_id="curate_canvas", elem_prefix="ds_curate_bbox_")
    return crop_rgb, canvas_html, nx1, ny1, nx2, ny2


def curate_apply_coords(x1, y1, x2, y2):
    """Apply manual or canvas-dragged coordinates to live defect crop and canvas."""
    frame_bgr = _CURATE_STATE.get("current_frame_bgr")
    frame_rgb = _CURATE_STATE.get("current_frame_rgb")
    if frame_bgr is None or frame_rgb is None:
        return None, render_bbox_editor_html(None, (0, 0, 0, 0))

    h, w = frame_bgr.shape[:2]
    try:
        nx1 = max(0, min(w, int(x1)))
        ny1 = max(0, min(h, int(y1)))
        nx2 = max(0, min(w, int(x2)))
        ny2 = max(0, min(h, int(y2)))
    except (ValueError, TypeError):
        return None, render_bbox_editor_html(frame_rgb, (0, 0, 0, 0), canvas_id="curate_canvas", elem_prefix="ds_curate_bbox_")

    if nx2 <= nx1 or ny2 <= ny1:
        return None, render_bbox_editor_html(frame_rgb, _CURATE_STATE.get("current_bbox", (0, 0, 0, 0)), canvas_id="curate_canvas", elem_prefix="ds_curate_bbox_")

    new_bbox = (nx1, ny1, nx2, ny2)
    _CURATE_STATE["current_bbox"] = new_bbox

    crop_bgr = frame_bgr[ny1:ny2, nx1:nx2].copy()
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr.size > 0 else None
    canvas_html = render_bbox_editor_html(frame_rgb, new_bbox, canvas_id="curate_canvas", elem_prefix="ds_curate_bbox_")
    return crop_rgb, canvas_html


def curate_reset_bbox():
    """Reset curate bbox to initial saved coordinates."""
    ibbox = _CURATE_STATE.get("initial_bbox", (0, 0, 0, 0))
    _CURATE_STATE["current_bbox"] = ibbox
    x1, y1, x2, y2 = ibbox
    frame_bgr = _CURATE_STATE.get("current_frame_bgr")
    frame_rgb = _CURATE_STATE.get("current_frame_rgb")
    if frame_bgr is None or frame_rgb is None:
        return None, render_bbox_editor_html(None, (0, 0, 0, 0)), x1, y1, x2, y2, 0

    crop_bgr = frame_bgr[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr is not None and crop_bgr.size > 0 else None
    canvas_html = render_bbox_editor_html(frame_rgb, ibbox, canvas_id="curate_canvas", elem_prefix="ds_curate_bbox_")
    return crop_rgb, canvas_html, x1, y1, x2, y2, 0


def curate_save_changes(chosen_class, x1, y1, x2, y2):
    """
    Save modifications for the active defect: updates crop file, YOLO label,
    annotations.csv, and dataset stats.
    """
    row = _CURATE_STATE.get("current_row")
    if not row or not RDD_DATASET_MANAGER or not RDD_ANNOTATION_MANAGER:
        return (
            None, render_bbox_editor_html(None, (0, 0, 0, 0)), "",
            chosen_class, 0, 0, 0, 0, 0, "", None,
            "❌ Dataset manager not initialized.",
            get_dataset_stats_md()
        )

    image_name = row.get("image_name")
    split = row.get("split", "train")
    old_class_name = row.get("class_name", "")
    crop_name = row.get("crop_name", f"{os.path.splitext(image_name)[0]}_crop.jpg")

    new_class_name = chosen_class.strip() if chosen_class else old_class_name
    new_class_id = RDD_CLASS_CONFIG.get_class_id(new_class_name) if RDD_CLASS_CONFIG else 0

    try:
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = max(0, int(x2))
        y2 = max(0, int(y2))
    except (ValueError, TypeError):
        x1, y1, x2, y2 = _CURATE_STATE.get("current_bbox", (0, 0, 100, 100))

    new_bbox = (x1, y1, x2, y2)
    bbox_w = x2 - x1
    bbox_h = y2 - y1

    # Extract crop from current frame
    frame_bgr = _CURATE_STATE.get("current_frame_bgr")
    new_crop_bgr = None
    if frame_bgr is not None and frame_bgr.size > 0 and bbox_h > 0 and bbox_w > 0:
        new_crop_bgr = frame_bgr[y1:y2, x1:x2].copy()

    # 1. Update on disk (crop + YOLO txt)
    try:
        RDD_DATASET_MANAGER.update_defect(
            image_name=image_name,
            split=split,
            old_class_name=old_class_name,
            new_class_id=new_class_id,
            new_class_name=new_class_name,
            bbox=new_bbox,
            crop_image=new_crop_bgr,
            crop_name=crop_name
        )
    except Exception as e:
        print(f"[Curate Error] update_defect failed: {e}")

    # 2. Update in annotations.csv
    try:
        RDD_ANNOTATION_MANAGER.update_record(
            image_name=image_name,
            class_id=new_class_id,
            class_name=new_class_name,
            bbox_x=x1,
            bbox_y=y1,
            bbox_width=bbox_w,
            bbox_height=bbox_h
        )
    except Exception as e:
        print(f"[Curate Error] update_record failed: {e}")

    # 3. Update in-memory state
    pos = _CURATE_STATE.get("current_pos", 0)
    df = _CURATE_STATE.get("df")
    if df is not None and pos < len(df):
        df.loc[pos, "class_name"] = new_class_name
        df.loc[pos, "class_id"] = new_class_id
        df.loc[pos, "bbox_x"] = x1
        df.loc[pos, "bbox_y"] = y1
        df.loc[pos, "bbox_width"] = bbox_w
        df.loc[pos, "bbox_height"] = bbox_h
        _CURATE_STATE["df"] = df

    _CURATE_STATE["initial_bbox"] = new_bbox
    _CURATE_STATE["current_bbox"] = new_bbox
    _CURATE_STATE["active_class_name"] = new_class_name
    _CURATE_STATE["status_msg"] = f"✅ **Saved Defect #{pos + 1}!** Updated class: `{new_class_name}`, Crop: `{crop_name}`, Box: `[{x1}, {y1}, {x2}, {y2}]` ({bbox_w}x{bbox_h}px)."

    disp = get_curate_defect_display(pos)
    stats_md = get_dataset_stats_md()
    return (*disp, stats_md)


def curate_delete_defect():
    """
    Permanently delete the current defect from RDD_Ahmedabad dataset
    (removes crop, YOLO label, frame image, and metadata record).
    """
    row = _CURATE_STATE.get("current_row")
    if not row or not RDD_DATASET_MANAGER or not RDD_ANNOTATION_MANAGER:
        return (
            None, render_bbox_editor_html(None, (0, 0, 0, 0)), "",
            "POTHOLE", 0, 0, 0, 0, 0, "", None,
            "❌ Dataset manager not initialized.",
            get_dataset_stats_md()
        )

    image_name = row.get("image_name")
    split = row.get("split", "train")
    class_name = row.get("class_name", "")
    crop_name = row.get("crop_name", f"{os.path.splitext(image_name)[0]}_crop.jpg")
    pos = _CURATE_STATE.get("current_pos", 0)

    # 1. Delete from disk
    try:
        RDD_DATASET_MANAGER.delete_defect(
            image_name=image_name,
            split=split,
            class_name=class_name,
            crop_name=crop_name
        )
    except Exception as e:
        print(f"[Curate Delete Error] delete_defect failed: {e}")

    # 2. Delete from metadata
    try:
        RDD_ANNOTATION_MANAGER.delete_record(image_name=image_name)
    except Exception as e:
        print(f"[Curate Delete Error] delete_record failed: {e}")

    # 3. Update in-memory dataframe
    df = _CURATE_STATE.get("df")
    if df is not None and not df.empty and pos < len(df):
        df = df.drop(index=pos).reset_index(drop=True)
        _CURATE_STATE["df"] = df
        if pos >= len(df):
            pos = max(0, len(df) - 1)
        _CURATE_STATE["current_pos"] = pos

    _CURATE_STATE["status_msg"] = f"🗑️ **Deleted Defect!** Removed `{image_name}` from `{split}` split and deleted crop `{crop_name}`."
    disp = get_curate_defect_display(pos)
    stats_md = get_dataset_stats_md()
    return (*disp, stats_md)


# ═══════════════════════════════════════════════════════════════════════════════
# Crop Box Adjustment — Backend Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_current_candidate():
    """Return the current CandidateAnnotation or None."""
    candidates = _ANNOTATION_STATE.get("candidates", [])
    idx = _ANNOTATION_STATE.get("current_index", 0)
    if not candidates or idx >= len(candidates):
        return None
    return candidates[idx]


def draw_bbox_on_frame(frame_rgb, bbox, color=(0, 255, 0), thickness=3):
    """
    Draw a bounding box overlay on a copy of the frame (RGB format).
    Returns the annotated frame image (RGB, numpy array).
    """
    if frame_rgb is None:
        return None
    annotated = frame_rgb.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    # Draw corner markers for better visibility
    corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    corner_color = (255, 165, 0)  # Orange corners
    # Top-left
    cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), corner_color, thickness + 1)
    cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), corner_color, thickness + 1)
    # Top-right
    cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), corner_color, thickness + 1)
    cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), corner_color, thickness + 1)
    # Bottom-left
    cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), corner_color, thickness + 1)
    cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), corner_color, thickness + 1)
    # Bottom-right
    cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), corner_color, thickness + 1)
    cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), corner_color, thickness + 1)
    # Label
    label = f"({x1},{y1}) - ({x2},{y2})"
    cv2.putText(annotated, label, (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def _get_frame_rgb_for_candidate(candidate):
    """Get the RGB frame for a candidate, fetching from video if needed."""
    if candidate is None:
        return None
    frame_bgr = candidate.original_frame
    if frame_bgr is None:
        # Try to fetch from video
        video_path = _ANNOTATION_STATE.get("last_video_path")
        if video_path and os.path.exists(video_path) and candidate.frame_number > 0:
            cap = None
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_number - 1)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        candidate.original_frame = frame.copy()
                        frame_bgr = frame
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
    if frame_bgr is None:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _encode_frame_base64(frame_rgb):
    """Encode an RGB frame to base64 JPEG for HTML canvas embedding."""
    if frame_rgb is None:
        return None
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def render_bbox_editor_html(frame_rgb, bbox, canvas_id="bbox_canvas", elem_prefix="ds_bbox_", staged_overlays=None):
    """
    Render an interactive HTML5 canvas with draggable/resizable bounding box.
    Supports:
    - Drag corner handles or edge midpoints to resize
    - Drag inside box to move
    - Click and drag outside box (min 10px drag guard) to draw a brand-new bounding box
    - Overlay already-staged defects in the same frame as read-only dashed boxes with tags
    - Single DOM input sync mechanism (inp.value set + input/change dispatch)
    """
    if frame_rgb is None:
        return '<div style="text-align:center;color:#94a3b8;padding:40px;font-family:Inter,sans-serif;">No frame loaded. Click 📥 Load Detections to begin.</div>'

    b64_str = _encode_frame_base64(frame_rgb)
    if b64_str is None:
        return '<div style="text-align:center;color:#94a3b8;padding:40px;">Failed to encode frame.</div>'

    x1, y1, x2, y2 = [int(v) for v in bbox]
    staged_json = json.dumps(staged_overlays if staged_overlays else [])

    inner_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
  canvas {{ display:block; max-width:100%; border-radius:8px; border:1px solid rgba(255,255,255,0.08); }}
  .hint {{ text-align:center; padding:6px 0; font-size:0.78rem; color:#94a3b8; font-family:Inter,sans-serif; }}
</style>
</head><body>
<canvas id="cv"></canvas>
<div class="hint">🔷 Drag corners/edges to resize &nbsp;·&nbsp; Drag inside box to move &nbsp;·&nbsp; Click &amp; drag outside (min 10px) to draw new box</div>
<script>
(function() {{
  const C = document.getElementById('cv');
  if (!C) return;
  const ctx = C.getContext('2d');
  const img = new Image();
  img.src = 'data:image/jpeg;base64,{b64_str}';

  let box = {{ x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2} }};
  let origBox = {{ x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2} }};
  const staged = {staged_json};
  let drag = null;
  let isPotentialDraw = false;
  let drawStart = {{ x:0, y:0 }};
  let ds = {{ x:0, y:0 }};
  let sc = 1;

  function resizeIframe() {{
    try {{
      const h = document.documentElement.scrollHeight;
      if (window.frameElement) window.frameElement.style.height = h + 'px';
    }} catch(e) {{}}
  }}

  img.onload = function() {{
    const maxW = Math.min(780, (C.parentElement ? C.parentElement.clientWidth : 780) - 2);
    sc = maxW / img.width;
    C.width = Math.round(img.width * sc);
    C.height = Math.round(img.height * sc);
    paint();
    resizeIframe();
  }};

  function paint() {{
    ctx.clearRect(0,0,C.width,C.height);
    ctx.drawImage(img,0,0,C.width,C.height);

    // 1. Render already-staged defects in this frame as read-only dashed overlay boxes with labels
    if (Array.isArray(staged)) {{
      staged.forEach(function(item) {{
        if (!item || !item.bbox) return;
        const ox1 = item.bbox[0] * sc, oy1 = item.bbox[1] * sc;
        const ox2 = item.bbox[2] * sc, oy2 = item.bbox[3] * sc;
        const obw = ox2 - ox1, obh = oy2 - oy1;
        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.strokeRect(ox1, oy1, obw, obh);
        ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
        ctx.fillRect(ox1, oy1, obw, obh);
        const tag = item.label || 'Staged Defect';
        ctx.font = 'bold 10px monospace';
        const tagW = ctx.measureText(tag).width + 8;
        ctx.fillStyle = 'rgba(2, 132, 199, 0.9)';
        ctx.fillRect(ox1, Math.max(0, oy1 - 16), Math.max(tagW, 60), 16);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(tag, ox1 + 4, Math.max(11, oy1 - 4));
        ctx.restore();
      }});
    }}

    // 2. Dim area outside active box
    const sx1 = box.x1 * sc, sy1 = box.y1 * sc, sx2 = box.x2 * sc, sy2 = box.y2 * sc;
    const bw = sx2 - sx1, bh = sy2 - sy1;

    ctx.fillStyle = 'rgba(0,0,0,0.42)';
    ctx.fillRect(0, 0, C.width, sy1);
    ctx.fillRect(0, sy2, C.width, C.height - sy2);
    ctx.fillRect(0, sy1, sx1, bh);
    ctx.fillRect(sx2, sy1, C.width - sx2, bh);

    // 3. Active box border
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.strokeRect(sx1, sy1, bw, bh);

    // 4. Corner handles
    const hs = 10;
    ctx.fillStyle = '#38bdf8';
    [[sx1, sy1], [sx2, sy1], [sx1, sy2], [sx2, sy2]].forEach(function(p) {{
      ctx.fillRect(p[0] - hs / 2, p[1] - hs / 2, hs, hs);
    }});

    // 5. Edge midpoint handles
    ctx.fillStyle = '#0284c7';
    const emid = 7;
    [[sx1 + bw / 2, sy1], [sx1 + bw / 2, sy2], [sx1, sy1 + bh / 2], [sx2, sy1 + bh / 2]].forEach(function(p) {{
      ctx.beginPath();
      ctx.arc(p[0], p[1], emid / 2, 0, Math.PI * 2);
      ctx.fill();
    }});

    // 6. Coordinate label
    ctx.fillStyle = '#22c55e';
    ctx.font = 'bold 11px monospace';
    ctx.fillText('(' + Math.round(box.x1) + ',' + Math.round(box.y1) + ') - (' + Math.round(box.x2) + ',' + Math.round(box.y2) + ')', sx1, Math.max(sy1 - 6, 13));

    // 7. Size label
    const ww = Math.round(box.x2 - box.x1), hh = Math.round(box.y2 - box.y1);
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fillRect(sx1, sy2 + 4, 90, 18);
    ctx.fillStyle = '#f1f5f9';
    ctx.font = '11px monospace';
    ctx.fillText(ww + 'x' + hh + 'px', sx1 + 4, sy2 + 16);
  }}

  function mpos(e) {{
    const r = C.getBoundingClientRect();
    return {{ x: e.clientX - r.left, y: e.clientY - r.top }};
  }}

  function hit(p) {{
    const g = 14;
    const sx1 = box.x1 * sc, sy1 = box.y1 * sc, sx2 = box.x2 * sc, sy2 = box.y2 * sc;
    const mx = (sx1 + sx2) / 2, my = (sy1 + sy2) / 2;
    if (Math.abs(p.x - sx1) < g && Math.abs(p.y - sy1) < g) return 'tl';
    if (Math.abs(p.x - sx2) < g && Math.abs(p.y - sy1) < g) return 'tr';
    if (Math.abs(p.x - sx1) < g && Math.abs(p.y - sy2) < g) return 'bl';
    if (Math.abs(p.x - sx2) < g && Math.abs(p.y - sy2) < g) return 'br';
    if (Math.abs(p.x - mx) < g && Math.abs(p.y - sy1) < g) return 'mt';
    if (Math.abs(p.x - mx) < g && Math.abs(p.y - sy2) < g) return 'mb';
    if (Math.abs(p.x - sx1) < g && Math.abs(p.y - my) < g) return 'ml';
    if (Math.abs(p.x - sx2) < g && Math.abs(p.y - my) < g) return 'mr';
    if (p.x > sx1 && p.x < sx2 && p.y > sy1 && p.y < sy2) return 'mv';
    return null;
  }}

  const cursors = {{
    tl: 'nwse-resize', br: 'nwse-resize', tr: 'nesw-resize', bl: 'nesw-resize',
    mt: 'ns-resize', mb: 'ns-resize', ml: 'ew-resize', mr: 'ew-resize', mv: 'move'
  }};

  C.addEventListener('mousedown', function(e) {{
    const p = mpos(e);
    const h = hit(p);
    ds = p;
    origBox = {{ x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 }};

    if (h) {{
      drag = h;
      isPotentialDraw = false;
    }} else {{
      isPotentialDraw = true;
      drag = null;
      drawStart = {{ x: p.x, y: p.y }};
    }}
    e.preventDefault();
  }});

  C.addEventListener('mousemove', function(e) {{
    const p = mpos(e);
    if (!drag) {{
      if (isPotentialDraw) {{
        // Minimum drag distance (~10px) guard before committing to draw mode
        const dist = Math.hypot(p.x - drawStart.x, p.y - drawStart.y);
        if (dist >= 10) {{
          drag = 'draw';
        }}
      }} else {{
        const h = hit(p);
        C.style.cursor = h ? cursors[h] || 'crosshair' : 'crosshair';
        return;
      }}
    }}

    if (drag === 'draw') {{
      C.style.cursor = 'crosshair';
      const curX = Math.max(0, Math.min(p.x / sc, img.width));
      const curY = Math.max(0, Math.min(p.y / sc, img.height));
      const sx = Math.max(0, Math.min(drawStart.x / sc, img.width));
      const sy = Math.max(0, Math.min(drawStart.y / sc, img.height));
      box.x1 = Math.min(sx, curX);
      box.y1 = Math.min(sy, curY);
      box.x2 = Math.max(sx, curX);
      box.y2 = Math.max(sy, curY);
      paint();
      return;
    }}

    const dx = (p.x - ds.x) / sc, dy = (p.y - ds.y) / sc;
    if (drag === 'tl') {{ box.x1 += dx; box.y1 += dy; }}
    else if (drag === 'tr') {{ box.x2 += dx; box.y1 += dy; }}
    else if (drag === 'bl') {{ box.x1 += dx; box.y2 += dy; }}
    else if (drag === 'br') {{ box.x2 += dx; box.y2 += dy; }}
    else if (drag === 'mt') {{ box.y1 += dy; }}
    else if (drag === 'mb') {{ box.y2 += dy; }}
    else if (drag === 'ml') {{ box.x1 += dx; }}
    else if (drag === 'mr') {{ box.x2 += dx; }}
    else if (drag === 'mv') {{ box.x1 += dx; box.y1 += dy; box.x2 += dx; box.y2 += dy; }}

    box.x1 = Math.max(0, Math.min(box.x1, img.width));
    box.y1 = Math.max(0, Math.min(box.y1, img.height));
    box.x2 = Math.max(0, Math.min(box.x2, img.width));
    box.y2 = Math.max(0, Math.min(box.y2, img.height));
    ds = p;
    paint();
  }});

  function endDrag() {{
    if (isPotentialDraw && drag !== 'draw') {{
      // Near-miss click outside handles (<10px drag): guard preserves original box
      box = {{ x1: origBox.x1, y1: origBox.y1, x2: origBox.x2, y2: origBox.y2 }};
      isPotentialDraw = false;
      drag = null;
      paint();
      return;
    }}

    if (!drag) return;

    if (drag === 'draw') {{
      // Guard: Check if drawn box is at least 10x10 px
      if (Math.abs(box.x2 - box.x1) < 10 || Math.abs(box.y2 - box.y1) < 10) {{
        box = {{ x1: origBox.x1, y1: origBox.y1, x2: origBox.x2, y2: origBox.y2 }};
      }}
    }}

    drag = null;
    isPotentialDraw = false;

    if (box.x1 > box.x2) {{ let t = box.x1; box.x1 = box.x2; box.x2 = t; }}
    if (box.y1 > box.y2) {{ let t = box.y1; box.y1 = box.y2; box.y2 = t; }}

    paint();
    syncGradio();
  }}

  C.addEventListener('mouseup', endDrag);
  C.addEventListener('mouseleave', endDrag);

  function syncGradio() {{
    var pwin;
    try {{ pwin = window.parent || window; }} catch(e) {{ pwin = window; }}
    var pdoc;
    try {{ pdoc = pwin.document; }} catch(e) {{ return; }}
    if (!pdoc) return;

    function setN(eid, v) {{
      var el = pdoc.getElementById(eid);
      if (!el) return;
      var inp = el.querySelector('input');
      if (!inp) return;
      var val = Math.round(v);
      var s = Object.getOwnPropertyDescriptor(pwin.HTMLInputElement.prototype, 'value');
      if (s && s.set) {{ s.set.call(inp, val); }}
      else {{ inp.value = val; }}
      inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
      inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
    }}
    setN('{elem_prefix}x1', box.x1);
    setN('{elem_prefix}y1', box.y1);
    setN('{elem_prefix}x2', box.x2);
    setN('{elem_prefix}y2', box.y2);
  }}
}})();
</script>
</body></html>"""

    # Escape for safe embedding in srcdoc attribute (double-quotes → &quot;)
    escaped_html = inner_html.replace('&', '&amp;').replace('"', '&quot;')

    return f"""<iframe srcdoc="{escaped_html}" sandbox="allow-scripts allow-same-origin" style="width:100%;border:none;min-height:400px;overflow:hidden;border-radius:8px;" scrolling="no"></iframe>"""


def get_full_frame_with_overlay():
    """
    Get the current candidate's full source frame as interactive canvas HTML.
    Returns: (canvas_html, x1, y1, x2, y2) or (placeholder_html, 0, 0, 0, 0)
    """
    c = _get_current_candidate()
    if c is None:
        return render_bbox_editor_html(None, (0,0,0,0)), 0, 0, 0, 0
    frame_rgb = _get_frame_rgb_for_candidate(c)
    if frame_rgb is None:
        return render_bbox_editor_html(None, (0,0,0,0)), 0, 0, 0, 0
    x1, y1, x2, y2 = [int(v) for v in c.bbox]
    overlays = _get_staged_overlays_for_frame(c.source_video, c.frame_number)
    html = render_bbox_editor_html(frame_rgb, c.bbox, staged_overlays=overlays)
    return html, x1, y1, x2, y2


def adjust_bbox_with_padding(padding_px):
    """
    Expand the current candidate's bbox by padding_px pixels on all sides.
    Updates the candidate's bbox and crop_image in place.
    Returns: (crop_rgb, canvas_html, x1, y1, x2, y2)
    """
    c = _get_current_candidate()
    if c is None:
        return None, render_bbox_editor_html(None, (0,0,0,0)), 0, 0, 0, 0

    frame_rgb = _get_frame_rgb_for_candidate(c)
    if frame_rgb is None:
        return None, render_bbox_editor_html(None, (0,0,0,0)), 0, 0, 0, 0

    h, w = frame_rgb.shape[:2]
    ix1, iy1, ix2, iy2 = c.initial_bbox
    padding = int(padding_px)

    # Expand from initial bbox
    nx1 = max(0, ix1 - padding)
    ny1 = max(0, iy1 - padding)
    nx2 = min(w, ix2 + padding)
    ny2 = min(h, iy2 + padding)

    if nx2 <= nx1 or ny2 <= ny1:
        return None, render_bbox_editor_html(None, (0,0,0,0)), 0, 0, 0, 0

    new_bbox = (nx1, ny1, nx2, ny2)
    c.bbox = new_bbox

    # Re-extract crop from BGR frame
    frame_bgr = c.original_frame
    new_crop = frame_bgr[ny1:ny2, nx1:nx2].copy()
    c.crop_image = new_crop

    crop_rgb = cv2.cvtColor(new_crop, cv2.COLOR_BGR2RGB)
    overlays = _get_staged_overlays_for_frame(c.source_video, c.frame_number)
    html = render_bbox_editor_html(frame_rgb, new_bbox, staged_overlays=overlays)
    return crop_rgb, html, nx1, ny1, nx2, ny2


def apply_bbox_coord_adjustment(x1, y1, x2, y2):
    """
    Apply fine-tuned bbox coordinates to the current candidate.
    Returns: (crop_rgb, canvas_html)
    """
    c = _get_current_candidate()
    if c is None:
        return None, render_bbox_editor_html(None, (0,0,0,0))

    frame_rgb = _get_frame_rgb_for_candidate(c)
    if frame_rgb is None:
        return None, render_bbox_editor_html(None, (0,0,0,0))

    h, w = frame_rgb.shape[:2]
    nx1 = max(0, int(x1))
    ny1 = max(0, int(y1))
    nx2 = min(w, int(x2))
    ny2 = min(h, int(y2))

    if nx2 <= nx1 or ny2 <= ny1:
        return None, render_bbox_editor_html(None, (0,0,0,0))

    new_bbox = (nx1, ny1, nx2, ny2)
    c.bbox = new_bbox

    # Re-extract crop from BGR frame
    frame_bgr = c.original_frame
    new_crop = frame_bgr[ny1:ny2, nx1:nx2].copy()
    c.crop_image = new_crop

    crop_rgb = cv2.cvtColor(new_crop, cv2.COLOR_BGR2RGB)
    overlays = _get_staged_overlays_for_frame(c.source_video, c.frame_number)
    html = render_bbox_editor_html(frame_rgb, new_bbox, staged_overlays=overlays)
    return crop_rgb, html


def reset_bbox_to_original():
    """
    Reset the current candidate's bbox to its original detection coordinates.
    Returns: (crop_rgb, canvas_html, x1, y1, x2, y2, padding_value)
    """
    c = _get_current_candidate()
    if c is None:
        return None, render_bbox_editor_html(None, (0,0,0,0)), 0, 0, 0, 0, 0

    frame_rgb = _get_frame_rgb_for_candidate(c)
    if frame_rgb is None:
        return None, render_bbox_editor_html(None, (0,0,0,0)), 0, 0, 0, 0, 0

    ix1, iy1, ix2, iy2 = c.initial_bbox
    c.bbox = c.initial_bbox

    # Re-extract crop from BGR frame
    frame_bgr = c.original_frame
    new_crop = frame_bgr[iy1:iy2, ix1:ix2].copy()
    c.crop_image = new_crop

    crop_rgb = cv2.cvtColor(new_crop, cv2.COLOR_BGR2RGB)
    overlays = _get_staged_overlays_for_frame(c.source_video, c.frame_number)
    html = render_bbox_editor_html(frame_rgb, c.initial_bbox, staged_overlays=overlays)
    return crop_rgb, html, ix1, iy1, ix2, iy2, 0


# ═══════════════════════════════════════════════════════════════════════
# CLOUD GPU PROCESSING LOGIC
# ═══════════════════════════════════════════════════════════════════════

def test_cloud_connection(url):
    """Test the connection to the Cloud GPU Server."""
    if not url:
        return "<div class='sys-bar'><span class='sys-metric'><span class='label' style='color:#f87171;'>✗ Please enter a valid URL</span></span></div>"
    
    url = url.rstrip("/")
    try:
        r = requests.get(f"{url}/health", timeout=5)
        if r.status_code == 200:
            return "<div class='sys-bar'><span class='sys-metric'><span class='label' style='color:#34d399;'>✓ Connected to Cloud GPU</span></span></div>"
        else:
            return f"<div class='sys-bar'><span class='sys-metric'><span class='label' style='color:#f87171;'>✗ Unreachable ({r.status_code})</span></span></div>"
    except Exception as e:
        return f"<div class='sys-bar'><span class='sys-metric'><span class='label' style='color:#f87171;'>✗ Unreachable: {e}</span></span></div>"

def process_video_cloud(video_path, model_name, url, enable_segmentation=True, progress=gr.Progress()):
    """Sends video and model to Cloud GPU for processing."""
    if isinstance(video_path, dict):
        video_path = video_path.get("path") or video_path.get("name") or video_path.get("video", {}).get("path")
    elif hasattr(video_path, "path"):
        video_path = video_path.path
    if not video_path or not os.path.exists(video_path):
        raise gr.Error("Invalid or missing video input file.")
    if not url:
        raise gr.Error("Please enter a valid Notebook Server URL.")
        
    url = url.rstrip("/")
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise gr.Error(f"Model file {model_name} not found locally.")

    progress(0, desc="Initiating Cloud GPU connection...")
    
    # We do a streaming POST request so it doesn't time out, but we don't have true progress.
    # The progress UI in Gradio will just spin in indeterminate mode during the request.
    try:
        with open(video_path, "rb") as vf, open(model_path, "rb") as mf:
            files = {
                "video": (os.path.basename(video_path), vf, "video/mp4"),
                "model": (os.path.basename(model_path), mf, "application/octet-stream")
            }
            progress(0.05, desc="Uploading video and model to Cloud GPU (this may take a minute)...")
            
            # 1. Start the job
            data = {"enable_segmentation": str(enable_segmentation).lower()}
            r = requests.post(f"{url}/process", files=files, data=data)
            
            if r.status_code != 200:
                try:
                    err_msg = r.json().get("detail", r.text)
                except Exception:
                    err_msg = r.text
                
                if "<!DOCTYPE html>" in err_msg or "<html" in err_msg:
                    raise gr.Error(f"Cloud Server Error {r.status_code}: The remote Colab server is unreachable or crashed (Bad Gateway).")
                else:
                    raise gr.Error(f"Cloud Server Error: {err_msg}")
                    
            job_data = r.json()
            job_id = job_data.get("job_id")
            if not job_id:
                raise gr.Error("Cloud Server Error: Missing job_id in response.")

            # 2. Poll for completion
            progress(0.1, desc="Job started. Polling status...")
            while True:
                time.sleep(3)
                sr = requests.get(f"{url}/status/{job_id}")
                if sr.status_code != 200:
                    raise gr.Error(f"Cloud Server Error {sr.status_code} while polling status.")
                
                status_data = sr.json()
                status = status_data.get("status", "processing")
                
                if status in ("completed", "done"):
                    break
                elif status == "failed":
                    err_msg = status_data.get("error", "Unknown error during cloud processing.")
                    raise gr.Error(f"Cloud Processing Failed: {err_msg}")
                else:
                    msg = status_data.get("message", f"Status: {status}...")
                    prog_val = status_data.get("progress", 0.5)
                    try:
                        prog_val = float(prog_val)
                    except (ValueError, TypeError):
                        prog_val = 0.5
                    progress(prog_val, desc=f"Cloud GPU: {msg}")

            progress(0.9, desc="Processing complete! Downloading results archive...")
            
            # 3. Download the result
            dr = requests.get(f"{url}/download/{job_id}", stream=True)
            if dr.status_code != 200:
                raise gr.Error(f"Failed to download archive. Server returned {dr.status_code}")
                
            import re as _re
            _video_name = os.path.splitext(os.path.basename(video_path))[0]
            _video_name = _re.sub(r'[^\w\-]', '_', _video_name).strip('_')
            _clean_model = model_name.split(' ')[0].strip()
            _model_name = os.path.splitext(os.path.basename(_clean_model))[0]
            _model_name = _re.sub(r'[^\w\-]', '_', _model_name).strip('_')

            cd = dr.headers.get("content-disposition", "")
            if "filename=" in cd:
                fname = cd.split("filename=")[1].strip('"\'')
                if fname.startswith(job_id) or fname == f"{job_id}.zip":
                    fname = f"{_video_name}_{_model_name}_{job_id[:8]}.zip"
            else:
                fname = f"{_video_name}_{_model_name}_{job_id[:8]}.zip"
                
            out_dir = os.path.join(OUTPUTS_DIR, "cloud_runs")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            
            base, ext = os.path.splitext(out_path)
            counter = 1
            while os.path.exists(out_path):
                out_path = f"{base}_{counter}{ext}"
                counter += 1
                
            with open(out_path, "wb") as f:
                for chunk in dr.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        
            # Unzip and load results into UI
            progress(0.95, desc="Extracting results for display...")
            run_folder_name = None
            try:
                with zipfile.ZipFile(out_path, 'r') as zip_ref:
                    # Find the root folder name inside the zip
                    top_level_dirs = set(os.path.normpath(name).split(os.sep)[0] for name in zip_ref.namelist())
                    if len(top_level_dirs) == 1:
                        run_folder_name = list(top_level_dirs)[0]
                        extract_path = OUTPUTS_DIR
                    else:
                        run_folder_name = os.path.splitext(os.path.basename(out_path))[0]
                        extract_path = os.path.join(OUTPUTS_DIR, run_folder_name)
                        
                    zip_ref.extractall(extract_path)
            except Exception as e:
                print(f"[Cloud GPU] Warning: failed to extract zip: {e}")
                return out_path, None, None, [], f"⚠️ Could not extract zip to display results: {e}", [], []
                
            # Load details using existing function
            v, csv, crops, summary, charts, segs, *_ = load_history_details(run_folder_name)
            
            # Reconstruct PipelineResult for Dataset Builder
            try:
                import json
                from rdd_dedup.tracking.defect_track import PipelineResult
                summary_json_path = os.path.join(OUTPUTS_DIR, run_folder_name, "summary.json")
                if os.path.exists(summary_json_path):
                    with open(summary_json_path, "r", encoding="utf-8") as f:
                        result_dict = json.load(f)
                    _ANNOTATION_STATE["last_pipeline_result"] = PipelineResult.from_dict(result_dict)
                    _ANNOTATION_STATE["last_video_path"] = video_path
                    _ANNOTATION_STATE["last_inspection_mode"] = "dedup"

                    # Store result for GIS enrichment
                    _GIS_STATE["last_video_path"] = video_path
                    _GIS_STATE["last_output_dir"] = os.path.join(OUTPUTS_DIR, run_folder_name)
                    _GIS_STATE["last_summary_path"] = summary_json_path
                    _GIS_STATE["last_pipeline_result"] = _ANNOTATION_STATE["last_pipeline_result"]
            except Exception as e:
                print(f"[Cloud GPU] Warning: failed to parse PipelineResult for dataset builder: {e}")
            
            # Delete the zip file to avoid duplication
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception as e:
                print(f"[Cloud GPU] Warning: failed to delete zip file: {e}")
                
            return None, v, csv, crops, summary, charts, segs
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Network error connecting to Cloud GPU: {str(e)}")

def fetch_cloud_runs(url):
    """Fetch run history from Cloud GPU."""
    if not url:
        return []
    url = url.rstrip("/")
    try:
        r = requests.get(f"{url}/runs", timeout=5)
        if r.status_code == 200:
            return r.json().get("runs", [])
    except Exception:
        pass
    return []

def download_cloud_run(run_name, url):
    """Download a specific run archive from the cloud."""
    if not url or not run_name:
        raise gr.Error("Missing URL or run name.")
    url = url.rstrip("/")
    try:
        r = requests.get(f"{url}/runs/{run_name}/download", stream=True)
        if r.status_code != 200:
            raise gr.Error(f"Failed to download. Server returned {r.status_code}")
            
        out_dir = os.path.join(OUTPUTS_DIR, "cloud_runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{run_name}.zip")
        
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return out_path
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Network error: {str(e)}")



def _process_video_legacy(video_path, model_name, conf_threshold, iou_threshold,
                          frame_skip, progress, custom_class_thresholds=None,
                          cam_height=1.2, cam_tilt=15.0, cam_focal=1200.0):
    """
    Original frame-by-frame detection without tracking or dedup.
    Used as fallback when dedup is disabled or in Demo mode.
    Supports dynamic per-class custom confidence thresholds.
    Estimates real-world ground area (m²) for Pothole, Manhole, and Patch road defects.
    """
    if not video_path or not os.path.exists(video_path):
        return None, None, [], "❌ Error: Invalid or missing video input file.", []

    start_time = time.time()
    import re as _re
    from rdd_dedup.measurement.area_calculator import (
        CameraParams, compute_defect_area, is_area_eligible_class
    )

    # Build videoname_modelname folder name
    _video_name = os.path.splitext(os.path.basename(video_path))[0]
    _video_name = _re.sub(r'[^\w\-]', '_', _video_name).strip('_')
    _clean_model = model_name.split(' ')[0].strip()
    _model_name = os.path.splitext(os.path.basename(_clean_model))[0]
    _model_name = _re.sub(r'[^\w\-]', '_', _model_name).strip('_')
    _base_folder = f"{_video_name}_{_model_name}"
    run_output_dir = os.path.join(OUTPUTS_DIR, _base_folder)
    if os.path.exists(run_output_dir):
        _counter = 2
        while os.path.exists(os.path.join(OUTPUTS_DIR, f"{_base_folder}_{_counter}")):
            _counter += 1
        run_output_dir = os.path.join(OUTPUTS_DIR, f"{_base_folder}_{_counter}")

    crops_dir = os.path.join(run_output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, [], "❌ Error: Could not open video file.", []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = 300  # Fallback estimate

    # Camera calibration for ground area calculation
    camera_params = CameraParams(
        yc=float(cam_height or 1.2),
        tilt_deg=float(cam_tilt or 15.0),
        f=float(cam_focal) if cam_focal else None,
        image_width=width,
        image_height=height,
    )

    # Output file paths
    temp_avi = os.path.join(run_output_dir, "processed_raw.avi")
    final_mp4 = os.path.join(run_output_dir, f"output_video_{timestamp_str}.mp4")
    csv_file_path = os.path.join(run_output_dir, f"detections_{timestamp_str}.csv")

    # Define OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_avi, fourcc, fps / max(1, frame_skip), (width, height))

    # Load RDD-YOLO Model and class names
    model, class_map = load_yolo_model(model_name)

    # Base confidence for YOLO prediction
    valid_thresholds = [v for v in custom_class_thresholds.values() if isinstance(v, (int, float))] if custom_class_thresholds else []
    base_conf = min(valid_thresholds) if valid_thresholds else float(conf_threshold)
    base_conf = max(0.05, min(base_conf, float(conf_threshold)))

    # Prepare CSV Writer
    csv_records = []
    csv_headers = [
        "detection_id", "frame_index", "timestamp_sec",
        "class_id", "class_name", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "real_area_m2",
        "crop_filename"
    ]

    # Statistics tracking
    frame_index = 0
    processed_count = 0
    total_detections = 0
    class_counts = {cfg["name"]: 0 for cfg in class_map.values()}
    crop_gallery_images = []  # For Gradio UI display gallery

    progress(0, desc=f"Loading RDD-YOLO Model ({model_name})...")

    try:
        while cap.isOpened():
            if _CANCEL_FLAG["requested"]:
                print("[RDD UI] Video processing revoked by user.")
                return None, None, [], "⚠️ **Process Revoked**: Video processing was cancelled by user revocation request.", []

            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if (frame_index - 1) % frame_skip != 0:
                continue

            processed_count += 1
            current_frame_detections = 0
            timestamp_sec = round(frame_index / fps, 2)

            # Raw frame copy for clean crops
            clean_frame = frame.copy()

            # Perform inference using RDD-YOLO
            if model is not None:
                results = model.predict(frame, conf=base_conf, iou=iou_threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box_idx, box in enumerate(boxes):
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        cls_cfg = class_map.get(cls_id, {"name": f"Class-{cls_id}", "color": (0, 0, 255)})
                        cls_name = cls_cfg["name"]
                        color = cls_cfg["color"]

                        # Check per-class threshold
                        thresh = custom_class_thresholds.get(
                            cls_name,
                            custom_class_thresholds.get(
                                str(cls_name).upper(),
                                custom_class_thresholds.get(cls_id, float(conf_threshold))
                            )
                        ) if custom_class_thresholds else float(conf_threshold)
                        if conf < thresh:
                            continue

                        # Coordinates
                        xyxy = [int(v.item()) for v in box.xyxy[0]]
                        x1, y1, x2, y2 = xyxy
                        x1_c, y1_c = max(0, x1), max(0, y1)
                        x2_c, y2_c = min(width, x2), min(height, y2)

                        # Extract segmentation polygon if available from YOLO-Seg
                        polygon = None
                        if hasattr(r, "masks") and r.masks is not None:
                            if hasattr(r.masks, "xy") and len(r.masks.xy) > box_idx:
                                poly_cand = r.masks.xy[box_idx]
                                if poly_cand is not None and len(poly_cand) >= 3:
                                    polygon = poly_cand

                        # Calculate real-world ground area ONLY for Pothole, Manhole, and Patch
                        area_m2 = None
                        if is_area_eligible_class(cls_name):
                            area_m2 = compute_defect_area((x1, y1, x2, y2), camera_params, cls_name, polygon)

                        total_detections += 1
                        current_frame_detections += 1
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                        # Save Cropped Defect Patch
                        crop_filename = f"crop_frame{frame_index:04d}_det{total_detections:03d}_{cls_id}_{conf:.2f}.jpg"
                        crop_filepath = os.path.join(crops_dir, crop_filename)

                        if x2_c > x1_c and y2_c > y1_c:
                            crop_patch = clean_frame[y1_c:y2_c, x1_c:x2_c]
                            cv2.imwrite(crop_filepath, crop_patch)

                            # Add to UI gallery (up to 60 crops max for fast UI rendering)
                            if len(crop_gallery_images) < 60:
                                crop_rgb = cv2.cvtColor(crop_patch, cv2.COLOR_BGR2RGB)
                                caption = f"{cls_name} ({conf:.2f})"
                                if area_m2 is not None and area_m2 > 0:
                                    caption += f" | {area_m2:.2f} m²"
                                caption += f" - t={timestamp_sec}s"
                                crop_gallery_images.append((crop_rgb, caption))

                        # Record to CSV
                        csv_records.append({
                            "detection_id": total_detections,
                            "frame_index": frame_index,
                            "timestamp_sec": timestamp_sec,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                            "bbox_x2": x2,
                            "bbox_y2": y2,
                            "real_area_m2": round(area_m2, 4) if area_m2 is not None else None,
                            "crop_filename": crop_filename
                        })

                        # Live YOLO segmentation overlay and crisp contour if available
                        if polygon is not None and len(polygon) >= 3:
                            try:
                                poly_pts = np.int32([polygon])
                                overlay = frame.copy()
                                cv2.fillPoly(overlay, poly_pts, color)
                                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                                cv2.polylines(frame, poly_pts, isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                            except Exception:
                                pass

                        # Draw Bounding Box & Label on Output Frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if area_m2 is not None and area_m2 > 0:
                            label_str = f"{cls_name} {conf:.2f} | {area_m2:.2f} m2"
                        else:
                            label_str = f"{cls_name} {conf:.2f}"
                        (w, h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + w + 10, y1), color, -1)
                        cv2.putText(frame, label_str, (x1 + 5, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                # Demo / Simulation Mode HUD & Crops
                if (frame_index // 15) % 3 == 1:
                    x1, y1 = int(width * 0.35), int(height * 0.65)
                    x2, y2 = int(width * 0.55), int(height * 0.85)
                    cls_cfg = DEFAULT_RDD_CLASSES[3]  # Pothole
                    color = cls_cfg["color"]

                    demo_area = compute_defect_area((x1, y1, x2, y2), camera_params, "Pothole")

                    total_detections += 1
                    current_frame_detections += 1
                    class_counts[cls_cfg["name"]] = class_counts.get(cls_cfg["name"], 0) + 1

                    crop_filename = f"crop_frame{frame_index:04d}_det{total_detections:03d}_3_0.89.jpg"
                    crop_filepath = os.path.join(crops_dir, crop_filename)

                    crop_patch = clean_frame[y1:y2, x1:x2]
                    if crop_patch.size > 0:
                        cv2.imwrite(crop_filepath, crop_patch)
                        if len(crop_gallery_images) < 60:
                            crop_rgb = cv2.cvtColor(crop_patch, cv2.COLOR_BGR2RGB)
                            caption = f"{cls_cfg['name']} (0.89 Demo)"
                            if demo_area:
                                caption += f" | {demo_area:.2f} m²"
                            crop_gallery_images.append((crop_rgb, caption))

                    csv_records.append({
                        "detection_id": total_detections,
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "class_id": 3,
                        "class_name": cls_cfg["name"],
                        "confidence": 0.89,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "real_area_m2": round(demo_area, 4) if demo_area else None,
                        "crop_filename": crop_filename
                    })

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label_str = f"{cls_cfg['name']} 0.89 (Demo) | {demo_area:.2f} m2" if demo_area else f"{cls_cfg['name']} 0.89 (Demo)"
                    cv2.rectangle(frame, (x1, y1 - 22), (x1 + 250, y1), color, -1)
                    cv2.putText(frame, label_str, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Render HUD Analytics Overlay on top of frame
            hud_text = f"RDD-YOLO | Frame: {frame_index}/{total_frames} | Defects: {current_frame_detections}"
            cv2.rectangle(frame, (10, 10), (460, 45), (0, 0, 0), -1)
            cv2.putText(frame, hud_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            out.write(frame)

            # Update progress bar with live ETA
            pct = min(1.0, frame_index / total_frames)
            time_str, finish_str = compute_live_eta(pct, start_time)
            if time_str:
                progress(pct, desc=f"⏳ {time_str} (Est. finish: {finish_str}) | Frame {frame_index}/{total_frames} ({total_detections} defects)")
            else:
                progress(pct, desc=f"RDD-YOLO Processing: Frame {frame_index}/{total_frames} ({total_detections} defects detected)")
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if out is not None:
            try:
                out.release()
            except Exception:
                pass

    elapsed_time = time.time() - start_time
    proc_fps = processed_count / max(0.001, elapsed_time)

    # Convert raw AVI video to web-compatible MP4 format
    final_output_video = reencode_to_h264(temp_avi, final_mp4)
    if os.path.exists(temp_avi):
        try:
            os.remove(temp_avi)
        except Exception:
            pass

    # Store result for dataset builder (Raw / Non-dedup mode compatibility)
    try:
        raw_finalized_defects = []
        for rec in csv_records:
            x1, y1, x2, y2 = rec["bbox_x1"], rec["bbox_y1"], rec["bbox_x2"], rec["bbox_y2"]
            crop_path = os.path.join(crops_dir, rec["crop_filename"])
            
            raw_defect = FinalizedDefect(
                defect_type=rec["class_name"],
                class_id=rec["class_id"],
                avg_confidence=rec["confidence"],
                median_confidence=rec["confidence"],
                max_confidence=rec["confidence"],
                max_confidence_frame=rec["frame_index"],
                first_frame=rec["frame_index"],
                last_frame=rec["frame_index"],
                detection_duration_frames=1,
                frames_observed=1,
                representative_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                avg_bbox_area=float(max(0, x2 - x1) * max(0, y2 - y1)),
                avg_aspect_ratio=float(max(1, x2 - x1) / max(1, y2 - y1)),
                bbox_stats={},
                representative_crop_path=crop_path,
                representative_bbox=(x1, y1, x2, y2),
                source_track_ids=[rec["detection_id"]],
                real_area_m2=rec.get("real_area_m2")
            )
            if raw_defect.real_area_m2 is not None:
                raw_defect.measurement_data = {"real_area_m2": raw_defect.real_area_m2}
            raw_finalized_defects.append(raw_defect)

        raw_pipeline_result = PipelineResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=processed_count,
            processing_time_sec=elapsed_time,
            fps=proc_fps,
            total_raw_detections=total_detections,
            total_unique_defects=total_detections,
            tracks_created=total_detections,
            tracks_merged=0,
            defect_counts=class_counts,
            finalized_defects=raw_finalized_defects,
            output_dir=run_output_dir,
            annotated_video_path=final_output_video
        )

        _ANNOTATION_STATE["last_pipeline_result"] = raw_pipeline_result
        _ANNOTATION_STATE["last_video_path"] = video_path
        _ANNOTATION_STATE["last_inspection_mode"] = "raw"

        # Store result for GIS enrichment
        _GIS_STATE["last_video_path"] = video_path
        _GIS_STATE["last_output_dir"] = run_output_dir
        _GIS_STATE["last_summary_path"] = os.path.join(run_output_dir, "summary.json")
        _GIS_STATE["last_pipeline_result"] = raw_pipeline_result
    except Exception as e:
        print(f"[Warning] Could not construct raw PipelineResult for dataset builder: {e}")

    # Save CSV File
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_records)

    # Save JSON summary metadata
    summary_data = {
        "timestamp_folder": os.path.basename(run_output_dir),
        "video_input": os.path.basename(video_path),
        "output_video": os.path.basename(final_output_video),
        "csv_log": os.path.basename(csv_file_path),
        "model_used": model_name,
        "confidence_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "processed_frames": processed_count,
        "total_frames": total_frames,
        "processing_time_sec": round(elapsed_time, 2),
        "fps": round(proc_fps, 1),
        "total_defects_detected": total_detections,
        "defect_counts": class_counts,
        "camera_calibration": {
            "camera_height_m": float(cam_height or 1.2),
            "camera_tilt_deg": float(cam_tilt or 15.0),
            "camera_focal_length_px": float(cam_focal) if cam_focal else None,
        },
        "finalized_defects": [d.to_dict() if hasattr(d, "to_dict") else d for d in raw_finalized_defects]
    }
    with open(os.path.join(run_output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Build Summary Report Markdown
    summary_md = f"""
### 📊 RDD-YOLO Inspection Summary (Legacy Mode — No Dedup)

- **Timestamp Run Folder**: `outputs/{os.path.basename(run_output_dir)}`
- **Output Video**: `{os.path.basename(final_output_video)}`
- **Detections CSV Log**: `{os.path.basename(csv_file_path)}`
- **Saved Defect Crop Patches**: `{total_detections} crop images` saved in `crops/` folder
- **Model Used**: `{model_name}`
- **Frames Processed**: `{processed_count}` / `{total_frames}`
- **Processing Time**: `{elapsed_time:.2f} seconds` (`{proc_fps:.1f} FPS`)
- **Total Defect Detections**: **{total_detections}**

> ⚠️ *Duplicate removal is disabled. Enable it in the Dedup Settings to get unique defect counts.*

#### 🛠️ Defect Category Breakdown:
"""
    for cls_name, count in class_counts.items():
        summary_md += f"- **{cls_name}**: `{count}` detected\n"

    # Defect Area Measurements (Pothole, Manhole, Patch only)
    legacy_area_entries = [r for r in csv_records if r.get("real_area_m2")]
    if legacy_area_entries:
        summary_md += "\n#### 📐 Real-World Defect Area Measurements (Pothole / Manhole / Patch):\n"
        summary_md += f"*Camera Height: `{cam_height}m` · Downward Tilt: `{cam_tilt}°` · Focal Length: `{cam_focal}px`*\n\n"
        for r in legacy_area_entries[:15]:
            summary_md += f"- **{r['class_name']}** (Frame #{r['frame_index']}): **{r['real_area_m2']:.3f} m²** (conf: {r['confidence']:.2f})\n"
        if len(legacy_area_entries) > 15:
            summary_md += f"- *...and {len(legacy_area_entries) - 15} more (see CSV for full details)*\n"

    summary_md += f"\n📁 *All files saved to: `{run_output_dir}`*"

    # Generate visualization charts
    charts_gallery = []
    if HAS_VISUALIZER:
        try:
            chart_data = generate_visualizations_from_directory(run_output_dir)
            for chart_path, caption in chart_data:
                if os.path.exists(chart_path):
                    charts_gallery.append((chart_path, caption))
        except Exception as e:
            print(f"[Warning] Failed to generate legacy visualization charts: {e}")

    return final_output_video, csv_file_path, crop_gallery_images, summary_md, charts_gallery, []


# Construct Gradio Modern Dashboard UI with Multi-Tab Pages
def build_ui():
    available_models = get_available_models()
    example_videos = get_example_videos()
    example_gpx_files = get_example_gpx_files()
    history_runs = get_history_runs()

    custom_css = """
    /* ─── Google Fonts ─── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ─── Root Variables ─── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-card: rgba(15, 15, 25, 0.85);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-glow: rgba(249, 115, 22, 0.3);
        --accent-orange: #f97316;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --accent-emerald: #10b981;
        --accent-sky: #38bdf8;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
        --shadow-glow: 0 0 30px rgba(249, 115, 22, 0.08);
    }

    /* ─── Global Overrides ─── */
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: var(--bg-primary) !important;
        max-width: 1440px !important;
    }

    /* ─── Custom Scrollbar ─── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(249, 115, 22, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(249, 115, 22, 0.5); }

    /* ─── Hero Header ─── */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem;
        position: relative;
    }
    .hero-container h1 {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #f97316 0%, #f59e0b 35%, #ef4444 70%, #f97316 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 4s ease infinite;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        margin: 0.5rem 0 0;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    /* ─── Feature Pills ─── */
    .pill-bar {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin: 1.2rem 0 0.5rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: var(--bg-glass);
        color: var(--text-secondary);
        border: 1px solid var(--border-subtle);
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }
    .pill:hover {
        border-color: var(--border-glow);
        color: var(--text-primary);
        box-shadow: 0 0 16px rgba(249, 115, 22, 0.12);
        transform: translateY(-1px);
    }
    .pill-active {
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .pill-active:hover {
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.15);
    }

    /* ─── Tabs Styling ─── */
    .tab-nav button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em;
        border-radius: 10px 10px 0 0 !important;
        transition: all 0.3s ease !important;
    }
    .tab-nav button.selected {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.15), rgba(245, 158, 11, 0.08)) !important;
        border-bottom: 2px solid var(--accent-orange) !important;
        color: var(--accent-orange) !important;
    }

    /* ─── Section Headers ─── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, var(--accent-orange), transparent) 1;
    }

    /* ─── Glassmorphic Card Panels ─── */
    .glass-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1rem;
        backdrop-filter: blur(16px);
        box-shadow: var(--shadow-glow);
        transition: border-color 0.3s ease;
    }
    .glass-panel:hover {
        border-color: rgba(255, 255, 255, 0.1);
    }

    /* ─── Run Button Glow ─── */
    .run-btn button {
        background: linear-gradient(135deg, #f97316, #ea580c) !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 20px rgba(249, 115, 22, 0.35) !important;
        transition: all 0.3s ease !important;
    }
    .run-btn button:hover {
        box-shadow: 0 6px 30px rgba(249, 115, 22, 0.5) !important;
        transform: translateY(-1px);
    }

    /* ─── System Telemetry Bar ─── */
    .sys-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        background: rgba(15, 15, 25, 0.9);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 16px;
        backdrop-filter: blur(16px);
    }
    .sys-metric {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .sys-metric .label { color: var(--text-secondary); font-weight: 500; }
    .sys-metric .value { font-size: 0.95rem; font-weight: 800; font-variant-numeric: tabular-nums; }
    .sys-metric .detail { color: var(--text-muted); font-weight: 400; font-size: 0.72rem; }

    /* ─── Progress Bar ─── */
    .progress-bar {
        width: 60px;
        height: 4px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 2px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.5s ease;
    }

    /* ─── Dedup Status ─── */
    .dedup-active {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 16px; border-radius: var(--radius-sm);
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #34d399; font-weight: 600; font-size: 0.88rem;
    }
    .dedup-inactive {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 16px; border-radius: var(--radius-sm);
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #f87171; font-weight: 600; font-size: 0.88rem;
    }
    .dedup-dot {
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block; animation: pulse-dot 2s ease infinite;
    }
    .dedup-dot-on { background: #34d399; box-shadow: 0 0 8px rgba(52, 211, 153, 0.6); }
    .dedup-dot-off { background: #f87171; box-shadow: 0 0 8px rgba(248, 113, 113, 0.6); animation: none; }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.85); }
    }

    /* ─── Gallery Enhancement ─── */
    .gallery-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-subtle);
    }

    /* ─── Accordion Tweaks ─── */
    .gradio-accordion > .label-wrap {
        font-weight: 600 !important;
    }

    /* ─── Divider ─── */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
        margin: 1.2rem 0;
        border: none;
    }

    /* ─── Dataset Builder Wizard Panels & Active Step Hierarchy ─── */
    .ds-wizard-panel {
        border-radius: var(--radius-lg) !important;
        padding: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative;
    }
    .ds-candidate-step {
        background: linear-gradient(180deg, rgba(14, 28, 48, 0.65) 0%, rgba(15, 15, 25, 0.88) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.08), inset 0 1px 0 rgba(56, 189, 248, 0.2) !important;
    }
    .ds-candidate-step::before {
        content: "STAGE 1 · CANDIDATE REVIEW & EDITING";
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 2px 10px;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
        width: fit-content;
    }
    .ds-review-step {
        background: linear-gradient(180deg, rgba(6, 44, 28, 0.55) 0%, rgba(15, 15, 25, 0.9) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.1), inset 0 1px 0 rgba(16, 185, 129, 0.25) !important;
    }
    .ds-review-step::before {
        content: "STAGE 2 · BATCH VERIFICATION & DATASET COMMIT";
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #34d399;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 2px 10px;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
        width: fit-content;
    }
    .ds-curate-step {
        background: linear-gradient(180deg, rgba(28, 14, 48, 0.65) 0%, rgba(15, 15, 25, 0.9) 100%) !important;
        border: 1px solid rgba(168, 85, 247, 0.4) !important;
        box-shadow: 0 0 35px rgba(168, 85, 247, 0.1), inset 0 1px 0 rgba(168, 85, 247, 0.25) !important;
    }
    .ds-curate-step::before {
        content: "RDD_AHMEDABAD · DEFECT INSPECTOR & CROP REPAIR";
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #c084fc;
        background: rgba(168, 85, 247, 0.12);
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 2px 10px;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
        width: fit-content;
    }

    /* ─── Compact Candidate Information Card ─── */
    .ds-candidate-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 8px 10px;
        margin: 4px 0 6px 0;
    }
    .ds-card-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .ds-detection-num {
        font-weight: 700;
        font-size: 0.88rem;
        color: var(--text-primary);
    }
    .ds-conf-pill {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 1px 7px;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .ds-meta-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        font-size: 0.8rem;
    }
    .ds-meta-item {
        display: flex;
        align-items: center;
        gap: 4px;
        overflow: hidden;
    }
    .ds-meta-lbl {
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 600;
    }
    .ds-meta-val {
        color: var(--text-primary);
        font-weight: 600;
        white-space: nowrap;
        text-overflow: ellipsis;
        overflow: hidden;
    }
    .ds-details-accordion {
        margin-top: 6px;
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: var(--radius-sm);
        overflow: hidden;
    }
    .ds-details-accordion[open] {
        border-color: rgba(56, 189, 248, 0.25);
    }
    .ds-details-accordion summary {
        padding: 4px 8px;
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--text-secondary);
        cursor: pointer;
        user-select: none;
    }
    .ds-details-accordion summary:hover {
        color: var(--text-primary);
    }
    .ds-details-content {
        padding: 6px 8px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.72rem;
        color: var(--text-muted);
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 4px;
    }
    .ds-details-content code {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-secondary);
        font-size: 0.72rem;
    }

    /* ─── Compact Inline Coordinate Row ─── */
    .ds-coords-row {
        display: flex !important;
        flex-direction: row !important;
        gap: 6px !important;
        margin-top: 4px !important;
        margin-bottom: 4px !important;
        padding: 4px 6px !important;
        background: rgba(255, 255, 255, 0.015) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: var(--radius-sm) !important;
        align-items: center !important;
    }
    .ds-coords-row > div {
        min-width: 0 !important;
        flex: 1 1 0 !important;
    }
    .ds-coords-row label {
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        margin-bottom: 0px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .ds-coords-row input {
        height: 26px !important;
        min-height: 26px !important;
        padding: 2px 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        text-align: center !important;
        background: rgba(0, 0, 0, 0.35) !important;
    }

    /* ─── Dataset Builder Card ─── */
    .ds-action-row button {
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .ds-action-row button:hover {
        transform: translateY(-1px);
    }

    /* ─── Subtle Animations ─── */
    .fade-in {
        animation: fade-in 0.4s ease-out;
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    """

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#0a0a0f",
        body_background_fill_dark="#0a0a0f",
        body_text_color="#f1f5f9",
        body_text_color_dark="#f1f5f9",
        body_text_color_subdued="#94a3b8",
        body_text_color_subdued_dark="#94a3b8",
        block_background_fill="rgba(15, 15, 25, 0.85)",
        block_background_fill_dark="rgba(15, 15, 25, 0.85)",
        block_border_color="rgba(255, 255, 255, 0.06)",
        block_border_color_dark="rgba(255, 255, 255, 0.06)",
        block_label_text_color="#94a3b8",
        block_label_text_color_dark="#94a3b8",
        block_title_text_color="#f1f5f9",
        block_title_text_color_dark="#f1f5f9",
        block_radius="12px",
        block_shadow="0 0 30px rgba(249, 115, 22, 0.05)",
        input_background_fill="#0f0f19",
        input_background_fill_dark="#0f0f19",
        input_border_color="rgba(255, 255, 255, 0.08)",
        input_border_color_dark="rgba(255, 255, 255, 0.08)",
        button_primary_background_fill="linear-gradient(135deg, #f97316, #ea580c)",
        button_primary_background_fill_dark="linear-gradient(135deg, #f97316, #ea580c)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="rgba(255, 255, 255, 0.05)",
        button_secondary_background_fill_dark="rgba(255, 255, 255, 0.05)",
        button_secondary_text_color="#f1f5f9",
        button_secondary_text_color_dark="#f1f5f9",
        button_secondary_border_color="rgba(255, 255, 255, 0.1)",
        button_secondary_border_color_dark="rgba(255, 255, 255, 0.1)",
        checkbox_background_color="#0f0f19",
        checkbox_background_color_dark="#0f0f19",
        checkbox_border_color="rgba(255, 255, 255, 0.15)",
        checkbox_border_color_dark="rgba(255, 255, 255, 0.15)",
        checkbox_label_text_color="#f1f5f9",
        checkbox_label_text_color_dark="#f1f5f9",
        slider_color="#f97316",
        slider_color_dark="#f97316",
        table_even_background_fill="rgba(255, 255, 255, 0.02)",
        table_even_background_fill_dark="rgba(255, 255, 255, 0.02)",
        table_odd_background_fill="transparent",
        table_odd_background_fill_dark="transparent",
        border_color_accent="#f97316",
        border_color_accent_dark="#f97316",
        shadow_spread="0px",
    )

    with gr.Blocks(title="RDD-YOLO Road Damage Detection Studio") as demo:

        # ─── Hero Header ───
        gr.HTML(f"""
        <div class="hero-container fade-in">
            <h1>🛣️ RDD-YOLO Detection Studio</h1>
            <p class="hero-subtitle">AI-powered road damage inspection — detect potholes, cracks & defects from video footage</p>
            <div class="pill-bar">
                <span class="pill pill-active">✦ Model: {available_models[0]}</span>
                <span class="pill pill-active">{'⚡ Dedup Pipeline: Active' if HAS_DEDUP_PIPELINE else '⚡ Dedup: N/A'}</span>
                <span class="pill">🔗 BoT-SORT Tracking</span>
                <span class="pill">📊 CSV Export</span>
                <span class="pill">🖼️ Defect Crops</span>
                <span class="pill">📜 History Explorer</span>
            </div>
        </div>
        """)

        with gr.Tabs():
            # ═══════════════════════════════════════════════
            # PAGE 1: New Video Analysis
            # ═══════════════════════════════════════════════
            with gr.TabItem("🚀 Run New Inspection"):
                with gr.Row():
                    # Left Column: Inputs & Settings
                    with gr.Column(scale=5):
                        gr.HTML('<div class="section-header">📥 Video Input & Settings</div>')
                        
                        mode_toggle = gr.Radio(
                            choices=["💻 Local (CPU)", "☁️ Cloud GPU (Remote Server)"],
                            value="💻 Local (CPU)",
                            label="Processing Mode",
                            info="Run locally or offload to a remote Cloud GPU"
                        )
                        
                        cloud_url_state = gr.BrowserState("", storage_key="rdd_cloud_server_url")
                        
                        with gr.Column(visible=False) as cloud_settings_col:
                            cloud_url_input = gr.Textbox(
                                label="Notebook Server URL", 
                                placeholder="https://xxxx.trycloudflare.com",
                                info="Paste the tunnel URL from your Colab notebook."
                            )
                            btn_test_cloud = gr.Button("Test Connection", size="sm")
                            cloud_status_md = gr.HTML("")
                            
                        input_video = gr.Video(
                            label="Upload Road Footage",
                            sources=["upload"],
                            interactive=True
                        )
                        with gr.Row():
                            quick_sample_dd = gr.Dropdown(
                                choices=example_videos,
                                value=None,
                                label="📁 Quick Select from inputs/ directory",
                                info="Pick existing footage directly without re-uploading",
                                scale=4
                            )
                            btn_refresh_samples = gr.Button("🔄", scale=1, min_width=48)
                        video_est_md = gr.Markdown("")

                        with gr.Accordion("⚙️ Detection Model & Parameters", open=True):
                            openvino_toggle = gr.Checkbox(
                                value=False,
                                label="⚡ Use OpenVINO Models (Folders)",
                                info="Toggle ON for OpenVINO model folders · Toggle OFF for Normal (.pt) weights",
                            )
                            with gr.Row():
                                model_selector = gr.Dropdown(
                                    choices=available_models,
                                    value=available_models[0] if available_models else None,
                                    label="Model Weights (.pt)",
                                    info="Auto-detected from models/ directory",
                                    scale=4
                                )
                                btn_refresh_models = gr.Button("🔄", scale=1, min_width=48)
                            
                            with gr.Row():
                                conf_slider = gr.Slider(
                                    minimum=0.05,
                                    maximum=0.95,
                                    value=0.25,
                                    step=0.05,
                                    label="Confidence",
                                    info="Min detection confidence"
                                )
                                iou_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.45,
                                    step=0.05,
                                    label="IoU NMS",
                                    info="Non-Max Suppression threshold"
                                )

                            frame_skip_slider = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                                label="Frame Skip",
                                info="1 = all frames · 2 = every 2nd frame"
                            )

                            # ── Dynamic Per-Class Confidence Thresholds ──
                            initial_model = available_models[0] if available_models else "rdd_yolo.pt"
                            initial_classes = get_model_class_dict(initial_model)
                            initial_items = list(initial_classes.items())

                            with gr.Accordion("🎯 Custom Per-Class Confidence Thresholds", open=True) as per_class_accordion:
                                per_class_info_md = gr.Markdown(
                                    f"🎯 **{len(initial_items)} classes detected** for `{os.path.basename(initial_model)}`. Configure independent detection sensitivity per class:"
                                )
                                with gr.Row():
                                    btn_sync_all_conf = gr.Button("⚡ Sync All to Master Confidence", size="sm", scale=1)
                                    btn_reset_all_conf = gr.Button("🔄 Reset to 0.25", size="sm", scale=1)

                                class_sliders = []
                                for r_idx in range(MAX_CLASS_SLIDERS // 2):
                                    with gr.Row():
                                        for c_idx in range(2):
                                            s_idx = r_idx * 2 + c_idx
                                            is_vis = s_idx < len(initial_items)
                                            c_lbl = f"[{initial_items[s_idx][0]}] {initial_items[s_idx][1]}" if is_vis else f"Class {s_idx}"
                                            s = gr.Slider(
                                                minimum=0.05,
                                                maximum=0.95,
                                                value=0.25,
                                                step=0.05,
                                                label=c_lbl,
                                                visible=is_vis,
                                                elem_id=f"class_conf_slider_{s_idx}"
                                            )
                                            class_sliders.append(s)

                        with gr.Accordion("🔄 Duplicate Removal Pipeline", open=True):
                            enable_dedup = gr.Checkbox(
                                value=True,
                                label="⚡ Activate Duplicate Removal Pipeline",
                                info="Toggle ON to merge repeat defect detections into unique counts. Toggle OFF for raw frame-by-frame detection.",
                                interactive=HAS_DEDUP_PIPELINE,
                            )
                            dedup_status_md = gr.HTML(
                                '<div class="dedup-active"><span class="dedup-dot dedup-dot-on"></span> Dedup Pipeline — ACTIVE <span style="font-weight:400; color:#94a3b8; font-size:0.8rem;">· BoT-SORT + Multi-Feature Verification</span></div>'
                            )
                            with gr.Row(visible=True) as dedup_params_row:
                                merge_threshold_slider = gr.Slider(
                                    minimum=0.3,
                                    maximum=0.95,
                                    value=0.65,
                                    step=0.05,
                                    label="Merge Threshold",
                                    info="Similarity to merge duplicates",
                                )
                                track_buffer_slider = gr.Slider(
                                    minimum=10,
                                    maximum=120,
                                    value=45,
                                    step=5,
                                    label="Track Buffer",
                                    info="Frames to keep lost tracks alive",
                                )

                        with gr.Accordion("📐 Camera Calibration & Area Estimation", open=True):
                            gr.HTML('<div style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px;">Calibrate flat-plane camera parameters to compute physical ground area (m²) for <b>Pothole</b>, <b>Manhole</b>, and <b>Patch</b> road defects.</div>')
                            with gr.Row():
                                cam_height_slider = gr.Slider(
                                    minimum=0.5,
                                    maximum=3.5,
                                    value=1.2,
                                    step=0.1,
                                    label="Camera Height (yc)",
                                    info="Height above road surface (meters)"
                                )
                                cam_tilt_slider = gr.Slider(
                                    minimum=1.0,
                                    maximum=60.0,
                                    value=15.0,
                                    step=1.0,
                                    label="Downward Tilt (θ)",
                                    info="Downward tilt in degrees"
                                )
                                cam_focal_slider = gr.Slider(
                                    minimum=300.0,
                                    maximum=3000.0,
                                    value=1200.0,
                                    step=50.0,
                                    label="Focal Length (f)",
                                    info="Focal length in pixels"
                                )

                        with gr.Accordion("✨ YOLO Segmentation", open=True):
                            enable_seg_ui = gr.Checkbox(
                                value=True,
                                label="✨ Enable YOLO Segmentation",
                                info="Toggle ON to segment detected defects using YOLO-Seg and measure exact polygon contour area. Toggle OFF for bounding box corners."
                            )

                        with gr.Row(elem_classes="run-btn"):
                            btn_process = gr.Button("🚀 Run RDD-YOLO Detection + Dedup", variant="primary", scale=2)
                            btn_process_cloud = gr.Button("🚀 Run on Cloud GPU", variant="primary", scale=2, visible=False)
                            btn_revoke = gr.Button("🛑 Revoke Process", variant="stop", scale=1)
                            btn_clear = gr.ClearButton(components=[input_video, quick_sample_dd], value="✕ Clear", scale=1)

                    # Right Column: Live Telemetry & Output Results
                    with gr.Column(scale=6):
                        # System Telemetry Bar
                        sys_cards_html = gr.HTML(value='<div class="sys-bar"><span class="sys-metric"><span class="label">Initializing telemetry...</span></span></div>')

                        gr.HTML('<div class="section-header">📤 Inspection Results</div>')
                        with gr.Column(visible=False) as cloud_output_col:
                            cloud_output_file = gr.File(label="📦 Cloud Output Archive (.zip)")
                            cloud_output_msg = gr.Markdown("Cloud GPU results will appear here as a downloadable ZIP archive.")

                        with gr.Column(visible=True) as local_output_col:
                            output_video = gr.Video(
                                label="Annotated Output Video",
                                interactive=False
                            )
                            output_csv = gr.File(
                                label="📥 Detections Log (.csv)",
                                interactive=False
                            )
                            output_summary = gr.Markdown(
                                "Upload road video footage and click **Run RDD-YOLO Detection + Dedup** to begin automated inspection."
                            )

                            # ── Geolocation & GIS Map Enrichment Section ──
                            with gr.Accordion("🗺️ Geolocation & GIS Map Enrichment", open=True) as gis_accordion:
                                gr.HTML('<div style="font-size:0.88rem; color:#94a3b8; margin-bottom:8px;">Synchronize finalized unique defects with GPX survey trackpoints to generate GeoJSON and an interactive map.</div>')
                                with gr.Row():
                                    gis_gpx_upload = gr.File(
                                        label="Upload GPX Survey Track (.gpx)",
                                        file_types=[".gpx"],
                                        file_count="single",
                                        interactive=True,
                                        scale=3
                                    )
                                    gis_gpx_dropdown = gr.Dropdown(
                                        choices=example_gpx_files,
                                        value=example_gpx_files[0] if example_gpx_files else None,
                                        label="📁 Quick Select GPX from inputs/",
                                        info="Pick existing GPX survey log",
                                        scale=3
                                    )
                                    btn_refresh_gpx = gr.Button("🔄", scale=1, min_width=48)
                                with gr.Row():
                                    gis_offset_slider = gr.Slider(
                                        minimum=-30.0,
                                        maximum=30.0,
                                        value=0.0,
                                        step=0.5,
                                        label="Time Offset (seconds)",
                                        info="Video start offset relative to GPX start (+/- sec)",
                                        scale=2
                                    )
                                    btn_generate_geo = gr.Button(
                                        "📍 Generate GEO Information",
                                        variant="primary",
                                        scale=2
                                    )
                                gis_status_md = gr.Markdown("")
                                gis_map_html = gr.HTML(value="", label="Interactive Defect Map")
                                gis_download_file = gr.File(label="📥 Download defects.geojson", interactive=False, visible=False)

                with gr.Column(visible=True) as local_galleries_col:
                    # Gallery Section
                    gr.HTML('<div class="gallery-section"></div>')
                    gr.HTML('<div class="section-header">🖼️ Detected Defect Crops</div>')
                    output_crops_gallery = gr.Gallery(
                        label="Representative crop patches — one per unique physical defect",
                        columns=6,
                        rows=2,
                        height="auto",
                        object_fit="contain"
                    )

                    gr.HTML('<div class="section-header">✨ YOLO Segmentation Results</div>')
                    output_seg_gallery = gr.Gallery(
                        label="YOLO Segmentation Overlays & Masks",
                        columns=4,
                        rows=2,
                        height="auto",
                        object_fit="contain"
                    )

                    # Data Visualization Charts Section
                    gr.HTML('<div class="styled-divider"></div>')
                    gr.HTML('<div class="section-header">📈 Data Visualization</div>')
                    output_charts_gallery = gr.Gallery(
                        label="Pipeline analytics — defect distribution, confidence, timeline, and spatial analysis",
                        columns=3,
                        rows=2,
                        height="auto",
                        object_fit="contain"
                    )

                # Example Videos
                if example_videos:
                    gr.HTML('<div class="styled-divider"></div>')
                    gr.HTML('<div class="section-header">🎞️ Sample Road Footage</div>')
                    gr.Examples(
                        examples=[[vid] for vid in example_videos],
                        inputs=[input_video],
                        outputs=[output_video, output_csv, output_crops_gallery, output_summary, output_charts_gallery, output_seg_gallery],
                        fn=lambda vid: process_video(vid, available_models[0], 0.25, 0.45, 1, True, 0.65, 45, True, 1.2, 15.0, 1200.0),
                        cache_examples=False,
                        label="Click a sample to load:"
                    )

            # ═══════════════════════════════════════════════
            # PAGE 2: Dataset Builder
            # ═══════════════════════════════════════════════
            with gr.TabItem("📦 Dataset Builder"):
                gr.HTML('<div class="section-header">📦 RDD Ahmedabad Dataset Builder & Curation Suite</div>')

                with gr.Tabs():
                    # ─── SUB-TAB 1: Collect & Stage Defects ───
                    with gr.TabItem("➕ Collect & Stage Defects", id="tab_collect_defects"):
                        with gr.Row():
                            btn_load_candidates = gr.Button("📥 Load Detections from Last Inspection", variant="primary", scale=3)
                            btn_add_from_image = gr.UploadButton("🖼️ Add from Image(s)", file_types=["image"], file_count="multiple", variant="secondary", scale=2)
                            ds_status_md = gr.Markdown("", scale=5)

                        # ── Container 1: Candidate Review Step ──
                        candidate_step_box = gr.Column(visible=True, elem_classes=["ds-wizard-panel", "ds-candidate-step"])
                        with candidate_step_box:
                            with gr.Row():
                                # Left: Full frame adjustment
                                with gr.Column(scale=6):
                                    ds_full_frame = gr.HTML(
                                        value='<div style="text-align:center;color:#94a3b8;padding:40px;font-family:Inter,sans-serif;">Click 📥 Load Detections or 🖼️ Add from Image(s) to begin.</div>'
                                    )
                                    with gr.Row():
                                        ds_padding_slider = gr.Slider(
                                            minimum=0, maximum=200, value=0, step=5,
                                            label="Expand Margin Padding (px)", scale=3
                                        )
                                        btn_reset_box = gr.Button("🔄 Reset Box", variant="secondary", scale=1)

                                # Right: Info, Crop Preview & Actions
                                with gr.Column(scale=4):
                                    ds_crop_image = gr.Image(
                                        label="Crop Preview (Will be staged)",
                                        type="numpy", interactive=False, height=190,
                                    )
                                    
                                    ds_progress_md = gr.Markdown("")
                                    ds_info_md = gr.Markdown("Click **Load Detections** or upload image(s) to start.")

                                    rdd_class_names = RDD_CLASS_CONFIG.get_class_names() if RDD_CLASS_CONFIG else ["POTHOLE"]
                                    ds_class_dropdown = gr.Dropdown(
                                        choices=rdd_class_names,
                                        value=rdd_class_names[0] if rdd_class_names else None,
                                        label="Assign Class",
                                        interactive=True,
                                    )

                                    with gr.Row(elem_classes="ds-action-row"):
                                        btn_accept = gr.Button("✅ Accept Defect", variant="primary", scale=2)
                                        btn_add_another = gr.Button("➕ Add Another in Frame", variant="secondary", scale=2)
                                        btn_skip = gr.Button("⏭️ Skip", variant="secondary", scale=1)
                                    
                                    with gr.Row(elem_classes="ds-action-row"):
                                        btn_prev_candidate = gr.Button("⏮️ Previous", variant="secondary", scale=1)
                                        btn_review_staged = gr.Button("📋 Review Staged (0)", variant="primary", scale=2)
                                    
                                    # Coordinates for manual fine-tuning and canvas sync
                                    with gr.Row(elem_classes="ds-coords-row"):
                                        ds_x1 = gr.Number(label="X1", value=0, precision=0, elem_id="ds_bbox_x1", scale=1)
                                        ds_y1 = gr.Number(label="Y1", value=0, precision=0, elem_id="ds_bbox_y1", scale=1)
                                        ds_x2 = gr.Number(label="X2", value=0, precision=0, elem_id="ds_bbox_x2", scale=1)
                                        ds_y2 = gr.Number(label="Y2", value=0, precision=0, elem_id="ds_bbox_y2", scale=1)

                        # ── Container 2: Confirmation & Review Step ──
                        review_step_box = gr.Column(visible=False, elem_classes=["ds-wizard-panel", "ds-review-step"])
                        with review_step_box:
                            with gr.Row():
                                with gr.Column(scale=6):
                                    ds_review_prompt_md = gr.Markdown("")
                                mantle_dist = gr.Column(scale=4)
                                with mantle_dist:
                                    ds_review_dist_md = gr.Markdown("")

                            gr.HTML('<div class="section-header" style="margin-top:0.4rem;margin-bottom:0.4rem;">🖼️ Staged Defect Crops for Confirmation</div>')
                            ds_review_gallery = gr.Gallery(
                                label="Staged Defect Crops",
                                columns=5,
                                rows=2,
                                height=240,
                                object_fit="contain",
                            )

                            with gr.Accordion("✏️ Edit a Specific Defect", open=False):
                                gr.Markdown("Select any staged defect below to adjust its crop margins, coordinates, or class before adding to the dataset:")
                                with gr.Row():
                                    ds_edit_selector = gr.Dropdown(
                                        choices=[],
                                        label="Select Defect to Edit",
                                        scale=4,
                                        interactive=True
                                    )
                                    btn_refresh_edit_list = gr.Button("🔄 Refresh List", scale=1)

                                with gr.Row():
                                    with gr.Column(scale=6):
                                        ds_edit_full_frame = gr.HTML(
                                            value='<div style="text-align:center;color:#94a3b8;padding:20px;">Select a defect from the dropdown above to edit.</div>'
                                        )
                                        with gr.Row():
                                            ds_edit_padding_slider = gr.Slider(
                                                minimum=0, maximum=200, value=0, step=5,
                                                label="Expand Margin Padding (px)", scale=3
                                            )
                                            btn_edit_reset_box = gr.Button("🔄 Reset Box", variant="secondary", scale=1)
                                    with gr.Column(scale=4):
                                        ds_edit_crop_image = gr.Image(
                                            label="Crop Preview", type="numpy", interactive=False, height=200
                                        )
                                        ds_edit_class_dropdown = gr.Dropdown(
                                            choices=rdd_class_names,
                                            label="Reassign Class",
                                            interactive=True,
                                        )
                                        ds_edit_status_md = gr.Markdown("")
                                        with gr.Row():
                                            btn_save_staged_edit = gr.Button("💾 Save Defect Changes", variant="primary", scale=2)
                                            btn_delete_staged = gr.Button("🗑️ Remove from Batch", variant="stop", scale=2)
                                        # Coordinates for manual fine-tuning and canvas sync
                                        with gr.Row(elem_classes="ds-coords-row"):
                                            ds_edit_x1 = gr.Number(label="X1", value=0, precision=0, elem_id="ds_edit_bbox_x1", scale=1)
                                            ds_edit_y1 = gr.Number(label="Y1", value=0, precision=0, elem_id="ds_edit_bbox_y1", scale=1)
                                            ds_edit_x2 = gr.Number(label="X2", value=0, precision=0, elem_id="ds_edit_bbox_x2", scale=1)
                                            ds_edit_y2 = gr.Number(label="Y2", value=0, precision=0, elem_id="ds_edit_bbox_y2", scale=1)

                            gr.HTML('<div class="styled-divider"></div>')
                            with gr.Row():
                                btn_confirm_dataset = gr.Button(
                                    "💾 Confirm & Add to RDD_Ahmedabad Dataset",
                                    variant="primary",
                                    scale=4,
                                )
                                btn_back_to_candidates = gr.Button("↩️ Back to Candidate Review", variant="secondary", scale=2)
                                btn_discard_batch = gr.Button("🗑️ Discard Batch", variant="stop", scale=1)

                        ds_commit_result_md = gr.Markdown("", visible=False)

                    # ─── SUB-TAB 2: Revive & Curate RDD_Ahmedabad Dataset ───
                    with gr.TabItem("🛠️ Revive & Curate Dataset", id="tab_curate_dataset"):
                        gr.HTML('<div class="section-header" style="font-size:0.95rem;margin-bottom:0.5rem;">🔍 Inspect, Repair Crops, Reclassify, or Clean Defect Annotations in RDD_Ahmedabad</div>')

                        curate_class_choices = get_curate_class_choices()
                        initial_jump_choices = load_curate_dataset()

                        with gr.Row():
                            curate_class_filter = gr.Dropdown(
                                choices=curate_class_choices,
                                value="ALL",
                                label="Filter by Defect Class",
                                scale=3,
                                interactive=True
                            )
                            curate_split_filter = gr.Dropdown(
                                choices=["ALL", "train", "val", "test"],
                                value="ALL",
                                label="Filter by Split",
                                scale=2,
                                interactive=True
                            )
                            curate_search_input = gr.Textbox(
                                label="Search Image / Video",
                                placeholder="Search filename (e.g. road_footage_7)...",
                                scale=3,
                                interactive=True
                            )
                            btn_curate_filter_apply = gr.Button("🔍 Apply Filter", variant="secondary", scale=1)
                            btn_curate_refresh = gr.Button("🔄 Refresh", variant="secondary", scale=1)

                        with gr.Row(elem_classes="ds-action-row"):
                            btn_curate_prev = gr.Button("⏮️ Previous Defect", variant="secondary", scale=2)
                            curate_jump_dropdown = gr.Dropdown(
                                choices=initial_jump_choices,
                                value=0 if initial_jump_choices else None,
                                label="Jump to Defect in Filtered Set",
                                scale=5,
                                interactive=True
                            )
                            btn_curate_next = gr.Button("⏭️ Next Defect", variant="secondary", scale=2)
                            btn_curate_random = gr.Button("🎲 Random", variant="secondary", scale=1)

                        curate_pos_indicator_md = gr.Markdown(
                            value=f"📍 Defect **1** of **{len(initial_jump_choices)}** · Filtered Total: **{len(initial_jump_choices)}**" if initial_jump_choices else "No defects loaded."
                        )

                        # ── Curate Workbench Panels ──
                        curate_workbench_box = gr.Row(elem_classes=["ds-wizard-panel", "ds-curate-step"])
                        with curate_workbench_box:
                            # Left Column: Canvas Bbox Editor & Margin Padding
                            with gr.Column(scale=6):
                                curate_full_frame = gr.HTML(
                                    value='<div style="text-align:center;color:#94a3b8;padding:40px;">Select a defect above or click Apply Filter to inspect.</div>'
                                )
                                with gr.Row():
                                    curate_padding_slider = gr.Slider(
                                        minimum=0, maximum=200, value=0, step=5,
                                        label="Expand Margin Padding (px)", scale=3
                                    )
                                    btn_curate_reset_box = gr.Button("🔄 Reset Box", variant="secondary", scale=1)

                            # Right Column: Crop Preview, Info Card, Class Reassign & Save/Delete
                            with gr.Column(scale=4):
                                curate_crop_image = gr.Image(
                                    label="Defect Crop Preview",
                                    type="numpy", interactive=False, height=200
                                )
                                curate_info_md = gr.Markdown("")
                                curate_class_dropdown = gr.Dropdown(
                                    choices=rdd_class_names,
                                    value=rdd_class_names[0] if rdd_class_names else None,
                                    label="Reassign Defect Class",
                                    interactive=True,
                                )
                                curate_status_md = gr.Markdown("")
                                with gr.Row(elem_classes="ds-action-row"):
                                    btn_curate_save = gr.Button("💾 Save Defect Changes", variant="primary", scale=2)
                                    btn_curate_delete = gr.Button("🗑️ Delete Defect", variant="stop", scale=2)

                                # Synchronized coordinates
                                with gr.Row(elem_classes="ds-coords-row"):
                                    curate_x1 = gr.Number(label="X1", value=0, precision=0, elem_id="ds_curate_bbox_x1", scale=1)
                                    curate_y1 = gr.Number(label="Y1", value=0, precision=0, elem_id="ds_curate_bbox_y1", scale=1)
                                    curate_x2 = gr.Number(label="X2", value=0, precision=0, elem_id="ds_curate_bbox_x2", scale=1)
                                    curate_y2 = gr.Number(label="Y2", value=0, precision=0, elem_id="ds_curate_bbox_y2", scale=1)

                gr.HTML('<div class="styled-divider"></div>')
                with gr.Accordion("📊 Dataset Statistics (RDD_Ahmedabad)", open=False):
                    ds_stats_md = gr.Markdown(
                        value=get_dataset_stats_md()
                    )
                    btn_refresh_stats = gr.Button("🔄 Refresh Stats")

            # ═══════════════════════════════════════════════
            # PAGE 3: History & Reports
            # ═══════════════════════════════════════════════
            with gr.TabItem("📜 History & Reports"):
                gr.HTML('<div class="section-header">📜 Previous Inspections</div>')
                
                with gr.Row():
                    history_selector = gr.Dropdown(
                        choices=history_runs,
                        value=history_runs[0] if history_runs else None,
                        label="Select Past Run",
                        info="Browse previous analysis folders from outputs/",
                        scale=4
                    )
                    btn_refresh_history = gr.Button("🔄", scale=1, min_width=48)

                with gr.Row():
                    with gr.Column(scale=5):
                        hist_video = gr.Video(
                            label="Annotated Output Video",
                            interactive=False
                        )
                        hist_csv = gr.File(
                            label="📥 Detections Log (.csv)",
                            interactive=False
                        )
                    with gr.Column(scale=6):
                        hist_summary = gr.Markdown("Select a run from the dropdown to view its inspection report.")

                        # ── History Geolocation & GIS Map Enrichment Section ──
                        with gr.Accordion("🗺️ Geolocation & GIS Map Enrichment", open=True) as hist_gis_accordion:
                            gr.HTML('<div style="font-size:0.88rem; color:#94a3b8; margin-bottom:8px;">Synchronize past inspection defects with GPX survey trackpoints to generate GeoJSON and view the interactive map.</div>')
                            with gr.Row():
                                hist_gis_gpx_upload = gr.File(
                                    label="Upload GPX Survey Track (.gpx)",
                                    file_types=[".gpx"],
                                    file_count="single",
                                    interactive=True,
                                    scale=3
                                )
                                hist_gis_gpx_dropdown = gr.Dropdown(
                                    choices=example_gpx_files,
                                    value=example_gpx_files[0] if example_gpx_files else None,
                                    label="📁 Quick Select GPX from inputs/",
                                    info="Pick existing GPX survey log",
                                    scale=3
                                )
                                btn_hist_refresh_gpx = gr.Button("🔄", scale=1, min_width=48)
                            with gr.Row():
                                hist_gis_offset_slider = gr.Slider(
                                    minimum=-30.0,
                                    maximum=30.0,
                                    value=0.0,
                                    step=0.5,
                                    label="Time Offset (seconds)",
                                    info="Video start offset relative to GPX start (+/- sec)",
                                    scale=2
                                )
                                btn_hist_generate_geo = gr.Button(
                                    "📍 Generate GEO Information",
                                    variant="primary",
                                    scale=2
                                )
                            hist_gis_status_md = gr.Markdown("")
                            hist_gis_map_html = gr.HTML(value="", label="Interactive Defect Map")
                            hist_gis_download_file = gr.File(label="📥 Download defects.geojson", interactive=False, visible=False)

                gr.HTML('<div class="gallery-section"></div>')
                gr.HTML('<div class="section-header">🖼️ Defect Crops from Selected Run</div>')
                hist_crops_gallery = gr.Gallery(
                    label="Saved crop patches from past run",
                    columns=6,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )

                gr.HTML('<div class="section-header">✨ SAM2 Segmentation Results</div>')
                hist_seg_gallery = gr.Gallery(
                    label="Segmentation Overlays",
                    columns=4,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )

                # History Data Visualization Charts
                gr.HTML('<div class="styled-divider"></div>')
                gr.HTML('<div class="section-header">📈 Data Visualization</div>')
                hist_charts_gallery = gr.Gallery(
                    label="Pipeline analytics from selected run",
                    columns=3,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )

                with gr.Accordion("☁️ Cloud Run History", open=False):
                    with gr.Row():
                        cloud_history_selector = gr.Dropdown(
                            choices=[],
                            label="Select Cloud Run to Download",
                            info="Requires valid Notebook Server URL in the New Inspection tab.",
                            scale=4
                        )
                        btn_refresh_cloud_history = gr.Button("🔄 Refresh", scale=1, min_width=48)
                    
                    btn_download_cloud_run = gr.Button("⬇️ Download Selected Run", variant="primary")
                    cloud_run_download = gr.File(label="📦 Downloaded Archive", visible=False)

        # ─── Timers & Event Handlers ───
        # (All event handler logic preserved identically)

        # Initial load and periodic timer for hardware status bar (every 2s)
        sys_timer = gr.Timer(2)
        demo.load(
            fn=lambda: update_system_dashboard()[0],
            inputs=[],
            outputs=[sys_cards_html]
        )
        sys_timer.tick(
            fn=lambda: update_system_dashboard()[0],
            inputs=[],
            outputs=[sys_cards_html]
        )

        # Event Handlers
        # Cloud URL State mapping
        demo.load(lambda x: x, inputs=cloud_url_state, outputs=cloud_url_input)
        cloud_url_input.change(lambda x: x, inputs=cloud_url_input, outputs=cloud_url_state)

        # Mode Toggle logic
        def toggle_mode(mode):
            is_cloud = mode == "☁️ Cloud GPU (Remote Server)"
            return (
                gr.update(visible=is_cloud),
                gr.update(visible=is_cloud),
                gr.update(visible=not is_cloud),
                gr.update(visible=is_cloud),
                gr.update(visible=not is_cloud),
                gr.update(visible=not is_cloud)
            )

        mode_toggle.change(
            fn=toggle_mode,
            inputs=[mode_toggle],
            outputs=[cloud_settings_col, btn_process_cloud, btn_process, cloud_output_col, local_output_col, local_galleries_col]
        )

        btn_test_cloud.click(
            fn=test_cloud_connection,
            inputs=[cloud_url_input],
            outputs=[cloud_status_md]
        )

        btn_process_cloud_event = btn_process_cloud.click(
            fn=process_video_cloud,
            inputs=[input_video, model_selector, cloud_url_input, enable_seg_ui],
            outputs=[
                cloud_output_file,
                output_video, output_csv, output_crops_gallery, output_summary, output_charts_gallery, output_seg_gallery
            ]
        ).then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=True)),
            inputs=[],
            outputs=[local_output_col, local_galleries_col]
        )

        btn_refresh_cloud_history.click(
            fn=lambda url: gr.Dropdown(choices=fetch_cloud_runs(url)),
            inputs=[cloud_url_input],
            outputs=[cloud_history_selector]
        )
        
        btn_download_cloud_run.click(
            fn=download_cloud_run,
            inputs=[cloud_history_selector, cloud_url_input],
            outputs=[cloud_run_download]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[cloud_run_download]
        )

        btn_process_event = btn_process.click(
            fn=process_video,
            inputs=[input_video, model_selector, conf_slider, iou_slider, frame_skip_slider,
                    enable_dedup, merge_threshold_slider, track_buffer_slider, enable_seg_ui,
                    cam_height_slider, cam_tilt_slider, cam_focal_slider,
                    *class_sliders],
            outputs=[output_video, output_csv, output_crops_gallery, output_summary, output_charts_gallery, output_seg_gallery]
        )

        btn_revoke.click(
            fn=revoke_processing,
            inputs=[],
            outputs=[output_summary],
            cancels=[btn_process_event]
        )

        # ── Per-Class Confidence Dynamic Sliders Event Handlers ──
        model_selector.change(
            fn=on_model_selection_change,
            inputs=[model_selector, conf_slider],
            outputs=[per_class_info_md, *class_sliders]
        )

        btn_sync_all_conf.click(
            fn=sync_all_class_sliders,
            inputs=[model_selector, conf_slider],
            outputs=class_sliders
        )

        btn_reset_all_conf.click(
            fn=lambda m: sync_all_class_sliders(m, 0.25),
            inputs=[model_selector],
            outputs=class_sliders
        )

        # OpenVINO Model Toggle Event Handler
        openvino_toggle.change(
            fn=on_toggle_openvino,
            inputs=[openvino_toggle],
            outputs=[model_selector]
        ).then(
            fn=on_model_selection_change,
            inputs=[model_selector, conf_slider],
            outputs=[per_class_info_md, *class_sliders]
        )

        # Model Refresh Event Handler
        btn_refresh_models.click(
            fn=on_toggle_openvino,
            inputs=[openvino_toggle],
            outputs=[model_selector]
        ).then(
            fn=on_model_selection_change,
            inputs=[model_selector, conf_slider],
            outputs=[per_class_info_md, *class_sliders]
        )

        # Dedup Toggle Event Handler
        enable_dedup.change(
            fn=on_toggle_dedup,
            inputs=[enable_dedup],
            outputs=[dedup_status_md, btn_process, dedup_params_row]
        )

        # Quick Sample Selector Event Handlers
        quick_sample_dd.change(
            fn=lambda v: v,
            inputs=[quick_sample_dd],
            outputs=[input_video]
        )
        btn_refresh_samples.click(
            fn=lambda: gr.Dropdown(choices=get_example_videos(), value=None),
            inputs=[],
            outputs=[quick_sample_dd]
        )

        # ── GIS Geolocation Enrichment Event Handlers (New Inspection Tab) ──
        btn_refresh_gpx.click(
            fn=lambda: gr.Dropdown(
                choices=get_example_gpx_files(),
                value=get_example_gpx_files()[0] if get_example_gpx_files() else None
            ),
            inputs=[],
            outputs=[gis_gpx_dropdown]
        )

        btn_generate_geo.click(
            fn=on_generate_geo,
            inputs=[gis_gpx_upload, gis_gpx_dropdown, gis_offset_slider],
            outputs=[gis_status_md, gis_map_html, gis_download_file]
        )

        # Video Estimation Event Handlers
        input_video.change(
            fn=estimate_video_info,
            inputs=[input_video, frame_skip_slider],
            outputs=[video_est_md]
        )
        frame_skip_slider.change(
            fn=estimate_video_info,
            inputs=[input_video, frame_skip_slider],
            outputs=[video_est_md]
        )

        # ── History Page Event Handlers ──
        history_selector.change(
            fn=load_history_details,
            inputs=[history_selector],
            outputs=[
                hist_video, hist_csv, hist_crops_gallery, hist_summary, hist_charts_gallery, hist_seg_gallery,
                hist_gis_status_md, hist_gis_map_html, hist_gis_download_file
            ]
        )

        btn_refresh_history.click(
            fn=lambda: gr.Dropdown(
                choices=get_history_runs(),
                value=get_history_runs()[0] if get_history_runs() else None
            ),
            inputs=[],
            outputs=[history_selector]
        )

        # ── History GIS Geolocation Enrichment Event Handlers ──
        btn_hist_refresh_gpx.click(
            fn=lambda: gr.Dropdown(
                choices=get_example_gpx_files(),
                value=get_example_gpx_files()[0] if get_example_gpx_files() else None
            ),
            inputs=[],
            outputs=[hist_gis_gpx_dropdown]
        )

        btn_hist_generate_geo.click(
            fn=on_generate_geo_history,
            inputs=[history_selector, hist_gis_gpx_upload, hist_gis_gpx_dropdown, hist_gis_offset_slider],
            outputs=[hist_gis_status_md, hist_gis_map_html, hist_gis_download_file]
        )

        if history_runs:
            demo.load(
                fn=load_history_details,
                inputs=[history_selector],
                outputs=[
                    hist_video, hist_csv, hist_crops_gallery, hist_summary, hist_charts_gallery, hist_seg_gallery,
                    hist_gis_status_md, hist_gis_map_html, hist_gis_download_file
                ]
            )

        # ── Dataset Builder Event Handlers ──

        def _on_load_candidates():
            msg, count = prepare_annotation_candidates()
            if count > 0:
                crop_img, info, suggested, progress, is_done = get_current_candidate_display()
                frame_overlay, x1, y1, x2, y2 = get_full_frame_with_overlay()
                return (
                    msg,
                    gr.Column(visible=True),   # candidate_step_box
                    gr.Column(visible=False),  # review_step_box
                    gr.Markdown(value="", visible=False),  # ds_commit_result_md
                    crop_img, info, progress, gr.Dropdown(value=suggested),
                    frame_overlay, x1, y1, x2, y2, 0,
                    gr.Button(value="📋 Review Staged (0)")
                )
            return (
                msg,
                gr.Column(visible=True),
                gr.Column(visible=False),
                gr.Markdown(value="", visible=False),
                None, "No candidates available.", "", gr.Dropdown(),
                None, 0, 0, 0, 0, 0,
                gr.Button(value="📋 Review Staged (0)")
            )

        btn_load_candidates.click(
            fn=_on_load_candidates,
            inputs=[],
            outputs=[
                ds_status_md, candidate_step_box, review_step_box, ds_commit_result_md,
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider, btn_review_staged
            ]
        )

        def _on_add_from_image(image_file):
            msg, count = prepare_image_annotation_candidates(image_file)
            staged = _ANNOTATION_STATE.get("staged_defects", [])
            staged_btn_label = f"📋 Review Staged ({len(staged)})"
            if count > 0:
                crop_img, info, suggested, progress, is_done = get_current_candidate_display()
                frame_overlay, x1, y1, x2, y2 = get_full_frame_with_overlay()
                return (
                    msg,
                    gr.Column(visible=True),   # candidate_step_box
                    gr.Column(visible=False),  # review_step_box
                    gr.Markdown(value="", visible=False),  # ds_commit_result_md
                    crop_img, info, progress, gr.Dropdown(value=suggested),
                    frame_overlay, x1, y1, x2, y2, 0,
                    gr.Button(value=staged_btn_label)
                )
            return (
                msg,
                gr.Column(visible=True),
                gr.Column(visible=False),
                gr.Markdown(value="", visible=False),
                None, "Failed to load image.", "", gr.Dropdown(),
                None, 0, 0, 0, 0, 0,
                gr.Button(value=staged_btn_label)
            )

        btn_add_from_image.upload(
            fn=_on_add_from_image,
            inputs=[btn_add_from_image],
            outputs=[
                ds_status_md, candidate_step_box, review_step_box, ds_commit_result_md,
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider, btn_review_staged
            ]
        )

        def _on_accept(chosen_class, x1, y1, x2, y2):
            try:
                coords = (int(x1), int(y1), int(x2), int(y2))
            except (ValueError, TypeError):
                coords = None
            stage_candidate_annotation(chosen_class, bbox=coords)
            candidates = _ANNOTATION_STATE.get("candidates", [])
            idx = _ANNOTATION_STATE.get("current_index", 0)
            staged = _ANNOTATION_STATE.get("staged_defects", [])
            
            if idx >= len(candidates):
                # Completed last defect! Switch automatically to Confirmation & Review step
                prompt_md, dist_md, gallery_items, choices = get_staged_review_data()
                return (
                    gr.Column(visible=False),  # candidate_step_box
                    gr.Column(visible=True),   # review_step_box
                    None,                      # ds_crop_image
                    "All candidates reviewed.",# ds_info_md
                    f"Review Complete — Staged: {len(staged)}", # ds_progress_md
                    gr.Dropdown(),             # ds_class_dropdown
                    None, 0, 0, 0, 0, 0,       # ds_full_frame, x1, y1, x2, y2, padding
                    gr.Button(value=f"📋 Review Staged ({len(staged)})"),
                    prompt_md, dist_md, gallery_items,
                    gr.Dropdown(choices=choices, value=choices[0] if choices else None)
                )
            else:
                crop_img, info, suggested, progress, is_done = get_current_candidate_display()
                frame_overlay, rx1, ry1, rx2, ry2 = get_full_frame_with_overlay()
                return (
                    gr.Column(visible=True),
                    gr.Column(visible=False),
                    crop_img, info, progress, gr.Dropdown(value=suggested),
                    frame_overlay, rx1, ry1, rx2, ry2, 0,
                    gr.Button(value=f"📋 Review Staged ({len(staged)})"),
                    "", "", [], gr.Dropdown()
                )

        btn_accept.click(
            fn=_on_accept,
            inputs=[ds_class_dropdown, ds_x1, ds_y1, ds_x2, ds_y2],
            outputs=[
                candidate_step_box, review_step_box,
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider, btn_review_staged,
                ds_review_prompt_md, ds_review_dist_md, ds_review_gallery, ds_edit_selector
            ]
        )

        def _on_add_another(chosen_class, x1, y1, x2, y2):
            try:
                coords = (int(x1), int(y1), int(x2), int(y2))
            except (ValueError, TypeError):
                coords = None
            stage_another_defect_annotation(chosen_class, bbox=coords)
            staged = _ANNOTATION_STATE.get("staged_defects", [])
            crop_img, info, suggested, progress, is_done = get_current_candidate_display()
            frame_overlay, rx1, ry1, rx2, ry2 = get_full_frame_with_overlay()
            return (
                crop_img, info, progress, gr.Dropdown(value=suggested),
                frame_overlay, rx1, ry1, rx2, ry2, 0,
                gr.Button(value=f"📋 Review Staged ({len(staged)})")
            )

        btn_add_another.click(
            fn=_on_add_another,
            inputs=[ds_class_dropdown, ds_x1, ds_y1, ds_x2, ds_y2],
            outputs=[
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider, btn_review_staged
            ]
        )

        def _on_skip():
            skip_annotation()
            candidates = _ANNOTATION_STATE.get("candidates", [])
            idx = _ANNOTATION_STATE.get("current_index", 0)
            staged = _ANNOTATION_STATE.get("staged_defects", [])

            if idx >= len(candidates):
                prompt_md, dist_md, gallery_items, choices = get_staged_review_data()
                return (
                    gr.Column(visible=False),
                    gr.Column(visible=True),
                    None, "All candidates reviewed.", f"Review Complete — Staged: {len(staged)}", gr.Dropdown(),
                    None, 0, 0, 0, 0, 0,
                    gr.Button(value=f"📋 Review Staged ({len(staged)})"),
                    prompt_md, dist_md, gallery_items,
                    gr.Dropdown(choices=choices, value=choices[0] if choices else None)
                )
            else:
                crop_img, info, suggested, progress, is_done = get_current_candidate_display()
                frame_overlay, x1, y1, x2, y2 = get_full_frame_with_overlay()
                return (
                    gr.Column(visible=True),
                    gr.Column(visible=False),
                    crop_img, info, progress, gr.Dropdown(value=suggested),
                    frame_overlay, x1, y1, x2, y2, 0,
                    gr.Button(value=f"📋 Review Staged ({len(staged)})"),
                    "", "", [], gr.Dropdown()
                )

        btn_skip.click(
            fn=_on_skip,
            inputs=[],
            outputs=[
                candidate_step_box, review_step_box,
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider, btn_review_staged,
                ds_review_prompt_md, ds_review_dist_md, ds_review_gallery, ds_edit_selector
            ]
        )

        def _on_prev_candidate():
            previous_annotation()
            staged = _ANNOTATION_STATE.get("staged_defects", [])
            crop_img, info, suggested, progress, is_done = get_current_candidate_display()
            frame_overlay, x1, y1, x2, y2 = get_full_frame_with_overlay()
            return (
                crop_img, info, progress, gr.Dropdown(value=suggested),
                frame_overlay, x1, y1, x2, y2, 0,
                gr.Button(value=f"📋 Review Staged ({len(staged)})")
            )

        btn_prev_candidate.click(
            fn=_on_prev_candidate,
            inputs=[],
            outputs=[
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider, btn_review_staged
            ]
        )

        def _on_go_to_review():
            prompt_md, dist_md, gallery_items, choices = get_staged_review_data()
            return (
                gr.Column(visible=False),  # candidate_step_box
                gr.Column(visible=True),   # review_step_box
                prompt_md, dist_md, gallery_items,
                gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            )

        btn_review_staged.click(
            fn=_on_go_to_review,
            inputs=[],
            outputs=[
                candidate_step_box, review_step_box,
                ds_review_prompt_md, ds_review_dist_md, ds_review_gallery, ds_edit_selector
            ]
        )

        def _on_back_to_candidates():
            staged = _ANNOTATION_STATE.get("staged_defects", [])
            crop_img, info, suggested, progress, is_done = get_current_candidate_display()
            frame_overlay, x1, y1, x2, y2 = get_full_frame_with_overlay()
            return (
                gr.Column(visible=True),
                gr.Column(visible=False),
                crop_img, info, progress, gr.Dropdown(value=suggested),
                frame_overlay, x1, y1, x2, y2, 0,
                gr.Button(value=f"📋 Review Staged ({len(staged)})")
            )

        btn_back_to_candidates.click(
            fn=_on_back_to_candidates,
            inputs=[],
            outputs=[
                candidate_step_box, review_step_box,
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider, btn_review_staged
            ]
        )

        # ── Staged Defect Edit Handlers ──

        def _on_select_edit(selected_label):
            return select_staged_defect_for_edit(selected_label)

        ds_edit_selector.change(
            fn=_on_select_edit,
            inputs=[ds_edit_selector],
            outputs=[
                ds_edit_crop_image, ds_edit_full_frame, ds_edit_class_dropdown,
                ds_edit_padding_slider, ds_edit_x1, ds_edit_y1, ds_edit_x2, ds_edit_y2,
                ds_edit_status_md
            ]
        )

        def _on_refresh_edit_list():
            prompt_md, dist_md, gallery_items, choices = get_staged_review_data()
            return prompt_md, dist_md, gallery_items, gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        btn_refresh_edit_list.click(
            fn=_on_refresh_edit_list,
            inputs=[],
            outputs=[ds_review_prompt_md, ds_review_dist_md, ds_review_gallery, ds_edit_selector]
        )

        def _on_edit_padding_change(pad):
            crop_rgb, html, x1, y1, x2, y2 = adjust_staged_bbox_padding(pad)
            return crop_rgb, html, x1, y1, x2, y2

        ds_edit_padding_slider.change(
            fn=_on_edit_padding_change,
            inputs=[ds_edit_padding_slider],
            outputs=[ds_edit_crop_image, ds_edit_full_frame, ds_edit_x1, ds_edit_y1, ds_edit_x2, ds_edit_y2]
        )

        def _on_edit_coords_change(x1, y1, x2, y2):
            crop_rgb, html = apply_staged_bbox_coords(x1, y1, x2, y2)
            return crop_rgb, html

        gr.on(
            triggers=[ds_edit_x1.change, ds_edit_y1.change, ds_edit_x2.change, ds_edit_y2.change],
            fn=_on_edit_coords_change,
            inputs=[ds_edit_x1, ds_edit_y1, ds_edit_x2, ds_edit_y2],
            outputs=[ds_edit_crop_image, ds_edit_full_frame]
        )

        def _on_edit_reset_box():
            crop_rgb, html, x1, y1, x2, y2, pad = reset_staged_bbox_to_original()
            return crop_rgb, html, x1, y1, x2, y2, pad

        btn_edit_reset_box.click(
            fn=_on_edit_reset_box,
            inputs=[],
            outputs=[
                ds_edit_crop_image, ds_edit_full_frame,
                ds_edit_x1, ds_edit_y1, ds_edit_x2, ds_edit_y2, ds_edit_padding_slider
            ]
        )

        def _on_save_staged_edit(chosen_class, x1, y1, x2, y2):
            try:
                coords = (int(x1), int(y1), int(x2), int(y2))
            except (ValueError, TypeError):
                coords = None
            return save_staged_defect_changes(chosen_class, bbox=coords)

        btn_save_staged_edit.click(
            fn=_on_save_staged_edit,
            inputs=[ds_edit_class_dropdown, ds_edit_x1, ds_edit_y1, ds_edit_x2, ds_edit_y2],
            outputs=[
                ds_review_prompt_md, ds_review_dist_md, ds_review_gallery,
                ds_edit_selector, ds_edit_status_md
            ]
        )

        def _on_delete_staged():
            return delete_staged_defect()

        btn_delete_staged.click(
            fn=_on_delete_staged,
            inputs=[],
            outputs=[
                ds_review_prompt_md, ds_review_dist_md, ds_review_gallery, ds_edit_selector,
                ds_edit_crop_image, ds_edit_full_frame, ds_edit_class_dropdown,
                ds_edit_padding_slider, ds_edit_x1, ds_edit_y1, ds_edit_x2, ds_edit_y2,
                ds_edit_status_md
            ]
        )

        # ── Final Confirmation & Commit Handlers ──

        def _on_confirm_dataset():
            success_md, stats_md, review_visible = commit_staged_defects_to_dataset()
            return (
                gr.Markdown(value=success_md, visible=True),  # ds_commit_result_md
                stats_md,                                     # ds_stats_md
                gr.Column(visible=False),                     # review_step_box
                gr.Column(visible=False),                     # candidate_step_box
                "✅ Successfully committed defects to RDD_Ahmedabad dataset!" # ds_status_md
            )

        btn_confirm_dataset.click(
            fn=_on_confirm_dataset,
            inputs=[],
            outputs=[
                ds_commit_result_md, ds_stats_md, review_step_box, candidate_step_box, ds_status_md
            ]
        )

        def _on_discard_batch():
            msg = discard_staged_batch()
            return (
                msg,                       # ds_status_md
                gr.Column(visible=True),   # candidate_step_box
                gr.Column(visible=False),  # review_step_box
                gr.Markdown(value="", visible=False), # ds_commit_result_md
                None, "Batch discarded. Click Load Detections to start a new session.", "",
                None, gr.Button(value="📋 Review Staged (0)")
            )

        btn_discard_batch.click(
            fn=_on_discard_batch,
            inputs=[],
            outputs=[
                ds_status_md, candidate_step_box, review_step_box, ds_commit_result_md,
                ds_crop_image, ds_info_md, ds_progress_md, ds_full_frame, btn_review_staged
            ]
        )

        btn_refresh_stats.click(
            fn=get_dataset_stats_md,
            inputs=[],
            outputs=[ds_stats_md]
        )

        # ── Crop Adjustment Event Handlers ──

        def _on_padding_change(padding_px):
            """Live update crop and frame overlay when padding slider changes."""
            crop_rgb, frame_annotated, x1, y1, x2, y2 = adjust_bbox_with_padding(padding_px)
            return crop_rgb, frame_annotated, x1, y1, x2, y2

        ds_padding_slider.change(
            fn=_on_padding_change,
            inputs=[ds_padding_slider],
            outputs=[ds_crop_image, ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2]
        )

        def _on_coords_change(x1, y1, x2, y2):
            """Apply manually entered bbox coordinates (or dynamically dragged coordinates from canvas)."""
            crop_rgb, frame_annotated = apply_bbox_coord_adjustment(x1, y1, x2, y2)
            return crop_rgb, frame_annotated

        gr.on(
            triggers=[ds_x1.change, ds_y1.change, ds_x2.change, ds_y2.change],
            fn=_on_coords_change,
            inputs=[ds_x1, ds_y1, ds_x2, ds_y2],
            outputs=[ds_crop_image, ds_full_frame]
        )

        def _on_reset_box():
            """Reset bbox to original detection coordinates."""
            crop_rgb, frame_annotated, x1, y1, x2, y2, pad = reset_bbox_to_original()
            return crop_rgb, frame_annotated, x1, y1, x2, y2, pad

        btn_reset_box.click(
            fn=_on_reset_box,
            inputs=[],
            outputs=[
                ds_crop_image, ds_full_frame,
                ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider
            ]
        )

        # ── Dataset Curation & Revival Event Handlers ──

        def _on_curate_filter(class_filter, split_filter, search_term):
            """Apply filters to dataset and reload defect view from position 0."""
            choices = load_curate_dataset(class_filter, split_filter, search_term)
            crop_rgb, html, info_md, cname, x1, y1, x2, y2, pad, pos_badge, pos, status_md = get_curate_defect_display(0)
            dropdown_val = 0 if choices else None
            return (
                crop_rgb, html, info_md, cname,
                x1, y1, x2, y2, pad, pos_badge,
                gr.Dropdown(choices=choices, value=dropdown_val),
                status_md
            )

        btn_curate_filter_apply.click(
            fn=_on_curate_filter,
            inputs=[curate_class_filter, curate_split_filter, curate_search_input],
            outputs=[
                curate_crop_image, curate_full_frame, curate_info_md, curate_class_dropdown,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider,
                curate_pos_indicator_md, curate_jump_dropdown, curate_status_md
            ]
        )

        def _on_curate_refresh():
            """Refresh annotations from disk, update class filter choices and reload view."""
            curate_classes = get_curate_class_choices()
            choices = load_curate_dataset("ALL", "ALL", "")
            crop_rgb, html, info_md, cname, x1, y1, x2, y2, pad, pos_badge, pos, status_md = get_curate_defect_display(0)
            dropdown_val = 0 if choices else None
            return (
                gr.Dropdown(choices=curate_classes, value="ALL"),
                gr.Dropdown(value="ALL"),
                "",
                crop_rgb, html, info_md, cname,
                x1, y1, x2, y2, pad, pos_badge,
                gr.Dropdown(choices=choices, value=dropdown_val),
                status_md,
                get_dataset_stats_md()
            )

        btn_curate_refresh.click(
            fn=_on_curate_refresh,
            inputs=[],
            outputs=[
                curate_class_filter, curate_split_filter, curate_search_input,
                curate_crop_image, curate_full_frame, curate_info_md, curate_class_dropdown,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider,
                curate_pos_indicator_md, curate_jump_dropdown, curate_status_md,
                ds_stats_md
            ]
        )

        def _on_curate_jump(pos):
            """Jump to defect selected in dropdown."""
            if pos is None:
                pos = 0
            crop_rgb, html, info_md, cname, x1, y1, x2, y2, pad, pos_badge, pos_idx, status_md = get_curate_defect_display(pos)
            return (
                crop_rgb, html, info_md, cname,
                x1, y1, x2, y2, pad, pos_badge,
                pos_idx, status_md
            )

        curate_jump_dropdown.change(
            fn=_on_curate_jump,
            inputs=[curate_jump_dropdown],
            outputs=[
                curate_crop_image, curate_full_frame, curate_info_md, curate_class_dropdown,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider,
                curate_pos_indicator_md, curate_jump_dropdown, curate_status_md
            ]
        )

        def _on_curate_prev():
            cur_pos = _CURATE_STATE.get("current_pos", 0)
            target = cur_pos - 1
            crop_rgb, html, info_md, cname, x1, y1, x2, y2, pad, pos_badge, pos_idx, status_md = get_curate_defect_display(target)
            return (
                crop_rgb, html, info_md, cname,
                x1, y1, x2, y2, pad, pos_badge,
                pos_idx, status_md
            )

        btn_curate_prev.click(
            fn=_on_curate_prev,
            inputs=[],
            outputs=[
                curate_crop_image, curate_full_frame, curate_info_md, curate_class_dropdown,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider,
                curate_pos_indicator_md, curate_jump_dropdown, curate_status_md
            ]
        )

        def _on_curate_next():
            cur_pos = _CURATE_STATE.get("current_pos", 0)
            target = cur_pos + 1
            crop_rgb, html, info_md, cname, x1, y1, x2, y2, pad, pos_badge, pos_idx, status_md = get_curate_defect_display(target)
            return (
                crop_rgb, html, info_md, cname,
                x1, y1, x2, y2, pad, pos_badge,
                pos_idx, status_md
            )

        btn_curate_next.click(
            fn=_on_curate_next,
            inputs=[],
            outputs=[
                curate_crop_image, curate_full_frame, curate_info_md, curate_class_dropdown,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider,
                curate_pos_indicator_md, curate_jump_dropdown, curate_status_md
            ]
        )

        def _on_curate_random():
            df = _CURATE_STATE.get("df")
            import random
            target = random.randint(0, len(df) - 1) if df is not None and len(df) > 0 else 0
            crop_rgb, html, info_md, cname, x1, y1, x2, y2, pad, pos_badge, pos_idx, status_md = get_curate_defect_display(target)
            return (
                crop_rgb, html, info_md, cname,
                x1, y1, x2, y2, pad, pos_badge,
                pos_idx, status_md
            )

        btn_curate_random.click(
            fn=_on_curate_random,
            inputs=[],
            outputs=[
                curate_crop_image, curate_full_frame, curate_info_md, curate_class_dropdown,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider,
                curate_pos_indicator_md, curate_jump_dropdown, curate_status_md
            ]
        )

        def _on_curate_padding_change(pad):
            crop_rgb, html, x1, y1, x2, y2 = curate_adjust_padding(pad)
            return crop_rgb, html, x1, y1, x2, y2

        curate_padding_slider.change(
            fn=_on_curate_padding_change,
            inputs=[curate_padding_slider],
            outputs=[curate_crop_image, curate_full_frame, curate_x1, curate_y1, curate_x2, curate_y2]
        )

        def _on_curate_coords_change(x1, y1, x2, y2):
            crop_rgb, html = curate_apply_coords(x1, y1, x2, y2)
            return crop_rgb, html

        gr.on(
            triggers=[curate_x1.change, curate_y1.change, curate_x2.change, curate_y2.change],
            fn=_on_curate_coords_change,
            inputs=[curate_x1, curate_y1, curate_x2, curate_y2],
            outputs=[curate_crop_image, curate_full_frame]
        )

        def _on_curate_reset():
            crop_rgb, html, x1, y1, x2, y2, pad = curate_reset_bbox()
            return crop_rgb, html, x1, y1, x2, y2, pad

        btn_curate_reset_box.click(
            fn=_on_curate_reset,
            inputs=[],
            outputs=[
                curate_crop_image, curate_full_frame,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider
            ]
        )

        def _on_curate_save(chosen_class, x1, y1, x2, y2):
            crop_rgb, html, info_md, cname, rx1, ry1, rx2, ry2, pad, pos_badge, pos_idx, status_md, stats_md = curate_save_changes(chosen_class, x1, y1, x2, y2)
            df = _CURATE_STATE.get("df")
            choices = []
            if df is not None:
                for i, r in df.iterrows():
                    cn = r.get("class_name", "")
                    img_name = str(r.get("image_name", ""))
                    short_img = img_name[:24] + "..." if len(img_name) > 24 else img_name
                    choices.append((f"#{i+1}: [{cn}] {short_img} ({r.get('split', '')})", i))
            return (
                crop_rgb, html, info_md, cname,
                rx1, ry1, rx2, ry2, pad, pos_badge,
                gr.Dropdown(choices=choices, value=pos_idx),
                status_md,
                stats_md
            )

        btn_curate_save.click(
            fn=_on_curate_save,
            inputs=[curate_class_dropdown, curate_x1, curate_y1, curate_x2, curate_y2],
            outputs=[
                curate_crop_image, curate_full_frame, curate_info_md, curate_class_dropdown,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider,
                curate_pos_indicator_md, curate_jump_dropdown, curate_status_md,
                ds_stats_md
            ]
        )

        def _on_curate_delete():
            crop_rgb, html, info_md, cname, rx1, ry1, rx2, ry2, pad, pos_badge, pos_idx, status_md, stats_md = curate_delete_defect()
            df = _CURATE_STATE.get("df")
            choices = []
            if df is not None:
                for i, r in df.iterrows():
                    cn = r.get("class_name", "")
                    img_name = str(r.get("image_name", ""))
                    short_img = img_name[:24] + "..." if len(img_name) > 24 else img_name
                    choices.append((f"#{i+1}: [{cn}] {short_img} ({r.get('split', '')})", i))
            return (
                crop_rgb, html, info_md, cname,
                rx1, ry1, rx2, ry2, pad, pos_badge,
                gr.Dropdown(choices=choices, value=pos_idx),
                status_md,
                stats_md
            )

        btn_curate_delete.click(
            fn=_on_curate_delete,
            inputs=[],
            outputs=[
                curate_crop_image, curate_full_frame, curate_info_md, curate_class_dropdown,
                curate_x1, curate_y1, curate_x2, curate_y2, curate_padding_slider,
                curate_pos_indicator_md, curate_jump_dropdown, curate_status_md,
                ds_stats_md
            ]
        )

    return demo, theme, custom_css


if __name__ == "__main__":
    app, theme, css = build_ui()
    target_port = int(os.environ.get("GRADIO_SERVER_PORT", 7860))
    print(f"[RDD-YOLO] Launching Gradio Web UI on http://127.0.0.1:{target_port} ...")
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=target_port,
            share=False,
            theme=theme,
            css=css,
            allowed_paths=[OUTPUTS_DIR, BASE_DIR]
        )
    except OSError:
        print(f"[RDD-YOLO] Port {target_port} occupied. Automatically searching for next available port...")
        app.launch(
            server_name="127.0.0.1",
            share=False,
            theme=theme,
            css=css,
            allowed_paths=[OUTPUTS_DIR, BASE_DIR]
        )



