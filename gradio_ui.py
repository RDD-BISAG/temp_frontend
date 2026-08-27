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
import numpy as np
import pandas as pd
import gradio as gr

# Silence WinError 10054 ConnectionResetError noise on Windows asyncio event loops
if sys.platform == "win32":
    try:
        def _silence_win_connection_reset(loop, context):
            exception = context.get("exception")
            if isinstance(exception, ConnectionResetError):
                return
            loop.default_exception_handler(context)

        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop and not loop.is_closed():
                loop.set_exception_handler(_silence_win_connection_reset)
        except Exception:
            pass
    except Exception:
        pass

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
    "accepted_count": 0,
    "skipped_count": 0,
    "changed_count": 0,
    "last_pipeline_result": None,
    "last_video_path": None,
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


def get_available_models():
    """Scans the models directory for available PyTorch (.pt) weights and prioritizes RDD-YOLO models."""
    model_files = []
    
    if os.path.exists(MODELS_DIR):
        for root, _, files in os.walk(MODELS_DIR):
            for file in files:
                if file.endswith(".pt") or file.endswith(".onnx"):
                    rel_path = os.path.relpath(os.path.join(root, file), MODELS_DIR)
                    rel_path = rel_path.replace("\\", "/")
                    model_files.append(rel_path)
                    
    if not model_files:
        model_files = ["rdd_yolo.pt (Default)", "yolov8n.pt (Baseline)", "Demo / Simulation Mode"]
    else:
        model_files.append("yolov8n.pt (Baseline)")
        model_files.append("Demo / Simulation Mode")

    return model_files


def get_example_videos():
    """Finds available sample videos in the inputs directory."""
    examples = []
    if os.path.exists(INPUTS_DIR):
        for file in sorted(os.listdir(INPUTS_DIR)):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                examples.append(os.path.join(INPUTS_DIR, file))
    return examples


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
    """Loads past analysis details (video, csv, crops gallery, summary markdown) for selected run."""
    if not selected_run_folder:
        return None, None, [], "Please select an analysis run from the dropdown.", []

    run_dir = os.path.join(OUTPUTS_DIR, selected_run_folder)
    if not os.path.exists(run_dir):
        return None, None, [], f"❌ Error: Run directory '{selected_run_folder}' not found.", []

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

    # Locate crops
    crop_gallery = []
    crops_dir = os.path.join(run_dir, "crops")
    if os.path.exists(crops_dir):
        for f in sorted(os.listdir(crops_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                crop_path = os.path.join(crops_dir, f)
                crop_gallery.append((crop_path, f))

    # Read summary.json or build summary markdown
    json_path = os.path.join(run_dir, "summary.json")
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
            defect_config = {
                        "D00" : "Longituduinal Crack",
                        "D10" : "Transverse Crack",
                        "D20" : "Alligator Crack",
                        "D40" : "Pothole"
                    }
            for cls_name, count in defect_counts.items():
                summary_md += f"- **{cls_name}**: `{count}` detected\n"
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

    return video_path, csv_path, crop_gallery, summary_md, charts_gallery


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


def process_video(video_path, model_name, conf_threshold, iou_threshold, frame_skip,
                  enable_dedup=True, merge_threshold=0.65, track_buffer=45,
                  progress=gr.Progress()):
    """
    Processes video for Road Damage Detection with integrated duplicate removal.

    When dedup is enabled, runs the full pipeline:
      YOLO + BoT-SORT tracking → TrackManager → DuplicateVerifier → DefectDatabase

    When dedup is disabled, falls back to the original frame-by-frame detection.

    Saves outputs in outputs/analysis_TIMESTAMP/:
      - Annotated Output Video (.mp4) with track IDs
      - Unique Defects CSV Log (.csv)
      - Deduplicated Defect Crops (crops/*.jpg)
      - Summary Report with dedup statistics
      - Merge audit log (merge_audit_log.json)
      - SQLite defect database (defects.db)
    """
    if not video_path or not os.path.exists(video_path):
        return None, None, [], "❌ Error: Invalid or missing video input file.", []

    # Reset revocation flag for new processing session
    _CANCEL_FLAG["requested"] = False

    # ── Dedup Pipeline Path ──
    if enable_dedup and HAS_DEDUP_PIPELINE and "Demo" not in model_name:
        return _process_video_with_dedup(
            video_path, model_name, conf_threshold, iou_threshold,
            frame_skip, merge_threshold, track_buffer, progress
        )

    # ── Fallback: Original frame-by-frame detection (no tracking/dedup) ──
    return _process_video_legacy(
        video_path, model_name, conf_threshold, iou_threshold,
        frame_skip, progress
    )


def _process_video_with_dedup(video_path, model_name, conf_threshold, iou_threshold,
                              frame_skip, merge_threshold, track_buffer, progress):
    """
    Full dedup pipeline: YOLO + BoT-SORT + TrackManager + DuplicateVerifier.
    Returns the same 4 outputs as the original process_video for Gradio compatibility.
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

    config = PipelineConfig(
        model_path=model_path,
        confidence_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        frame_skip=int(frame_skip),
        merge_threshold=merge_threshold,
        track_buffer_frames=int(track_buffer),
        save_annotated_video=True,
        save_crops=True,
        database_type="sqlite",
        _base_dir=BASE_DIR,
    )

    # Create pipeline and run
    pipeline = RDDPipeline(config=config)
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
        return None, None, [], f"⚠️ **Process Revoked**: {e}", []
    except Exception as e:
        if _CANCEL_FLAG["requested"]:
            return None, None, [], "⚠️ **Process Revoked**: Video processing was cancelled by user revocation request.", []
        import traceback
        traceback.print_exc()
        return None, None, [], f"❌ Pipeline error: {e}", []

    # Store result for dataset builder
    _ANNOTATION_STATE["last_pipeline_result"] = result
    _ANNOTATION_STATE["last_video_path"] = video_path
    _ANNOTATION_STATE["last_inspection_mode"] = "dedup"

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

    summary_md += f"""
---
#### ⚙️ Pipeline Configuration
- Confidence Threshold: `{conf_threshold}`
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

    return final_video, csv_path, crop_gallery_images, summary_md, charts_gallery


# ═══════════════════════════════════════════════════════════════════════════════
# RDD_Ahmedabad Dataset Builder — Backend Functions
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
    _ANNOTATION_STATE["accepted_count"] = 0
    _ANNOTATION_STATE["skipped_count"] = 0
    _ANNOTATION_STATE["changed_count"] = 0

    mode_label = "Deduplicated" if mode == "dedup" else "Raw Detections (No Dedup)"
    return f"✅ {len(candidates)} defect crops ready for review ({mode_label}).", len(candidates)


def get_current_candidate_display():
    """
    Return display data for the current annotation candidate.
    Returns: (crop_image, info_markdown, suggested_class, progress_text, is_done)
    """
    candidates = _ANNOTATION_STATE.get("candidates", [])
    idx = _ANNOTATION_STATE.get("current_index", 0)

    if not candidates or idx >= len(candidates):
        # Session complete
        accepted = _ANNOTATION_STATE.get("accepted_count", 0)
        skipped = _ANNOTATION_STATE.get("skipped_count", 0)
        changed = _ANNOTATION_STATE.get("changed_count", 0)
        stats_md = RDD_DATASET_MANAGER.get_stats_markdown() if RDD_DATASET_MANAGER else ""
        done_md = f"""
### ✅ Dataset Collection Complete!

| Action | Count |
|---|---|
| Accepted | **{accepted}** |
| Class Changed | **{changed}** |
| Skipped | **{skipped}** |
| Total Reviewed | **{accepted + skipped}** |

---
{stats_md}
"""
        return None, done_md, "", f"Done — {accepted + skipped}/{len(candidates)} reviewed", True

    c = candidates[idx]

    # Auto-suggest class using model mapping
    suggested = c.class_name
    if RDD_CLASS_CONFIG:
        suggested = RDD_CLASS_CONFIG.map_model_class(c.predicted_class)
    
    # Ensure uppercase matching dropdown options
    if isinstance(suggested, str):
        suggested = suggested.upper()

    info_md = f"""
### Detection #{idx + 1} / {len(candidates)}

| Field | Value |
|---|---|
| **Predicted Class** | `{c.predicted_class}` |
| **Suggested (RDD_Ahmedabad)** | `{suggested}` |
| **Confidence** | `{c.confidence:.2f}` |
| **Frame** | `{c.frame_number}` |
| **Track ID** | `{c.track_id}` |
| **Source Video** | `{c.source_video}` |
| **Bbox** | `{c.bbox}` |
"""

    accepted = _ANNOTATION_STATE.get("accepted_count", 0)
    skipped = _ANNOTATION_STATE.get("skipped_count", 0)
    progress_text = f"Crop {idx + 1} / {len(candidates)} | Accepted: {accepted} | Skipped: {skipped}"

    # Convert BGR crop to RGB for Gradio display
    crop_rgb = None
    if c.crop_image is not None:
        crop_rgb = cv2.cvtColor(c.crop_image, cv2.COLOR_BGR2RGB)

    return crop_rgb, info_md, suggested, progress_text, False


def accept_annotation(chosen_class):
    """
    Accept the current candidate with the chosen class and save to dataset.
    Returns: next candidate display tuple.
    """
    candidates = _ANNOTATION_STATE.get("candidates", [])
    idx = _ANNOTATION_STATE.get("current_index", 0)

    if not candidates or idx >= len(candidates):
        return get_current_candidate_display()

    c = candidates[idx]

    if not RDD_CLASS_CONFIG or not RDD_DATASET_MANAGER or not RDD_ANNOTATION_MANAGER:
        _ANNOTATION_STATE["current_index"] = idx + 1
        return get_current_candidate_display()

    # Resolve class
    class_name = chosen_class.strip().upper() if chosen_class else c.class_name
    class_id = RDD_CLASS_CONFIG.get_class_id(class_name)
    if class_id < 0:
        class_id = 0
        class_name = RDD_CLASS_CONFIG.get_class_name(0)

    # Track if class was changed
    original_suggested = RDD_CLASS_CONFIG.map_model_class(c.predicted_class)
    if class_name != original_suggested:
        _ANNOTATION_STATE["changed_count"] = _ANNOTATION_STATE.get("changed_count", 0) + 1

    # Save to dataset
    saved = RDD_DATASET_MANAGER.add_annotation(
        frame=c.original_frame,
        bbox=c.bbox,
        class_id=class_id,
        class_name=class_name,
        source_video=c.source_video,
        frame_number=c.frame_number,
        track_id=c.track_id,
        confidence=c.confidence,
        crop_image=c.crop_image,
    )

    # Record metadata
    x1, y1, x2, y2 = c.bbox
    RDD_ANNOTATION_MANAGER.add_record(
        image_name=saved.get("image_filename", ""),
        crop_name=saved.get("crop_filename", ""),
        class_id=class_id,
        class_name=class_name,
        confidence=c.confidence,
        source_video=c.source_video,
        frame_number=c.frame_number,
        track_id=c.track_id,
        bbox_x=x1,
        bbox_y=y1,
        bbox_width=x2 - x1,
        bbox_height=y2 - y1,
        split=saved.get("split", "train"),
    )

    _ANNOTATION_STATE["accepted_count"] = _ANNOTATION_STATE.get("accepted_count", 0) + 1
    _ANNOTATION_STATE["current_index"] = idx + 1

    return get_current_candidate_display()


def skip_annotation():
    """Skip the current candidate and advance to the next."""
    _ANNOTATION_STATE["skipped_count"] = _ANNOTATION_STATE.get("skipped_count", 0) + 1
    _ANNOTATION_STATE["current_index"] = _ANNOTATION_STATE.get("current_index", 0) + 1
    return get_current_candidate_display()


def get_dataset_stats_md():
    """Return dataset statistics as markdown."""
    if RDD_DATASET_MANAGER:
        return RDD_DATASET_MANAGER.get_stats_markdown()
    return "Dataset system not initialized."


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


def render_bbox_editor_html(frame_rgb, bbox, canvas_id="bbox_canvas"):
    """
    Render an interactive HTML5 canvas with draggable bounding box corners.
    The user can drag corner handles, edge midpoints, or move the entire box.
    On release, the JS auto-updates the Gradio X1/Y1/X2/Y2 Number inputs.
    """
    if frame_rgb is None:
        return '<div style="text-align:center;color:#94a3b8;padding:40px;font-family:Inter,sans-serif;">No frame loaded. Click ✏️ Adjust Crop to load the source frame.</div>'

    b64_str = _encode_frame_base64(frame_rgb)
    if b64_str is None:
        return '<div style="text-align:center;color:#94a3b8;padding:40px;">Failed to encode frame.</div>'

    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Build the inner HTML document for the iframe srcdoc.
    # Gradio 5+/6+ sanitizes <script> tags in gr.HTML, so we must use an
    # <iframe srcdoc> to get JavaScript execution for the interactive canvas.
    inner_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
  canvas {{ display:block; max-width:100%; border-radius:8px; border:1px solid rgba(255,255,255,0.08); }}
  .hint {{ text-align:center; padding:6px 0; font-size:0.78rem; color:#94a3b8; font-family:Inter,sans-serif; }}
</style>
</head><body>
<canvas id="cv"></canvas>
<div class="hint">🟠 Drag corners to resize &nbsp;·&nbsp; Drag inside box to move &nbsp;·&nbsp; Click 📐 Apply to update crop</div>
<script>
(function() {{
  const C = document.getElementById('cv');
  if (!C) return;
  const ctx = C.getContext('2d');
  const img = new Image();
  img.src = 'data:image/jpeg;base64,{b64_str}';

  let box = {{ x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2} }};
  let drag = null;
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
    const sx1=box.x1*sc, sy1=box.y1*sc, sx2=box.x2*sc, sy2=box.y2*sc;
    const bw=sx2-sx1, bh=sy2-sy1;

    // Dim area outside box
    ctx.fillStyle='rgba(0,0,0,0.45)';
    ctx.fillRect(0,0,C.width,sy1);
    ctx.fillRect(0,sy2,C.width,C.height-sy2);
    ctx.fillRect(0,sy1,sx1,bh);
    ctx.fillRect(sx2,sy1,C.width-sx2,bh);

    // Box border
    ctx.strokeStyle='#22c55e'; ctx.lineWidth=2;
    ctx.strokeRect(sx1,sy1,bw,bh);

    // Corner handles
    const hs=10;
    ctx.fillStyle='#f97316';
    [[sx1,sy1],[sx2,sy1],[sx1,sy2],[sx2,sy2]].forEach(function(p){{
      ctx.fillRect(p[0]-hs/2,p[1]-hs/2,hs,hs);
    }});

    // Edge midpoint handles
    ctx.fillStyle='#38bdf8';
    const emid=7;
    [[sx1+bw/2,sy1],[sx1+bw/2,sy2],[sx1,sy1+bh/2],[sx2,sy1+bh/2]].forEach(function(p){{
      ctx.beginPath(); ctx.arc(p[0],p[1],emid/2,0,Math.PI*2); ctx.fill();
    }});

    // Coordinate label
    ctx.fillStyle='#22c55e'; ctx.font='bold 11px monospace';
    ctx.fillText('('+Math.round(box.x1)+','+Math.round(box.y1)+') - ('+Math.round(box.x2)+','+Math.round(box.y2)+')', sx1, Math.max(sy1-6,13));

    // Size label
    const ww=Math.round(box.x2-box.x1), hh=Math.round(box.y2-box.y1);
    ctx.fillStyle='rgba(0,0,0,0.7)'; ctx.fillRect(sx1,sy2+4,90,18); ctx.fillStyle='#f1f5f9'; ctx.font='11px monospace';
    ctx.fillText(ww+'x'+hh+'px', sx1+4, sy2+16);
  }}

  function mpos(e) {{
    const r=C.getBoundingClientRect();
    return {{ x:e.clientX-r.left, y:e.clientY-r.top }};
  }}

  function hit(p) {{
    const g=14;
    const sx1=box.x1*sc, sy1=box.y1*sc, sx2=box.x2*sc, sy2=box.y2*sc;
    const mx=(sx1+sx2)/2, my=(sy1+sy2)/2;
    if(Math.abs(p.x-sx1)<g && Math.abs(p.y-sy1)<g) return 'tl';
    if(Math.abs(p.x-sx2)<g && Math.abs(p.y-sy1)<g) return 'tr';
    if(Math.abs(p.x-sx1)<g && Math.abs(p.y-sy2)<g) return 'bl';
    if(Math.abs(p.x-sx2)<g && Math.abs(p.y-sy2)<g) return 'br';
    if(Math.abs(p.x-mx)<g && Math.abs(p.y-sy1)<g) return 'mt';
    if(Math.abs(p.x-mx)<g && Math.abs(p.y-sy2)<g) return 'mb';
    if(Math.abs(p.x-sx1)<g && Math.abs(p.y-my)<g) return 'ml';
    if(Math.abs(p.x-sx2)<g && Math.abs(p.y-my)<g) return 'mr';
    if(p.x>sx1 && p.x<sx2 && p.y>sy1 && p.y<sy2) return 'mv';
    return null;
  }}

  const cursors = {{tl:'nwse-resize',br:'nwse-resize',tr:'nesw-resize',bl:'nesw-resize',mt:'ns-resize',mb:'ns-resize',ml:'ew-resize',mr:'ew-resize',mv:'move'}};

  C.addEventListener('mousedown',function(e){{
    const p=mpos(e); drag=hit(p); ds=p; e.preventDefault();
  }});

  C.addEventListener('mousemove',function(e){{
    const p=mpos(e);
    const h=hit(p); C.style.cursor=h?cursors[h]||'crosshair':'crosshair';
    if(!drag) return;
    const dx=(p.x-ds.x)/sc, dy=(p.y-ds.y)/sc;
    if(drag==='tl'){{ box.x1+=dx; box.y1+=dy; }}
    else if(drag==='tr'){{ box.x2+=dx; box.y1+=dy; }}
    else if(drag==='bl'){{ box.x1+=dx; box.y2+=dy; }}
    else if(drag==='br'){{ box.x2+=dx; box.y2+=dy; }}
    else if(drag==='mt'){{ box.y1+=dy; }}
    else if(drag==='mb'){{ box.y2+=dy; }}
    else if(drag==='ml'){{ box.x1+=dx; }}
    else if(drag==='mr'){{ box.x2+=dx; }}
    else if(drag==='mv'){{ box.x1+=dx; box.y1+=dy; box.x2+=dx; box.y2+=dy; }}
    box.x1=Math.max(0,Math.min(box.x1,img.width));
    box.y1=Math.max(0,Math.min(box.y1,img.height));
    box.x2=Math.max(0,Math.min(box.x2,img.width));
    box.y2=Math.max(0,Math.min(box.y2,img.height));
    ds=p; paint();
  }});

  function endDrag(){{
    if(!drag) return;
    drag=null;
    if(box.x1>box.x2){{ let t=box.x1; box.x1=box.x2; box.x2=t; }}
    if(box.y1>box.y2){{ let t=box.y1; box.y1=box.y2; box.y2=t; }}
    paint(); syncGradio();
  }}
  C.addEventListener('mouseup', endDrag);
  C.addEventListener('mouseleave', endDrag);

  function syncGradio(){{
    // Reach into parent document (Gradio page) to update the number inputs
    var pwin;
    try {{ pwin = window.parent; }} catch(e) {{ pwin = window; }}
    var pdoc = pwin.document;
    function setN(eid, v){{
      var el = pdoc.getElementById(eid);
      if(!el) return;
      var inp = el.querySelector('input[type="number"]');
      if(!inp) return;
      var s = Object.getOwnPropertyDescriptor(pwin.HTMLInputElement.prototype,'value');
      if(s && s.set) {{ s.set.call(inp, Math.round(v)); }}
      else {{ inp.value = Math.round(v); }}
      inp.dispatchEvent(new Event('input',{{bubbles:true}}));
      inp.dispatchEvent(new Event('change',{{bubbles:true}}));
    }}
    setN('ds_bbox_x1', box.x1);
    setN('ds_bbox_y1', box.y1);
    setN('ds_bbox_x2', box.x2);
    setN('ds_bbox_y2', box.y2);
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
    x1, y1, x2, y2 = c.bbox
    html = render_bbox_editor_html(frame_rgb, c.bbox)
    return html, int(x1), int(y1), int(x2), int(y2)


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
    html = render_bbox_editor_html(frame_rgb, new_bbox)
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
    html = render_bbox_editor_html(frame_rgb, new_bbox)
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
    html = render_bbox_editor_html(frame_rgb, c.initial_bbox)
    return crop_rgb, html, ix1, iy1, ix2, iy2, 0


def _process_video_legacy(video_path, model_name, conf_threshold, iou_threshold,
                          frame_skip, progress):
    """
    Original frame-by-frame detection without tracking or dedup.
    Used as fallback when dedup is disabled or in Demo mode.
    """
    if not video_path or not os.path.exists(video_path):
        return None, None, [], "❌ Error: Invalid or missing video input file.", []

    start_time = time.time()
    import re as _re

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

    # Output file paths
    temp_avi = os.path.join(run_output_dir, "processed_raw.avi")
    final_mp4 = os.path.join(run_output_dir, f"output_video_{timestamp_str}.mp4")
    csv_file_path = os.path.join(run_output_dir, f"detections_{timestamp_str}.csv")

    # Define OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_avi, fourcc, fps / max(1, frame_skip), (width, height))

    # Load RDD-YOLO Model and class names
    model, class_map = load_yolo_model(model_name)

    # Prepare CSV Writer
    csv_records = []
    csv_headers = [
        "detection_id", "frame_index", "timestamp_sec",
        "class_id", "class_name", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
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
                results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())

                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        # Map class info dynamically
                        cls_cfg = class_map.get(cls_id, {"name": f"Class-{cls_id}", "color": (0, 0, 255)})
                        cls_name = cls_cfg["name"]
                        color = cls_cfg["color"]

                        # Coordinates
                        x1, y1, x2, y2 = xyxy
                        x1_c, y1_c = max(0, x1), max(0, y1)
                        x2_c, y2_c = min(width, x2), min(height, y2)

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
                                crop_gallery_images.append((crop_rgb, f"{cls_name} ({conf:.2f}) - t={timestamp_sec}s"))

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
                            "crop_filename": crop_filename
                        })

                        # Draw Bounding Box & Label on Output Frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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
                            crop_gallery_images.append((crop_rgb, f"{cls_cfg['name']} (0.89 Demo)"))

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
                        "crop_filename": crop_filename
                    })

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label_str = f"{cls_cfg['name']} 0.89 (Demo)"
                    cv2.rectangle(frame, (x1, y1 - 22), (x1 + 220, y1), color, -1)
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
                avg_center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                avg_bbox_area=float(max(0, x2 - x1) * max(0, y2 - y1)),
                avg_aspect_ratio=float(max(1, x2 - x1) / max(1, y2 - y1)),
                bbox_stats={},
                representative_crop_path=crop_path,
                source_track_ids=[rec["detection_id"]]
            )
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
        "defect_counts": class_counts
    }
    with open(os.path.join(run_output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

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

    return final_output_video, csv_file_path, crop_gallery_images, summary_md, charts_gallery


# Construct Gradio Modern Dashboard UI with Multi-Tab Pages
def build_ui():
    available_models = get_available_models()
    example_videos = get_example_videos()
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
                        input_video = gr.Video(
                            label="Upload Road Footage",
                            sources=["upload"],
                            interactive=True
                        )
                        video_est_md = gr.Markdown("")

                        with gr.Accordion("⚙️ Detection Model & Parameters", open=True):
                            with gr.Row():
                                model_selector = gr.Dropdown(
                                    choices=available_models,
                                    value=available_models[0],
                                    label="Model Weights",
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

                        with gr.Row(elem_classes="run-btn"):
                            btn_process = gr.Button("🚀 Run RDD-YOLO Detection + Dedup", variant="primary", scale=2)
                            btn_revoke = gr.Button("🛑 Revoke Process", variant="stop", scale=1)
                            btn_clear = gr.ClearButton(components=[input_video], value="✕ Clear", scale=1)

                    # Right Column: Live Telemetry & Output Results
                    with gr.Column(scale=6):
                        # System Telemetry Bar
                        sys_cards_html = gr.HTML(value='<div class="sys-bar"><span class="sys-metric"><span class="label">Initializing telemetry...</span></span></div>')

                        gr.HTML('<div class="section-header">📤 Inspection Results</div>')
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
                        outputs=[output_video, output_csv, output_crops_gallery, output_summary, output_charts_gallery],
                        fn=lambda vid: process_video(vid, available_models[0], 0.25, 0.45, 1, True, 0.65, 45),
                        cache_examples=False,
                        label="Click a sample to load:"
                    )

            # ═══════════════════════════════════════════════
            # PAGE 2: Dataset Builder
            # ═══════════════════════════════════════════════
            with gr.TabItem("📦 Dataset Builder"):
                gr.HTML('<div class="section-header">📦 Semi-Automatic Dataset Collection</div>')
                
                with gr.Row():
                    btn_load_candidates = gr.Button("📥 Load Detections from Last Inspection", variant="primary", scale=3)
                    ds_status_md = gr.Markdown("", scale=7)

                with gr.Row():
                    # Left: Full frame adjustment
                    with gr.Column(scale=6):
                        ds_full_frame = gr.HTML(
                            value='<div style="text-align:center;color:#94a3b8;padding:40px;font-family:Inter,sans-serif;">Click 📥 Load Detections to begin.</div>'
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
                            label="Crop Preview (Will be saved)",
                            type="numpy", interactive=False, height=280,
                        )
                        
                        ds_progress_md = gr.Markdown("")
                        ds_info_md = gr.Markdown("Click **Load Detections** after running an inspection.")

                        rdd_class_names = RDD_CLASS_CONFIG.get_class_names() if RDD_CLASS_CONFIG else ["POTHOLE"]
                        ds_class_dropdown = gr.Dropdown(
                            choices=rdd_class_names,
                            value=rdd_class_names[0] if rdd_class_names else None,
                            label="Assign Class",
                            info="Change if the model prediction is incorrect",
                            interactive=True,
                        )

                        with gr.Row(elem_classes="ds-action-row"):
                            btn_accept = gr.Button("✅ Accept & Save", variant="primary", scale=2)
                            btn_skip = gr.Button("⏭️ Skip", variant="secondary", scale=1)
                        
                        # Hidden coordinates for JS sync (dynamically applies bounding box changes)
                        with gr.Row(visible=False):
                            ds_x1 = gr.Number(label="X1", value=0, elem_id="ds_bbox_x1")
                            ds_y1 = gr.Number(label="Y1", value=0, elem_id="ds_bbox_y1")
                            ds_x2 = gr.Number(label="X2", value=0, elem_id="ds_bbox_x2")
                            ds_y2 = gr.Number(label="Y2", value=0, elem_id="ds_bbox_y2")

                gr.HTML('<div class="styled-divider"></div>')
                gr.HTML('<div class="section-header">📊 Dataset Statistics</div>')
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

                gr.HTML('<div class="gallery-section"></div>')
                gr.HTML('<div class="section-header">🖼️ Defect Crops from Selected Run</div>')
                hist_crops_gallery = gr.Gallery(
                    label="Saved crop patches from past run",
                    columns=6,
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
        btn_process_event = btn_process.click(
            fn=process_video,
            inputs=[input_video, model_selector, conf_slider, iou_slider, frame_skip_slider,
                    enable_dedup, merge_threshold_slider, track_buffer_slider],
            outputs=[output_video, output_csv, output_crops_gallery, output_summary, output_charts_gallery]
        )

        btn_revoke.click(
            fn=revoke_processing,
            inputs=[],
            outputs=[output_summary],
            cancels=[btn_process_event]
        )

        # Model Refresh Event Handler
        btn_refresh_models.click(
            fn=lambda: gr.Dropdown(
                choices=get_available_models(),
                value=get_available_models()[0] if get_available_models() else None
            ),
            inputs=[],
            outputs=[model_selector]
        )

        # Dedup Toggle Event Handler
        enable_dedup.change(
            fn=on_toggle_dedup,
            inputs=[enable_dedup],
            outputs=[dedup_status_md, btn_process, dedup_params_row]
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

        # History Page Event Handlers
        history_selector.change(
            fn=load_history_details,
            inputs=[history_selector],
            outputs=[hist_video, hist_csv, hist_crops_gallery, hist_summary, hist_charts_gallery]
        )

        btn_refresh_history.click(
            fn=lambda: gr.Dropdown(
                choices=get_history_runs(),
                value=get_history_runs()[0] if get_history_runs() else None
            ),
            inputs=[],
            outputs=[history_selector]
        )

        # ── Dataset Builder Event Handlers ──

        def _on_load_candidates():
            msg, count = prepare_annotation_candidates()
            if count > 0:
                crop_img, info, suggested, progress, is_done = get_current_candidate_display()
                # Also load the full frame overlay for adjustment panel
                frame_overlay, x1, y1, x2, y2 = get_full_frame_with_overlay()
                return (
                    msg, crop_img, info, progress, gr.Dropdown(value=suggested),
                    frame_overlay, x1, y1, x2, y2, 0
                )
            return (
                msg, None, "No candidates available.", "", gr.Dropdown(),
                None, 0, 0, 0, 0, 0
            )

        btn_load_candidates.click(
            fn=_on_load_candidates,
            inputs=[],
            outputs=[
                ds_status_md, ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider
            ]
        )

        def _on_accept(chosen_class):
            crop_img, info, suggested, progress, is_done = accept_annotation(chosen_class)
            stats = get_dataset_stats_md()
            # Load frame overlay for the next candidate
            frame_overlay, x1, y1, x2, y2 = get_full_frame_with_overlay()
            return (
                crop_img, info, progress, gr.Dropdown(value=suggested), stats,
                frame_overlay, x1, y1, x2, y2, 0
            )

        btn_accept.click(
            fn=_on_accept,
            inputs=[ds_class_dropdown],
            outputs=[
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown, ds_stats_md,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider
            ]
        )

        def _on_skip():
            crop_img, info, suggested, progress, is_done = skip_annotation()
            # Load frame overlay for the next candidate
            frame_overlay, x1, y1, x2, y2 = get_full_frame_with_overlay()
            return (
                crop_img, info, progress, gr.Dropdown(value=suggested),
                frame_overlay, x1, y1, x2, y2, 0
            )

        btn_skip.click(
            fn=_on_skip,
            inputs=[],
            outputs=[
                ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown,
                ds_full_frame, ds_x1, ds_y1, ds_x2, ds_y2, ds_padding_slider
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

    return demo, theme, custom_css


if __name__ == "__main__":
    app, theme, css = build_ui()
    print("[RDD-YOLO] Launching Gradio Web UI on http://127.0.0.1:7860 ...")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=theme,
        css=css,
        allowed_paths=[OUTPUTS_DIR, BASE_DIR]
    )

