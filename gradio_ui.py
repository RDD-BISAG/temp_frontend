import os
import sys
import time
import csv
import json
import datetime
import tempfile
import logging
import collections
import cv2
import numpy as np
import pandas as pd
import gradio as gr

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
    HAS_DEDUP_PIPELINE = True
except ImportError as e:
    print(f"[Warning] rdd_dedup pipeline not available: {e}")
    HAS_DEDUP_PIPELINE = False

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
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; margin-bottom: 6px;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 6px;">
            <span> CPU:</span> <span style="color: {cpu_color}; font-size: 0.95rem; font-weight: 800;">{cpu['percent']}%</span>
            <span style="color: #94a3b8; font-weight: normal; font-size: 0.72rem;">({cpu['cores']}C/{cpu['threads']}T)</span>
        </div>
        <div style="font-size: 0.8rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 6px;">
            <span>🧠 RAM:</span> <span style="color: {ram_color}; font-size: 0.95rem; font-weight: 800;">{ram['percent']}%</span>
            <span style="color: #94a3b8; font-weight: normal; font-size: 0.72rem;">({ram['used_gb']}/{ram['total_gb']} GB)</span>
        </div>
        <div style="font-size: 0.8rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 6px;">
            <span>🎮 GPU:</span> <span style="color: #38bdf8; font-size: 0.8rem;">{gpu['name']}</span>
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8;">
            ⏱️ {uptime_min}m {uptime_sec}s
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

    # Sort with priority to rdd2022_st1, rdd_yolo, rdd2022
    def model_priority(name):
        lower = name.lower()
        if "rdd2022_st2" in lower:
            return 0
        if "rdd2022_st1" in lower:
            return 1
        if "rdd_yolo.pt" in lower and "/" not in lower:
            return 2
        if "rdd2022" in lower:
            return 3
        if "rdd" in lower:
            return 4
        return 10

    model_files.sort(key=model_priority)

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
        return None, None, [], "Please select an analysis run from the dropdown."

    run_dir = os.path.join(OUTPUTS_DIR, selected_run_folder)
    if not os.path.exists(run_dir):
        return None, None, [], f"❌ Error: Run directory '{selected_run_folder}' not found."

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

    return video_path, csv_path, crop_gallery, summary_md


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

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ""

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

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
        try:
            reader = imageio.get_reader(input_path)
            fps = reader.get_meta_data().get('fps', 30)
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p')
            for frame in reader:
                writer.append_data(frame)
            writer.close()
            reader.close()
            return output_path
        except Exception as e:
            print(f"[Warning] H.264 re-encoding failed: {e}")
    return input_path


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
        return None, None, [], "❌ Error: Invalid or missing video input file."

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
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, [], f"❌ Pipeline error: {e}"

    # Store result for dataset builder
    _ANNOTATION_STATE["last_pipeline_result"] = result
    _ANNOTATION_STATE["last_video_path"] = video_path

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

    return final_video, csv_path, crop_gallery_images, summary_md


# ═══════════════════════════════════════════════════════════════════════════════
# RDD_Ahmedabad Dataset Builder — Backend Functions
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_annotation_candidates():
    """
    Extract annotation candidates from the last pipeline run.
    Returns: (status_message, total_count)
    """
    result = _ANNOTATION_STATE.get("last_pipeline_result")
    video_path = _ANNOTATION_STATE.get("last_video_path")

    if result is None or video_path is None:
        return "❌ No pipeline result available. Run a video inspection first on the '🚀 Run New Inspection' tab.", 0

    if not HAS_DATASET_SYSTEM or RDD_CROP_SELECTOR is None:
        return "❌ Dataset system not available. Check configs/rdd_ahmedabad_classes.yaml.", 0

    candidates = RDD_CROP_SELECTOR.select_from_pipeline_result(result, video_path)

    if not candidates:
        return "⚠️ No annotation candidates found. The pipeline may not have detected any defects.", 0

    _ANNOTATION_STATE["candidates"] = candidates
    _ANNOTATION_STATE["current_index"] = 0
    _ANNOTATION_STATE["accepted_count"] = 0
    _ANNOTATION_STATE["skipped_count"] = 0
    _ANNOTATION_STATE["changed_count"] = 0

    return f"✅ {len(candidates)} unique defect crops ready for review.", len(candidates)


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


def _process_video_legacy(video_path, model_name, conf_threshold, iou_threshold,
                          frame_skip, progress):
    """
    Original frame-by-frame detection without tracking or dedup.
    Used as fallback when dedup is disabled or in Demo mode.
    """
    if not video_path or not os.path.exists(video_path):
        return None, None, [], "❌ Error: Invalid or missing video input file."

    start_time = time.time()
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped run folder inside outputs/
    run_output_dir = os.path.join(OUTPUTS_DIR, f"analysis_{timestamp_str}")
    crops_dir = os.path.join(run_output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, [], "❌ Error: Could not open video file."

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

    while cap.isOpened():
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

    cap.release()
    out.release()

    elapsed_time = time.time() - start_time
    proc_fps = processed_count / max(0.001, elapsed_time)

    # Convert raw AVI video to web-compatible MP4 format
    final_output_video = reencode_to_h264(temp_avi, final_mp4)
    if os.path.exists(temp_avi):
        try:
            os.remove(temp_avi)
        except Exception:
            pass

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

    return final_output_video, csv_file_path, crop_gallery_images, summary_md


# Construct Gradio Modern Dashboard UI with Multi-Tab Pages
def build_ui():
    available_models = get_available_models()
    example_videos = get_example_videos()
    history_runs = get_history_runs()

    custom_css = """
    .main-title {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .main-title h1 {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ef4444, #f97316, #eab308);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge-bar {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .badge {
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
    }
    .badge-active {
        background: #065f46;
        color: #34d399;
        border: 1px solid #059669;
    }
    """

    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate"
    )

    with gr.Blocks(title="RDD-YOLO Road Damage Detection Studio") as demo:
        gr.HTML(f"""
        <div class="main-title">
            <h1>🚧 RDD-YOLO Road Damage Detection Studio</h1>
            <p>Automated video inspection for potholes, longitudinal cracks, transverse cracks, and road defects.</p>
        </div>
        <div class="badge-bar">
            <span class="badge badge-active">Active Model: {available_models[0]}</span>
            <span class="badge badge-active">{'Dedup Pipeline: ON' if HAS_DEDUP_PIPELINE else 'Dedup: N/A'}</span>
            <span class="badge">BoT-SORT Tracking</span>
            <span class="badge">CSV Export & Defect Crops</span>
            <span class="badge">Analysis History Explorer</span>
        </div>
        """)

        with gr.Tabs():
            # PAGE 1: New Video Analysis
            with gr.TabItem("🚀 Run New Inspection"):
                with gr.Row():
                    # Left Column: Inputs & Settings
                    with gr.Column(scale=5):
                        gr.Markdown("### 📥 Video Input & Inspection Settings")
                        input_video = gr.Video(
                            label="Input Road Footage Video",
                            sources=["upload"],
                            interactive=True
                        )
                        video_est_md = gr.Markdown("")

                        with gr.Accordion("⚙️ RDD-YOLO Model & Detection Parameters", open=True):
                            with gr.Row():
                                model_selector = gr.Dropdown(
                                    choices=available_models,
                                    value=available_models[0],
                                    label="Select Model Weights (auto-detected from models/ directory)",
                                    info="Automatically scans all .pt/.onnx files in models/ directory. Select any model to use.",
                                    scale=4
                                )
                                btn_refresh_models = gr.Button("🔄 Refresh Models", scale=1)
                            
                            with gr.Row():
                                conf_slider = gr.Slider(
                                    minimum=0.05,
                                    maximum=0.95,
                                    value=0.25,
                                    step=0.05,
                                    label="Confidence Threshold",
                                    info="Minimum confidence score"
                                )
                                iou_slider = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.9,
                                    value=0.45,
                                    step=0.05,
                                    label="IoU NMS Threshold",
                                    info="Non-Maximum Suppression threshold"
                                )

                            frame_skip_slider = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=1,
                                step=1,
                                label="Frame Subsampling (Speedup)",
                                info="1 = Process all frames; 2 = Process every 2nd frame"
                            )

                        with gr.Accordion("🔄 Duplicate Removal (BoT-SORT + Track Manager)", open=True):
                            enable_dedup = gr.Checkbox(
                                value=True,
                                label="Enable Duplicate Removal",
                                info="Uses BoT-SORT tracking + multi-feature verification to count each defect once",
                                interactive=HAS_DEDUP_PIPELINE,
                            )
                            with gr.Row():
                                merge_threshold_slider = gr.Slider(
                                    minimum=0.3,
                                    maximum=0.95,
                                    value=0.65,
                                    step=0.05,
                                    label="Merge Threshold",
                                    info="Similarity score needed to merge duplicate tracks (lower = more aggressive merging)",
                                )
                                track_buffer_slider = gr.Slider(
                                    minimum=10,
                                    maximum=120,
                                    value=45,
                                    step=5,
                                    label="Track Buffer (frames)",
                                    info="How long to keep lost tracks alive (~1.5s at 30fps = 45 frames)",
                                )

                        with gr.Row():
                            btn_process = gr.Button("🚀 Run RDD-YOLO Detection + Dedup", variant="primary", scale=2)
                            btn_clear = gr.ClearButton(components=[input_video], value="🧹 Clear", scale=1)

                    # Right Column: Live Telemetry & Output Results
                    with gr.Column(scale=6):
                        # Top-Right Section: Ultra-Compact Hardware Status Bar
                        sys_cards_html = gr.HTML(value="<p>Initializing hardware telemetry...</p>")

                        gr.Markdown("### 📤 Inspection Results & Downloadable Artifacts")
                        output_video = gr.Video(
                            label="Annotated Output Video (HTML5 H.264)",
                            interactive=False
                        )
                        output_csv = gr.File(
                            label="📥 Download Detections Log (.csv)",
                            interactive=False
                        )
                        output_summary = gr.Markdown("Ready for inspection. Upload road video footage and click **Run RDD-YOLO Detection + Dedup**. The pipeline will detect defects, track them with BoT-SORT, and remove duplicates automatically.")

                # Gallery Section for Detected Defect Crops
                gr.Markdown("---")
                gr.Markdown("### 🖼️ Unique Defect Crop Patches (Deduplicated)")
                output_crops_gallery = gr.Gallery(
                    label="Representative crops — one per unique physical defect (best confidence frame)",
                    columns=6,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )

                # Example section if sample videos are available in inputs directory
                if example_videos:
                    gr.Markdown("---")
                    gr.Markdown("### 🎞️ Sample Road Footage (inputs/)")
                    gr.Examples(
                        examples=[[vid] for vid in example_videos],
                        inputs=[input_video],
                        outputs=[output_video, output_csv, output_crops_gallery, output_summary],
                        fn=lambda vid: process_video(vid, available_models[0], 0.25, 0.45, 1, True, 0.65, 45),
                        cache_examples=False,
                        label="Click a sample video to load:"
                    )

            # PAGE 2: RDD_Ahmedabad Dataset Builder
            with gr.TabItem("📦 RDD_Ahmedabad Dataset Builder"):
                gr.Markdown("### 📦 Semi-Automatic Dataset Collection & Annotation")
                gr.Markdown("Run a video inspection first, then come here to review detected defects and add them to the **RDD_Ahmedabad** training dataset.")

                with gr.Row():
                    btn_load_candidates = gr.Button(
                        "📥 Load Detections from Last Inspection",
                        variant="primary", scale=3
                    )
                    ds_status_md = gr.Markdown("")

                with gr.Row():
                    # Left: Crop preview
                    with gr.Column(scale=4):
                        ds_crop_image = gr.Image(
                            label="Detection Crop Preview",
                            type="numpy",
                            interactive=False,
                            height=320,
                        )

                    # Right: Info + Actions
                    with gr.Column(scale=6):
                        ds_info_md = gr.Markdown("Click **Load Detections** after running an inspection.")
                        ds_progress_md = gr.Markdown("")

                        rdd_class_names = RDD_CLASS_CONFIG.get_class_names() if RDD_CLASS_CONFIG else ["POTHOLE"]
                        ds_class_dropdown = gr.Dropdown(
                            choices=rdd_class_names,
                            value=rdd_class_names[0],
                            label="Assign Class (change if prediction is wrong)",
                            interactive=True,
                        )

                        with gr.Row():
                            btn_accept = gr.Button("✅ Accept & Save", variant="primary", scale=2)
                            btn_skip = gr.Button("⏭️ Skip", variant="secondary", scale=1)

                gr.Markdown("---")
                gr.Markdown("### 📊 Current Dataset Statistics")
                ds_stats_md = gr.Markdown(
                    value=get_dataset_stats_md()
                )
                btn_refresh_stats = gr.Button("🔄 Refresh Stats")

            # PAGE 3: History & Past Reports Explorer
            with gr.TabItem("📜 Analysis History & Reports"):
                gr.Markdown("### 📜 Previous Inspection Runs & Output Explorer")
                
                with gr.Row():
                    history_selector = gr.Dropdown(
                        choices=history_runs,
                        value=history_runs[0] if history_runs else None,
                        label="Select Past Analysis Run (outputs/ directory)",
                        info="Select any previous run folder to load its video, CSV log, summary metrics, and defect crops.",
                        scale=4
                    )
                    btn_refresh_history = gr.Button("🔄 Refresh List", scale=1)

                with gr.Row():
                    with gr.Column(scale=5):
                        hist_video = gr.Video(
                            label="Annotated Output Video",
                            interactive=False
                        )
                        hist_csv = gr.File(
                            label="📥 Download Detections Log (.csv)",
                            interactive=False
                        )
                    with gr.Column(scale=6):
                        hist_summary = gr.Markdown("Select an analysis run from the dropdown above to view past inspection report.")

                gr.Markdown("---")
                gr.Markdown("### 🖼️ Defect Crop Patches from Selected Run")
                hist_crops_gallery = gr.Gallery(
                    label="Cropped Image Patches from Past Run",
                    columns=6,
                    rows=2,
                    height="auto",
                    object_fit="contain"
                )

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
        btn_process.click(
            fn=process_video,
            inputs=[input_video, model_selector, conf_slider, iou_slider, frame_skip_slider,
                    enable_dedup, merge_threshold_slider, track_buffer_slider],
            outputs=[output_video, output_csv, output_crops_gallery, output_summary]
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
            outputs=[hist_video, hist_csv, hist_crops_gallery, hist_summary]
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
                return msg, crop_img, info, progress, gr.Dropdown(value=suggested)
            return msg, None, "No candidates available.", "", gr.Dropdown()

        btn_load_candidates.click(
            fn=_on_load_candidates,
            inputs=[],
            outputs=[ds_status_md, ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown]
        )

        def _on_accept(chosen_class):
            crop_img, info, suggested, progress, is_done = accept_annotation(chosen_class)
            stats = get_dataset_stats_md()
            return crop_img, info, progress, gr.Dropdown(value=suggested), stats

        btn_accept.click(
            fn=_on_accept,
            inputs=[ds_class_dropdown],
            outputs=[ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown, ds_stats_md]
        )

        def _on_skip():
            crop_img, info, suggested, progress, is_done = skip_annotation()
            return crop_img, info, progress, gr.Dropdown(value=suggested)

        btn_skip.click(
            fn=_on_skip,
            inputs=[],
            outputs=[ds_crop_image, ds_info_md, ds_progress_md, ds_class_dropdown]
        )

        btn_refresh_stats.click(
            fn=get_dataset_stats_md,
            inputs=[],
            outputs=[ds_stats_md]
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

