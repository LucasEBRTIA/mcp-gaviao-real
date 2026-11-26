"""Machine hardware detection and adaptive execution policy resolution."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import subprocess
from typing import Literal

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.models import ExecutionPolicy, HardwareProfile, HardwareTier


def _read_ram_gb() -> tuple[float, float]:
    """Read (total_ram_gb, available_ram_gb) without requiring heavy dependencies."""
    system = platform.system()
    if system == "Windows":
        try:

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                gb = 1024**3
                return round(stat.ullTotalPhys / gb, 2), round(stat.ullAvailPhys / gb, 2)
        except Exception:
            pass

    if system == "Linux":
        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    key, _, val = line.partition(":")
                    parts = val.strip().split()
                    if parts and parts[0].isdigit():
                        meminfo[key] = int(parts[0])
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            if total_kb > 0:
                kb_to_gb = 1024**2
                return round(total_kb / kb_to_gb, 2), round(avail_kb / kb_to_gb, 2)
        except Exception:
            pass

    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], timeout=2, text=True
            ).strip()
            total_gb = round(int(out) / (1024**3), 2)
            return total_gb, round(total_gb * 0.5, 2)
        except Exception:
            pass

    try:
        import psutil

        vm = psutil.virtual_memory()
        gb = 1024**3
        return round(vm.total / gb, 2), round(vm.available / gb, 2)
    except Exception:
        return 8.0, 4.0


def _probe_nvidia_gpu() -> tuple[str | None, int | None]:
    """Probe NVIDIA GPU via nvidia-smi without importing torch."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None, None
        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) >= 2:
            name = parts[0]
            try:
                vram_mb = int(float(parts[1]))
            except ValueError:
                vram_mb = None
            return name, vram_mb
        return first_line, None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, None


def detect_hardware(settings: GaviaoSettings | None = None) -> HardwareProfile:
    """Detect local machine capabilities without importing torch or loading ML models."""
    os_name = platform.system()
    arch = platform.machine()
    cpu_cores = os.cpu_count() or 1
    ram_total_gb, ram_available_gb = _read_ram_gb()

    gpu_vendor: Literal["nvidia", "apple"] | None = None
    gpu_name: str | None = None
    vram_total_mb: int | None = None

    nv_name, nv_vram = _probe_nvidia_gpu()
    if nv_name:
        gpu_vendor = "nvidia"
        gpu_name = nv_name
        vram_total_mb = nv_vram
    elif os_name == "Darwin" and arch.lower() in {"arm64", "aarch64"}:
        gpu_vendor = "apple"
        gpu_name = "Apple Silicon (MPS)"

    has_docling = importlib.util.find_spec("docling") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    has_rapidocr = importlib.util.find_spec("rapidocr_onnxruntime") is not None
    has_vlm = bool(settings and settings.vlm_base_url and settings.vlm_model)

    return HardwareProfile(
        os_name=os_name,
        arch=arch,
        cpu_cores=cpu_cores,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        vram_total_mb=vram_total_mb,
        has_docling=has_docling,
        has_torch=has_torch,
        has_rapidocr=has_rapidocr,
        has_vlm=has_vlm,
    )


def _recommended_install_cmd(profile: HardwareProfile) -> str:
    if profile.has_docling:
        return "Ambiente já possui extra 'accurate' instalado para esta máquina."
    if profile.gpu_vendor == "nvidia":
        return "uv sync --extra accurate --extra cuda"
    if profile.gpu_vendor == "apple":
        return "uv sync --extra accurate"
    if profile.ram_total_gb >= 12.0:
        return "uv sync --extra accurate --extra cpu"
    return "uv sync (perfil 'lite' recomendado para máquinas com pouca RAM)"


def resolve_policy(
    profile: HardwareProfile,
    settings: GaviaoSettings,
) -> ExecutionPolicy:
    """Compute adaptive ExecutionPolicy for the detected hardware and user overrides."""
    notes: list[str] = []
    cores = max(1, profile.cpu_cores)
    omp_threads = max(1, min(8, cores // 2 if cores >= 4 else cores))
    recommended_cmd = _recommended_install_cmd(profile)

    forced_profile = settings.profile
    forced_device = settings.device

    if forced_profile != "auto":
        tier = HardwareTier(forced_profile)
        notes.append(f"Perfil forçado via configuração: {tier.value}")
    else:
        if not profile.has_docling or profile.ram_total_gb < 8.0:
            tier = HardwareTier.LITE
            if not profile.has_docling:
                notes.append(
                    "Extra 'accurate' (docling) não instalado; operando em modo leve (PyMuPDF + MarkItDown)."
                )
            if profile.ram_total_gb < 8.0:
                notes.append(
                    f"RAM total ({profile.ram_total_gb} GB) < 8 GB; ativando perfil 'lite'."
                )
        elif (
            profile.gpu_vendor == "nvidia"
            and (profile.vram_total_mb or 0) >= 8000
            and forced_device != "cpu"
        ):
            tier = HardwareTier.GPU_HIGH
        elif (
            (profile.gpu_vendor == "nvidia" and (profile.vram_total_mb or 0) >= 3500)
            or profile.gpu_vendor == "apple"
        ) and forced_device != "cpu":
            tier = HardwareTier.GPU_LOW
        else:
            tier = HardwareTier.CPU_RICH

    if tier == HardwareTier.LITE:
        return ExecutionPolicy(
            tier=tier,
            device="cpu",
            can_use_docling=False,
            docling_batch_size=0,
            omp_threads=omp_threads,
            max_ocr_pages_per_call=0,
            clear_vram_after_job=False,
            vram_idle_unload_s=0,
            recommended_install_cmd=recommended_cmd,
            notes=notes,
        )

    if tier == HardwareTier.GPU_HIGH:
        device: Literal["cuda", "mps", "cpu"] = "cpu" if forced_device == "cpu" else "cuda"
        return ExecutionPolicy(
            tier=tier,
            device=device,
            can_use_docling=profile.has_docling,
            docling_batch_size=4,
            omp_threads=omp_threads,
            max_ocr_pages_per_call=20,
            clear_vram_after_job=False,
            vram_idle_unload_s=settings.gpu_idle_unload_seconds,
            recommended_install_cmd=recommended_cmd,
            notes=notes,
        )

    if tier == HardwareTier.GPU_LOW:
        if forced_device == "cpu":
            device = "cpu"
        elif profile.gpu_vendor == "apple":
            device = "mps"
        else:
            device = "cuda"
        return ExecutionPolicy(
            tier=tier,
            device=device,
            can_use_docling=profile.has_docling,
            docling_batch_size=1,
            omp_threads=omp_threads,
            max_ocr_pages_per_call=5,
            clear_vram_after_job=True,
            vram_idle_unload_s=60,
            recommended_install_cmd=recommended_cmd,
            notes=notes,
        )

    # CPU_RICH
    return ExecutionPolicy(
        tier=HardwareTier.CPU_RICH,
        device="cpu",
        can_use_docling=profile.has_docling,
        docling_batch_size=1,
        omp_threads=max(2, omp_threads),
        max_ocr_pages_per_call=3,
        clear_vram_after_job=False,
        vram_idle_unload_s=0,
        recommended_install_cmd=recommended_cmd,
        notes=notes,
    )
