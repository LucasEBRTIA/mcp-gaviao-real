"""Tests for the machine hardware adaptation system."""

from __future__ import annotations

import subprocess
import sys

from mcp_gaviao_real.config import GaviaoSettings
from mcp_gaviao_real.hardware import resolve_policy
from mcp_gaviao_real.models import HardwareProfile, HardwareTier


def test_detect_hardware_never_imports_torch() -> None:
    # Run in an isolated Python process so earlier tests importing torch don't pollute sys.modules
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from mcp_gaviao_real.config import GaviaoSettings; "
                "from mcp_gaviao_real.hardware import detect_hardware; "
                "p = detect_hardware(GaviaoSettings()); "
                "assert 'torch' not in sys.modules, 'torch was imported!'; "
                "assert p.cpu_cores >= 1 and p.ram_total_gb > 0"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_resolve_policy_gpu_high() -> None:
    prof = HardwareProfile(
        os_name="Windows",
        arch="AMD64",
        cpu_cores=16,
        ram_total_gb=32.0,
        ram_available_gb=20.0,
        gpu_vendor="nvidia",
        gpu_name="NVIDIA GeForce RTX 4060 Ti",
        vram_total_mb=16380,
        has_docling=True,
        has_torch=True,
    )
    policy = resolve_policy(prof, GaviaoSettings())
    assert policy.tier == HardwareTier.GPU_HIGH
    assert policy.device == "cuda"
    assert policy.can_use_docling is True
    assert policy.docling_batch_size == 4
    assert policy.clear_vram_after_job is False


def test_resolve_policy_gpu_low_and_apple_mps() -> None:
    prof_nv = HardwareProfile(
        os_name="Linux",
        arch="x86_64",
        cpu_cores=8,
        ram_total_gb=16.0,
        ram_available_gb=10.0,
        gpu_vendor="nvidia",
        gpu_name="RTX 3060 Laptop",
        vram_total_mb=6144,
        has_docling=True,
    )
    p_nv = resolve_policy(prof_nv, GaviaoSettings())
    assert p_nv.tier == HardwareTier.GPU_LOW
    assert p_nv.device == "cuda"
    assert p_nv.clear_vram_after_job is True
    assert p_nv.docling_batch_size == 1

    prof_mac = HardwareProfile(
        os_name="Darwin",
        arch="arm64",
        cpu_cores=10,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        gpu_vendor="apple",
        gpu_name="Apple Silicon (MPS)",
        has_docling=True,
    )
    p_mac = resolve_policy(prof_mac, GaviaoSettings())
    assert p_mac.tier == HardwareTier.GPU_LOW
    assert p_mac.device == "mps"


def test_resolve_policy_cpu_rich() -> None:
    prof = HardwareProfile(
        os_name="Linux",
        arch="x86_64",
        cpu_cores=12,
        ram_total_gb=32.0,
        ram_available_gb=24.0,
        gpu_vendor=None,
        has_docling=True,
    )
    policy = resolve_policy(prof, GaviaoSettings())
    assert policy.tier == HardwareTier.CPU_RICH
    assert policy.device == "cpu"
    assert policy.omp_threads == 6
    assert policy.max_ocr_pages_per_call == 3


def test_resolve_policy_lite_without_docling_and_low_ram() -> None:
    prof_no_docling = HardwareProfile(
        os_name="Windows",
        arch="AMD64",
        cpu_cores=16,
        ram_total_gb=32.0,
        ram_available_gb=24.0,
        gpu_vendor="nvidia",
        gpu_name="RTX 4060 Ti",
        vram_total_mb=16380,
        has_docling=False,
    )
    policy = resolve_policy(prof_no_docling, GaviaoSettings())
    assert policy.tier == HardwareTier.LITE
    assert policy.can_use_docling is False
    assert "cuda" in policy.recommended_install_cmd

    prof_low_ram = HardwareProfile(
        os_name="Linux",
        arch="x86_64",
        cpu_cores=4,
        ram_total_gb=4.0,
        ram_available_gb=1.5,
        has_docling=True,
    )
    policy_low = resolve_policy(prof_low_ram, GaviaoSettings())
    assert policy_low.tier == HardwareTier.LITE


def test_profile_env_override() -> None:
    prof = HardwareProfile(
        os_name="Windows",
        arch="AMD64",
        cpu_cores=8,
        ram_total_gb=32.0,
        ram_available_gb=20.0,
        gpu_vendor="nvidia",
        vram_total_mb=16380,
        has_docling=True,
    )
    policy = resolve_policy(prof, GaviaoSettings(profile="cpu_rich"))
    assert policy.tier == HardwareTier.CPU_RICH
    assert policy.device == "cpu"
