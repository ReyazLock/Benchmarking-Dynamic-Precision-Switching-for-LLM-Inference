from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import asdict, dataclass

import psutil


@dataclass
class ResourceSnapshot:
    peak_benchmark_rss_bytes: int = 0
    peak_ollama_rss_bytes: int = 0
    peak_system_memory_used_bytes: int = 0
    peak_gpu_memory_used_mib: float | None = None
    mean_gpu_utilization_pct: float | None = None
    mean_gpu_power_w: float | None = None

    def to_dict(self):
        return asdict(self)


class ResourceSampler:
    """Samples process/system/GPU usage while one inference call is active.

    GPU values are whole-device measurements. Run benchmarks in an otherwise
    idle environment so deltas remain attributable to the model under test.
    """

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss: list[int] = []
        self._ollama_rss: list[int] = []
        self._system_used: list[int] = []
        self._gpu_memory: list[float] = []
        self._gpu_util: list[float] = []
        self._gpu_power: list[float] = []

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 3))

    def _sample_gpu(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return
        rows = [row for row in completed.stdout.splitlines() if row.strip()]
        memory_total = util_total = power_total = 0.0
        valid = 0
        for row in rows:
            try:
                memory, utilization, power = [float(value.strip()) for value in row.split(",")]
            except (ValueError, TypeError):
                continue
            memory_total += memory
            util_total += utilization
            power_total += power
            valid += 1
        if valid:
            self._gpu_memory.append(memory_total)
            self._gpu_util.append(util_total / valid)
            self._gpu_power.append(power_total)

    def _run(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop.is_set():
            try:
                self._rss.append(process.memory_info().rss)
                self._system_used.append(psutil.virtual_memory().used)
                ollama_rss = 0
                for candidate in psutil.process_iter(["name", "memory_info"]):
                    if "ollama" in (candidate.info.get("name") or "").lower():
                        ollama_rss += candidate.info["memory_info"].rss
                self._ollama_rss.append(ollama_rss)
            except (psutil.Error, KeyError):
                pass
            self._sample_gpu()
            self._stop.wait(self.interval_s)

    def snapshot(self) -> ResourceSnapshot:
        mean = lambda values: sum(values) / len(values) if values else None
        return ResourceSnapshot(
            peak_benchmark_rss_bytes=max(self._rss, default=0),
            peak_ollama_rss_bytes=max(self._ollama_rss, default=0),
            peak_system_memory_used_bytes=max(self._system_used, default=0),
            peak_gpu_memory_used_mib=max(self._gpu_memory) if self._gpu_memory else None,
            mean_gpu_utilization_pct=mean(self._gpu_util),
            mean_gpu_power_w=mean(self._gpu_power),
        )

