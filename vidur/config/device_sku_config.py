from dataclasses import dataclass, field

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int
    # Peak HBM bandwidth (GB/s). Drives memory-bound decode ops (weight loads,
    # KV writes, HBM hot-window reads) in the HBF predictor. Numerically 1 GB/s
    # = 1 byte/ns. Published per-SKU specs.
    mem_bandwidth_gbps: float = 2039.0


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45
    mem_bandwidth_gbps: float = 696.0       # A40 GDDR6

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80
    mem_bandwidth_gbps: float = 2039.0      # A100 80GB SXM HBM2e

    @staticmethod
    def get_type():
        return DeviceSKUType.A100


@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80
    mem_bandwidth_gbps: float = 3350.0      # H100 80GB SXM HBM3

    @staticmethod
    def get_type():
        return DeviceSKUType.H100


@dataclass
class H200DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1979
    total_memory_gb: int = 141
    mem_bandwidth_gbps: float = 4800.0      # H200 141GB HBM3e

    @staticmethod
    def get_type():
        return DeviceSKUType.H200


@dataclass
class BlackwellDeviceSKUConfig(BaseDeviceSKUConfig):
    # NVIDIA RTX PRO 6000 Blackwell Server Edition: 96GB GDDR7, 1.6 TB/s.
    # fp16_tflops approximate (not used by the memory-bound decode path; compute
    # comes from profiled traces). Refine if compute-bound regimes matter.
    fp16_tflops: int = 500
    total_memory_gb: int = 96
    mem_bandwidth_gbps: float = 1600.0

    @staticmethod
    def get_type():
        return DeviceSKUType.BLACKWELL
