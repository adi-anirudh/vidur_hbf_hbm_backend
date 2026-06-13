from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import ActivationType, NormType

logger = init_logger(__name__)


@dataclass
class BaseModelConfig(BaseFixedConfig):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    embedding_dim: int
    mlp_hidden_dim: int
    max_position_embeddings: int
    use_gated_mlp: bool
    use_bias: bool
    use_qkv_bias: bool
    activation: ActivationType
    norm: NormType
    post_attn_norm: bool
    vocab_size: int
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False
    # Explicit attention head dim. None -> embedding_dim // num_q_heads (default).
    # Qwen3 sets head_dim independently of hidden size (hidden=4096, q=64,
    # head_dim=128 -> q_proj 4096->8192); set it so KV size + weight bytes are
    # correct. Read via head_size().
    head_dim: Optional[int] = None

    def head_size(self) -> int:
        return self.head_dim if self.head_dim else self.embedding_dim // self.num_q_heads


@dataclass
class Llama2ModelConfig(BaseModelConfig):
    max_position_embeddings: int = 16384
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 32768
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 10000
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-Config"


@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 22016
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "codellama/CodeLlama-34b-Instruct-hf"


@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-7b-hf"


@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-70b-hf"


@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-8B"


@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-70B"


@dataclass
class InternLMModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 4096
    vocab_size: int = 103168


@dataclass
class InternLM_20BModelConfig(InternLMModelConfig):
    num_layers: int = 60
    num_q_heads: int = 40
    num_kv_heads: int = 40
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 13824

    @staticmethod
    def get_name():
        return "internlm/internlm-20b"


@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 32768
    vocab_size: int = 92544


@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 48
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    mlp_hidden_dim: int = 16384
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "internlm/internlm2-20b"


@dataclass
class Phi2ModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 10240
    max_position_embeddings: int = 2048
    use_gated_mlp: bool = False
    use_bias: bool = True
    use_qkv_bias: bool = True
    activation: ActivationType = ActivationType.GELU
    norm: NormType = NormType.LAYER_NORM
    post_attn_norm: bool = False
    vocab_size: int = 51200
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = 10000
    partial_rotary_factor: float = 0.4
    no_tensor_parallel: bool = True

    @staticmethod
    def get_name():
        return "microsoft/phi-2"


@dataclass
class QwenModelConfig(Llama2ModelConfig):
    use_qkv_bias: bool = True
    max_position_embeddings: int = 32768
    vocab_size: int = 152064

    @staticmethod
    def get_name():
        return "Qwen/Qwen-Config"


@dataclass
class Qwen72BModelConfig(QwenModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 64
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 24576
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "Qwen/Qwen-72B"


@dataclass
class Mistral7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 32768
    rope_theta: Optional[float] = 10000
    vocab_size: int = 32000   # real HF config.json (was faked to 128256 to reuse Llama-3-8B traces)

    @staticmethod
    def get_name():
        return "mistralai/Mistral-7B-v0.1"


@dataclass
class Mixtral8x7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336   # per-expert intermediate size (HF). NOTE: MoE
    # (8 experts, top-2). Vidur's reference GPTModel profiles a DENSE MLP, so
    # this undercounts Mixtral's per-token MLP cost — handle before paper use.
    max_position_embeddings: int = 32768
    rope_theta: Optional[float] = 1000000
    vocab_size: int = 32000   # real HF config.json (was faked to 128256)

    @staticmethod
    def get_name():
        return "mistralai/Mixtral-8x7B-v0.1"


@dataclass
class DeepSeek67BModelConfig(Llama2ModelConfig):
    num_layers: int = 95
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 22016   # real HF config.json (was faked to 28672 for Llama-2-70b reuse)
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 10000
    vocab_size: int = 102400   # real HF config.json (was faked to 32768)

    @staticmethod
    def get_name():
        return "deepseek-ai/deepseek-llm-67b-chat"


@dataclass
class Qwen2_72BModelConfig(QwenModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 29568   # real HF config.json (was faked to 28672 for Llama-2-70b reuse)
    max_position_embeddings: int = 131072
    rope_theta: Optional[float] = 1000000
    vocab_size: int = 152064   # real HF config.json (was faked to 32768)
    use_qkv_bias: bool = True   # Qwen2 uses QKV bias

    @staticmethod
    def get_name():
        return "Qwen/Qwen2-72B"


@dataclass
class Llama31_405BModelConfig(Llama2ModelConfig):
    num_layers: int = 126
    num_q_heads: int = 128
    num_kv_heads: int = 8
    embedding_dim: int = 16384
    mlp_hidden_dim: int = 53248
    max_position_embeddings: int = 131072
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3.1-405B"


@dataclass
class Llama33_70BModelConfig(Llama2ModelConfig):
    # Llama-3.3-70B: identical architecture to Llama-3-70B (dense GQA).
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 131072
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Llama-3.3-70B-Instruct"


@dataclass
class Mixtral8x22BModelConfig(Llama2ModelConfig):
    # MoE (8 experts, 2/tok). Standard head_dim (6144/48=128). mlp_hidden_dim =
    # active experts x intermediate (2 x 16384) — per-token active MLP. Vidur
    # profiles this as a DENSE MLP (no routing): MoE-compute caveat.
    num_layers: int = 56
    num_q_heads: int = 48
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    mlp_hidden_dim: int = 32768
    max_position_embeddings: int = 65536
    rope_theta: Optional[float] = 1000000
    vocab_size: int = 32000

    @staticmethod
    def get_name():
        return "mistralai/Mixtral-8x22B-v0.1"


@dataclass
class Qwen3_235B_A22BModelConfig(Llama2ModelConfig):
    # MoE (128 experts, 8/tok), GQA 64q/4kv, EXPLICIT head_dim=128 (!= 4096/64).
    # mlp_hidden_dim = active experts x moe_intermediate (8 x 1536). MoE caveat.
    num_layers: int = 94
    num_q_heads: int = 64
    num_kv_heads: int = 4
    embedding_dim: int = 4096
    head_dim: Optional[int] = 128
    mlp_hidden_dim: int = 12288
    max_position_embeddings: int = 40960
    rope_theta: Optional[float] = 10000000
    vocab_size: int = 151936

    @staticmethod
    def get_name():
        return "Qwen/Qwen3-235B-A22B"


@dataclass
class Qwen3_Coder_480BModelConfig(Llama2ModelConfig):
    # MoE (160 experts, 8/tok), GQA 96q/8kv, EXPLICIT head_dim=128 (!= 6144/96).
    # mlp_hidden_dim = active experts x moe_intermediate (8 x 2560). MoE caveat.
    num_layers: int = 62
    num_q_heads: int = 96
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    head_dim: Optional[int] = 128
    mlp_hidden_dim: int = 20480
    max_position_embeddings: int = 262144
    rope_theta: Optional[float] = 10000000
    vocab_size: int = 151936

    @staticmethod
    def get_name():
        return "Qwen/Qwen3-Coder-480B-A35B-Instruct"
