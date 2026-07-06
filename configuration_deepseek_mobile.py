"""
DeepSeek Coder Mobile - 移动端优化配置
针对手机端内存限制和 ARM CPU 架构进行深度优化

优化策略:
- 5 档模型规格: nano(80M) / tiny(160M) / small(410M) / base(1.3B) / large(2.7B)
- GQA 分组查询注意力，减少 KV 缓存内存
- 滑动窗口注意力，降低长序列计算量
- RoPE 缩放支持，扩展上下文窗口
- KV 缓存量化: 4位/8位
- 设备自动检测与适配
"""
import os
import json
from typing import Optional, Dict, Any
from transformers import PretrainedConfig


class DeepseekMobileConfig(PretrainedConfig):
    """
    移动端优化的 DeepSeek 配置

    属性:
        vocab_size: 词表大小
        hidden_size: 隐藏层维度
        intermediate_size: MLP 中间层维度
        num_hidden_layers: Transformer 层数
        num_attention_heads: 查询头数
        num_key_value_heads: KV 头数 (GQA，小于 num_attention_heads)
        hidden_act: 激活函数
        max_position_embeddings: 最大序列长度
        rms_norm_eps: RMSNorm epsilon
        rope_theta: RoPE 基频
        rope_scaling: RoPE 缩放配置
        sliding_window: 滑动窗口大小 (0=禁用)
        mobile_optimize: 启用移动端优化
        kv_cache_quant_bits: KV 缓存量化位数 (4/8/0)
        torch_dtype: 推理数据类型 ("float16"/"float32"/"bfloat16")
    """
    model_type = "deepseek_mobile"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # 移动端专属配置
        mobile_optimize=True,
        kv_cache_quant_bits=8,
        torch_dtype="float16",
        sliding_window=0,
        # MLA (Multi-head Latent Attention) 配置
        mla_enabled=False,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        # MoE 配置 (移动端默认关闭)
        moe_enabled=False,
        n_routed_experts=0,
        n_shared_experts=0,
        n_activated_experts=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # 移动端配置
        self.mobile_optimize = mobile_optimize
        self.kv_cache_quant_bits = kv_cache_quant_bits
        self.torch_dtype = torch_dtype
        self.sliding_window = sliding_window

        # MLA 配置
        self.mla_enabled = mla_enabled
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # MoE 配置
        self.moe_enabled = moe_enabled
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "hidden_act": self.hidden_act,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "mobile_optimize": self.mobile_optimize,
            "kv_cache_quant_bits": self.kv_cache_quant_bits,
            "torch_dtype": self.torch_dtype,
            "sliding_window": self.sliding_window,
        }

    @classmethod
    def get_mobile_preset(cls, size: str = "small") -> "DeepseekMobileConfig":
        """
        获取移动端预设配置

        Args:
            size: 模型大小规格
                - "nano":   ~80M  参数 (超轻量，低端机可用)
                - "tiny":   ~160M 参数 (轻量，入门机可用)
                - "small":  ~410M 参数 (平衡，主流中端机)
                - "base":   ~1.3B  参数 (性能优先，旗舰机)
                - "large":  ~2.7B  参数 (高性能，平板/旗舰)
        """
        presets = {
            "nano": cls(
                vocab_size=32000,
                hidden_size=256,
                intermediate_size=704,
                num_hidden_layers=6,
                num_attention_heads=8,
                num_key_value_heads=2,
                max_position_embeddings=512,
                kv_cache_quant_bits=4,
                torch_dtype="float16",
                sliding_window=256,
            ),
            "tiny": cls(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=1408,
                num_hidden_layers=8,
                num_attention_heads=8,
                num_key_value_heads=2,
                max_position_embeddings=1024,
                kv_cache_quant_bits=4,
                torch_dtype="float16",
                sliding_window=512,
            ),
            "small": cls(
                vocab_size=32000,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=16,
                num_attention_heads=8,
                num_key_value_heads=2,
                max_position_embeddings=2048,
                kv_cache_quant_bits=8,
                torch_dtype="float16",
                sliding_window=1024,
            ),
            "base": cls(
                vocab_size=32000,
                hidden_size=2048,
                intermediate_size=5632,
                num_hidden_layers=24,
                num_attention_heads=16,
                num_key_value_heads=4,
                max_position_embeddings=4096,
                kv_cache_quant_bits=8,
                torch_dtype="float16",
                sliding_window=2048,
            ),
            "large": cls(
                vocab_size=32000,
                hidden_size=2560,
                intermediate_size=7168,
                num_hidden_layers=32,
                num_attention_heads=20,
                num_key_value_heads=4,
                max_position_embeddings=4096,
                kv_cache_quant_bits=8,
                torch_dtype="float16",
                sliding_window=2048,
            ),
        }
        return presets.get(size, presets["small"])

    @staticmethod
    def auto_detect_size() -> str:
        """
        根据设备内存自动推荐模型规格

        Returns:
            推荐的模型规格字符串
        """
        try:
            # 尝试读取系统内存
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()

            mem_total_kb = 0
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                    break

            mem_total_mb = mem_total_kb // 1024

            if mem_total_mb < 2048:
                return "nano"
            elif mem_total_mb < 4096:
                return "tiny"
            elif mem_total_mb < 6144:
                return "small"
            elif mem_total_mb < 8192:
                return "base"
            else:
                return "large"
        except Exception:
            return "small"

    def estimate_memory_mb(self, quant_bits: int = 0) -> Dict[str, float]:
        """
        估算模型内存占用

        Args:
            quant_bits: 量化位数 (0=FP32, 4, 8, 16=FP16)

        Returns:
            各部分内存占用明细
        """
        # 参数量估算
        embed_params = self.vocab_size * self.hidden_size * 2  # embed + lm_head
        attn_params = (self.hidden_size * (self.num_attention_heads + 2 * self.num_key_value_heads)
                       * self.head_dim + self.hidden_size * self.hidden_size) * self.num_hidden_layers
        mlp_params = (3 * self.hidden_size * self.intermediate_size) * self.num_hidden_layers
        norm_params = self.hidden_size * 2 * (self.num_hidden_layers + 1)

        total_params = embed_params + attn_params + mlp_params + norm_params

        # 每参数字节数
        if quant_bits == 0:
            bytes_per_param = 4  # FP32
        elif quant_bits == 16:
            bytes_per_param = 2  # FP16
        else:
            bytes_per_param = quant_bits / 8  # 量化

        weight_mb = total_params * bytes_per_param / (1024 * 1024)

        # KV 缓存估算 (单批次，最大序列)
        kv_head_dim = self.head_dim
        kv_cache_params = (2 * self.num_key_value_heads * kv_head_dim
                           * self.max_position_embeddings * self.num_hidden_layers)
        kv_bytes = self.kv_cache_quant_bits / 8 if self.kv_cache_quant_bits > 0 else 2
        kv_cache_mb = kv_cache_params * kv_bytes / (1024 * 1024)

        return {
            "total_params": total_params,
            "weight_mb": weight_mb,
            "kv_cache_mb": kv_cache_mb,
            "total_mb": weight_mb + kv_cache_mb,
        }

    def summary(self) -> str:
        """生成配置摘要字符串"""
        mem = self.estimate_memory_mb()
        return (
            f"DeepseekMobileConfig:\n"
            f"  hidden_size:       {self.hidden_size}\n"
            f"  num_layers:        {self.num_hidden_layers}\n"
            f"  num_heads:         {self.num_attention_heads} (KV: {self.num_key_value_heads})\n"
            f"  max_seq_len:       {self.max_position_embeddings}\n"
            f"  sliding_window:    {self.sliding_window}\n"
            f"  kv_cache_quant:    {self.kv_cache_quant_bits}-bit\n"
            f"  torch_dtype:       {self.torch_dtype}\n"
            f"  params:            {mem['total_params']:,}\n"
            f"  weight_memory:     {mem['weight_mb']:.1f} MB\n"
            f"  kv_cache_memory:   {mem['kv_cache_mb']:.1f} MB\n"
            f"  total_memory:      {mem['total_mb']:.1f} MB"
        )
