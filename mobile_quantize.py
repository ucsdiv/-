"""
DeepSeek Coder Mobile - 模型量化工具
支持 4-bit / 8-bit / FP16 量化，大幅减小模型体积和内存占用

使用方式:
    python mobile_quantize.py --model_path ./model --output_path ./model_q4 --bits 4
    python mobile_quantize.py --model_path ./model --output_path ./model_fp16 --bits 16
    python mobile_quantize.py --eval  # 量化质量评估
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path


class QuantizedLinear(nn.Module):
    """
    量化线性层 - 支持 4-bit / 8-bit 对称量化

    优化点:
    - 向量化分组量化，避免 Python 循环
    - 权重量化后存储，推理时反量化计算
    - 移动端 CPU 友好的内存布局
    """
    def __init__(self, in_features, out_features, bias=False, bits=8, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.use_bias = bias

        self.register_buffer('qweight', torch.empty((out_features, in_features), dtype=torch.int8))
        n_groups = in_features // group_size
        self.register_buffer('scales', torch.empty((out_features, n_groups), dtype=torch.float16))

        if bias:
            self.register_buffer('bias', torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits=8, group_size=128):
        """从现有线性层创建量化版本"""
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
            group_size=group_size
        )
        q_linear.quantize(linear.weight.data, linear.bias.data if linear.bias is not None else None)
        return q_linear

    def quantize(self, weight, bias=None):
        """向量化量化权重 - 避免循环，大幅加速"""
        weight = weight.float()
        out_features, in_features = weight.shape

        # 确保能被 group_size 整除
        if in_features % self.group_size != 0:
            self.group_size = in_features

        n_groups = in_features // self.group_size

        # 向量化: reshape 为 (out_features, n_groups, group_size)
        weight_grouped = weight.view(out_features, n_groups, self.group_size)

        if self.bits == 8:
            qmax = 127.0
            scale = weight_grouped.abs().amax(dim=-1, keepdim=True) / qmax
            scale = scale.clamp(min=1e-8)
            q = torch.round(weight_grouped / scale).clamp(-128, 127).to(torch.int8)
        elif self.bits == 4:
            qmax = 7.0
            scale = weight_grouped.abs().amax(dim=-1, keepdim=True) / qmax
            scale = scale.clamp(min=1e-8)
            q = torch.round(weight_grouped / scale).clamp(-8, 7).to(torch.int8)
        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

        self.qweight = q.view(out_features, in_features)
        self.scales = scale.squeeze(-1).half()

        if bias is not None:
            self.bias = bias.half()

    def dequantize(self):
        """反量化权重 - 向量化"""
        out_features, in_features = self.qweight.shape
        n_groups = in_features // self.group_size

        # reshape 并广播 scales
        q_grouped = self.qweight.view(out_features, n_groups, self.group_size)
        scales = self.scales.unsqueeze(-1)  # (out, n_groups, 1)

        weight = q_grouped.half() * scales
        return weight.view(out_features, in_features)

    def forward(self, x):
        weight = self.dequantize()
        return nn.functional.linear(x, weight, self.bias)


class FP16Linear(nn.Module):
    """FP16 量化线性层 - 简单但有效"""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight', torch.empty((out_features, in_features), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        q = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        q.weight = linear.weight.data.half()
        if linear.bias is not None:
            q.bias = linear.bias.data.half()
        return q

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


def quantize_model(model, bits=8, group_size=128, skip_lm_head=False):
    """
    量化整个模型

    Args:
        model: 待量化模型
        bits: 量化位数 (4, 8, 16)
        group_size: 量化组大小
        skip_lm_head: 是否跳过 lm_head (输出层通常不量化以保持精度)

    Returns:
        量化后的模型
    """
    model.eval()
    replaced_count = 0
    skipped_count = 0

    def _replace_linear(module, prefix=""):
        nonlocal replaced_count, skipped_count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and not isinstance(child, (QuantizedLinear, FP16Linear)):
                # 跳过 lm_head (输出投影层)
                if skip_lm_head and "lm_head" in full_name:
                    skipped_count += 1
                    continue

                if bits == 16:
                    setattr(module, name, FP16Linear.from_linear(child))
                else:
                    setattr(module, name, QuantizedLinear.from_linear(
                        child, bits=bits, group_size=group_size
                    ))
                replaced_count += 1
            else:
                _replace_linear(child, full_name)

    _replace_linear(model)
    print(f"  Quantized {replaced_count} layers, skipped {skipped_count} layers")
    return model


def save_quantized_model(model, output_path, config=None):
    """保存量化模型"""
    os.makedirs(output_path, exist_ok=True)

    state_dict = model.state_dict()

    # 保存为 safetensors 格式 (如果可用) 或 pt
    try:
        from safetensors.torch import save_file
        # safetensors 要求 tensors 是连续的
        safe_state = {k: v.contiguous() for k, v in state_dict.items()}
        save_file(safe_state, os.path.join(output_path, 'quantized_model.safetensors'))
        print(f"  Saved as safetensors format")
    except ImportError:
        torch.save(state_dict, os.path.join(output_path, 'quantized_model.pt'))
        print(f"  Saved as PyTorch format")

    if config is not None:
        config.save_pretrained(output_path)

    # 计算实际大小
    total_size = 0
    for name, param in state_dict.items():
        total_size += param.element_size() * param.nelement()

    print(f"  Quantized model size: {total_size / (1024*1024):.2f} MB")
    return total_size


def estimate_model_size(model, bits=32):
    """估算模型大小（MB）"""
    total = 0
    for param in model.parameters():
        if bits == 32:
            total += param.element_size() * param.nelement()
        else:
            total += (bits / 8) * param.nelement()
    return total / (1024 * 1024)


def evaluate_quantization(original_model, quantized_model, test_inputs=10, vocab_size=32000):
    """
    量化质量评估 - 比较原始模型和量化模型的输出差异

    Returns:
        包含 MSE、余弦相似度等指标的字典
    """
    print("\n  Evaluating quantization quality...")
    original_model.eval()
    quantized_model.eval()

    mse_total = 0
    cosine_total = 0
    max_diff_total = 0

    with torch.no_grad():
        for i in range(test_inputs):
            input_ids = torch.randint(0, vocab_size, (1, 32))

            orig_out = original_model(input_ids).logits
            quant_out = quantized_model(input_ids).logits

            # MSE
            mse = torch.mean((orig_out - quant_out) ** 2).item()
            mse_total += mse

            # 余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(
                orig_out.flatten(), quant_out.flatten(), dim=0
            ).item()
            cosine_total += cos_sim

            # 最大差异
            max_diff = torch.max(torch.abs(orig_out - quant_out)).item()
            max_diff_total += max_diff

    return {
        "mse": mse_total / test_inputs,
        "cosine_similarity": cosine_total / test_inputs,
        "max_diff": max_diff_total / test_inputs,
        "test_samples": test_inputs,
    }


def main():
    parser = argparse.ArgumentParser(
        description='DeepSeek Coder Mobile - Model Quantization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to the original model')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save quantized model')
    parser.add_argument('--bits', type=int, default=4, choices=[4, 8, 16],
                        help='Quantization bits (4=int4, 8=int8, 16=fp16)')
    parser.add_argument('--group_size', type=int, default=128, help='Quantization group size')
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['nano', 'tiny', 'small', 'base', 'large'],
                        help='Model size preset')
    parser.add_argument('--skip_lm_head', action='store_true', default=True,
                        help='Skip lm_head quantization for better output quality')
    parser.add_argument('--eval', action='store_true', help='Evaluate quantization quality')

    args = parser.parse_args()

    print("=" * 60)
    print("  DeepSeek Coder Mobile - Quantization Tool")
    print("=" * 60)
    print(f"  Model path:   {args.model_path}")
    print(f"  Output path:  {args.output_path}")
    print(f"  Model size:   {args.model_size}")
    print(f"  Quantization: {args.bits}-bit")
    print(f"  Group size:   {args.group_size}")
    print()

    from modeling_deepseek_mobile import MobileDeepseekForCausalLM
    from configuration_deepseek_mobile import DeepseekMobileConfig

    # 加载原始模型
    print("[1/4] Loading original model...")
    start = time.time()
    config = DeepseekMobileConfig.get_mobile_preset(args.model_size)
    model = MobileDeepseekForCausalLM(config)

    model_file = os.path.join(args.model_path, 'pytorch_model.bin')
    if os.path.exists(model_file):
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights from {model_file}")
    else:
        print(f"  No weights found, using random init")

    original_size = estimate_model_size(model, bits=32)
    original_params = sum(p.numel() for p in model.parameters())
    print(f"  Original (FP32): {original_size:.2f} MB ({original_params:,} params)")
    print(f"  Load time: {time.time()-start:.2f}s")

    # 保存原始模型副本用于评估
    if args.eval:
        print("  Cloning model for evaluation...")
        import copy
        original_model = copy.deepcopy(model)

    # 量化
    print(f"\n[2/4] Quantizing to {args.bits}-bit...")
    start = time.time()
    model = quantize_model(
        model, bits=args.bits, group_size=args.group_size,
        skip_lm_head=args.skip_lm_head
    )
    print(f"  Quantization time: {time.time()-start:.2f}s")

    quant_size = estimate_model_size(model, bits=32)  # 实际存储大小
    estimated_size = original_size * (args.bits / 32) if args.bits < 16 else original_size / 2
    print(f"  Estimated {args.bits}-bit size: ~{estimated_size:.2f} MB")
    print(f"  Compression ratio: {original_size / max(quant_size, 0.001):.1f}x")

    # 评估
    if args.eval:
        print(f"\n[3/4] Evaluating quantization quality...")
        metrics = evaluate_quantization(original_model, model, vocab_size=config.vocab_size)
        print(f"  MSE:              {metrics['mse']:.6f}")
        print(f"  Cosine sim:       {metrics['cosine_similarity']:.6f}")
        print(f"  Max diff:         {metrics['max_diff']:.6f}")
        print(f"  Quality score:    {'Excellent' if metrics['cosine_similarity'] > 0.99 else 'Good' if metrics['cosine_similarity'] > 0.95 else 'Fair'}")

    # 保存
    print(f"\n[4/4] Saving quantized model...")
    save_quantized_model(model, args.output_path, config)

    print("\n" + "=" * 60)
    print("  Quantization completed!")
    print(f"  Original: {original_size:.2f} MB → Quantized: ~{estimated_size:.2f} MB")
    print(f"  Saved to: {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
