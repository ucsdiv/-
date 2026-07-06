# DeepSeek Coder Mobile - 手机端优化版

> 将 DeepSeek Coder AI 代码助手装进你的手机！

针对移动端（Android/iOS）深度优化的 DeepSeek Coder 版本，在保证代码质量的同时，极致压缩内存占用和计算量。

---

## 核心特性

### 移动端深度优化
- **内存优化**: KV缓存量化(4-bit/8-bit)、梯度检查点、增量推理、轻量级因果掩码
- **计算优化**: GQA分组查询注意力、滑动窗口注意力、Fused RMSNorm、PyTorch SDPA加速
- **存储优化**: 支持 4-bit/8-bit/FP16 权重量化，模型体积减少 75%+
- **流式输出**: 逐 token 生成，提升实时交互体验
- **模型预热**: 首次推理加速，避免冷启动延迟
- **自动设备检测**: CPU/GPU/MPS 自动选择
- **交互式 REPL**: 多轮对话模式，支持历史管理

### 5 档模型规格

| 规格 | 参数量 | FP32 | 4-bit | 8-bit | 适用设备 | 序列长度 |
|------|--------|------|-------|-------|----------|----------|
| **nano** | ~80M | ~320MB | ~80MB | ~160MB | 入门机 (2GB) | 512 |
| **tiny** | ~160M | ~640MB | ~160MB | ~320MB | 入门机 (3GB) | 1024 |
| **small** | ~410M | ~1.6GB | ~410MB | ~820MB | 中端机 (4GB) | 2048 |
| **base** | ~1.3B | ~5.2GB | ~650MB | ~1.3GB | 旗舰机 (6GB) | 4096 |
| **large** | ~2.7B | ~10.8GB | ~1.35GB | ~2.7GB | 平板 (8GB+) | 4096 |

> 设备内存不足时，脚本会自动推荐合适的规格

---

## 快速开始

### 方式一：一键部署（推荐）

```bash
# 克隆仓库
git clone https://github.com/deepseek-ai/DeepSeek-Coder.git
cd DeepSeek-Coder

# 运行一键部署脚本
chmod +x setup_mobile.sh
./setup_mobile.sh          # 交互式安装
./setup_mobile.sh --yes    # 非交互式安装
```

### 方式二：手动安装

```bash
# 1. 安装依赖
pip install -r requirements_mobile.txt

# 2. 运行推理
python mobile_inference.py --model_path ./model --prompt "写一个快速排序"

# 3. 交互模式
python mobile_inference.py --interactive
```

---

## Termux (Android) 安装指南

### 前置要求
- Android 7.0+
- 至少 2GB RAM（推荐 4GB+）
- 至少 1GB 存储空间

### 安装步骤

```bash
# 1. 从 F-Droid 安装 Termux (不要用 Google Play 版)
# https://f-droid.org/packages/com.termux/

# 2. 更新并安装工具
pkg update && pkg upgrade -y
pkg install git python clang wget -y

# 3. 克隆项目
cd /storage/emulated/0/
git clone https://github.com/deepseek-ai/DeepSeek-Coder.git
cd DeepSeek-Coder

# 4. 一键部署
chmod +x setup_mobile.sh
./setup_mobile.sh
```

### 常用命令

```bash
cd ~/deepseek-coder-mobile

./start.sh --prompt "写一个快速排序"    # 单次生成
./chat.sh                                # 交互对话
./benchmark.sh                           # 性能测试
./quantize.sh 4                          # 4-bit 量化
./quantize.sh 8                          # 8-bit 量化
```

---

## 模型量化

将大模型压缩到手机可运行的大小：

```bash
# 4-bit 量化（体积最小，推荐）
python mobile_quantize.py \
    --model_path ./original_model \
    --output_path ./model_q4 \
    --bits 4 \
    --model_size small \
    --eval

# 8-bit 量化（精度更高）
python mobile_quantize.py \
    --model_path ./original_model \
    --output_path ./model_q8 \
    --bits 8

# FP16 量化（最高精度）
python mobile_quantize.py \
    --model_path ./original_model \
    --output_path ./model_fp16 \
    --bits 16
```

### 量化效果对比

| 量化方式 | 模型大小 (1.3B) | 精度损失 | 推荐场景 |
|----------|-----------------|----------|----------|
| FP32 | ~5.2GB | 无 | 服务器/PC |
| FP16 | ~2.6GB | 极小 | 高端手机 |
| 8-bit | ~1.3GB | <1% | 旗舰手机 |
| 4-bit | ~650MB | ~2-5% | 主流手机 |

> `--eval` 参数可自动评估量化质量（MSE、余弦相似度）

---

## 使用示例

### 命令行

```bash
# 基本生成
python mobile_inference.py --prompt "写一个快速排序"

# 流式输出
python mobile_inference.py --stream --prompt "解释递归"

# 交互模式
python mobile_inference.py --interactive

# 性能测试
python mobile_inference.py --benchmark

# 自动检测模型大小
python mobile_inference.py --model_size auto

# 使用量化模型
python mobile_inference.py --quantized

# FP32 模式
python mobile_inference.py --fp32
```

### Python API

```python
from mobile_inference import MobileInference

# 初始化
engine = MobileInference(
    model_path="./model",
    model_size="small",
    fp16=True,
    warmup=True
)

# 生成代码
code = engine.generate("写一个快速排序", max_new_tokens=256)

# 流式生成
for token in engine.generate_stream("写一个快速排序"):
    print(token, end="", flush=True)

# 对话模式
response = engine.chat("如何优化 Python 性能？")

# 交互式 REPL
engine.interactive()

# 性能测试
engine.benchmark()

# 清理
engine.cleanup()
```

---

## 性能优化技巧

### 1. 内存优化
```python
engine = MobileInference(
    model_path="./model",
    model_size="small",     # 选择合适的规格
    low_memory=True,        # 限制线程数
    use_quantized=True,     # 使用量化模型
)
```

### 2. 生成速度优化
```python
# 贪心解码（最快）
result = engine.generate(prompt, temperature=0, do_sample=False)

# 减少生成长度
result = engine.generate(prompt, max_new_tokens=64)

# 使用 Top-k 和 Top-p 控制
result = engine.generate(
    prompt,
    temperature=0.3,        # 低温度更确定
    top_k=20,               # 限制候选词
    top_p=0.85,             # 核采样
    repetition_penalty=1.1, # 重复惩罚
)
```

### 3. 省电技巧
- 降低 `max_new_tokens`，减少生成时间
- 使用 `temperature=0` 贪心解码
- 充电时使用
- 关闭后台应用释放内存

---

## 配置选项

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `./model` | 模型路径 |
| `--model_size` | `auto` | 模型规格 (auto/nano/tiny/small/base/large) |
| `--prompt` | - | 输入提示 |
| `--max_tokens` | 128 | 最大生成 token 数 |
| `--temperature` | 0.7 | 采样温度 (0=贪心) |
| `--top_p` | 0.9 | Top-p 采样 |
| `--top_k` | 50 | Top-k 采样 |
| `--repetition_penalty` | 1.1 | 重复惩罚 |
| `--stream` | False | 流式输出 |
| `--interactive` | False | 交互式 REPL |
| `--benchmark` | False | 性能测试 |
| `--device` | auto | 设备 (auto/cpu/cuda/mps) |
| `--quantized` | False | 使用量化模型 |
| `--fp32` | False | 使用 FP32 |
| `--no_warmup` | False | 跳过预热 |

### 配置文件

`config.env` 记录了安装时的配置，可手动修改后重新部署。

---

## 技术细节

### 核心优化技术

1. **GQA (Grouped Query Attention)**
   - KV 头数减少 4-8 倍
   - KV 缓存内存大幅降低
   - 精度损失极小

2. **滑动窗口注意力**
   - 只计算窗口内的注意力
   - 长序列场景计算量降低
   - 支持窗口大小可配置

3. **KV 缓存量化**
   - 4-bit / 8-bit 量化
   - 推理时动态反量化
   - 长序列内存节省显著

4. **PyTorch SDPA**
   - 调用 `scaled_dot_product_attention` fused kernel
   - 自动选择最优后端
   - 减少 GPU/CPU 内存访问

5. **向量化量化**
   - 分组量化避免 Python 循环
   - reshape + 广播加速
   - 量化速度提升 10x+

6. **模型预热**
   - 首次推理前运行 dummy input
   - 预编译计算图
   - 避免冷启动延迟

---

## 常见问题

### Q: 手机能跑起来吗？
A: 可以！4-bit 量化的 1.3B 模型在 6GB RAM 手机上可流畅运行。2GB RAM 手机推荐 nano 规格。

### Q: 生成速度怎么样？
A: 根据手机性能不同，通常 2-15 tokens/秒。运行 `./benchmark.sh` 测试实际速度。

### Q: 会不会很耗电？
A: 推理时 CPU 占用较高，建议充电时使用。贪心解码(`temperature=0`)速度最快、最省电。

### Q: 支持哪些编程语言？
A: 支持 87+ 种编程语言，包括 Python、JavaScript、Java、C++、Go、Rust 等。

### Q: 模型从哪里下载？
A: HuggingFace 或 ModelScope 下载 DeepSeek-Coder 系列模型，推荐 1.3B 及以下版本。

### Q: 量化后精度损失大吗？
A: 8-bit 几乎无感(<1%)，4-bit 约 2-5%。使用 `--eval` 参数可评估实际精度。

### Q: 支持 iOS 吗？
A: 通过 iSH 支持。安装方式与 Termux 类似，使用 `apk` 替代 `pkg`。

---

## 项目结构

```
DeepSeek-Coder/
├── configuration_deepseek_mobile.py  # 移动端配置 (5档预设/自动检测)
├── modeling_deepseek_mobile.py        # 移动端模型 (GQA/滑动窗口/KV量化)
├── mobile_inference.py               # 推理引擎 (流式/交互/benchmark)
├── mobile_quantize.py                # 量化工具 (4-bit/8-bit/FP16)
├── setup_mobile.sh                   # 一键部署 (Termux/iSH/Linux)
├── requirements_mobile.txt            # 移动端依赖
├── README_MOBILE.md                  # 本文档
├── LICENSE-CODE                      # 代码许可证
├── LICENSE-MODEL                     # 模型许可证
└── README.md                         # 原版文档
```

---

## 许可证

- 代码: MIT License (LICENSE-CODE)
- 模型: DeepSeek Model License (LICENSE-MODEL)

---

**把 AI 装进口袋，随时随地写代码！**
