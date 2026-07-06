"""
DeepSeek Coder Mobile - 轻量级推理引擎
专为手机端优化的代码补全和对话推理工具

使用方式:
    python mobile_inference.py --model_path ./model --prompt "写一个快速排序"
    python mobile_inference.py --model_path ./model --interactive
    python mobile_inference.py --benchmark
"""
import argparse
import sys
import os
import time
import gc
import torch
from typing import Optional, List, Generator, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling_deepseek_mobile import MobileDeepseekForCausalLM
from configuration_deepseek_mobile import DeepseekMobileConfig


class MobileInference:
    """
    移动端推理引擎 - 专为手机 CPU 优化

    优化特性:
    - 低内存模式: 自动管理内存，避免 OOM
    - 流式输出: 逐 token 输出，提升体验
    - 自动设备检测: CPU/GPU/MPS 自动选择
    - 温度/Top-p/Top-k 采样控制
    - 模型预热: 首次推理加速
    - 交互式 REPL: 多轮对话模式
    - 性能基准: 内置 benchmark 工具
    """

    def __init__(
        self,
        model_path: str,
        model_size: str = "small",
        device: str = "auto",
        low_memory: bool = True,
        use_quantized: bool = False,
        fp16: bool = True,
        warmup: bool = True,
    ):
        self.model_path = model_path
        self.model_size = model_size
        self.low_memory = low_memory
        self.use_quantized = use_quantized
        self.fp16 = fp16

        # 自动设备检测
        self.device = self._detect_device(device)
        self.model = None
        self.tokenizer = None
        self._load_tokenizer()
        self._load_model(warmup)

    def _detect_device(self, device: str) -> str:
        """自动检测最优设备"""
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_tokenizer(self):
        """加载 tokenizer"""
        try:
            from transformers import AutoTokenizer
            tokenizer_path = self.model_path
            if os.path.exists(os.path.join(self.model_path, "tokenizer.json")):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
                print(f"[Mobile] Tokenizer loaded from {self.model_path}")
            else:
                print(f"[Mobile] No tokenizer found at {self.model_path}, using test mode")
                self.tokenizer = None
        except Exception as e:
            print(f"[Mobile] Tokenizer load failed: {e}, using test mode")
            self.tokenizer = None

    def _load_model(self, do_warmup: bool = True):
        """加载模型（低内存模式）"""
        print(f"[Mobile] Loading model: {self.model_path}")
        print(f"[Mobile] Device: {self.device}")
        print(f"[Mobile] Size: {self.model_size}")
        print(f"[Mobile] Low memory: {self.low_memory}")
        print(f"[Mobile] FP16: {self.fp16}")

        start_time = time.time()

        config = DeepseekMobileConfig.get_mobile_preset(self.model_size)

        self.model = MobileDeepseekForCausalLM(config)

        # 加载权重
        model_file = os.path.join(self.model_path, 'quantized_model.pt') if self.use_quantized \
            else os.path.join(self.model_path, 'pytorch_model.bin')

        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[Mobile] Weights loaded: {model_file}")
        else:
            print(f"[Mobile] Warning: no weights found, using random init")

        self.model.eval()
        self.model.to(self.device)

        # FP16 优化
        if self.fp16 and self.device != "cpu":
            try:
                self.model.half()
                print(f"[Mobile] Model converted to FP16")
            except Exception:
                pass

        # 低内存模式: 限制线程数
        if self.low_memory:
            num_threads = min(4, os.cpu_count() or 4)
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(2)
            print(f"[Mobile] Threads: {num_threads}")

        load_time = time.time() - start_time
        mem_usage = self.model.get_memory_usage()
        num_params = self.model.get_num_params()
        print(f"[Mobile] Loaded in {load_time:.2f}s")
        print(f"[Mobile] Params: {num_params:,}")
        print(f"[Mobile] Memory: {mem_usage:.2f} MB")

        # 预热
        if do_warmup:
            print(f"[Mobile] Warming up...")
            warmup_start = time.time()
            self.model.warmup(seq_len=32, device=self.device)
            print(f"[Mobile] Warmup done in {time.time() - warmup_start:.2f}s")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """生成文本（完整输出）"""
        if self.tokenizer is None:
            return self._generate_simple(prompt, max_new_tokens, temperature, top_p, do_sample)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        all_tokens = input_ids
        with torch.no_grad():
            for token_id in self.model.generate_stream(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            ):
                all_tokens = torch.cat(
                    [all_tokens, torch.tensor([[token_id]], device=self.device)], dim=1
                )

        generated = all_tokens[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> Generator[str, None, None]:
        """流式生成文本（逐 token 输出）"""
        if self.tokenizer is None:
            yield from self._generate_stream_simple(prompt, max_new_tokens, temperature, top_p, do_sample)
            return

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        for token_id in self.model.generate_stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        ):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield token_text

    def _generate_simple(self, prompt, max_new_tokens, temperature, top_p, do_sample):
        """简化版生成（无 tokenizer）"""
        print("[Mobile] No tokenizer, test mode only")
        input_ids = torch.randint(0, 1000, (1, len(prompt.split())), device=self.device)
        count = 0
        for _ in self.model.generate_stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0,
            top_p=top_p,
        ):
            count += 1
        return f"[Generated {count} tokens (no tokenizer)]"

    def _generate_stream_simple(self, prompt, max_new_tokens, temperature, top_p, do_sample):
        """简化版流式生成（无 tokenizer）"""
        input_ids = torch.randint(0, 1000, (1, 10), device=self.device)
        for token_id in self.model.generate_stream(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0,
            top_p=top_p,
        ):
            yield f"token_{token_id} "

    def chat(self, message: str, history: Optional[List] = None) -> str:
        """对话模式"""
        if history is None:
            history = []
        prompt = self._build_chat_prompt(message, history)
        response = self.generate(prompt, max_new_tokens=512)
        return response

    def chat_stream(self, message: str, history: Optional[List] = None) -> Generator[str, None, None]:
        """流式对话模式"""
        if history is None:
            history = []
        prompt = self._build_chat_prompt(message, history)
        yield from self.generate_stream(prompt, max_new_tokens=512)

    def _build_chat_prompt(self, message: str, history: List) -> str:
        """构建对话提示词"""
        system_prompt = (
            "You are an AI programming assistant, utilizing the DeepSeek Coder model. "
            "You only answer questions related to computer science. "
            "For non-computer-science questions, you will refuse to answer.\n\n"
        )
        prompt = system_prompt
        for user_msg, assistant_msg in history:
            prompt += f"### Instruction:\n{user_msg}\n### Response:\n{assistant_msg}\n<|EOT|>\n"
        prompt += f"### Instruction:\n{message}\n### Response:\n"
        return prompt

    def interactive(self):
        """交互式 REPL 模式"""
        print("\n" + "=" * 60)
        print("  DeepSeek Coder Mobile - 交互模式")
        print("  输入 'quit' 或 'exit' 退出")
        print("  输入 'clear' 清空历史")
        print("  输入 'stats' 查看统计信息")
        print("=" * 60 + "\n")

        history = []
        total_tokens = 0
        total_time = 0.0

        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("再见！")
                break
            if user_input.lower() == "clear":
                history = []
                print("[历史已清空]")
                continue
            if user_input.lower() == "stats":
                print(f"  对话轮数: {len(history)}")
                print(f"  总生成 token: {total_tokens}")
                print(f"  总耗时: {total_time:.1f}s")
                if total_time > 0:
                    print(f"  平均速度: {total_tokens/total_time:.1f} tokens/s")
                continue

            print("AI: ", end="", flush=True)
            start = time.time()
            token_count = 0

            try:
                for token in self.chat_stream(user_input, history):
                    print(token, end="", flush=True)
                    token_count += 1
            except Exception as e:
                print(f"\n[错误] {e}")

            elapsed = time.time() - start
            total_tokens += token_count
            total_time += elapsed
            print(f"\n  [{token_count} tokens, {elapsed:.1f}s, {token_count/elapsed:.1f} tok/s]" if elapsed > 0 else "")

            # 保存到历史
            response_text = self.generate(user_input, max_new_tokens=512) if self.tokenizer else ""
            history.append((user_input, response_text))

            # 限制历史长度
            if len(history) > 10:
                history = history[-10:]

    def benchmark(self, seq_len=128, num_runs=5):
        """性能基准测试"""
        print("\n" + "=" * 60)
        print("  Benchmark")
        print("=" * 60)

        results = self.model.benchmark(seq_len=seq_len, num_runs=num_runs, device=self.device)

        print(f"  Device:         {results['device']}")
        print(f"  Sequence len:   {results['seq_len']}")
        print(f"  Runs:           {results['num_runs']}")
        print(f"  Total tokens:   {results['total_tokens']}")
        print(f"  Total time:     {results['total_time_s']:.2f}s")
        print(f"  Speed:          {results['tokens_per_sec']:.1f} tokens/s")
        print(f"  Per token:      {results['time_per_token_ms']:.1f} ms")
        print(f"  Model memory:   {results['memory_mb']:.1f} MB")

        return results

    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        info = {
            "model_memory_mb": self.model.get_memory_usage() if self.model else 0,
            "model_params": self.model.get_num_params() if self.model else 0,
        }
        try:
            import psutil
            process = psutil.Process(os.getpid())
            info["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
            info["system_available_mb"] = psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            # psutil 未安装时使用 torch 内存信息
            if torch.cuda.is_available():
                info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        return info

    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Mobile] Resources cleaned up")


def main():
    parser = argparse.ArgumentParser(
        description='DeepSeek Coder Mobile - 移动端推理引擎',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model_path', type=str, default='./model', help='模型路径')
    parser.add_argument('--model_size', type=str, default='auto',
                        choices=['auto', 'nano', 'tiny', 'small', 'base', 'large'],
                        help='模型大小规格 (auto=自动检测)')
    parser.add_argument('--prompt', type=str, default='写一个快速排序算法', help='输入提示')
    parser.add_argument('--max_tokens', type=int, default=128, help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度 (0=贪心)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p 采样')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k 采样 (0=禁用)')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='重复惩罚')
    parser.add_argument('--stream', action='store_true', help='流式输出')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互式 REPL 模式')
    parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')
    parser.add_argument('--device', type=str, default='auto', help='运行设备 (auto/cpu/cuda/mps)')
    parser.add_argument('--low_memory', action='store_true', default=True, help='低内存模式')
    parser.add_argument('--quantized', action='store_true', help='使用量化模型')
    parser.add_argument('--fp32', action='store_true', help='使用 FP32 (默认 FP16)')
    parser.add_argument('--no_warmup', action='store_true', help='跳过模型预热')

    args = parser.parse_args()

    # 自动检测模型大小
    model_size = args.model_size
    if model_size == "auto":
        model_size = DeepseekMobileConfig.auto_detect_size()
        print(f"[Mobile] Auto-detected model size: {model_size}")

    print("=" * 60)
    print("  DeepSeek Coder Mobile - 推理引擎")
    print("=" * 60)

    engine = MobileInference(
        model_path=args.model_path,
        model_size=model_size,
        device=args.device,
        low_memory=args.low_memory,
        use_quantized=args.quantized,
        fp16=not args.fp32,
        warmup=not args.no_warmup,
    )

    try:
        if args.benchmark:
            engine.benchmark()
        elif args.interactive:
            engine.interactive()
        else:
            print(f"\n输入: {args.prompt}")
            print("-" * 60)

            if args.stream:
                print("输出 (流式):")
                start_time = time.time()
                token_count = 0
                for token in engine.generate_stream(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                ):
                    print(token, end='', flush=True)
                    token_count += 1
                elapsed = time.time() - start_time
                print(f"\n\n[{token_count} tokens, {elapsed:.2f}s, {token_count/elapsed:.1f} tok/s]" if elapsed > 0 else "")
            else:
                start_time = time.time()
                result = engine.generate(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                )
                elapsed = time.time() - start_time
                print(f"输出:\n{result}")
                print(f"\n耗时: {elapsed:.2f}s")

        # 内存信息
        mem_info = engine.get_memory_info()
        print(f"\n内存: 模型 {mem_info.get('model_memory_mb', 0):.1f} MB", end="")
        if 'process_memory_mb' in mem_info:
            print(f" | 进程 {mem_info['process_memory_mb']:.1f} MB", end="")
        if 'system_available_mb' in mem_info:
            print(f" | 可用 {mem_info['system_available_mb']:.1f} MB", end="")
        print()

    except KeyboardInterrupt:
        print("\n\n[已中断]")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
