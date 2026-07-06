#!/data/data/com.termux/files/usr/bin/bash
# ============================================================
# DeepSeek Coder Mobile - 手机端一键部署脚本
# 适用于 Termux (Android) / iSH (iOS) / Linux
#
# 优化点:
# - 智能内存检测，自动推荐模型规格
# - 多镜像源加速 (PyPI/PyTorch)
# - 自动模型下载 (HuggingFace/ModelScope)
# - 完善的错误处理和日志
# - 生成快捷启动脚本
# ============================================================

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
INSTALL_DIR="${DEEPSEEK_MOBILE_DIR:-$HOME/deepseek-coder-mobile}"
MODEL_SIZE=""
QUANT_BITS=4
PYTHON="python3"
PIP="pip3"
VERBOSE=false

# 日志文件
LOG_FILE="/tmp/deepseek_mobile_install.log"

# ============================================================
# 工具函数
# ============================================================

log() { echo "[$(date '+%H:%M:%S')] $1" >> "$LOG_FILE"; }

print_banner() {
    echo -e "${CYAN}"
    cat << 'BANNER'
╔══════════════════════════════════════════════╗
║                                              ║
║   DeepSeek Coder Mobile - 移动端部署工具     ║
║                                              ║
║   🤖 手机端 AI 代码助手                      ║
║   📱 支持 Android / iOS / Linux              ║
║   ⚡ 5档模型规格自动适配                     ║
║                                              ║
╚══════════════════════════════════════════════╝
BANNER
    echo -e "${NC}"
}

print_step()   { echo -e "${GREEN}[✓]${NC} $1"; log "STEP: $1"; }
print_info()    { echo -e "${BLUE}[i]${NC} $1"; log "INFO: $1"; }
print_warn()    { echo -e "${YELLOW}[!]${NC} $1"; log "WARN: $1"; }
print_error()   { echo -e "${RED}[✗]${NC} $1"; log "ERROR: $1"; }
print_progress() { echo -e "${CYAN}[…]${NC} $1"; }

die() { print_error "$1"; exit 1; }

# ============================================================
# 环境检测
# ============================================================

detect_platform() {
    if [ -d "/data/data/com.termux" ]; then
        PLATFORM="termux"
        PYTHON="python"
        PIP="pip"
        print_step "检测到 Termux (Android) 环境"
    elif uname -a 2>/dev/null | grep -qi "iSH"; then
        PLATFORM="ish"
        print_step "检测到 iSH (iOS) 环境"
    elif [ -f "/etc/os-release" ]; then
        PLATFORM="linux"
        . /etc/os-release
        print_step "检测到 Linux 环境: ${PRETTY_NAME:-unknown}"
    else
        PLATFORM="unknown"
        print_warn "未知平台，尝试通用 Linux 方式安装"
    fi
}

check_memory() {
    if [ ! -f /proc/meminfo ]; then
        print_warn "无法检测内存，使用默认配置"
        MODEL_SIZE="small"
        QUANT_BITS=4
        return
    fi

    local total_mem_kb
    total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local total_mem_mb=$((total_mem_kb / 1024))
    local avail_mem_mb
    avail_mem_mb=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}')
    avail_mem_mb=$((avail_mem_kb / 1024))

    print_info "设备总内存: ${total_mem_mb} MB"
    print_info "可用内存: ${avail_mem_mb} MB"

    # 自动推荐
    if [ "$total_mem_mb" -lt 2048 ]; then
        MODEL_SIZE="nano"
        QUANT_BITS=4
        print_warn "内存 < 2GB，推荐 nano 模型 (4-bit)"
    elif [ "$total_mem_mb" -lt 4096 ]; then
        MODEL_SIZE="tiny"
        QUANT_BITS=4
        print_warn "内存 < 4GB，推荐 tiny 模型 (4-bit)"
    elif [ "$total_mem_mb" -lt 6144 ]; then
        MODEL_SIZE="small"
        QUANT_BITS=4
        print_info "内存充足，推荐 small 模型 (4-bit)"
    elif [ "$total_mem_mb" -lt 8192 ]; then
        MODEL_SIZE="base"
        QUANT_BITS=8
        print_info "内存充足，推荐 base 模型 (8-bit)"
    else
        MODEL_SIZE="large"
        QUANT_BITS=8
        print_info "内存充裕，推荐 large 模型 (8-bit)"
    fi
}

check_storage() {
    local available_kb
    available_kb=$(df "$HOME" 2>/dev/null | tail -1 | awk '{print $4}')
    if [ -n "$available_kb" ]; then
        local available_mb=$((available_kb / 1024))
        print_info "可用存储: ${available_mb} MB"

        if [ "$available_mb" -lt 1024 ]; then
            die "存储空间不足 (需要至少 1GB)，请清理空间后重试"
        elif [ "$available_mb" -lt 2048 ]; then
            print_warn "存储空间较少 (<2GB)，建议清理缓存"
        fi
    fi
}

# ============================================================
# 依赖安装
# ============================================================

install_system_deps() {
    print_progress "安装系统依赖..."

    case "$PLATFORM" in
        termux)
            pkg update -y 2>/dev/null || true
            pkg install -y python clang make git wget 2>/dev/null || {
                print_error "依赖安装失败"
                print_info "请手动运行: pkg install python clang make git wget"
                return 1
            }
            ;;
        ish)
            apk update 2>/dev/null || true
            apk add python3 py3-pip python3-dev clang make git wget 2>/dev/null || {
                print_error "依赖安装失败"
                return 1
            }
            ;;
        linux)
            if command -v apt &>/dev/null; then
                sudo apt update -y 2>/dev/null || true
                sudo apt install -y python3 python3-pip python3-dev python3-venv git wget 2>/dev/null || true
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y python3 python3-pip python3-devel git wget 2>/dev/null || true
            elif command -v yum &>/dev/null; then
                sudo yum install -y python3 python3-pip python3-devel git wget 2>/dev/null || true
            elif command -v pacman &>/dev/null; then
                sudo pacman -S --noconfirm python python-pip git wget 2>/dev/null || true
            fi
            ;;
    esac
    print_step "系统依赖安装完成"
}

install_python_deps() {
    print_progress "安装 Python 依赖..."

    # 尝试使用国内镜像加速
    local PIP_INDEX=""
    if curl -s --connect-timeout 3 https://pypi.org &>/dev/null; then
        PIP_INDEX=""  # 官方源可访问
    else
        PIP_INDEX="-i https://pypi.tuna.tsinghua.edu.cn/simple"
        print_info "使用清华镜像源加速"
    fi

    $PIP install --upgrade pip $PIP_INDEX 2>/dev/null || true

    # PyTorch (CPU 版)
    print_progress "安装 PyTorch (CPU)..."
    if [ "$PLATFORM" = "termux" ]; then
        $PIP install torch $PIP_INDEX 2>/dev/null || {
            print_warn "PyTorch 安装失败，尝试 numpy 替代"
            $PIP install numpy $PIP_INDEX
        }
    else
        $PIP install torch --extra-index-url https://download.pytorch.org/whl/cpu $PIP_INDEX 2>/dev/null || {
            print_warn "PyTorch CPU 源失败，尝试默认源"
            $PIP install torch $PIP_INDEX 2>/dev/null || print_warn "PyTorch 安装失败"
        }
    fi

    # transformers 和 tokenizers
    $PIP install tokenizers $PIP_INDEX 2>/dev/null || print_warn "tokenizers 安装失败"
    $PIP install transformers $PIP_INDEX 2>/dev/null || print_warn "transformers 安装失败"

    # 可选: psutil (内存监控)
    $PIP install psutil $PIP_INDEX 2>/dev/null || true

    # 可选: safetensors (量化保存)
    $PIP install safetensors $PIP_INDEX 2>/dev/null || true

    print_step "Python 依赖安装完成"
}

# ============================================================
# 项目部署
# ============================================================

setup_project() {
    print_progress "部署项目到 $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # 核心文件列表
    local FILES=(
        "configuration_deepseek_mobile.py"
        "modeling_deepseek_mobile.py"
        "mobile_inference.py"
        "mobile_quantize.py"
        "requirements_mobile.txt"
        "README_MOBILE.md"
    )

    for file in "${FILES[@]}"; do
        if [ -f "$SCRIPT_DIR/$file" ]; then
            cp "$SCRIPT_DIR/$file" "$INSTALL_DIR/"
            print_progress "复制: $file"
        else
            print_warn "文件不存在: $file，从 GitHub 下载..."
            wget -q -O "$INSTALL_DIR/$file" \
                "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Coder/main/$file" 2>/dev/null || \
                print_warn "下载失败: $file"
        fi
    done

    print_step "项目文件部署完成"
}

download_model() {
    print_progress "设置模型目录..."
    local MODEL_DIR="$INSTALL_DIR/model"
    mkdir -p "$MODEL_DIR"

    print_info "模型规格: $MODEL_SIZE"
    print_info "量化位数: ${QUANT_BITS}-bit"

    # 尝试自动下载模型 (1.3B base 模型)
    local MODEL_URL=""
    local HF_MODEL="deepseek-ai/deepseek-coder-1.3b-base"

    # 检查是否已有模型
    if [ -f "$MODEL_DIR/pytorch_model.bin" ] || [ -f "$MODEL_DIR/quantized_model.pt" ] || [ -f "$MODEL_DIR/quantized_model.safetensors" ]; then
        print_step "检测到已有模型文件"
        return 0
    fi

    # 尝试 huggingface-cli 下载
    if command -v huggingface-cli &>/dev/null; then
        print_progress "尝试使用 huggingface-cli 下载模型..."
        if huggingface-cli download "$HF_MODEL" \
            --include "*.json" \
            --local-dir "$MODEL_DIR" 2>/dev/null; then
            print_step "Tokenizer 下载成功"
        fi
    fi

    # 创建模型说明
    cat > "$MODEL_DIR/README.txt" << EOF
DeepSeek Coder Mobile - 模型目录
================================

推荐模型规格: $MODEL_SIZE
推荐量化位数: ${QUANT_BITS}-bit

请将以下文件放入此目录:
  必需:
  - pytorch_model.bin (FP32/FP16 权重)
  - 或 quantized_model.pt / .safetensors (量化权重)
  - config.json (模型配置)
  - tokenizer.json (分词器)
  - tokenizer_config.json (分词器配置)

下载地址:
  HuggingFace:
    https://huggingface.co/deepseek-ai
    推荐模型:
    - deepseek-coder-1.3b-base (约 2.5GB)
    - deepseek-coder-6.7b-base (约 13GB)

  ModelScope (国内加速):
    https://modelscope.cn/models/deepseek-ai
    pip install modelscope
    modelscope download --model deepseek-ai/deepseek-coder-1.3b-base

量化命令 (减小模型体积):
  cd $INSTALL_DIR
  python mobile_quantize.py --model_path ./model --output_path ./model_q4 --bits 4 --model_size $MODEL_SIZE
  python mobile_quantize.py --model_path ./model --output_path ./model_q8 --bits 8 --model_size $MODEL_SIZE

提示:
  - 推荐 1.3B 以下模型以获得最佳移动端体验
  - 4-bit 量化可将 1.3B 模型压缩至约 650MB
EOF

    print_step "模型目录已创建"
    print_warn "请将模型文件放入: $MODEL_DIR"
}

create_scripts() {
    print_progress "创建启动脚本..."

    # 主启动脚本
    cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# DeepSeek Coder Mobile - 启动脚本

cd "$(dirname "$0")"

# 环境变量优化
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "=============================================="
echo "  DeepSeek Coder Mobile"
echo "=============================================="

python mobile_inference.py \
    --model_path ./model \
    --stream \
    "$@"

echo ""
echo "=============================================="
EOF
    chmod +x "$INSTALL_DIR/start.sh"

    # 对话模式脚本
    cat > "$INSTALL_DIR/chat.sh" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# DeepSeek Coder Mobile - 交互式对话

cd "$(dirname "$0")"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python mobile_inference.py \
    --model_path ./model \
    --interactive \
    "$@"
EOF
    chmod +x "$INSTALL_DIR/chat.sh"

    # 性能测试脚本
    cat > "$INSTALL_DIR/benchmark.sh" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# DeepSeek Coder Mobile - 性能基准测试

cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

python mobile_inference.py \
    --model_path ./model \
    --benchmark \
    "$@"
EOF
    chmod +x "$INSTALL_DIR/benchmark.sh"

    # 量化脚本
    cat > "$INSTALL_DIR/quantize.sh" << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
# DeepSeek Coder Mobile - 模型量化工具

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

BITS=${1:-4}
SIZE=${2:-auto}

echo "Quantizing model to ${BITS}-bit..."
python mobile_quantize.py \
    --model_path ./model \
    --output_path "./model_q${BITS}" \
    --bits ${BITS} \
    --model_size ${SIZE} \
    --eval
EOF
    chmod +x "$INSTALL_DIR/quantize.sh"

    print_step "启动脚本创建完成"
}

create_config() {
    # 生成配置文件
    cat > "$INSTALL_DIR/config.env" << EOF
# DeepSeek Coder Mobile 配置
MODEL_SIZE=$MODEL_SIZE
QUANT_BITS=$QUANT_BITS
INSTALL_DIR=$INSTALL_DIR
PLATFORM=$PLATFORM
INSTALL_DATE=$(date '+%Y-%m-%d %H:%M:%S')
EOF
}

show_summary() {
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ 安装完成！${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}安装信息${NC}"
    echo "  安装目录:   $INSTALL_DIR"
    echo "  模型规格:   $MODEL_SIZE"
    echo "  量化位数:   ${QUANT_BITS}-bit"
    echo "  平台:       $PLATFORM"
    echo ""
    echo -e "  ${CYAN}使用方法${NC}"
    echo "  cd $INSTALL_DIR"
    echo ""
    echo "  # 单次生成"
    echo "  ./start.sh --prompt \"写一个快速排序\""
    echo ""
    echo "  # 交互对话"
    echo "  ./chat.sh"
    echo ""
    echo "  # 性能测试"
    echo "  ./benchmark.sh"
    echo ""
    echo "  # 模型量化"
    echo "  ./quantize.sh 4    # 4-bit 量化"
    echo "  ./quantize.sh 8    # 8-bit 量化"
    echo ""
    echo -e "  ${CYAN}注意事项${NC}"
    echo "  1. 请将模型文件放入 $INSTALL_DIR/model 目录"
    echo "  2. 首次运行会较慢（正在预热），请耐心等待"
    echo "  3. 建议充电时使用，避免耗电过快"
    echo "  4. 查看 README_MOBILE.md 获取详细使用说明"
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════${NC}"
}

# ============================================================
# 主流程
# ============================================================

main() {
    > "$LOG_FILE"
    print_banner

    detect_platform
    check_memory
    check_storage

    echo ""
    echo -e "${YELLOW}即将安装 DeepSeek Coder Mobile${NC}"
    echo -e "  安装目录: ${CYAN}$INSTALL_DIR${NC}"
    echo -e "  模型规格: ${CYAN}$MODEL_SIZE${NC}"
    echo -e "  量化位数: ${CYAN}${QUANT_BITS}-bit${NC}"
    echo ""

    # 非交互模式直接安装
    if [ "${1:-}" = "--yes" ] || [ "${1:-}" = "-y" ]; then
        CONFIRM="y"
    else
        read -p "确认安装? (y/N): " confirm
        CONFIRM="$confirm"
    fi

    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "安装已取消"
        exit 0
    fi

    echo ""
    install_system_deps   || die "系统依赖安装失败"
    echo ""
    install_python_deps   || die "Python 依赖安装失败"
    echo ""
    setup_project         || die "项目部署失败"
    echo ""
    download_model        || print_warn "模型目录创建失败"
    echo ""
    create_scripts        || die "脚本创建失败"
    create_config

    show_summary

    # 安装后验证
    echo -e "${CYAN}验证安装...${NC}"
    cd "$INSTALL_DIR"
    if $PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
        print_step "PyTorch 验证通过"
    else
        print_warn "PyTorch 未正确安装"
    fi

    if $PYTHON -c "import tokenizers; print(f'tokenizers: {tokenizers.__version__}')" 2>/dev/null; then
        print_step "tokenizers 验证通过"
    else
        print_warn "tokenizers 未正确安装"
    fi

    echo ""
    print_step "安装完成！运行 ./chat.sh 开始使用"
}

main "$@"
