#!/bin/bash

# 设置环境变量
# 使用临时目录存放IR dumps
export TRITON_SHARED_DUMP_PATH="$HOME/ir_dumps"
export TRITON_DUMP_DIR="$HOME/ir_dumps"

# 创建输出目录
# 在当前仓库根目录下创建test_dumps目录
OUTPUT_DIR="$(git rev-parse --show-toplevel)/test_dumps"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TRITON_SHARED_DUMP_PATH"

# 清理函数
cleanup() {
    echo "清理缓存..."
    rm -rf ~/.triton/cache/
    rm -rf .pytest_cache __pycache__
    rm -rf "$TRITON_SHARED_DUMP_PATH"/*
}

# 确保脚本退出时也清理
trap cleanup EXIT

# 进入测试目录
cd "$(dirname "$0")/../python/examples" || exit 1

# 对每个测试文件运行测试并收集结果
for test_file in test_*.py; do
    if [ -f "$test_file" ]; then
        echo "====================================="
        echo "测试文件: $test_file"
        echo "====================================="
        
        # 清理旧的IR dumps
        rm -rf "$TRITON_SHARED_DUMP_PATH"/*
        
        # 运行测试
        pytest "$test_file"
        
        # 检查是否有生成的IR文件
        if ls "$TRITON_SHARED_DUMP_PATH"/*.{ir,mlir} >/dev/null 2>&1; then
            # 创建以测试文件命名的压缩包（不含.py扩展名）
            test_name=$(basename "$test_file" .py)
            tar -czf "$OUTPUT_DIR/${test_name}_dumps.tar.gz" -C "$TRITON_SHARED_DUMP_PATH" .
            echo "IR dumps已保存到: $OUTPUT_DIR/${test_name}_dumps.tar.gz"
        else
            echo "警告: $test_file 没有生成IR dumps"
        fi
        
        # 清理缓存
        cleanup
        
        echo "完成: $test_file"
        echo
    fi
done

echo "所有测试完成！"
echo "压缩包保存在: $OUTPUT_DIR"