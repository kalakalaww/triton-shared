# triton-shared（中文说明）

这是 Triton 编译器的一个共享中间层（middle-layer）。

当前中间层还未完全实现，但已有足够功能用于演示其工作流程。总体思路是将 Triton IR 降低（lower）到一个基于 MLIR 的核心方言，以便在不同 Triton 目标间共享，并让后端能够与其它语言/编译器共享。

大致架构：

[Triton IR] -> [中间层（Middle Layer）] -> [硬件专用 IR]

中间层使用 MLIR 的 Linalg 和 Tensor 方言来处理 Triton 的 block 值，指针相关操作使用 Memref 方言。

## 动机

可以参考 2023 年 Triton 开发者大会上的这次演讲（了解背景与目标）：
https://www.youtube.com/watch?v=y2V3ucS1pfQ

## 使用方法

该仓库已把 `triton` 作为子模块包含进来，并作为一个 out-of-tree backend 构建。

注意：请把仓库克隆到名为 `triton_shared` 的目录（注意下划线），Triton 会根据这个目录名在 `triton.runtime` 下创建一个模块，用于引用该参考 CPU 后端。

你需要设置 `TRITON_PLUGIN_DIRS` 环境变量，指向你本地的 `triton_shared` 目录，以便 `triton` 能找到该插件。

示例：

```sh
export TRITON_PLUGIN_DIRS=$(pwd)/triton_shared

git clone --recurse-submodules https://github.com/microsoft/triton-shared.git triton_shared
cd triton_shared/triton
```

在使用 Clang 构建前，先安装依赖：

```sh
python3 -m pip install --upgrade pip
python3 -m pip install cmake==3.24 ninja pytest-xdist pybind11 setuptools
sudo apt-get update -y
sudo apt-get install -y ccache clang lld
TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install --no-build-isolation -vvv '.[tests]'
```

使用虚拟环境构建的示例：

```sh
python3 -m venv .venv --prompt triton
source .venv/bin/activate

pip3 install ninja cmake wheel pytest pybind11 setuptools
pip3 install -e . --no-build-isolation
```

构建完成后，生成的 `triton-shared` 二进制会放在 `triton/build/{current_cmake_version}/third_party/triton_shared` 下。

### 1. 独立使用（Stand-Alone）
中间层可以作为独立工具使用，用来把 Triton 方言转成中间层方言，适合测试与验证，也可在将 IR 传给其他 MLIR 编译器前先做转换。

独立命令示例：
```
triton-shared-opt --triton-to-linalg %file
```

### 2. 作为后端组件（Backend Component）
中间层的主要用途是在 Triton 后端中作为组件被复用。可以把它生成的 CMake 目标和头文件加入到后端工程中。示例后端将在日后发布。

### 3. 参考 CPU 后端（Reference CPU backend）
仓库包含了一个实验性的参考 CPU 后端，利用现有的 MLIR passes。构建完成后，可通过设置 Triton 的 driver 来使用该 CPU 后端：

```python
import triton
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
```

更多示例请参考 `python/examples` 目录。

## 实现细节

尽管一个合法的 Triton 程序可以在任意内存位置读写，当前原型仅支持具有结构化内存访问模式的程序降低（lowering）。

### 分析（Analyses）

转换过程中有三项重要分析：

1. 指针分析（Pointer analysis）：
   - 负责在 load/store 阶段从 Triton 程序中提取结构化的内存访问模式；它遍历 IR，并访问相关指令来构建 Memref 方言的 strided memory accesses。该分析仍在早期阶段，尚未支持所有情况。

2. 使用分析（Use analysis）：
   - 在“指针分析”之后，用于地址计算的指令会被转换为 Memref 操作表达的内存访问，从而原本用于地址计算的指令可能不再需要。为安全删除这些指令，需要进行使用分析，将仅用于地址计算的标记为 `MetaUse`，同时既用于地址计算又用于数据计算的标记为 `MixedUse`。对 `MixedUse` 的操作会被克隆并调整用户（users），目的是把 `MetaUse` 的部分隔离出来以便安全删除。

3. Mask 分析（Mask analysis）：
   - 负责处理带掩码的 load/store。

### 转换策略

我们引入了 `TritonToLinalg` 这个 pass，将 `triton` 方言转换为基于张量（tensor）的 `linalg` 方言。这样转换后的 IR 可以直接与 `linalg` 的 tiling、fusion 等变换 pass 兼容。如在“指针分析”中所述，我们仍需在 load/store 的边界处处理 memref 指令，并使用 `bufferization.to_tensor` 将其转换回张量。下面给出一个简单的转换前后 IR 示例（保留原始代码示例）：

```mlir
tt.func @kernel(%afloat : !tt.ptr<bf16>, %res : !tt.ptr<bf16>) {
   %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
   %1 = tt.splat %afloat : (!tt.ptr<bf16>) -> tensor<128x!tt.ptr<bf16>>
   %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
   %afm = tt.load %2 : tensor<128x!tt.ptr<bf16>>
   %3 = "tt.reduce"(%afm) ({
   ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
   }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
   tt.store %res, %3 : !tt.ptr<bf16>
   tt.return
}
```

转换后示例：

```mlir
func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32) {
      %cst = arith.constant 0.000000e+00 : f32
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1] :
            memref<*xbf16> to memref<128xbf16, strided<[1]>>
      %alloc = memref.alloc() : memref<128xbf16>
      memref.copy %reinterpret_cast, %alloc : memref<128xbf16, strided<[1]>> to memref<128xbf16>
      %0 = bufferization.to_tensor %alloc restrict writable : memref<128xbf16>
      %1 = bufferization.alloc_tensor() : tensor<f32>
      %inserted = tensor.insert %cst into %1[] : tensor<f32>
      %reduced = linalg.reduce ins(%0 : tensor<128xbf16>) outs(%inserted : tensor<f32>) dimensions = [0]
         (%in: bf16, %init: f32) {
            %3 = arith.extf %in : bf16 to f32
            %4 = arith.addf %3, %init : f32
            linalg.yield %4 : f32
         }
      %extracted = tensor.extract %reduced[] : tensor<f32>
      %2 = arith.truncf %extracted : f32 to bf16
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1], strides: [1] :
            memref<*xbf16> to memref<1xbf16, strided<[1]>>
      affine.store %2, %reinterpret_cast_0[0] : memref<1xbf16, strided<[1]>>
      return

}
```

需要注意的要点：

- `tt.load`（以及与其相关的地址计算指令，例如 `tt.addptr` 和 `tt.splat`）会被降低为 `memref.reinterpret_cast`、`memref.alloc`、`memref.copy` 的组合。在初始化本地缓冲区后，我们使用 `bufferization.to_tensor` 将 memref 转回张量；该操作在后续的 bufferization 阶段会被自动移除。

- `tt.store` 会被降低为 `memref.reinterpret_cast` 与 `affine.store` 或 `memref.tensor_store` 的组合：

```
%reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [...] memref<*xf32> to memref<1024xf32>
%extracted_slice = tensor.extract_slice %15[0] [%21] [1] : tensor<1024xf32> to tensor<?xf32>
%subview = memref.subview %reinterpret_cast[0] [%21] [1] : memref<1024xf32> to memref<?xf32>
bufferization.materialize_in_destination %extracted_slice in writable %subview
```

- 元素级的 `arith` 和 `math` 操作会被转换为对应的 `linalg.generic` 实现。
- `tt.dot` 会被转换为 `linalg.matmul`。
- `tt.reduce` 会被转换为 `linalg.reduce`；当前已知限制：只支持在 reduction body 中的 `addf` 和 `maxf`。

### 测试

该原型在以下 Triton kernel 示例上进行了测试：

1. [向量加法（vector addition）](./python/examples/test_vec_add.py)
2. [融合的 softmax（fused softmax）](./python/examples/test_softmax.py)
3. [矩阵乘法（matrix multiplication）](./python/examples/test_matmul.py)
4. 层归一化（layer normalization）
5. 融合注意力（fused attention）

Python 测试使用 Pytest 运行，运行测试前需要设置以下环境变量：

```sh
export LLVM_BINARY_DIR=<path-to-your-llvm-binaries>
export TRITON_SHARED_OPT_PATH=$TRITON_PLUGIN_DIRS/triton/build/<your-cmake-directory>/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt

pytest <path-to-triton-shared>/python/examples
```

除了教程示例的测试外，仓库中还有大量 lit 测试覆盖不同场景。

## 中间表示（IR）转储

为了便于调试和分析，triton-shared 项目现在支持在编译过程中导出所有中间表示（IR）。该功能由环境变量 `TRITON_SHARED_DUMP_PATH` 控制。

### 工作原理

设置 `TRITON_SHARED_DUMP_PATH` 环境变量，指定一个目录，编译器会在该目录中保存各个阶段生成的 IR 转储文件，便于开发者检查和分析编译器对代码所做的变换。

### 使用方法

1. 创建一个用于保存 IR 转储的目录（例如 `/path/to/dump_dir`）。
2. 设置环境变量：

```sh
export TRITON_SHARED_DUMP_PATH=/path/to/dump_dir
```

3. 按平常方式运行 Triton 编译，编译器会把 IR 文件写入指定目录。

### 示例

假设转储目录为 `/tmp/ir_dumps`，执行前设置：

```sh
export TRITON_SHARED_DUMP_PATH=/tmp/ir_dumps
```

编译完成后检查目录：

```sh
$ ls /tmp/ir_dumps
ll.ir  ll.mlir  tt.mlir  ttshared.mlir
```

## 调试 Triton 程序

triton-shared 提供了一个构建选项来启用 LLVM 的 Sanitizer（例如 AddressSanitizer（ASan）和 ThreadSanitizer（TSan）），帮助检测 Triton 程序中的内存安全和并发问题。这些 sanitizers 会在运行时动态分析程序，定位缓冲区溢出、数据竞争等错误。详细的设置与使用说明请参见：`scripts/SANITIZER.md`。

## 贡献（Contributing）

欢迎贡献！大多数贡献需要你同意一份贡献者许可协议（Contributor License Agreement, CLA），声明你有权并确实授权我们使用你的贡献。详情请访问：https://cla.opensource.microsoft.com。

当你提交 Pull Request 时，仓库会运行 CLA bot 自动判断你是否需要签署 CLA 并在 PR 中给出相应提示（例如状态检查或评论）。请按 bot 的提示操作；通常你只需完成一次 CLA 签署即可在所有采用该 CLA 的仓库中生效。

本项目采用 [Microsoft 开源行为准则（Code of Conduct）](https://opensource.microsoft.com/codeofconduct/)。更多信息请参见 Code of Conduct FAQ 或者联系 `opencode@microsoft.com`。

## 商标（Trademarks）

本项目可能包含项目、产品或服务的商标或徽标。对 Microsoft 商标或徽标的授权使用需遵循 [Microsoft 的商标与品牌使用指南](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general)。修改版本中对 Microsoft 商标或徽标的使用不得造成混淆或暗示由 Microsoft 提供支持。第三方商标或徽标的使用需遵循其各自的使用政策。

