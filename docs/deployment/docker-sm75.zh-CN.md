# SM75 Docker 路线

语言：[English](docker-sm75.md) | 简体中文

**状态：UNVALIDATED（容器路线尚未验证）。** 源码构建或 CPU 检查通过不等于 GPU
验证通过。推广此路线之前，必须记录容器冷构建、GPU、模型/API、缓存复用以及 GHCR
拉取的实际结果。已有原生主机 benchmark 不能作为此镜像的验证。一张 22 GiB RTX
2080 Ti 的本地测试也不能证明双卡、NVLink、容量或吞吐表现。

## 范围与来源

仓库已有沿用上游的 `docker/Dockerfile` 及相关 Docker 工具。独立的
`docker/Dockerfile.sm75` 接入本 fork 的 Torch 和 FlashQLA 补丁，以及仅面向 SM75
的构建/发布工作流；原有 Dockerfile 本身不代表已有经过验证的 2080 Ti 镜像或仓库
GitHub Actions 发布流水线。原生 `build.sh`、`launcher.sh` 和已发布的 Profile
保持不变。

本路线仅面向 **Linux amd64 / SM75**，容器使用 CUDA 13.0.3 devel、Ubuntu 24.04
和 `/opt/venv` 中的 Python 3.12，与项目原生 Ubuntu 26.04+/kernel 7+/GCC 15
目标环境不同。宿主机仍需要兼容的 NVIDIA 驱动、Docker Linux 引擎及 NVIDIA
Container Toolkit/GPU 透传；容器不能替代宿主机驱动。此路线不推广 ARM/QEMU
或其他 GPU 架构。

镜像从本 fork 源码构建非 editable wheel，不使用上游预编译 vLLM 内核 wheel。
Torch/CUDA 策略取自 `PROJECT_RELEASE.env` 和构建依赖；因 `.git` 不进入构建
上下文，包版本使用 `pyproject.toml` 的 fallback。`/opt/vllm-tools/` 保留发布
元数据、`pyproject.toml` 及 Torch E8M0 检查器，供核验使用。

FlashQLA 使用固定的
[`weicj/FlashQLA-SM70-SM75` 源码](https://github.com/weicj/FlashQLA-SM70-SM75/tree/3ab27d77d8ca01d7a4718903b726add1a8886c0e)
并应用本仓库 SM75 补丁，放在 `/opt/FlashQLA`，通过 Python import 路径及
`FLASHQLA_ROOT` / `FLASHQLA_DIR` 暴露。这是源码集成，不是另行安装带有旧依赖
固定版本的 FlashQLA distribution。镜像保留 CUDA devel 工具、C++ 编译器、Ninja、
头文件和补丁后源码：**FlashQLA 编译延后到首次 GPU 使用时进行**，不能称为镜像内
已 AOT 编译或构建阶段已完成 GPU 验证。FlashInfer 所选内核也可能需要 JIT。
Smoke 使用 FlashInfer 的 CUDA-core decode 路径，不强制要求 SM75 通常不支持的
FlashAttention 2（FA2）后端。

再分发镜像或派生版本时，保留 [上游 vLLM](https://github.com/vllm-project/vllm)
（Apache-2.0）、**vLLM 2080 Ti Definitive Edition**、作者
[`github.com/weicj`](https://github.com/weicj)、
[QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA)、其
[SM70/SM75 适配版本](https://github.com/weicj/FlashQLA-SM70-SM75) 及其他内含第三方
组件的署名和许可证。

## 构建或选择镜像

在仓库根目录使用 Docker Buildx：

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.sm75 --target vllm-openai \
  --build-arg MAX_JOBS=2 --build-arg NVCC_THREADS=1 \
  --load -t vllm-sm75:local .
export IMAGE=vllm-sm75:local
```

这是 CUDA/原生扩展/Rust 源码构建，不是轻量镜像组装。冷构建需要较多内存、磁盘和
时间；CI 限制编译并行度并设置约 350 分钟超时，但不保证在此时间内完成。Hosted
runner 失败时应记录实际资源瓶颈。GHA layer cache 不会自动持久化
`RUN --mount=type=cache` 内部的编译器缓存。

维护者实际发布镜像之后，命名空间为小写仓库名称，例如：

```bash
export IMAGE='ghcr.io/<owner>/vllm-2080ti-definitive:main'
docker pull "$IMAGE"
```

将 `<owner>` 替换为发布仓库的所有者。这是命名示例，不代表父仓库已经发布镜像。
GHCR 新包可能默认私有：需要认证访问，或另行明确确认包已公开。核验镜像 OCI
source/revision 标签；可复现部署优先使用镜像 digest。

| 工作流事件 | 检查成功后发布的标签 |
| --- | --- |
| 非默认分支 push 或 PR，命中构建路径 | 不发布；仅构建及 CPU smoke |
| 默认分支 push | `main` 和源码 SHA 标签；不更新 `latest` |
| GitHub Release 已发布 | 原始 release 标签，如 `v0.2.2-post3`；仅非 prerelease 同时更新 `latest` |
| 手动触发 | 默认仅构建；明确要求发布时生成唯一测试标签，不覆盖稳定标签 |

稳定性由 GitHub Release 的 prerelease 标记决定，不用通用 semver 规则重新解释
`post3`。仅推送 Git tag 不会发布 release 镜像。PR 不获得包写入凭据，也不会自动
在个人 GPU runner 上运行。这些工作流不会创建 Release 或 Git tag。

## CPU 与 GPU smoke（不下载模型）

CPU 工具刻意不初始化 CUDA、不加载 vLLM 原生内核；没有 `libcuda` 时也不声称已
执行原生扩展。它检查已安装 wheel 元数据、精确的 Torch/CUDA 和 vLLM 版本、
`import vllm`、E8M0 补丁、FlashQLA 源码/补丁符号、路径和 JIT 编译工具可用性。
依赖一致性和真实 `vllm serve --help` 入口另外检查：

```bash
docker run --rm --entrypoint python "$IMAGE" \
  /opt/vllm-tools/smoke_docker_sm75.py
docker run --rm --entrypoint uv "$IMAGE" pip check --python /opt/venv/bin/python
docker run --rm "$IMAGE" --help
```

只有显式传入 `--gpu` 才执行真实内核。GPU 不可用、不是 SM75 或检查失败时直接
报错，不会把失败转成 skip。工具仅测试 device 0：

```bash
docker run --rm --gpus all --ipc=host \
  -v vllm-sm75-jit:/root/.cache/torch_extensions \
  -v vllm-sm75-flashinfer:/root/.cache/flashinfer \
  -v vllm-sm75-vllm:/root/.cache/vllm \
  --entrypoint python "$IMAGE" /opt/vllm-tools/smoke_docker_sm75.py --gpu
```

它检查 vLLM 核心 SiLU 操作、FlashInfer CUDA-core decode 与 FP32 参考结果，以及
FlashQLA JIT `gdn_forward` / `gdn_forward_varlen`，包含原生 FP16 Q/K/V 与 FP32
staging 的对照、packed 序列与独立 dense 调用的对照。Gate 和 recurrent state
保持 FP32。这里只使用小张量，不是 benchmark、模型下载、全部 attention 后端
资格验证或 TP 测试。

首次 JIT 成功后，以相同缓存卷在新容器中再次运行该命令。分别记录首次编译和第二次
缓存复用；仅再次运行成功不能证明没有发生重新编译。
`TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions` 是 FlashQLA 持久化缓存。
不兼容的镜像/工具链更新应使用不同缓存；不要把旧主机虚拟环境挂载覆盖 `/opt/venv`。

## 真实模型/API smoke

入口是 `ENTRYPOINT ["vllm", "serve"]`：直接传入模型位置参数（或 `--model`）与
原生 vLLM 参数，**不要再添加 `vllm serve` 前缀**。它不调用 `launcher.sh`，也不会
自动应用 Profile、GPU 编号、TP、上下文、MTP、reasoning 或 chat-template 设置。
与官方一致的启动方式不代表支持上游所有 GPU/模型/后端组合。

以下真实小模型**仅用于 smoke，不是推荐部署 Profile**。显式的 2048-token 上限
和其他设置只是限制测试规模，不是容量或性能测量。此步骤会下载模型权重，Python
smoke 工具则不会。必要时在宿主机环境设置 `HF_TOKEN`，不要把 token 烘焙到镜像或
写入提交的命令。

```bash
docker run --rm --name vllm-sm75-smoke --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-sm75-jit:/root/.cache/torch_extensions \
  -v vllm-sm75-flashinfer:/root/.cache/flashinfer \
  -v vllm-sm75-vllm:/root/.cache/vllm \
  -e HF_TOKEN "$IMAGE" Qwen/Qwen3-0.6B \
  --dtype half --tensor-parallel-size 1 --max-model-len 2048 \
  --gpu-memory-utilization 0.8 --enforce-eager \
  --host 0.0.0.0 --port 8000
```

启动完成后，在另一个终端检查 health、实际 served model ID 和非空真实生成结果
（不能只检查 HTTP 状态码）：

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
curl --fail --silent --show-error http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Docker 发布端口绑定 `127.0.0.1`，因此即使服务监听容器全部接口，这个无认证 smoke
仍只供本机访问。实际远程服务需另外配置认证和网络访问控制。

已有本地权重时，替换模型参数并添加只读挂载：

```bash
docker run --rm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /absolute/path/to/Qwen3-0.6B:/models/smoke:ro \
  -v vllm-sm75-jit:/root/.cache/torch_extensions \
  -v vllm-sm75-flashinfer:/root/.cache/flashinfer \
  -v vllm-sm75-vllm:/root/.cache/vllm \
  "$IMAGE" /models/smoke --served-model-name local-smoke \
  --dtype half --tensor-parallel-size 1 --max-model-len 2048 \
  --gpu-memory-utilization 0.8 --enforce-eager --host 0.0.0.0 --port 8000
```

此例生成请求的模型名使用 `local-smoke`。Qwen3.5 系列使用本 fork FlashQLA 后端的
GDN 路线，需要显式传入 `--gdn-prefill-backend flashqla_legacy`（不是
`--mamba-backend flashqla_legacy`），以及 `--dtype half` 和其余路线专用设置。
Qwen3-0.6B API smoke 不会执行 GDN。实际部署时，查阅
[硬件 Profile 指南](../../profiles/README.zh-CN.md)，将目标 launcher 有效配置
（`launcher.sh --print-config`）转换为原生参数/环境变量，不要假设镜像自动应用
Profile。复用原生主机性能结论前，必须验证确切的模型、精度、MTP、上下文、拓扑及
benchmark 口径。
