# SM75 Docker route

Language: English | [简体中文](docker-sm75.zh-CN.md)

**Status: UNVALIDATED container route.** A successful source build or CPU check
is not GPU validation. Container cold-build, GPU, model/API, cache-reuse, and
GHCR pull results must be recorded before promoting this route. Existing native
host benchmarks do not validate this image. A local test on one 22 GiB RTX 2080
Ti would not establish dual-GPU, NVLink, capacity, or throughput claims.

## Scope and provenance

The repository already has the upstream-derived `docker/Dockerfile` and related
Docker tooling. The separate `docker/Dockerfile.sm75` adds the fork's Torch and
FlashQLA patch integration and an SM75-only build/publish workflow; the existing
Dockerfile did not itself establish a validated 2080 Ti image or a repository
GitHub Actions publishing pipeline. Native `build.sh`, `launcher.sh`, and shipped
profiles remain unchanged.

This route targets **Linux amd64, SM75 only**, using CUDA 13.0.3 devel, Ubuntu
24.04, and Python 3.12 in `/opt/venv`. It is distinct from the project's native
Ubuntu 26.04+/kernel 7+/GCC 15 target. The host needs a compatible NVIDIA driver,
Docker's Linux engine, and NVIDIA Container Toolkit/GPU passthrough; a container
does not replace the host driver. ARM/QEMU and other GPU architectures are not
promoted by this route.

The image builds a non-editable wheel from this fork, not an upstream prebuilt
vLLM kernel wheel. Torch/CUDA policy comes from `PROJECT_RELEASE.env` and build
requirements; the package version is the `pyproject.toml` fallback because `.git`
is excluded from the build context. `/opt/vllm-tools/` retains the release
metadata, `pyproject.toml`, and Torch E8M0 checker for inspection.

FlashQLA uses pinned
[`weicj/FlashQLA-SM70-SM75` source](https://github.com/weicj/FlashQLA-SM70-SM75/tree/3ab27d77d8ca01d7a4718903b726add1a8886c0e)
with this repository's SM75 patches, at `/opt/FlashQLA`, exposed on Python's
import path and through `FLASHQLA_ROOT` / `FLASHQLA_DIR`. It is source integration,
not a separately installed FlashQLA distribution with its older dependency pins.
The image retains CUDA devel tools, C++ compiler, Ninja, headers, and patched
source: **FlashQLA compilation is deferred to first GPU use**, not AOT-baked or
GPU-validated at image build time. FlashInfer may also JIT its selected kernels.
The smoke uses FlashInfer's CUDA-core decode path, not a mandatory FlashAttention
2 (FA2) backend, which is not generally supported on SM75.

Retain credit and licenses for [upstream vLLM](https://github.com/vllm-project/vllm)
(Apache-2.0), **vLLM 2080 Ti Definitive Edition**, author
[`github.com/weicj`](https://github.com/weicj),
[QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA), its
[SM70/SM75 adaptation](https://github.com/weicj/FlashQLA-SM70-SM75), and other
bundled third-party components when redistributing an image or derivative.

## Build or select an image

Run from the repository root with Docker Buildx:

```bash
docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.sm75 --target vllm-openai \
  --build-arg MAX_JOBS=2 --build-arg NVCC_THREADS=1 \
  --load -t vllm-sm75:local .
export IMAGE=vllm-sm75:local
```

This is a CUDA/native/Rust source build, not a lightweight image assembly. Cold
builds need substantial RAM, disk, and time; CI limits compiler concurrency and
has an approximately 350-minute timeout, not a completion guarantee. Record the
actual resource bottleneck if a hosted runner fails. GHA layer caching does not
automatically persist compiler caches inside `RUN --mount=type=cache`.

After a maintainer has actually published an image, the namespace is the
lowercase repository name, for example:

```bash
export IMAGE='ghcr.io/<owner>/vllm-2080ti-definitive:main'
docker pull "$IMAGE"
```

Replace `<owner>` with the publishing repository owner. This is a naming example,
not a claim that the parent repository has published a package. GHCR packages may
initially be private: authenticated access or an explicitly verified public
visibility setting is needed. Inspect the image's OCI source/revision labels and
prefer an image digest for repeatable deployment.

| Workflow event | Published tags after successful checks |
| --- | --- |
| Non-default branch push or PR, matching build paths | None; build and CPU smoke only |
| Default-branch push | `main` and a source-SHA tag; does not move `latest` |
| Published GitHub Release | Exact release tag, e.g. `v0.2.2-post3`; only a non-prerelease also moves `latest` |
| Manual dispatch | Build-only by default; explicit publishing creates unique test tags, not stable tags |

The GitHub Release prerelease flag determines stability; `post3` is not
reinterpreted by a generic semver rule. Pushing a Git tag alone does not publish a
release image. PRs receive no package-write credentials and do not automatically
run on a personal GPU runner. These workflows do not create releases or Git tags.

## CPU and GPU smoke (no model download)

The CPU tool intentionally does not initialize CUDA, load native vLLM kernels, or claim
native extension execution when `libcuda` is absent. It checks installed wheel
metadata, exact Torch/CUDA and vLLM versions, `import vllm`, the E8M0 patch,
FlashQLA source/patch symbols, paths, and JIT compiler availability. Dependency
consistency and the actual `vllm serve --help` entrypoint are separate checks:

```bash
docker run --rm --entrypoint python "$IMAGE" \
  /opt/vllm-tools/smoke_docker_sm75.py
docker run --rm --entrypoint uv "$IMAGE" pip check --python /opt/venv/bin/python
docker run --rm "$IMAGE" --help
```

Only an explicit `--gpu` runs real kernels. It fails on unavailable/non-SM75 GPUs
or failed checks; it does not turn failures into skips. It tests device 0 only:

```bash
docker run --rm --gpus all --ipc=host \
  -v vllm-sm75-jit:/root/.cache/torch_extensions \
  -v vllm-sm75-flashinfer:/root/.cache/flashinfer \
  -v vllm-sm75-vllm:/root/.cache/vllm \
  --entrypoint python "$IMAGE" /opt/vllm-tools/smoke_docker_sm75.py --gpu
```

This checks a vLLM core SiLU operation, FlashInfer CUDA-core decode against a
FP32 reference, and FlashQLA JIT `gdn_forward` / `gdn_forward_varlen`, including
native FP16 Q/K/V versus FP32 staging and packed sequences versus separate dense
calls. Gates and recurrent state remain FP32. It uses small tensors, not a
benchmark, model download, full attention-backend qualification, or TP test.

Repeat the same command in a fresh container with the same cache volumes after
the first successful JIT. Record first-run compilation and second-run cache
reuse separately; rerun success alone is not proof that no recompilation
occurred. `TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions` is the persistent
FlashQLA cache. Keep caches separate across incompatible image/toolchain updates;
do not mount an old host virtual environment over `/opt/venv`.

## Real model/API smoke

The entrypoint is `ENTRYPOINT ["vllm", "serve"]`: pass the model positionally (or
with `--model`) and native vLLM flags, **without another `vllm serve` prefix**.
It does not invoke `launcher.sh` or automatically apply profiles, GPU indices,
TP, context, MTP, reasoning, or chat-template settings. Official-style invocation
does not mean support for every upstream GPU/model/backend combination.

This small real checkpoint is **smoke-only, not a recommended deployment
profile**. The explicit 2048-token limit and other settings below bound the test;
they are not capacity or performance measurements. Model weights are downloaded
here, unlike the Python smoke tool. Set `HF_TOKEN` in the host environment if
needed; do not bake it into an image or write it into a committed command.

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

After startup completes, in another terminal check health, the served model ID,
and an actual nonempty completion (not just HTTP status):

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
curl --fail --silent --show-error http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

Binding Docker's published port to `127.0.0.1` keeps this unauthenticated smoke
local even though the server listens on all container interfaces. For actual
remote service, configure authentication and network access separately.

For existing local weights, replace the model and add a read-only mount:

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

Use `local-smoke` in the completion request for that example. For Qwen3.5-family
GDN routes that use the fork's FlashQLA backend, explicitly pass
`--gdn-prefill-backend flashqla_legacy` (not `--mamba-backend flashqla_legacy`)
alongside `--dtype half` and the remaining route-specific settings. The Qwen3-0.6B
API smoke does not exercise GDN. For real deployment, consult the
[hardware profile guides](../../profiles/README.md) and translate the intended
launcher's effective configuration (`launcher.sh --print-config`) to native
flags/environment variables; do not assume the image applies a profile for you.
Validate the exact model, precision, MTP, context, topology, and benchmark method
before reusing any native-host performance claims.
