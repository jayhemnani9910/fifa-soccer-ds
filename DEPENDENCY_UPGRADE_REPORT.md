# Dependency and platform modernization report

Research date: **2026-07-18**
Access date for every linked source: **2026-07-18**
Companion repository audit: [AUDIT_REPORT.md](AUDIT_REPORT.md)

## Decision method and security caveat

Versions in this report were not inferred from model memory. Current release/version claims were checked
against official PyPI project pages, official Python/PyTorch/PyG/ONNX Runtime/NVIDIA compatibility and
release documentation, official Docker Hub metadata, and official GitHub release pages.

The matrices distinguish:

- **Latest stable** — newest non-pre-release upstream release observed on the research date.
- **Latest verified compatible** — newest release compatible with the repository's selected Python/API/
  CUDA group that was actually exercised locally, or explicitly marked metadata-only when local
  installation was blocked.
- **Recommended target** — the version this repository should declare now, considering coordinated
  ecosystems, migration size, hardware, and test evidence.

Security code **A** means the selected Python package/version was covered by at least one of two
`pip-audit==2.10.1` scans on 2026-07-18 with no known vulnerability reported: the exact installed clean
environment and the fully pinned public-version application closure. The installed scan skipped the
editable first-party package and custom `torch==2.13.0+cpu`/`torchvision==0.28.0+cpu` labels; the
public-version scan covered their upstream `2.13.0`/`0.28.0` identities. **N/A** means pip-audit is not
the applicable scanner (for example,
CPython, Docker, CUDA, FFmpeg, or GitHub Actions). A clean scan is time-bounded, not a promise that no future
advisory exists, and it does not cover OS packages, drivers, CUDA/cuDNN native libraries, model artifacts,
or container layers. PyPI upload dates below are the release dates shown by the official index.

## Source of truth and installation model

- **Verified fact:** `pyproject.toml` is now the only dependency source of truth.
- **Verified fact:** runtime, optional, and development dependencies use exact versions.
- **Verified fact:** `requirements.txt` now contains only `-e ".[dev]"`; it no longer duplicates pins.
- **Verified fact:** optional integrations are grouped as `audio`, `data`, `export`, and `mlops`; heavy
  dependencies are no longer forced into the core API/runtime install. The unused `experiments` group was
  removed.
- **Verified fact:** no new package manager was introduced. setuptools/pip remain appropriate for the
  existing project and Docker/CI workflows.
- **Verified fact:** the long-lived `.venv` is Python 3.12.3 and contains historical/orphaned packages, so
  it is not dependency-authoritative. Its final `pip check` fails because PyDrive2 1.21.3 and pyOpenSSL
  24.2.1 require Cryptography `<44`, conflicting with the secure 48.0.1 target. A fresh disposable Python
  3.12.3 environment installed every supported extra and is the installed-version source below; its full
  suite and `pip check` pass.

## Platform and packaging toolchain matrix

| Component | Purpose | Declared in / declared version | Installed | Latest stable (release date) | Latest verified compatible | Recommended target | Compatibility / breaking changes | Security | Official sources | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CPython | Runtime | `pyproject >=3.12,<3.13`; `.python-version 3.12.13`; CI `3.12` | 3.12.3 | 3.14.6 (2026-06-10) | 3.12.3 locally; 3.12.13 is latest selected-line source release | 3.12.13 | 3.12 is security-only through 2028-10; upstream stopped binary installers after 3.12.10, so source/OS builds may be needed | N/A | [3.12.13](https://www.python.org/downloads/release/python-31213/), [3.14.6](https://www.python.org/downloads/release/python-3146/), [status](https://devguide.python.org/versions/) | Upgrade host now; keep project on 3.12 until compiled/GPU matrix is requalified |
| pip | Installer/resolver | Docker/CI 26.1.2 | 26.1.2 | 26.1.2 (2026-05-31) | 26.1.2 | 26.1.2 | Tool-only patch/minor update; exact target exercised | A | [PyPI](https://pypi.org/project/pip/) | Upgraded and verified |
| setuptools | Build backend | build system/Docker 83.0.0 | 83.0.0 | 83.0.0 (2026-07-04) | 83.0.0 | 83.0.0 | Standard isolated build passed with modern SPDX license metadata | A | [PyPI](https://pypi.org/project/setuptools/) | Upgraded and verified |
| wheel | Wheel tooling | build system/Docker 0.47.0 | 0.47.0 | 0.47.0 (2026-04-22) | 0.47.0 | 0.47.0 | Compatible tool update; standard build passed | A | [PyPI](https://pypi.org/project/wheel/) | Upgraded and verified |
| CUDA toolkit/container line | GPU runtime | CUDA 12.6.3 cuDNN Ubuntu 24.04 image pinned by digest | No driver/GPU; clean tests use CPU wheels | CUDA 13.3 (2026) | CUDA 12.6 declaration/BuildKit resolution only | CUDA 12.6.3 | Torch 2.13 and Vision 0.28 publish cu126 wheels; CUDA 12.x minor compatibility requires driver >=525 and full 12.6 U3 requires >=560.35.05 | N/A | [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/), [minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [official cu126 index](https://download.pytorch.org/whl/cu126), [NVIDIA images](https://hub.docker.com/r/nvidia/cuda/) | Retain cu126 as deployable target; GPU execution still required |
| Docker base | Runtime OS/GPU image | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:8aef...4356` | BuildKit daemon available for checks | NVIDIA publishes newer CUDA 13.3 images | Exact digest resolved by BuildKit | Exact current digest | New CUDA major requires Torch/ORT/driver requalification; apt packages inside remain floating | N/A | [NVIDIA CUDA images](https://hub.docker.com/r/nvidia/cuda/) | Retain digest; Dockerfile check passes; rebuild/scan routinely |

## Direct runtime dependency matrix

All runtime declarations are in `pyproject.toml [project].dependencies`; the installed project metadata
was refreshed editable with dependencies disabled before the final `pip check`.

| Component | Purpose | Declared / installed | Latest stable (date) | Latest verified compatible | Recommended target | Python/platform/CUDA and breaking-change notes | Security | Official source | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aiohttp | Async health/network client | 3.14.1 / 3.14.1 | 3.14.1 (2026-06-07) | 3.14.1 | 3.14.1 | Python >=3.10; async API exercised | A | [PyPI](https://pypi.org/project/aiohttp/) | Already current |
| FastAPI | Batch/live HTTP API | 0.139.2 / 0.139.2 | 0.139.2 (2026-07-16) | 0.139.2 | 0.139.2 | Python >=3.10; Pydantic 2 path exercised | A | [PyPI](https://pypi.org/project/fastapi/), [release notes](https://fastapi.tiangolo.com/release-notes/) | Upgraded and verified |
| NumPy | Arrays/numerics | 2.4.4 / 2.4.4 | 2.5.1 (2026-07-04) | 2.4.4 | 2.4.4 | Latest requires Python >=3.12; 2.5 must be tested with SciPy/sklearn/OpenCV/Torch as a compiled batch | A | [PyPI](https://pypi.org/project/numpy/), [2.0 migration](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html), [release notes](https://numpy.org/doc/stable/release.html) | Defer one tested numerical-stack batch |
| OpenCV Python | Image/video I/O | 4.13.0.92 / 4.13.0.92 | 5.0.0.93 (2026-07-02) | 4.13.0.92 | 4.13.0.92 | 5.0 is a major API/ABI transition and needs codec/GUI/container regression | A | [PyPI](https://pypi.org/project/opencv-python/) | Defer major; upgraded within 4.x |
| Pillow | Image metadata/I/O | 12.3.0 / 12.3.0 | 12.3.0 (2026-07-01) | 12.3.0 | 12.3.0 | Python >=3.10 | A | [PyPI](https://pypi.org/project/Pillow/) | Already current |
| prometheus-client | Metrics | 0.25.0 / 0.25.0 | 0.25.0 (2026-04-09) | 0.25.0 | 0.25.0 | Python >=3.9; API endpoint exercised | A | [PyPI](https://pypi.org/project/prometheus-client/) | Already current |
| psutil | Process/system health | 7.2.2 / 7.2.2 | 7.2.2 (2026-01-28) | 7.2.2 | 7.2.2 | Linux wheels available | A | [PyPI](https://pypi.org/project/psutil/) | Already current |
| Pydantic | API/config schemas | 2.13.4 / 2.13.4 | 2.13.4 (2026-05-06) | 2.13.4 | 2.13.4 | Python >=3.9; schema migrations tested | A | [PyPI](https://pypi.org/project/pydantic/), [migration guide](https://docs.pydantic.dev/latest/migration/) | Upgraded and verified |
| PyYAML | YAML config | 6.0.3 / 6.0.3 | 6.0.3 (2025-09-25) | 6.0.3 | 6.0.3 | Safe loaders retained | A | [PyPI](https://pypi.org/project/PyYAML/) | Already current |
| Requests | HTTP compatibility/client use | 2.34.2 / 2.34.2 | 2.34.2 (2026-05-14) | 2.34.2 | 2.34.2 | Python >=3.10; now publishes inline annotations | A | [PyPI](https://pypi.org/project/requests/) | Upgraded; redundant `types-requests` removed |
| scikit-learn | Classifier/metrics utilities | 1.8.0 / 1.8.0 | 1.9.0 (2026-06-02) | 1.8.0 | 1.8.0 | 1.9 needs coordinated NumPy/SciPy compiled-wheel and behavior tests | A | [PyPI](https://pypi.org/project/scikit-learn/), [what's new](https://scikit-learn.org/stable/whats_new.html) | Defer numerical-stack batch |
| SciPy | Scientific numerics | 1.17.1 / 1.17.1 | 1.18.0 (2026-06-19) | 1.17.1 | 1.17.1 | 1.18 requires Python >=3.12; coordinate with NumPy/sklearn | A | [PyPI](https://pypi.org/project/scipy/), [release notes](https://docs.scipy.org/doc/scipy/release.html) | Defer numerical-stack batch |
| SlowAPI | FastAPI rate limiting | 0.1.10 / 0.1.10 | 0.1.10 (2026-06-13) | 0.1.10 | 0.1.10 | Uses in-process memory backend here; not distributed quota | A | [PyPI](https://pypi.org/project/slowapi/) | Upgraded and verified |
| PyTorch | Training/inference | 2.13.0 / 2.13.0+cpu | 2.13.0 (2026-07-08) | 2.13.0+cpu locally; official cu126 wheel exists | 2.13.0 | Paired with Vision 0.28, PyG 2.8, CUDA 12.6 and cuDNN 9; replaces advisory-affected 2.8 | A | [PyPI](https://pypi.org/project/torch/), [2.13 release](https://pytorch.org/blog/pytorch-2-13-release-blog/), [CPU index](https://download.pytorch.org/whl/cpu), [cu126 index](https://download.pytorch.org/whl/cu126), [advisories](https://osv.dev/list?q=package%3APyPI%2Ftorch) | Upgrade performed as security/compatibility batch |
| PyTorch Geometric | Graph construction/GNN | 2.8.0 / 2.8.0 | 2.8.0 (2026-06-05) | 2.8.0 with Torch 2.13 CPU | 2.8.0 | Base package is exercised; optional compiled extensions need their own official wheel matrix on GPU | A | [PyPI](https://pypi.org/project/torch-geometric/), [installation](https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html) | Upgraded and verified; third-party JIT deprecation remains |
| TorchVision | Vision operators/model utilities | 0.28.0 / 0.28.0+cpu | 0.28.0 (2026-07-08) | 0.28.0+cpu locally; official cu126 wheel exists | 0.28.0 | Official release line paired with Torch 2.13; Python 3.12 tested | A | [PyPI](https://pypi.org/project/torchvision/), [0.28 docs](https://docs.pytorch.org/vision/0.28/), [CPU index](https://download.pytorch.org/whl/cpu), [cu126 index](https://download.pytorch.org/whl/cu126) | Upgraded and verified |
| Ultralytics | YOLO inference/training/export | 8.4.100 / 8.4.100 | 8.4.100 (2026-07-18) | 8.4.100 | 8.4.100 | Current exporter API adopted; restricted checkpoint load and real ONNX parity exercised | A | [PyPI](https://pypi.org/project/ultralytics/), [docs](https://docs.ultralytics.com/) | Upgraded and verified |
| Uvicorn standard | ASGI server | 0.51.0 / 0.51.0 | 0.51.0 (2026-07-08) | 0.51.0 | 0.51.0 | Python >=3.10; standard extras installed | A | [PyPI](https://pypi.org/project/uvicorn/) | Upgraded and verified |
| yt-dlp | YouTube metadata/download | 2026.7.4 / 2026.7.4 | 2026.7.4 (2026-07-04) | 2026.7.4 | 2026.7.4 | Python >=3.10; network behavior changes frequently, so exact pin is intentional | A | [PyPI](https://pypi.org/project/yt-dlp/), [releases](https://github.com/yt-dlp/yt-dlp/releases) | Upgraded and verified with mocks/local CLI health only |

## Optional dependency matrix

| Component | Purpose / declared in | Declared / installed | Latest stable (date) | Latest verified compatible | Recommended target | Compatibility / breaking changes | Security | Official source | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ffmpeg-python | Audio wrapper / `audio` | 0.2.0 / 0.2.0 | 0.2.0 (2019-07-06) | 0.2.0 | 0.2.0 | Old but no stable successor in this package; system FFmpeg does real work | A | [PyPI](https://pypi.org/project/ffmpeg-python/) | Retain optional; consider direct subprocess wrapper replacement later |
| librosa | Audio features / `audio` | 0.11.0 / 0.11.0 | 0.11.0 (2025-03-11) | 0.11.0 | 0.11.0 | Compatible with the selected NumPy stack in clean imports/tests | A | [PyPI](https://pypi.org/project/librosa/) | Already current |
| openai-whisper | Optional transcription / `audio` | 20250625 / 20250625 | 20250625 (2025-06-26) | 20250625 | 20250625 | Expensive models were not downloaded by setup/audit | A | [PyPI](https://pypi.org/project/openai-whisper/) | Retain optional; no model download |
| PyAV | Video decode / `data` | 17.0.1 / 17.0.1 | 18.0.0 (2026-07-02) | 17.0.1 | 17.0.1 | 18 is a major codec/API transition; requires real codec fixture matrix | A | [PyPI](https://pypi.org/project/av/) | Defer major |
| Cryptography | Secure dependency floor for Kaggle/MLflow closures / `data`, `mlops` | 48.0.1 / 48.0.1 | 49.0.0 (2026-06-12) | 48.0.1 (2026-06-09) with MLflow 3.14 (`<49`) | 48.0.1 | 48.0.1 wheels use OpenSSL 4.0.1; fixes the vulnerable older wheel line while respecting MLflow's upper bound | A | [PyPI](https://pypi.org/project/cryptography/), [changelog](https://cryptography.io/en/stable/changelog/) | Add exact security-compatible target in both relevant extras |
| Kaggle | Dataset API / `data` | 2.2.3 / 2.2.3 | 2.2.3 (2026-06-25) | 2.2.3 | 2.2.3 | Python >=3.11; credential/network E2E unavailable | A | [PyPI](https://pypi.org/project/kaggle/) | Upgraded and clean-installed; service run blocked |
| ONNX | Exchange format/export / `export` | 1.22.0 / 1.22.0 | 1.22.0 (2026-06-15) | 1.22.0 | 1.22.0 | Replaces advisory-affected 1.21; checker and real exported model exercised | A | [PyPI](https://pypi.org/project/onnx/), [releases](https://github.com/onnx/onnx/releases), [advisory](https://osv.dev/vulnerability/GHSA-537c-gmf6-5ccf) | Upgraded and verified |
| ONNX Runtime GPU | CUDA/CPU execution providers / `export` | 1.26.0 / 1.26.0 | 1.27.0 (2026-06-15) | 1.26.0 for CUDA 12; CPU provider exercised | 1.26.0 | Official 1.26 release says CUDA 12 support is removed in 1.27; therefore 1.26 is latest compatible with selected cu126 | A | [PyPI](https://pypi.org/project/onnxruntime-gpu/), [1.26 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0), [CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) | Upgrade to latest CUDA-12-compatible release; GPU run still required |
| MLflow | Experiment tracking / `mlops` | 3.14.0 / 3.14.0 | 3.14.0 (2026-06-17) | 3.14.0 locally | 3.14.0 | Local run lifecycle/API tests pass; remote backend migration/outage behavior remains deployment-owned | A | [PyPI](https://pypi.org/project/mlflow/), [3.14 release](https://mlflow.org/releases/), [MLflow 3 migration](https://mlflow.org/docs/latest/ml/getting-started/migration/) | Upgraded and verified locally |

## Development dependency matrix

All declarations are in `pyproject.toml [project.optional-dependencies].dev`.

| Component | Purpose | Declared / installed | Latest stable (date) | Latest verified compatible | Recommended target | Breaking-change notes | Security | Official source | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AnyIO | Async tests/runtime support | 4.14.2 / 4.14.2 | 4.14.2 (2026-07-12) | 4.14.2 | 4.14.2 | Current pytest/FastAPI stack passes | A | [PyPI](https://pypi.org/project/anyio/) | Already current |
| Bandit | Python security lint | 1.9.4 / 1.9.4 | 1.9.4 (2026-02-25) | 1.9.4 | 1.9.4 | Exact line-scoped reviewed exclusions only | A | [PyPI](https://pypi.org/project/bandit/) | Upgraded; clean result |
| build | PEP 517 frontend | 1.5.0 / 1.5.0 | 1.5.0 (2026-04-30) | 1.5.0 | 1.5.0 | Standard isolated sdist/wheel build passed | A | [PyPI](https://pypi.org/project/build/) | Already current |
| coverage | Coverage engine | 7.15.2 / 7.15.2 | 7.15.2 (2026-07-15) | 7.15.2 | 7.15.2 | Branch coverage enabled | A | [PyPI](https://pypi.org/project/coverage/) | Already current |
| HTTPX | ASGI/API tests | 0.28.1 / 0.28.1 | 0.28.1 (2024-12-06) | 0.28.1 | 0.28.1 | FastAPI metadata constrains optional standard HTTPX <1 | A | [PyPI](https://pypi.org/project/httpx/) | Already current |
| mypy | Type checking | 1.20.2 / 1.20.2 | 2.3.0 (2026-07-13) | 1.20.2 | 1.20.2 | 2.x is a major migration; direct and aggregate checks are green on tested 1.x | A | [PyPI](https://pypi.org/project/mypy/), [changelog](https://mypy-lang.org/news.html) | Defer major as isolated quality-tool batch |
| pre-commit | Hook orchestration | 4.6.0 / 4.6.0 | 4.6.0 (2026-04-21) | 4.6.0 | 4.6.0 | Remote SHA-pinned hooks initialize; mypy intentionally uses the installed project environment | A | [PyPI](https://pypi.org/project/pre-commit/) | Already current; aggregate gate passes |
| pytest | Test runner | 9.1.1 / 9.1.1 | 9.1.1 (2026-06-19) | 9.1.1 | 9.1.1 | 251 tests pass | A | [PyPI](https://pypi.org/project/pytest/) | Upgraded and verified |
| pytest-cov | Coverage plugin | 7.1.0 / 7.1.0 | 7.1.0 (2026-03-21) | 7.1.0 | 7.1.0 | Compatible with coverage 7.15; 54% branch-aware result | A | [PyPI](https://pypi.org/project/pytest-cov/) | Upgraded and verified |
| Ruff | Lint/format | 0.15.22 / 0.15.22 | 0.15.22 (2026-07-16) | 0.15.22 | 0.15.22 | Config migrated and repository formatted | A | [PyPI](https://pypi.org/project/ruff/), [docs](https://docs.astral.sh/ruff/) | Upgraded and verified |
| types-PyYAML | PyYAML typing | 6.0.12.20260518 / same | same (2026-05-18) | same | same | Only remaining third-party stub required by mypy | A | [PyPI](https://pypi.org/project/types-PyYAML/) | Already current |

## Important transitive and native compatibility matrix

| Component | Introduced by / purpose | Installed | Latest stable | Latest verified compatible / target | Compatibility and security | Official source / decision |
| --- | --- | --- | --- | --- | --- | --- |
| Starlette | FastAPI ASGI core | 1.3.1 | 1.3.1 (2026-06-12) | 1.3.1 in clean environment | FastAPI 0.139.2 resolver selected it; full API suite passes; A | [PyPI](https://pypi.org/project/starlette/) — resolver-owned and verified |
| limits | SlowAPI rate-limit engine | 5.8.0 | 5.8.0 (2026-02-05) | 5.8.0 | Python >=3.10; in-memory backend is per process; A | [PyPI](https://pypi.org/project/limits/) — already current |
| fsspec | PyG/filesystem abstraction | 2026.4.0 | 2026.6.0 (2026-06-16) | 2026.4.0 in clean environment | Important transitive; do not direct-pin outside generated constraints; A | [PyPI](https://pypi.org/project/fsspec/) — manage through future constraints |
| DiskCache | DVC cache serialization | Not installed | 5.6.3 (2023-07-24) | No fixed release | Versions through 5.6.3 deserialize cache values with pickle; keep DVC outside supported extras | [OSV CVE-2025-69872](https://osv.dev/vulnerability/CVE-2025-69872), [PyPI](https://pypi.org/project/diskcache/) — remove/defer |
| CUDA/cuDNN wheel dependencies | Torch cu126 runtime | Not installed in CPU audit environment | Resolver-owned | Official Torch 2.13 cu126 set | Must follow the official Torch cu126 index, never independently chase individual NVIDIA package versions; GPU `pip check` remains required | [cu126 index](https://download.pytorch.org/whl/cu126), [ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) — coordinated target |
| FFmpeg | Decode/encode/audio system CLI | Host available; Docker apt unpinned | Distribution-dependent | Host/Docker repository version | Native codec behavior and CVEs depend on OS snapshot; no exact package version is declared; N/A | [FFmpeg](https://ffmpeg.org/download.html) — snapshot or rebuild/scan routinely |

## Removed/replaced declarations

| Historical component | Historical role/version | Evidence-based decision |
| --- | --- | --- |
| `supervision==0.27.0.post2` | CV helper | Removed: no first-party import/use remained; reduces CV compatibility surface |
| `hydra-core==1.3.2` and Hydra configs | Config framework | Removed: runtime did not consume it; CLI/env/YAML are the actual configuration paths |
| `torchmetrics==1.5.2` | Metrics | Removed: first-party metrics are used; no direct import remained |
| `pandas==2.2.3` | Tables | Removed as a direct dependency: no required runtime import; may remain transitive/optional elsewhere |
| `rich`, `tqdm`, `click` | CLI presentation | Removed as direct declarations where first-party code did not import them; transitive use remains resolver-owned |
| `black`, `isort` | Formatting/import sorting | Removed: Ruff provides both configured functions |
| `types-requests` | Requests typing stub | Removed from manifest and pre-commit: [official package page](https://pypi.org/project/types-requests/) says Requests >=2.34 includes annotations and users should uninstall the stub |
| ImageIO / `experiments` extra | One removed experimental script | Removed after confirming no first-party import/use remains |
| `dvc[all]`, then base DVC 3.67.1 | Data/versioning CLI and every remote backend | Removed from supported extras: DVC's current closure includes DiskCache 5.6.3, whose unsafe-pickle advisory has no fixed release. DVC YAML/code remains and operators may use a separately isolated, reviewed CLI. |
| Core ONNX/ORT/MLflow/audio/media packages | Previously always-installed heavy stack | Moved to named optional groups to make core API/CPU installation smaller and less conflict-prone; all supported extras are exercised in CI |

## Coordinated ecosystem decisions and migrations

### Python, packaging, and compiled scientific stack

The repository now deliberately targets the Python 3.12 line. `.python-version` moves to 3.12.13 while
CI remains `3.12` to receive current hosted patch releases. NumPy 2 migration work is already absorbed.
NumPy 2.5.1, SciPy 1.18, and scikit-learn 1.9 resolve for Python 3.12, but were deliberately not adopted:
they should be upgraded together with OpenCV/PyAV compiled-wheel and numerical-behavior tests, not by
changing one line independently. Packaging tools were upgraded to pip 26.1.2, setuptools 83, and wheel
0.47; a standard isolated build and installed-wheel import pass.

### PyTorch, TorchVision, PyG, CUDA, and ONNX Runtime

The recommended group is:

```text
Python 3.12
torch 2.13.0 + cu126
torchvision 0.28.0 + cu126
torch-geometric 2.8.0
CUDA 12.6.3 container / CUDA 12.6 wheel runtime
cuDNN 9 (official Torch cu126 resolver-owned set)
onnx 1.22.0
onnxruntime-gpu 1.26.0
```

The exact coordinated set is Torch 2.13.0, Vision 0.28.0, PyG 2.8.0, CUDA 12.6, ONNX 1.22.0, and ORT
1.26.0. Official PyTorch indexes publish both CPU and cu126 variants of the Torch/Vision pair. The CPU
pair and PyG passed 251 tests. ONNX/ORT passed a real export, checker, session-creation, and raw-output
parity run. ORT 1.26 is intentionally selected instead of numerically newer 1.27 because the official
1.26 announcement states that CUDA 12 support is removed in 1.27. GPU execution remains an explicit
deployment blocker, not a reason to keep advisory-affected Torch/ONNX versions.

### Ultralytics, OpenCV, and media

Ultralytics was upgraded to the current 8.4.100 API and export code was migrated. OpenCV remains on the
latest tested 4.x because 5.0 is a major transition. PyAV remains 17 for the same reason. yt-dlp is exact
pinned because extractor behavior changes rapidly. Real redirect/codec/network tests remain required.

### FastAPI, Pydantic, Starlette, Uvicorn, and SlowAPI

Direct components were upgraded together and the ASGI/API suite passes. Starlette remains transitive;
the clean resolver selected current 1.3.1 and the full API/test suite passed. A reviewed constraints
workflow would freeze that successful transitive choice per supported platform.

### MLflow and DVC

MLflow moved from 2.x to current 3.14.0 with run ownership and artifact logging repaired. Its `<49`
Cryptography constraint is satisfied by fixed-compatible 48.0.1. DVC itself is current at 3.67.1, but its
DiskCache 5.6.3 dependency has an unfixed unsafe-deserialization advisory; DVC was therefore removed from
normal extras while its stage/configuration remain. No remote service or credentials were modified.

### Quality tools

pytest 9, Ruff 0.15, coverage 7.15, Bandit 1.9, pre-commit 4.6, build 1.5, and pytest-cov 7 are current and
locally compatible. mypy 2.3 is a deliberate deferred major; 1.20.2 is green. Pre-commit no longer creates
an unusable dependency-empty mypy environment: the local mypy hook runs against the already installed
project environment, and the complete aggregate gate passes.

## GitHub Actions matrix

Every action is pinned to an immutable commit SHA and annotated with a human-readable release. Official
`releases/latest` pages were checked on 2026-07-18.

| Action | Declared/current | Latest stable observed | Installed | Compatibility/security | Official source | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `actions/checkout` | SHA `9c091bb...` (`v7.0.0`) | v7.0.0 (2026-06-18) | N/A | Immutable SHA, current Node runtime | [releases](https://github.com/actions/checkout/releases/latest) | Already current |
| `actions/setup-python` | SHA `ece7cb0...` (`v6.3.0`) | v6.3.0 (2026-06-24) | N/A | Immutable SHA | [releases](https://github.com/actions/setup-python/releases/latest) | Already current |
| `actions/configure-pages` | SHA `45b...` (`v6.0.0`) | v6.0.0 | N/A | Immutable SHA, Pages-scoped permission | [releases](https://github.com/actions/configure-pages/releases/latest) | Already current |
| `actions/upload-pages-artifact` | SHA `fc...` (`v5`) | v5 | N/A | Immutable SHA | [releases](https://github.com/actions/upload-pages-artifact/releases/latest) | Already current |
| `actions/deploy-pages` | SHA `cd2...` (`v5`) | v5 | N/A | Immutable SHA, deployment environment | [releases](https://github.com/actions/deploy-pages/releases/latest) | Already current |
| `actions/labeler` | SHA `b8dd...` (`v6.2`) | v6.2 (2026-07-09) | N/A | Immutable SHA, pull-request write only in job | [releases](https://github.com/actions/labeler/releases/latest) | Already current |
| `actions/stale` | SHA `1e223...` (`v10.4`) | v10.4 (2026-07-10) | N/A | Immutable SHA, issues/PR write only | [releases](https://github.com/actions/stale/releases/latest) | Already current |
| `github/codeql-action` | SHA `7188...` (`v4`) | v4 / CodeQL bundle 2.26.1 | N/A | Immutable SHA, security-events permission isolated | [releases](https://github.com/github/codeql-action/releases/latest) | Already current |
| `googleapis/release-please-action` | SHA `45996...` (`v5`) | v5 (2026-04-22) | N/A | Immutable SHA, release job-scoped write permission | [releases](https://github.com/googleapis/release-please-action/releases/latest) | Already current |

Remote pre-commit repositories are immutable-SHA pinned: `pre-commit-hooks` v6, Ruff 0.15.22, and Bandit
1.9.4. Mypy is a local system hook so it can see the installed ML/CV dependency graph. Remote initialization
and the entire aggregate gate passed.

## Docker and system dependencies

The runtime image is multi-stage, digest-pinned, non-root (`10001:10001`), `tini`-managed, and has a local
health check. Python toolchain versions and the Torch cu126 pair are exact. System packages installed by
name are `ca-certificates`, `ffmpeg`, GLib/OpenGL/OpenMP/X11 runtime libraries, Python/pip/venv, and `tini`.
They are not version-pinned, so an unchanged Dockerfile can receive different OS packages when rebuilt.

Compose rendering passes for the base `dev`/`mlops` profiles and the explicit GPU overlay. The app now
mounts `/app/models` read-only, owns a writable local MLflow SQLite target by default, can be pointed at the
profile service explicitly, and uses writable Matplotlib/Ultralytics config paths under `/tmp`. Docker
BuildKit resolved the pinned base and reported no Dockerfile warnings. A full CUDA image build and health
run were not attempted because they require a large image/toolchain fetch; no such result is claimed.

## Upgrade summary

### Performed

- Consolidated dependency authority in `pyproject.toml`; synchronized README, Makefile, Docker, CI,
  pre-commit, `.python-version`, and the editable requirements wrapper.
- Upgraded compatible runtime/API/CV/tooling packages to the recommended targets shown above.
- Coordinated Torch 2.13/TorchVision 0.28/PyG 2.8/CUDA 12.6 and ONNX 1.22/ORT 1.26 target declarations.
- Migrated code for current Ultralytics export, MLflow 3 lifecycle, Pydantic/FastAPI schemas, NumPy 2,
  pytest 9, Ruff, and modern typing.
- Removed unused/conflicting direct dependencies and the unused ImageIO experiment extra; moved heavy
  supported integrations into optional extras.
- Removed DVC from supported extras because no fixed DiskCache release exists; preserved its configuration
  for explicitly isolated operator use.
- Pinned the CUDA base by digest and every remote GitHub Action/pre-commit repository by immutable SHA.
- Added a fixed-compatible Cryptography target, real ONNX validation/atomic publication, and all-extras CI.

### Deferred with evidence

- CUDA 13: requires a new base/driver/Torch/ORT GPU qualification; current application target remains cu126.
- ONNX Runtime 1.27: incompatible with the selected CUDA 12 line according to the official 1.26 announcement.
- NumPy 2.5/SciPy 1.18/scikit-learn 1.9: requires a compiled numerical-stack resolver/test batch.
- OpenCV 5 and PyAV 18: major versions require real codec/image/video regression.
- mypy 2.3: major quality-tool migration should be isolated.
- DVC/DiskCache: defer normal installation until a fixed DiskCache release exists or an operator documents
  explicit isolation and risk acceptance.
- Cross-platform transitive locks and OS-package snapshots: generate from the successful Linux closure and
  then validate CPU/CUDA/other supported platforms before committing them.

### Security advisory decisions

- The prior Torch 2.8 line was reachable from normal installs and matched multiple official advisory records,
  including [PYSEC-2025-203](https://osv.dev/vulnerability/PYSEC-2025-203),
  [PYSEC-2025-204](https://osv.dev/vulnerability/PYSEC-2025-204),
  [PYSEC-2025-206](https://osv.dev/vulnerability/PYSEC-2025-206), and
  [PYSEC-2026-139](https://osv.dev/vulnerability/PYSEC-2026-139). The tested target is 2.13.0.
- ONNX 1.21 matched [GHSA-537c-gmf6-5ccf](https://osv.dev/vulnerability/GHSA-537c-gmf6-5ccf);
  the tested target is 1.22.0.
- Older Cryptography wheels were below the fixed compatible 48.0.1 line; the target follows the
  [official changelog](https://cryptography.io/en/stable/changelog/) while respecting MLflow 3.14's `<49`
  constraint.
- DiskCache through current 5.6.3 matches [CVE-2025-69872](https://osv.dev/vulnerability/CVE-2025-69872)
  and has no fixed release; this is why DVC is not in a supported installation extra.

## Final dependency validation

| Check | Result |
| --- | --- |
| Standard isolated sdist/wheel build | PASS with declared setuptools 83.0.0 and wheel 0.47.0; SPDX MIT metadata; no deprecation |
| Wheel installation in isolated target directory | PASS; distribution/import version 0.1.0, installed package path outside repository, 15 API routes |
| Clean editable all-extras installation | PASS for `audio,data,dev,export,mlops`; every declared direct version matches |
| Clean-environment `pip check` | PASS, no broken requirements |
| Pre-existing ignored `.venv` `pip check` | FAIL: orphaned PyDrive2 1.21.3 and pyOpenSSL 24.2.1 require Cryptography `<44`; recreate rather than weakening the secure target |
| Runtime/API/import/test compatibility | PASS, 251 tests; 54% branch coverage; Ruff/format/mypy/Bandit/pre-commit pass |
| Real ONNX/ORT compatibility | PASS, ONNX 1.22 checker and ORT 1.26 CPU provider; raw PyTorch/ORT parity within configured tolerance |
| Installed-environment `pip-audit --local` | PASS, no known vulnerabilities; tool-reported skips limited to editable project and custom `+cpu` Torch/Vision labels |
| Fully pinned public-version closure audit | PASS, no known vulnerabilities; explicitly covers Torch 2.13.0 and Vision 0.28.0; CI dynamically repeats their public-version scan |
| Docker Compose / Dockerfile dependency model | PASS for base and GPU overlay rendering; BuildKit check reports no warnings |
| GPU binary compatibility | Environment blocker: no driver/GPU/TensorRT execution; exact cu126 group is declared but not runtime-certified |

## Recommended next dependency steps

1. Recreate the ignored local `.venv` from `pyproject.toml`; do not downgrade Cryptography or preserve
   orphaned PyDrive2/pyOpenSSL solely to repair that historical environment.
2. On supported NVIDIA hardware, create a fresh Python 3.12/CUDA 12.6 environment from the declared
   official wheel index; run `pip check`, vulnerability/container scans, the full suite, ONNX GPU parity,
   TensorRT parity, and a representative soak test.
3. Generate reviewed, platform-specific constraints from the successful clean Linux resolution; keep
   `pyproject.toml` as the direct-dependency authority and regenerate constraints in CI.
4. Test the NumPy/SciPy/scikit-learn and OpenCV/PyAV major/minor batches independently with real media.
5. Add routine container rebuild/scan policy and an OS-package snapshot strategy.
6. Re-evaluate DVC only after DiskCache publishes a fixed release or after explicit operational risk
   acceptance; do not silently restore it to broad developer/CI environments.
