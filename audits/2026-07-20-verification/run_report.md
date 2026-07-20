# fifa-soccer-ds End-to-End Run Report

Repo: `/home/po/projects/work/fifa-soccer-ds`
State: detached HEAD (mid-rebase), large uncommitted working tree. No git state was changed during this run (only read-only git commands were used). No tracked files were modified. No packages were installed or uninstalled.

---

## 1. Environment assessment

Used the existing venv at `/home/po/projects/work/fifa-soccer-ds/.venv` throughout.

**`python --version`**
```
$ .venv/bin/python --version
Python 3.12.3
```

**`pip check`**
```
$ .venv/bin/python -m pip check
pydrive2 1.21.3 has requirement cryptography<44, but you have cryptography 48.0.1.
pyopenssl 24.2.1 has requirement cryptography<44,>=41.0.5, but you have cryptography 48.0.1.
EXIT_CODE=1
```
Both conflicts are the two known ones flagged in advance (pydrive2/pyopenssl vs cryptography 48). Nothing else showed up. These packages don't appear to be exercised by the pipeline/API/test paths exercised below, so they look orphaned/harmless as expected, but `pip check` does exit non-zero.

**Core import check**
```
$ .venv/bin/python -c "import src.pipeline_full; print('IMPORT_OK')"
IMPORT_OK
EXIT_CODE=0
```

**Verdict: WORKS.** Venv is intact, correct Python version, only the two pre-flagged dependency conflicts, core module imports cleanly.

---

## 2. Smoke tests

`make smoke` fails — the Makefile's `PYTHON` variable defaults to `python3` (system Python), not the venv, so it can't find torch/cv2/etc:
```
$ make smoke
...
ERROR tests/test_api_youtube.py
ERROR tests/test_barca_api.py
...
E   ModuleNotFoundError: No module named 'torch'
E   ModuleNotFoundError: No module named 'cv2'
=========================== short test summary info ============================
SKIPPED [1] tests/test_gnn_inference.py:7: could not import 'torch': No module named 'torch'
SKIPPED [1] tests/test_graph_builder.py:3: could not import 'torch': No module named 'torch'
!!!!!!!!!!!!!!!!!!! Interrupted: 23 errors during collection !!!!!!!!!!!!!!!!!!!
2 skipped, 75 deselected, 23 errors in 2.35s
make: *** [Makefile:43: smoke] Error 2
```
This is a Makefile/environment wiring issue (`PYTHON ?= python3` doesn't resolve to the venv), not a code defect — confirmed by running the same marker through the venv's pytest directly:
```
$ .venv/bin/python -m pytest -m smoke -q
..........                                                               [100%]
10 passed, 241 deselected, 2 warnings in 6.61s
```

**Verdict: PARTIAL.** The smoke tests themselves pass 10/10 when run through the correct interpreter, but the documented entry point (`make smoke`) is broken as shipped because it invokes system `python3` instead of the project's venv.

---

## 3. Full test suite

pytest-timeout is not installed in the venv (`ModuleNotFoundError: No module named 'pytest_timeout'`), so `--timeout=120` was dropped and only the outer 1200s (20 min) wall-clock cap via `timeout` was used, per the brief's fallback instruction.

```
$ timeout 1200 .venv/bin/python -m pytest tests/ -q --no-header
........................................................................ [ 28%]
........................................................................ [ 57%]
........................................................................ [ 86%]
...................................                                      [100%]
251 passed, 2 warnings in 14.94s
```
No FAILED/ERROR lines anywhere in the captured output. Totals: **251 passed, 0 failed, 0 skipped, 0 errors**, runtime ~15s (well inside the 20-minute cap). Full log captured at `run_outputs/full_test_output.log`.

This **confirms** the Jul 18 audit's claim of 251 passed, exactly.

**Verdict: WORKS.** Full suite is green and fast when run through the venv interpreter.

---

## 4. Real pipeline run

Used existing real extracted frames already checked into the repo at `data/processed/real_sample` (20 JPEG frames from real match footage) — no video extraction or network download needed. Weights file `./yolov8n.pt` exists at repo root.

```
$ timeout 600 .venv/bin/python -m src.pipeline_full \
    --frames-dir data/processed/real_sample \
    --output-dir <scratchpad>/run_outputs/pipeline_run \
    --weights ./yolov8n.pt \
    --max-frames 10

...
2026-07-20 16:09:12,561 - src.graph.build_graph - INFO - Graph built: 76 nodes, 110 edges, 0.00MB
2026-07-20 16:09:12,561 - __main__ - INFO - Pipeline complete! Summary saved to: .../pipeline_summary.json

============================================================
PIPELINE SUMMARY
============================================================
Frames processed: 10
Total detections: 82
Unique tracks: 34
Track IDs: [0, 1, ..., 33]
Graph nodes: 76
Graph edges: 110
Output: <scratchpad>/run_outputs/pipeline_run
============================================================
PIPE_EXIT=0
```

Verified outputs on disk:
- `pipeline_summary.json` — present
- `graphs/final_graph.json` — present
- `detections/frame_00000{1..10}_detections.json` — 10 files present
- `tracks/frame_00000{1..10}_tracks.json` — 10 files present
- `overlays/frame_00000{1..10}_overlay.jpg` — 10 files present

`pipeline_summary.json` contents:
```json
{
  "total_frames": 10,
  "attempted_frames": 10,
  "successful_frames": 10,
  "processing_success_rate": 1.0,
  "partial_failure": false,
  "failures": {
    "unreadable_frames": 0,
    "detection_failures": 0,
    "tracking_failures": 0,
    "overlay_failures": 0
  },
  "total_detections": 82,
  "num_unique_tracks": 34,
  "graph_nodes": 76,
  "graph_edges": 110,
  "output_dir": ".../pipeline_run",
  "config": {
    "weights": "./yolov8n.pt",
    "confidence": 0.4,
    "min_confidence": 0.25,
    "distance_threshold": 80.0,
    "max_age": 15,
    "graph_window": 30,
    "graph_distance_threshold": 80.0
  }
}
```

Detection/track counts are non-zero on real match footage (82 detections across 10 frames, 34 unique tracks, a 76-node/110-edge spatial-temporal graph) — the pipeline is actually detecting players, not just running and producing empty output.

**Verdict: WORKS.** End-to-end detect → track → graph pipeline runs clean on real frames with sane, non-trivial output.

---

## 5. API check

Started the batch API on a non-default port to avoid clashing with anything already running:
```
$ nohup .venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8123 &
INFO:     Started server process [242077]
INFO:     Waiting for application startup.
INFO:src.api.main:Starting FIFA Soccer DS API service
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8123 (Press CTRL+C to quit)
```

```
$ curl -s -w "\nHTTP_STATUS=%{http_code}\n" http://127.0.0.1:8123/health
{"status":"healthy","timestamp":"2026-07-20T10:41:52.095706Z","version":"0.1.0","system_metrics":{"cpu_usage_percent":18.5,"gpu_memory_bytes":0,"memory_total_bytes":16441294848,"memory_used_bytes":8208687104,"memory_percent":49.9},"gpu_memory":0,"active_tasks":0}
HTTP_STATUS=200

$ curl -s -w "\nHTTP_STATUS=%{http_code}\n" http://127.0.0.1:8123/
{"service":"FIFA Soccer DS YouTube Analysis API","version":"0.1.0","status":"running","docs":"/docs","health":"/health"}
HTTP_STATUS=200

$ curl -s -w "\nHTTP_STATUS=%{http_code}\n" http://127.0.0.1:8123/pipeline/info
{"pipeline_version":"0.1.0","supported_formats":["YouTube URLs"],"max_video_duration":7200,"supported_languages":["en","es","fr","de","it"],"confidence_threshold_default":0.75,"features":["Video metadata analysis","Heuristic thumbnail analysis","Optional audio transcription and heuristics","YOLO object detection and ByteTrack-style tracking","Spatial-temporal graph construction","Optional tactical analytics when both teams can be classified"]}
HTTP_STATUS=200
```

Server shutdown and port verification:
```
$ kill 242077
$ ps -p 242077 -o pid,cmd
    PID CMD          <- no matching process, confirmed dead
$ ss -ltnp | grep 8123
(no output)          <- port 8123 confirmed free
```

**Verdict: WORKS.** API boots cleanly, health/info endpoints return valid 200 JSON responses, server was cleanly killed with no orphan and the port confirmed free afterward.

---

## 6. Verdict table

| Surface | Verdict | Evidence |
|---|---|---|
| Environment (venv) | WORKS | Python 3.12.3, `pip check` shows only the 2 pre-flagged pydrive2/pyopenssl vs cryptography conflicts, `import src.pipeline_full` succeeds |
| Smoke tests | PARTIAL | `make smoke` broken (Makefile defaults to system `python3`, not venv → 23 collection errors); `.venv/bin/python -m pytest -m smoke` passes 10/10 |
| Full test suite | WORKS | `251 passed, 0 failed, 0 skipped, 0 errors` in ~15s via venv pytest — matches the Jul 18 audit's claim of 251 passed exactly |
| Pipeline CLI | WORKS | Real frames from `data/processed/real_sample` → 82 detections, 34 tracks, 76-node/110-edge graph; `pipeline_summary.json` and `graphs/final_graph.json` both produced with non-zero, sane values |
| API | WORKS | `/health`, `/`, `/pipeline/info` all return HTTP 200 with valid JSON; server cleanly killed, port 8123 confirmed free afterward |

**Overall: the project works end-to-end on this machine**, provided you invoke Python through the venv directly. The one real defect found is that `make smoke` (and by extension likely `make run`, which uses the same `PYTHON ?= python3` default) doesn't resolve to the project's venv, so the documented make targets don't work out of the box even though the underlying code and tests are fine.
