import subprocess
import sys


def test_package_imports():
    import src  # noqa: F401
    from src import calib, detect, graph, live, models, track, utils  # noqa: F401

    assert hasattr(detect, "run_inference")
    assert hasattr(track, "ByteTrackRuntime")


def test_cli_packages_do_not_preload_executable_modules() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys, src.detect, src.live; "
            "assert 'src.detect.infer' not in sys.modules; "
            "assert 'src.live.run_live' not in sys.modules"
        ),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
