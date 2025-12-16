def test_package_imports():
    import src  # noqa: F401
    from src import calib, detect, graph, live, models, track, utils  # noqa: F401

    assert hasattr(detect, "run_inference")
    assert hasattr(track, "ByteTrackRuntime")
