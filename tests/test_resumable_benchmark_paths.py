import importlib.util
from pathlib import Path


def _load_resumable_module():
    module_path = Path(__file__).resolve().parents[1] / "kaggle_setup" / "resumable_benchmark.py"
    spec = importlib.util.spec_from_file_location("resumable_benchmark", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_benchmark_path_uses_nested_json_directory(tmp_path):
    resumable = _load_resumable_module()
    root = tmp_path / "benchmarks" / "nosbench"
    nested = root / "nosbench-json"
    nested.mkdir(parents=True)
    (nested / "toy_problem.json").write_text("{}", encoding="utf-8")

    resolved = resumable._normalize_benchmark_json_path(str(root), "nosbench")

    assert resolved == str(nested)


def test_normalize_benchmark_path_keeps_existing_json_directory(tmp_path):
    resumable = _load_resumable_module()
    json_dir = tmp_path / "benchmarks" / "nosbench" / "nosbench-json"
    json_dir.mkdir(parents=True)
    (json_dir / "toy_problem.json").write_text("{}", encoding="utf-8")

    resolved = resumable._normalize_benchmark_json_path(str(json_dir), "nosbench")

    assert resolved == str(json_dir)
