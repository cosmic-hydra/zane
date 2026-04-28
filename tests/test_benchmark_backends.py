from drug_discovery.benchmarking.backends import BaseBenchmarkBackend, BenchmarkResult, BenchmarkRunner


class DummyBenchmarkBackend(BaseBenchmarkBackend):
    name = "dummy"

    def __init__(self, succeed: bool = True):
        self.succeed = succeed

    def is_available(self) -> bool:
        return True

    def run(self, dataset_path: str | None = None) -> BenchmarkResult:
        if not self.succeed:
            return BenchmarkResult.failure(self.name, "fail")
        return BenchmarkResult(suite=self.name, success=True, metrics={"score": 0.9})


def test_benchmark_runner_calls_backend():
    runner = BenchmarkRunner(backends=[DummyBenchmarkBackend()])
    result = runner.run("dummy")

    assert result["success"] is True
    assert result["metrics"]["score"] == 0.9


def test_benchmark_runner_handles_missing_suite():
    runner = BenchmarkRunner(backends=[])
    result = runner.run("unknown")

    assert result["success"] is False
