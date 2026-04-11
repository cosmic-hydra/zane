from drug_discovery.generation.backends import BaseGeneratorBackend, GenerationManager, GenerationResult


class DummyGenBackend(BaseGeneratorBackend):
    name = "dummy"

    def __init__(self, succeed: bool = True):
        self.succeed = succeed

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str | None, num: int = 5, **kwargs) -> GenerationResult:
        if not self.succeed:
            return GenerationResult.failure(self.name, "fail")
        return GenerationResult(backend=self.name, success=True, molecules=[f"X{i}" for i in range(num)])


def test_generation_manager_picks_first_successful_backend():
    manager = GenerationManager(backends=[DummyGenBackend(succeed=False), DummyGenBackend()])
    result = manager.generate(prompt="test", num=3)

    assert result["success"] is True
    assert result["backend"] == "dummy"
    assert result["molecules"] == ["X0", "X1", "X2"]
    assert len(result["attempts"]) == 2


def test_generation_manager_reports_failure_when_none_succeed():
    manager = GenerationManager(backends=[DummyGenBackend(succeed=False)])
    result = manager.generate(prompt=None, num=2)

    assert result["success"] is False
    assert result["molecules"] == []
