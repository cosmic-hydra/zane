from drug_discovery.synthesis.backends import BackendResult, BaseRetrosynthesisBackend, RouteCandidate
from drug_discovery.synthesis.retrosynthesis import RetrosynthesisPlanner


class DummyBackend(BaseRetrosynthesisBackend):
    name = "dummy"

    def __init__(self, succeed: bool = True):
        self.succeed = succeed

    def is_available(self) -> bool:
        return True

    def plan(self, smiles: str, max_depth: int = 5) -> BackendResult:
        if not self.succeed:
            return BackendResult.failure(self.name, "no route")
        route = RouteCandidate(score=0.1, steps=3, precursors=["CCC"])
        return BackendResult(backend=self.name, success=True, routes=[route])


def test_plan_synthesis_uses_first_successful_backend():
    planner = RetrosynthesisPlanner(backends=[DummyBackend()])
    result = planner.plan_synthesis("CCO", max_depth=6)

    assert result["success"] is True
    assert result["backend_used"] == "dummy"
    assert result["num_steps"] == 3
    assert result["precursors"] == ["CCC"]
    assert result["backend_results"][0]["backend"] == "dummy"


def test_plan_synthesis_falls_back_when_backend_fails():
    planner = RetrosynthesisPlanner(backends=[DummyBackend(succeed=False)])
    result = planner.plan_synthesis("CCO", max_depth=4)

    assert result["success"] is True
    assert result["backend_results"]
    assert result["backend_results"][0]["success"] is False
