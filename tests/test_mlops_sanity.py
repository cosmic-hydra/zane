from drug_discovery.synthesis.retrosynthesis import RetrosynthesisPlanner


def test_sascore_calculator():
    planner = RetrosynthesisPlanner()
    # Simple molecule
    score_easy = planner.score_synthetic_accessibility("CCO")
    # Complex molecule (Paclitaxel)
    score_hard = planner.score_synthetic_accessibility(
        "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@H]([C@@H]([C@]2(C)C)O)OC(=O)C5=CC=CC=C5)(CO4)OC(=O)C)O)C)OC(=O)[C@H]([C@@H](C1(C)C)O)NC(=O)C6=CC=CC=C6"
    )

    assert score_easy < score_hard
    assert 1.0 <= score_easy <= 10.0
    assert 1.0 <= score_hard <= 10.0
    print(f"Easy score: {score_easy}, Hard score: {score_hard}")


if __name__ == "__main__":
    test_sascore_calculator()
