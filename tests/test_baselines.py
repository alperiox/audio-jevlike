from prosodia.evaluation.baselines import ARMS, build_jev_request
from prosodia.schema import QuestionSpec


def test_ablation_grid_is_three_losses_by_three_encoders():
    assert len(ARMS) == 9
    assert {a.encoder for a in ARMS} == {"wavlm", "whisper", "prosody"}
    names = {(a.brier_weight > 0, a.temperature_scale) for a in ARMS}
    assert names == {(False, False), (True, False), (False, True)}


def test_every_arm_has_a_unique_name():
    assert len({a.name for a in ARMS}) == len(ARMS)


def test_jev_request_matches_the_documented_schema():
    specs = [
        QuestionSpec("emotion", "choice", "Which emotion?",
                     {"anger": None, "joy": None}),
        QuestionSpec("sentiment", "score", "Rate sentiment.",
                     ["negative", "neutral", "positive"]),
        QuestionSpec("is_negative", "noul", "Is it negative?", None),
    ]
    req = build_jev_request("Joey: You fell asleep!", specs)
    assert req["model"] == "jev-latest"
    assert req["questions"]["emotion"]["type"] == "choice"
    assert set(req["questions"]["emotion"]["criteria"]) == {"anger", "joy"}
    assert req["questions"]["sentiment"]["criteria"] == \
        ["negative", "neutral", "positive"]
    assert req["questions"]["is_negative"]["type"] == "noul"
