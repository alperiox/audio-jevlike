import pytest
import torch

from prosodia.config import RunConfig
from prosodia.evaluation.baselines import (
    ARMS, TextOnlyBaseline, build_jev_request, companion_arm_a_name,
)
from prosodia.schema import QuestionSpec

_LOSS_COMBOS = {(False, False), (True, False), (False, True)}


def test_ablation_grid_is_three_losses_by_three_encoders():
    """The 9-arm encoder ablation grid, unchanged by the text-only wiring."""
    encoder_arms = [a for a in ARMS if not a.text_only]
    assert len(encoder_arms) == 9
    assert {a.encoder for a in encoder_arms} == {"wavlm", "whisper", "prosody"}
    combos = {(a.brier_weight > 0, a.temperature_scale) for a in encoder_arms}
    assert combos == _LOSS_COMBOS
    # every encoder gets all three loss regimes, not just some
    for enc in ("wavlm", "whisper", "prosody"):
        enc_combos = {(a.brier_weight > 0, a.temperature_scale)
                      for a in encoder_arms if a.encoder == enc}
        assert enc_combos == _LOSS_COMBOS


def test_text_only_baseline_has_one_arm_per_loss_regime_not_per_encoder():
    """Decision 2: audio is absent for every text-only example, so the
    encoder axis is meaningless there -- exactly one text-only arm per loss
    regime (3 total), not one per (loss, encoder) pair (which would be 9
    bit-identical duplicates logged under different names)."""
    text_only_arms = [a for a in ARMS if a.text_only]
    assert len(text_only_arms) == 3
    combos = {(a.brier_weight > 0, a.temperature_scale) for a in text_only_arms}
    assert combos == _LOSS_COMBOS


def test_ablation_grid_is_nine_encoder_arms_plus_three_text_only_arms():
    assert len(ARMS) == 12


def test_every_arm_has_a_unique_name():
    assert len({a.name for a in ARMS}) == len(ARMS)


def test_mute_audio_forces_audio_present_false_without_touching_anything_else():
    """Fault this catches: a `TextOnlyBaseline` wiring that forgets to mute
    audio (or mutes the wrong field) would let real audio reach the model in
    what is supposed to be a text-only run, silently invalidating the
    controlled comparison success criterion #1 depends on."""
    batch = {
        "audio": torch.randn(3, 5, 4),
        "audio_present": torch.tensor([True, False, True]),
        "context_present": torch.tensor([True, True, True]),
    }
    muted = TextOnlyBaseline.mute_audio(batch)
    assert torch.equal(muted["audio_present"], torch.tensor([False, False, False]))
    # nothing else in the batch changes
    assert torch.equal(muted["audio"], batch["audio"])
    assert torch.equal(muted["context_present"], batch["context_present"])
    # the original batch's tensor must be untouched (mute_audio clones)
    assert torch.equal(batch["audio_present"], torch.tensor([True, False, True]))


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


# --- I9: companion_arm_a_name ----------------------------------------------

def test_companion_arm_a_name_matches_the_real_arms_naming_for_every_encoder_c_arm():
    """Fault this catches: any string-surgery-on-cfg.name implementation
    (e.g. `cfg.name.replace("C-temp", "A-ce")`) that happens to work for
    the literal ARMS entries but would silently misname a config built
    outside that convention. Checking against the REAL `ARMS` list (not a
    hand-rolled RunConfig) proves the production naming is correct, not
    just a hand-picked example."""
    encoder_c_arms = [a for a in ARMS if a.temperature_scale and not a.text_only]
    assert len(encoder_c_arms) == 3  # one per encoder
    for cfg in encoder_c_arms:
        assert companion_arm_a_name(cfg) == f"{cfg.encoder}__A-ce"


def test_companion_arm_a_name_routes_text_only_c_to_text_only_a_never_an_encoder_arm():
    """The exact 'impossible to get wrong' guarantee I9 asks for: a
    text-only Arm C must resolve to the text-only Arm A, never to an
    encoder arm's name -- even though `TEXT_ONLY_CACHE_ENCODER` is
    "wavlm", a text-only Arm C's companion must NOT be "wavlm__A-ce".
    Fault this catches: a naive implementation that always uses
    `cfg.encoder` (which is "wavlm" for every text-only arm, purely for
    cache-shape purposes) instead of branching on `cfg.text_only`."""
    text_only_c_arms = [a for a in ARMS if a.temperature_scale and a.text_only]
    assert len(text_only_c_arms) == 1
    cfg = text_only_c_arms[0]
    assert cfg.encoder == "wavlm"  # sanity: this IS the wavlm-cache-backed arm
    name = companion_arm_a_name(cfg)
    assert name == "text_only__A-ce"
    assert name != "wavlm__A-ce"


def test_companion_arm_a_name_rejects_a_non_arm_c_config():
    """Calling this for an Arm A or Arm B config is a caller bug (it has no
    Arm A to derive from -- it may BE Arm A), and must fail loudly rather
    than silently returning a nonsense name."""
    cfg = RunConfig(name="wavlm__A-ce", encoder="wavlm", temperature_scale=False)
    with pytest.raises(ValueError):
        companion_arm_a_name(cfg)
