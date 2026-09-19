import math
from unittest import mock

import numpy as np
import torch

from prosodia.features import ENCODERS, SAMPLE_RATE, FeatureCache


def test_encoder_registry_has_the_three_ablation_arms():
    assert set(ENCODERS) == {"wavlm", "whisper", "prosody"}


def test_cache_roundtrip_is_fp16_on_disk_fp32_in_memory(tmp_path):
    cache = FeatureCache(tmp_path / "feats")
    x = torch.randn(37, 16, dtype=torch.float32)
    cache.write("utt-1", x)
    assert "utt-1" in cache
    back = cache.read("utt-1")
    assert back.dtype is torch.float32          # fp32 at measurement time
    assert back.shape == x.shape
    torch.testing.assert_close(back, x, rtol=1e-2, atol=1e-2)  # fp16 on disk


def test_cache_reports_missing_keys(tmp_path):
    assert "nope" not in FeatureCache(tmp_path / "feats")


def test_prosody_extractor_returns_explicit_f0_energy_voicing():
    from prosodia.features import FeatureExtractor
    ex = FeatureExtractor("prosody", torch.device("cpu"))
    sr = 16000
    t = np.arange(sr) / sr
    wav = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    feats = ex.encode(wav)
    assert feats.ndim == 2 and feats.shape[1] == 3  # f0, energy, voicing
    assert feats.shape[0] > 10


def test_whisper_encoder_output_is_trimmed_to_real_audio_frames():
    """R1: Whisper's feature extractor zero-pads every clip to 30s, so its
    encoder always emits 1500 frames. A short MELD-length utterance must come
    back trimmed to the frame count that corresponds to its real duration
    (50 frames/sec), not the full padded 1500.
    """
    from prosodia.features import FeatureExtractor

    ex = FeatureExtractor("whisper", torch.device("cpu"))
    duration_s = 2.0
    wav = np.zeros(int(duration_s * SAMPLE_RATE), dtype=np.float32)
    feats = ex.encode(wav)

    expected_frames = math.ceil(duration_s * 50)  # 100
    assert feats.shape[0] == expected_frames
    assert feats.shape[0] < 1500


def _sine(duration_s: float, freq: float = 150.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_duration_cap_truncates_a_clip_longer_than_the_cap():
    """A clip past the cap must come back with exactly the frame count a
    clip of exactly-cap-length would produce -- not more.

    Fault this catches: WavLM's gated relative-position bias builds a T x T
    index tensor, so an uncapped long clip doesn't just cost more memory
    linearly, it can blow the allocator outright (a 305s MELD segmentation
    artifact implies a ~950GB fp32 tensor; a real extraction run died with
    `RuntimeError: Invalid buffer size: 13.85 GiB` on a merely-long clip
    under memory pressure). Confirmed by fault injection (see the task
    report): with the `truncate_to_max_seconds` call removed from `encode`,
    this test's long clip produces MORE frames than the capped reference
    clip (proportional to its real duration), instead of the same count.

    Uses the "prosody" arm with a tiny cap (0.5s) rather than a real 30s/45s
    pair: the cap is applied identically regardless of its value (a plain
    length check before dispatch), and this keeps the test from paying for
    tens of seconds of librosa.pyin on every run.
    """
    from prosodia.features import FeatureExtractor

    cap_s = 0.5
    ex = FeatureExtractor("prosody", torch.device("cpu"), max_audio_seconds=cap_s)

    exact_cap_feats = ex.encode(_sine(cap_s))
    long_feats = ex.encode(_sine(cap_s + 1.5))

    assert long_feats.shape[0] == exact_cap_feats.shape[0]
    assert ex.last_truncated is True


def test_duration_cap_leaves_a_clip_shorter_than_the_cap_untouched():
    """A clip under the cap must be bit-identical to the same clip encoded
    with no meaningful cap at all -- the cap must not perturb the common
    case (MELD's median utterance is ~2.5s, far under any sane cap).

    Fault this catches: an off-by-one or `<` vs `<=` slip in the length
    comparison that clips exactly-at-the-boundary or near-boundary audio
    that should pass through untouched.
    """
    from prosodia.features import FeatureExtractor

    wav = _sine(1.0)
    capped = FeatureExtractor("prosody", torch.device("cpu"), max_audio_seconds=5.0)
    uncapped = FeatureExtractor(
        "prosody", torch.device("cpu"), max_audio_seconds=1000.0
    )

    capped_feats = capped.encode(wav.copy())
    assert capped.last_truncated is False
    uncapped_feats = uncapped.encode(wav.copy())
    assert uncapped.last_truncated is False

    assert capped_feats.shape == uncapped_feats.shape
    torch.testing.assert_close(capped_feats, uncapped_feats)


def test_duration_cap_applies_uniformly_to_the_whisper_arm_too():
    """The cap must not be a wavlm-only guardrail: `assert_uniform_cache_coverage`
    requires every encoder's cache to cover an identical uid set, so a cap
    that behaved differently per arm would break the cross-arm comparison.

    Fault this catches: a cap wired into only the wavlm/prosody code paths
    (e.g. added after the `if self.key == "whisper"` branch instead of at
    the top of `encode`), which would leave whisper unbounded while the
    other two arms were capped.
    """
    from prosodia.features import FeatureExtractor

    cap_s = 0.5
    ex = FeatureExtractor("whisper", torch.device("cpu"), max_audio_seconds=cap_s)
    ex.encode(_sine(cap_s + 1.5))
    assert ex.last_truncated is True


def test_whisper_feature_extractor_is_constructed_once_in_init():
    """R2: WhisperFeatureExtractor.from_pretrained must be hoisted into
    __init__ (constructed once per FeatureExtractor), not called on every
    encode() invocation, or a 13k-utterance extraction pass runs overnight.
    """
    from transformers import WhisperFeatureExtractor

    from prosodia.features import FeatureExtractor

    with mock.patch.object(
        WhisperFeatureExtractor, "from_pretrained",
        wraps=WhisperFeatureExtractor.from_pretrained,
    ) as mocked:
        ex = FeatureExtractor("whisper", torch.device("cpu"))
        assert mocked.call_count == 1

        wav = np.zeros(SAMPLE_RATE, dtype=np.float32)
        ex.encode(wav)
        ex.encode(wav)
        assert mocked.call_count == 1  # not re-constructed per encode() call
