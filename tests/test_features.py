import math

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


def test_whisper_feature_extractor_is_constructed_once_in_init():
    """R2: WhisperFeatureExtractor.from_pretrained must be hoisted into
    __init__ (constructed once per FeatureExtractor), not called on every
    encode() invocation, or a 13k-utterance extraction pass runs overnight.
    """
    from prosodia.features import FeatureExtractor

    ex = FeatureExtractor("whisper", torch.device("cpu"))
    assert ex._feature_extractor is not None

    wav = np.zeros(SAMPLE_RATE, dtype=np.float32)
    fe_before = ex._feature_extractor
    ex.encode(wav)
    assert ex._feature_extractor is fe_before
