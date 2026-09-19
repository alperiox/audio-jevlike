"""Frozen-encoder feature extraction, cached once to disk.

Three ablation arms (spec §6):
  wavlm   — primary; SSL objective retains paralinguistic information
  whisper — control; ASR objective may discard prosody at the feature boundary
  prosody — diagnostic; explicit F0/energy/voicing. Distinguishes "the encoder
            threw prosody away" from "the task does not need prosody" on a null.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16_000
HOP_LENGTH = 320  # 20ms at 16kHz -> 50Hz frames, matching WavLM

# Whisper's encoder always emits frames at 50/sec of *input* audio, regardless
# of how long that audio actually is (see the trimming note in `encode` below).
WHISPER_FRAMES_PER_SECOND = 50

# WavLM's gated relative-position bias builds a T x T index tensor, so its
# memory scales with the *square* of clip length: a 305s MELD segmentation
# artifact (dia38_utt4) implies a ~950GB fp32 tensor, and a real extraction
# run over MELD died mid-corpus with `RuntimeError: Invalid buffer size:
# 13.85 GiB` at a merely-long (not pathological) clip once memory was under
# concurrent pressure. 30s affects 3 of 13,706 MELD utterances (0.02%) while
# keeping peak WavLM memory around 3.3GB -- comfortable headroom even under
# pressure. Tune via `FeatureExtractor(..., max_audio_seconds=...)`, not by
# editing `encode`.
DEFAULT_MAX_AUDIO_SECONDS = 30.0

ENCODERS: dict[str, str] = {
    "wavlm": "microsoft/wavlm-large",
    "whisper": "openai/whisper-small",
    "prosody": "__explicit__",
}


def truncate_to_max_seconds(
    wav: np.ndarray, max_seconds: float = DEFAULT_MAX_AUDIO_SECONDS
) -> tuple[np.ndarray, bool]:
    """Hard-cap audio length before it reaches any encoder arm.

    Applied at the single call site every encoder arm's `encode()` goes
    through (prosody, whisper, wavlm alike), so the cap is identical across
    arms -- `assert_uniform_cache_coverage` (scripts/run_ablation.py) needs
    every encoder's cache to cover the same uid set, so a cap that behaved
    differently per arm would break the cross-arm comparison.

    Returns `(possibly-truncated wav, whether truncation happened)` so
    callers can report it. Silent truncation is exactly the failure class
    this project exists to avoid -- see `scripts/extract_features.py`'s
    end-of-run summary.
    """
    max_samples = int(max_seconds * SAMPLE_RATE)
    if len(wav) <= max_samples:
        return wav, False
    return wav[:max_samples], True


class FeatureExtractor:
    def __init__(
        self,
        encoder_key: str,
        device: torch.device,
        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
    ) -> None:
        if encoder_key not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder_key!r}")
        self.key = encoder_key
        self.device = device
        self.max_audio_seconds = max_audio_seconds
        self.last_truncated = False
        self._model = None
        self._feature_extractor = None
        if encoder_key == "wavlm":
            from transformers import WavLMModel
            self._model = WavLMModel.from_pretrained(ENCODERS[encoder_key])
        elif encoder_key == "whisper":
            from transformers import WhisperFeatureExtractor, WhisperModel
            self._model = WhisperModel.from_pretrained(ENCODERS[encoder_key]).encoder
            # R2: hoisted out of encode() — from_pretrained() here is a
            # filesystem/network load. Called once per utterance across the
            # ~13k-utterance MELD corpus, that turns a ~1 hour extraction
            # pass into an overnight one.
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(
                ENCODERS["whisper"]
            )
        if self._model is not None:
            self._model.eval().to(device)
            for p in self._model.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, wav: np.ndarray) -> torch.Tensor:
        """(samples,) float32 @16kHz -> (T, D) float32 frame features.

        Duration-capped at `self.max_audio_seconds` (default
        `DEFAULT_MAX_AUDIO_SECONDS` = 30s) before the waveform reaches any
        encoder arm below -- see `truncate_to_max_seconds`. Sets
        `self.last_truncated` so a caller can tell whether this call's clip
        was affected (see `scripts/extract_features.py`'s summary).

        Interaction with Whisper's own 30s window: `WhisperFeatureExtractor`
        already hard-pads/truncates its mel input to a fixed 30s (see the R1
        note below), so at the default cap the two truncations land on the
        exact same sample boundary (30s * 16kHz = 480,000 samples either
        way) and this cap is a no-op for the whisper arm -- it only changes
        behavior for wavlm/prosody, which have no such internal bound. If
        `max_audio_seconds` is ever tuned *above* 30s, Whisper's internal
        window still silently wins for that arm (it cannot see past 30s
        regardless of what we pass it), so the caps would stop matching;
        tuning it *below* 30s tightens all three arms uniformly with no such
        mismatch.
        """
        wav, self.last_truncated = truncate_to_max_seconds(
            wav, self.max_audio_seconds
        )
        if self.key == "prosody":
            return _explicit_prosody(wav)
        if self.key == "whisper":
            mel = self._feature_extractor(
                wav, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            out = self._model(mel.input_features.to(self.device)).last_hidden_state
            out = out.squeeze(0).float().cpu()
            # R1: WhisperFeatureExtractor zero-pads every clip to a fixed 30s
            # window, so the encoder always emits a fixed 1500 frames (50/sec
            # * 30s) no matter how short the input actually was. MELD
            # utterances average ~3s (~165 real frames), so left untrimmed:
            #   (a) the cache balloons to ~59GB instead of ~8.5GB, and
            #   (b) every downstream attention mask would mark all 1500
            #       frames valid, so ~89% of what the state encoder attends
            #       over would be silent padding — quietly destroying the
            #       control arm this experiment depends on (see module
            #       docstring: whisper is the "did the ASR objective throw
            #       prosody away" test, and a broken control still produces
            #       a clean-looking comparison).
            # Trim back to the frame count that corresponds to the real
            # audio: Whisper's encoder runs at 50 frames/sec of *original*
            # audio (not of the padded 30s), so that count is
            # ceil(len(wav) / SAMPLE_RATE * 50), clamped to whatever the
            # encoder actually produced (it can't exceed the padded max).
            valid_frames = min(
                math.ceil(len(wav) / SAMPLE_RATE * WHISPER_FRAMES_PER_SECOND),
                out.shape[0],
            )
            return out[:valid_frames]
        x = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        out = self._model(x).last_hidden_state
        return out.squeeze(0).float().cpu()


def _explicit_prosody(wav: np.ndarray) -> torch.Tensor:
    """F0, RMS energy and voicing probability at 50Hz.

    Uses librosa.pyin (YIN pitch tracking): F0 is what the interventions in
    Phase 2 manipulate, so the diagnostic arm reads exactly the manipulated
    quantity.
    """
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        wav, sr=SAMPLE_RATE, fmin=60, fmax=400, hop_length=HOP_LENGTH,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    rms = librosa.feature.rms(y=wav, hop_length=HOP_LENGTH).squeeze(0)
    n = min(len(f0), len(rms), len(voiced_prob))
    stack = np.stack([f0[:n], rms[:n], np.nan_to_num(voiced_prob[:n])], axis=-1)
    return torch.from_numpy(stack).float()


class FeatureCache:
    """fp16 on disk (halves an ~8.5GB cache), fp32 in memory (spec §10)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, uid: str) -> Path:
        return self.path / f"{uid}.pt"

    def __contains__(self, uid: str) -> bool:
        return self._file(uid).exists()

    def write(self, uid: str, tensor: torch.Tensor) -> None:
        torch.save(tensor.to(torch.float16).contiguous(), self._file(uid))

    def read(self, uid: str) -> torch.Tensor:
        return torch.load(self._file(uid), map_location="cpu").float()
