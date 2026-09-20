"""Local mic demo: record in the browser, answered by the model on this machine.

Stdlib http.server only -- no web framework, because this is a localhost tool
and a dependency is not worth it.

Out-of-domain warning, surfaced in the UI rather than buried here: the model
is trained on MELD (acted sitcom audio, six known speakers). On your own voice
through a laptop mic:

  * `speaker` is meaningless -- the option set IS the six Friends leads, so it
    must answer one of them. Shown as a novelty, not a prediction.
  * `emotion`/`sentiment` are acted TV affect and may not transfer at all.
  * the acoustic three should transfer best, AND their gold answers are
    computed by a fixed rule -- so the page shows the true answer for YOUR
    clip beside the model's, which makes the demo self-verifying.
"""
from __future__ import annotations

import argparse, io, json, os, shutil, subprocess, sys, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prosodia.config import RunConfig  # noqa: E402
from prosodia.device import get_device  # noqa: E402
from prosodia.features import SAMPLE_RATE, FeatureExtractor  # noqa: E402
from prosodia.labels.acoustic import (  # noqa: E402
    LOUDNESS_NAMES, PITCH_NAMES, RATE_NAMES, TertileBinner, mean_loudness_db,
    pitch_slope_semitones_per_second, speaking_rate_wps,
)
from prosodia.labels.bank import demo_question_specs  # noqa: E402
from prosodia.model.prosodia import ProsodiaModel  # noqa: E402
from prosodia.schema import QuestionSpec  # noqa: E402
from prosodia.train.loop import load_checkpoint  # noqa: E402

NOUL_NAMES = ["no", "yes"]
WHISPER_DIM = 768          # openai/whisper-small encoder hidden size
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENTS = ["negative", "neutral", "positive"]


def meld_question_specs() -> list[QuestionSpec]:
    """The three MELD questions, inlined.

    Imported from MeldCorpus in the local server, which dragged the corpus
    loader (and therefore the corpus) into a deployment that needs neither.
    """
    return [
        QuestionSpec(key="emotion", qtype="choice",
                     instructions="Which emotion is the speaker expressing?",
                     criteria={e: None for e in EMOTIONS}),
        QuestionSpec(key="sentiment", qtype="score",
                     instructions="Rate the sentiment the speaker conveys.",
                     criteria=list(SENTIMENTS)),
        QuestionSpec(key="is_negative", qtype="noul",
                     instructions="Is the speaker expressing something negative?",
                     criteria={"true": "Negative sentiment",
                               "false": "Neutral or positive"}),
    ]
STATE: dict = {}


def ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for c in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.exists(c):
            return c
    raise FileNotFoundError("ffmpeg not found; needed to decode browser audio")


def to_wav16k(blob: bytes) -> np.ndarray:
    """Browser MediaRecorder gives webm/opus; the model wants 16k mono float32."""
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in", Path(td) / "out.wav"
        src.write_bytes(blob)
        r = subprocess.run(
            [ffmpeg_bin(), "-y", "-i", str(src), "-ac", "1",
             "-ar", str(SAMPLE_RATE), "-f", "wav", str(dst)],
            capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode()[-400:])
        import soundfile as sf
        wav, _ = sf.read(str(dst), dtype="float32")
    return wav if wav.ndim == 1 else wav.mean(axis=1)


def nearest_trained(instructions: str) -> dict:
    """Which TRAINED question is this typed question closest to?

    The model accepts any question text, so it will always answer -- fluently
    and confidently -- including for questions nothing in training resembles.
    Without this the demo looks like zero-shot generalisation when the real
    mechanism is nearest-neighbour in a frozen MiniLM space.

    Surfacing the nearest trained question and its cosine similarity makes
    that mechanism visible and the generalisation claim testable. A low
    similarity is the signal that an answer is unsupported, not impressive.
    """
    qe = STATE["model"].question_encoder
    vecs = qe._raw([instructions] + STATE["trained_instructions"])
    v = vecs[0] / (vecs[0].norm() + 1e-9)
    rest = vecs[1:] / (vecs[1:].norm(dim=-1, keepdim=True) + 1e-9)
    sims = (rest @ v)
    i = int(sims.argmax())
    return {"key": STATE["trained_keys"][i], "similarity": float(sims[i])}


def build_spec(q: dict) -> QuestionSpec:
    opts = [o.strip() for o in q["options"] if o.strip()]
    if len(opts) < 2:
        raise ValueError(f"question {q.get('key')!r} needs at least 2 options")
    return QuestionSpec(key=q["key"], qtype="choice",
                        instructions=q["instructions"],
                        criteria={o: None for o in opts})


def predict(wav: np.ndarray, specs=None) -> dict:
    feats = STATE["whisper"].encode(wav)
    pros = STATE["prosody"].encode(wav)

    specs = specs or STATE["specs"]
    batch = {
        "uid": ["live"],
        "audio": feats.unsqueeze(0),
        "audio_mask": torch.ones(1, feats.shape[0], dtype=torch.bool),
        "audio_present": torch.tensor([True]),
        "context": [""],
        "context_present": torch.tensor([False]),
        "questions": [{s.key: {"instructions": s.instructions,
                               "options": s.options, "qtype": s.qtype}
                       for s in specs}],
        "targets": [{}],
        "gold": [{}],
        "speaker": [None],
    }
    dev = STATE["device"]
    batch["audio"] = batch["audio"].to(dev)
    batch["audio_mask"] = batch["audio_mask"].to(dev)
    with torch.no_grad():
        out = STATE["model"](batch)[0]

    # deterministic truth for the acoustic questions, on THIS clip
    truth: dict[str, str | None] = {}
    slope = pitch_slope_semitones_per_second(pros)
    truth["pitch_direction"] = STATE["bins"]["pitch"](slope) if slope is not None else None
    truth["loudness"] = STATE["bins"]["loud"](mean_loudness_db(pros))
    truth["speaking_rate"] = None       # needs a word count we do not have live

    qs = {}
    for s in specs:
        if s.key not in out:
            continue
        probs = torch.softmax(out[s.key].detach().float().cpu(), -1).tolist()
        opts = (list(s.criteria.keys()) if s.qtype == "choice"
                else list(s.criteria) if s.qtype == "score" else NOUL_NAMES)
        qs[s.key] = {"instructions": s.instructions, "options": opts,
                     "probs": probs, "truth": truth.get(s.key),
                     "nearest": nearest_trained(s.instructions)}
    return {"questions": qs, "duration_s": len(wav) / SAMPLE_RATE}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        body = STATE["page"].encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw)
            import base64 as _b64
            blob = _b64.b64decode(body["audio"])
            specs = ([build_spec(q) for q in body["questions"]]
                     if body.get("questions") else None)
            result = predict(to_wav16k(blob), specs)
        except Exception as exc:                       # surface, never swallow
            result = {"error": f"{type(exc).__name__}: {exc}"}
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--bins", type=Path, required=True)
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args()

    specs = meld_question_specs() + demo_question_specs()
    cfg = RunConfig(name="live", encoder="whisper", brier_weight=0.5)

    # Declared, not derived. The local server read in_dim off a training
    # batch, which meant standing up the demo required the 1.3 GB corpus and
    # the 7.3 GB feature cache. WHISPER_DIM is a property of
    # openai/whisper-small's encoder, so a mismatch means the checkpoint was
    # trained on a different encoder -- and load_checkpoint raises on the
    # resulting shape mismatch rather than silently half-loading.
    in_dim = WHISPER_DIM

    device = get_device()
    model = ProsodiaModel(in_dim=in_dim, d_model=cfg.d_model,
                          state_layers=cfg.state_layers,
                          branch_layers=cfg.branch_layers,
                          n_heads=cfg.n_heads, stride=cfg.stride)
    load_checkpoint(args.ckpt, model)
    model.to(device).eval()

    # Tertile boundaries must be the TRAIN-fitted ones the model was trained
    # against; refitting on live clips would move the goalposts per recording.
    if not args.bins.exists():
        raise SystemExit(
            f"missing {args.bins} -- the live demo must score against the SAME "
            "train-fitted thresholds the model was trained on. Refitting per "
            "recording would define the truth from the clip being judged and "
            "make agreement unfalsifiable.")
    bins = json.loads(args.bins.read_text())

    STATE.update(
        specs=specs,
        trained_keys=[s.key for s in specs],
        trained_instructions=[s.instructions for s in specs], model=model, device=device,
        whisper=FeatureExtractor("whisper", device),
        prosody=FeatureExtractor("prosody", device),
        page=args.page.read_text(),
        bins={
            "pitch": TertileBinner(bins["pitch"][0], bins["pitch"][1], PITCH_NAMES),
            "loud": TertileBinner(bins["loud"][0], bins["loud"][1], LOUDNESS_NAMES),
        },
    )
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
