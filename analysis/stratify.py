"""Stratified re-analysis: does audio help most where the transcript is ambiguous?

Ambiguity comes from EmotionLines' five text-only annotators (vote counts
joined to MELD on speaker+utterance, 2610/2610 test coverage). Two
stratifiers:
  H       -- entropy of the 5-annotator vote distribution
  agree   -- whether argmax(q_text) matches MELD's audio-visual label

Arm C is derived here exactly as the grid does: temperature fit on DEV
logits, applied to TEST. The 1-D NLL objective is convex in log T, so a
golden-section search reaches the same optimum as the grid's LBFGS.
"""
import json, glob, math, os, random

QS = ["emotion", "sentiment", "is_negative"]

def softmax(z, T=1.0):
    m = max(z); e = [math.exp((v - m) / T) for v in z]; s = sum(e)
    return [v / s for v in e]

def nll_at(rows, T):
    return sum(-math.log(max(softmax(lg, T)[t], 1e-12)) for lg, t in rows) / len(rows)

def fit_temperature(rows):
    lo, hi = math.log(0.05), math.log(20.0)
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(80):
        if nll_at(rows, math.exp(c)) < nll_at(rows, math.exp(d)): b = d
        else: a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return math.exp((a + b) / 2)

def brier_of(p, t):
    return sum((pi - (1.0 if i == t else 0.0)) ** 2 for i, pi in enumerate(p))

def agg(items):
    n = len(items)
    if not n: return None
    acc = sum(1 for p, t in items if max(range(len(p)), key=lambda i: p[i]) == t) / n
    br = sum(brier_of(p, t) for p, t in items) / n
    return dict(n=n, accuracy=acc, brier=br)

def boot_diff(a_items, b_items, iters=2000, seed=0):
    """Paired bootstrap on mean Brier difference (a - b); same indices both arms."""
    rng = random.Random(seed); n = len(a_items)
    da = [brier_of(p, t) for p, t in a_items]
    db = [brier_of(p, t) for p, t in b_items]
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(da[i] - db[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]

# ---- load ----
qt = {x["uid"]: x for x in json.load(open("qtext_test.json"))}
Hs = sorted(x["H"] for x in qt.values()); med = Hs[len(Hs) // 2]

preds = {}
for f in glob.glob("preds/*.json"):
    preds[os.path.basename(f)[:-5]] = json.load(open(f))

# derive Arm C from each A arm
for arm in [a for a in list(preds) if a.endswith("__A-ce")]:
    base = arm[:-6]
    dev, test = preds[arm]["dev"], preds[arm]["test"]
    newtest = {}
    for q in QS:
        rows = [(v[q]["logits"], v[q]["target"]) for v in dev.values() if q in v]
        T = fit_temperature(rows)
        print(f"  {base}__C-temp  {q:<12} T={T:.4f}")
        for uid, v in test.items():
            if q in v:
                newtest.setdefault(uid, {})[q] = {"logits": [z / T for z in v[q]["logits"]],
                                                  "target": v[q]["target"]}
    preds[f"{base}__C-temp"] = {"test": newtest}

def items_for(arm, q, uids):
    d = preds[arm]["test"]
    return [(softmax(d[u][q]["logits"]), d[u][q]["target"]) for u in uids if u in d and q in d[u]]

STRATA = {
    "text-CLEAR  (H<=med)":  [u for u, x in qt.items() if x["H"] <= med],
    "text-AMBIG  (H> med)":  [u for u, x in qt.items() if x["H"] > med],
    "text-SUFFICIENT (argmax q = AV label)": [u for u, x in qt.items() if x["q_argmax"] == x["meld"]],
    "text-MISLEADING (argmax q != AV label)": [u for u, x in qt.items() if x["q_argmax"] != x["meld"]],
}

ARMS = ["wavlm__B-brier", "whisper__C-temp", "whisper__B-brier", "prosody__B-brier"]
BASE = "text_only__B-brier"

for q in QS:
    print(f"\n{'='*94}\n{q.upper()}   — Brier vs the text-only baseline, by stratum (negative = audio better)\n{'='*94}")
    print(f"{'stratum':<40}{'n':>6}{'text_only':>11}" + "".join(f"{a.split('__')[0][:7]:>11}" for a in ARMS))
    for name, uids in STRATA.items():
        b = agg(items_for(BASE, q, uids))
        cells = []
        for a in ARMS:
            m = agg(items_for(a, q, uids))
            cells.append(f"{m['brier']-b['brier']:>+11.4f}")
        print(f"  {name:<38}{b['n']:>6}{b['brier']:>11.4f}" + "".join(cells))

print(f"\n{'='*94}\nKEY CONTRAST — whisper__C-temp minus text_only__B-brier, with 95% paired bootstrap CI\n{'='*94}")
for q in QS:
    print(f"\n  {q}:")
    for name in ["text-CLEAR  (H<=med)", "text-AMBIG  (H> med)",
                 "text-SUFFICIENT (argmax q = AV label)", "text-MISLEADING (argmax q != AV label)"]:
        uids = STRATA[name]
        ai = items_for("whisper__C-temp", q, uids); bi = items_for(BASE, q, uids)
        d = agg(ai)["brier"] - agg(bi)["brier"]
        lo, hi = boot_diff(ai, bi)
        sig = "  *" if hi < 0 or lo > 0 else ""
        print(f"    {name:<40} n={len(uids):<5} dBrier={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{sig}")
