"""Arm D verdict: did class weighting recover the tail, or just trade precision for recall?

Macro-F1 alone cannot tell those apart -- a flat macro-F1 can hide a
redistribution, and a rising one can be bought entirely with recall at
near-zero precision. Both are reported per class.

Slot order at EVAL is the spec's canonical order (permute_candidates is
gated behind `augment`, which is train-only), so slot index IS the class.
"""
import json, os, sys
from stratify import softmax

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
TRAIN_N = {"anger":1109,"disgust":271,"fear":268,"joy":1743,
           "neutral":4710,"sadness":683,"surprise":1205}

def prf(rows, K):
    preds=[max(range(len(p)),key=lambda i:p[i]) for p,_ in rows]
    tgts=[t for _,t in rows]
    out={}
    for c in range(K):
        tp=sum(1 for p,t in zip(preds,tgts) if p==c and t==c)
        fp=sum(1 for p,t in zip(preds,tgts) if p==c and t!=c)
        fn=sum(1 for p,t in zip(preds,tgts) if p!=c and t==c)
        prec=tp/(tp+fp) if tp+fp else 0.0
        rec =tp/(tp+fn) if tp+fn else 0.0
        f1  =2*prec*rec/(prec+rec) if prec+rec else 0.0
        out[c]=(prec,rec,f1,tp+fn,tp+fp)
    return out

def load(arm):
    p=f"preds/{arm}.json"
    if not os.path.exists(p): return None
    d=json.load(open(p))["test"]
    return [(softmax(v["emotion"]["logits"]), v["emotion"]["target"]) for v in d.values() if "emotion" in v]

for enc in ["whisper","prosody","wavlm"]:
    a, dd = load(f"{enc}__A-ce"), load(f"{enc}__D-balanced")
    if a is None or dd is None:
        print(f"{enc}: missing dump, skipping"); continue
    pa, pd = prf(a,7), prf(dd,7)
    print(f"\n{'='*94}\n{enc.upper()}  —  A-ce (unweighted)  vs  D-balanced (inverse-frequency)\n{'='*94}")
    print(f"{'class':<10}{'train n':>8}   {'--- A-ce ---':^24}   {'--- D-balanced ---':^24}   {'F1':>7}")
    print(f"{'':<10}{'':>8}   {'prec':>7}{'rec':>8}{'F1':>8}   {'prec':>7}{'rec':>8}{'F1':>8}   {'delta':>7}")
    print("-"*94)
    for c,name in enumerate(EMOTIONS):
        (p1,r1,f1,sup,_)=pa[c]; (p2,r2,f2,_,npred2)=pd[c]
        flag=""
        if f2-f1 > 0.05: flag="  <-- recovered"
        elif f1 < 0.05 and f2 < 0.05: flag="  <-- still dead"
        print(f"{name:<10}{TRAIN_N[name]:>8}   {p1:>7.3f}{r1:>8.3f}{f1:>8.3f}   {p2:>7.3f}{r2:>8.3f}{f2:>8.3f}   {f2-f1:>+7.3f}{flag}")
    ma=sum(v[2] for v in pa.values())/7; md=sum(v[2] for v in pd.values())/7
    print(f"{'macro-F1':<10}{'':>8}   {'':>7}{'':>8}{ma:>8.3f}   {'':>7}{'':>8}{md:>8.3f}   {md-ma:>+7.3f}")
