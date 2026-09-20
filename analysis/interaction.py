"""The actual thesis test: is an arm's advantage over text LARGER where the
transcript is ambiguous? That is a difference-in-differences, so it needs its
own CI -- two separately-significant strata do not establish that they differ.

CLEAR and AMBIG are disjoint example sets, so each is resampled independently
and the interaction recombined per draw.
"""
import json, glob, math, os, random
from stratify import softmax, brier_of, fit_temperature, QS

qt={x["uid"]:x for x in json.load(open("qtext_test.json"))}
Hs=sorted(x["H"] for x in qt.values()); med=Hs[len(Hs)//2]
preds={os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("preds/*.json")}
for arm in [a for a in list(preds) if a.endswith("__A-ce")]:
    base=arm[:-6]; dev,test=preds[arm]["dev"],preds[arm]["test"]; nt={}
    for q in QS:
        T=fit_temperature([(v[q]["logits"],v[q]["target"]) for v in dev.values() if q in v])
        for uid,v in test.items():
            if q in v: nt.setdefault(uid,{})[q]={"logits":[z/T for z in v[q]["logits"]],"target":v[q]["target"]}
    preds[f"{base}__C-temp"]={"test":nt}

def bs(arm,q,uids):
    d=preds[arm]["test"]
    return [brier_of(softmax(d[u][q]["logits"]), d[u][q]["target"]) for u in uids if u in d and q in d[u]]

CLEAR=[u for u,x in qt.items() if x["H"]<=med]
AMBIG=[u for u,x in qt.items() if x["H"]>med]
SUFF =[u for u,x in qt.items() if x["q_argmax"]==x["meld"]]
MISL =[u for u,x in qt.items() if x["q_argmax"]!=x["meld"]]
BASE="text_only__B-brier"

def interaction(arm,q,A,B,iters=4000,seed=1):
    """(arm-text) on B  minus  (arm-text) on A."""
    rng=random.Random(seed)
    aA,bA=bs(arm,q,A),bs(BASE,q,A)
    aB,bB=bs(arm,q,B),bs(BASE,q,B)
    pt=lambda a,b: sum(a)/len(a)-sum(b)/len(b)
    obs=pt(aB,bB)-pt(aA,bA)
    out=[]
    for _ in range(iters):
        iA=[rng.randrange(len(aA)) for _ in range(len(aA))]
        iB=[rng.randrange(len(aB)) for _ in range(len(aB))]
        dA=sum(aA[i]-bA[i] for i in iA)/len(iA)
        dB=sum(aB[i]-bB[i] for i in iB)/len(iB)
        out.append(dB-dA)
    out.sort()
    return obs, out[int(.025*iters)], out[int(.975*iters)]

for label,A,B in [("H-split : AMBIG minus CLEAR", CLEAR, AMBIG),
                  ("agreement: MISLEADING minus SUFFICIENT", SUFF, MISL)]:
    print(f"\n{'='*92}\n{label}   (negative = audio's edge GROWS where text fails -> thesis prediction)\n{'='*92}")
    for q in QS:
        print(f"\n  {q}:")
        for arm in ["whisper__C-temp","whisper__B-brier","wavlm__B-brier","prosody__B-brier","prosody__A-ce"]:
            o,lo,hi=interaction(arm,q,A,B)
            sig="  *" if hi<0 or lo>0 else ""
            print(f"    {arm:<20} interaction={o:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{sig}")
