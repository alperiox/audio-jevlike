import json, glob, math, os
def softmax(z):
    m=max(z); e=[math.exp(v-m) for v in z]; s=sum(e); return [v/s for v in e]
def metrics(rows, nbins=10):
    n=len(rows); acc=0; brier=0.0; nll=0.0; bins=[[0,0.0,0] for _ in range(nbins)]
    for lg,t in rows:
        p=softmax(lg); pred=max(range(len(p)), key=lambda i:p[i])
        ok = 1 if pred==t else 0; acc+=ok
        brier += sum((pi-(1.0 if i==t else 0.0))**2 for i,pi in enumerate(p))
        nll += -math.log(max(p[t],1e-12))
        conf=p[pred]; b=min(int(conf*nbins), nbins-1)
        bins[b][0]+=ok; bins[b][1]+=conf; bins[b][2]+=1
    ece=sum(abs(c/cnt - s/cnt)*cnt/n for c,s,cnt in bins if cnt)
    return dict(accuracy=acc/n, ece=ece, brier=brier/n, nll=nll/n, n=n)
if __name__=="__main__":
    pub=json.load(open("published.json")) if os.path.exists("published.json") else {}
    for f in sorted(glob.glob("preds/*.json")):
        arm=os.path.basename(f)[:-5]; d=json.load(open(f))["test"]
        print(f"\n{arm}")
        for q in ["emotion","sentiment","is_negative"]:
            rows=[(v[q]["logits"], v[q]["target"]) for v in d.values() if q in v]
            m=metrics(rows)
            p=pub.get(arm,{}).get(q)
            tag=""
            if p: tag=f"   published acc={p['accuracy']:.4f} ece={p['ece']:.4f} brier={p['brier']:.4f}"
            print(f"  {q:<12} acc={m['accuracy']:.4f} ece={m['ece']:.4f} brier={m['brier']:.4f} n={m['n']}{tag}")
