import json, csv, re, math, collections
EMO=["neutral","joy","sadness","fear","anger","surprise","disgust"]
def norm(s):
    s=s.replace("\x92","'").replace("\x93",'"').replace("\x94",'"').replace("\x85","...")
    s=s.replace("’","'").replace("‘","'")
    return re.sub(r"[^a-z0-9]+"," ", s.lower()).strip()
pool={}
for f in ["el/Friends/friends.json","el/2019_Eval_Labeled/friends_eval_gold.json",
          "el/2018_EmotionX/friends.train.json","el/2018_EmotionX/friends.dev.json",
          "el/2018_EmotionX/friends.test.json"]:
    for dlg in json.load(open(f)):
        for u in dlg:
            a=u.get("annotation")
            if a: pool.setdefault((norm(u["speaker"]), norm(u["utterance"])), a)

def build(split):
    out=[]
    for r in csv.DictReader(open(f"meld_csv/{split}_sent_emo.csv",encoding="utf-8",errors="replace")):
        a=pool.get((norm(r["Speaker"]), norm(r["Utterance"])))
        if not a: continue
        v=[int(c) for c in a]; n=sum(v)
        if n==0: continue
        q=[c/n for c in v]
        H=-sum(p*math.log(p) for p in q if p>0)
        out.append(dict(uid=f"meld-{split}-{r['Dialogue_ID']}-{r['Utterance_ID']}",
                        votes=v, n=n, q=q, H=H,
                        q_argmax=EMO[max(range(7), key=lambda i:q[i])],
                        meld=r["Emotion"], sent=r["Sentiment"]))
    return out

if __name__=="__main__":
    te=build("test")
    json.dump(te, open("qtext_test.json","w"))
    print(f"test utterances with votes: {len(te)}")
    print(f"annotators per utt: {collections.Counter(x['n'] for x in te).most_common()}")
    agree=sum(1 for x in te if x["q_argmax"]==x["meld"])
    print(f"\nargmax(q_text) == MELD audio-visual label: {agree}/{len(te)} = {agree/len(te)*100:.1f}%")
    unan=[x for x in te if max(x['q'])==1.0]
    print(f"unanimous text annotators: {len(unan)} ({len(unan)/len(te)*100:.1f}%)")
    ua=sum(1 for x in unan if x['q_argmax']==x['meld'])
    print(f"   of those, MELD AV label agrees: {ua}/{len(unan)} = {ua/len(unan)*100:.1f}%")
    Hs=sorted(x["H"] for x in te)
    qs=[Hs[int(p*len(Hs))] for p in (0.25,0.5,0.75)]
    print(f"\nH(q_text) quartiles: p25={qs[0]:.3f}  p50={qs[1]:.3f}  p75={qs[2]:.3f}  max={Hs[-1]:.3f}")
    lo=[x for x in te if x["H"]<=qs[1]]; hi=[x for x in te if x["H"]>qs[1]]
    for nm,g in (("text-CLEAR (H<=median)",lo),("text-AMBIGUOUS (H>median)",hi)):
        ag=sum(1 for x in g if x["q_argmax"]==x["meld"])
        mj=collections.Counter(x["meld"] for x in g).most_common(1)[0]
        print(f"  {nm:<26} n={len(g):<5} text/AV agree={ag/len(g)*100:5.1f}%  majority={mj[0]} {mj[1]/len(g)*100:.1f}%")
