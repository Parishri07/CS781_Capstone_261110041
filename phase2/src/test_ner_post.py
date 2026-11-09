import ner_post as npp, ner_data as nd
LABELS = ["APP","RESP","A.COUNSEL","R.COUNSEL","JUDGE","WIT","AUTH","COURT","STAT","PREC","DATE","CASENO"]
base = ("The Tribunal held. The Industrial Tribunal Delhi said. Section 34 of the Arbitration and Conciliation Act, 1996 applies. "
        "See R. Prakash v. State of Karnataka (2004) 9 SCC 27 and K.M. Nanavati v. State of Maharashtra, AIR 1962 SC 605. "
        "The High Court of Punjab and Haryana. ")
train_texts = {f"t{i}": base * 3 for i in range(4)}
def find_all(t, s):
    i = t.find(s); out = []
    while i >= 0: out.append(i); i = t.find(s, i + 1)
    return out
train_gold = {}
for d, t in train_texts.items():
    g = []
    for i in find_all(t, "Industrial Tribunal Delhi"): g.append((i, i + 25, "AUTH"))
    for i in find_all(t, "Arbitration and Conciliation Act, 1996"): g.append((i, i + 38, "STAT"))
    s = "R. Prakash v. State of Karnataka (2004) 9 SCC 27"
    for i in find_all(t, s): g.append((i, i + len(s), "PREC"))
    s = "K.M. Nanavati v. State of Maharashtra, AIR 1962 SC 605"
    for i in find_all(t, s): g.append((i, i + len(s), "PREC"))
    for i in find_all(t, "High Court of Punjab and Haryana"): g.append((i, i + 32, "COURT"))
    train_gold[d] = sorted(g)
M = npp.fit(train_texts, train_gold)
print("year rule P(include)", M["stat_year"], "| cite rule", M["prec_cite"], "| gazetteer", M["gazetteer"][:4])

# a test document and a "model" prediction with the typical errors
t = base
gold = {"x": [g for g in train_gold["t0"] if g[0] < len(base)]}
pred = []
pred.append((t.index("Tribunal"), t.index("Tribunal") + 8, "AUTH"))                         # false positive
i = t.index("Arbitration and Conciliation Act"); pred.append((i, i + 32, "STAT"))           # too short (no year)
i = t.index("R. Prakash"); pred.append((i, i + len("R. Prakash v. State of Karnataka"), "PREC"))   # too short (no citation)
i = t.index("K.M. Nanavati"); pred.append((i, i + len("K.M. Nanavati v. State of Maharashtra"), "PREC"))
pred = {"x": pred}
texts = {"x": t}
print("before:", nd.strict_scores(pred, gold, LABELS)["micro_F1"], [(t[s:e], l) for s, e, l in pred["x"]])
for rules in [("filter",), ("stat_year",), ("prec_cite",), ("gazetteer",), npp.RULES]:
    p, log = npp.apply(pred, texts, M, rules)
    print(rules, "->", round(nd.strict_scores(p, gold, LABELS)["micro_F1"], 3), log)
p, _ = npp.apply(pred, texts, M, npp.RULES)
print([(t[s:e], l) for s, e, l in p["x"]])
