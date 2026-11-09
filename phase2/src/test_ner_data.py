import os
import sys, pickle
ROOT = os.environ.get("PHASE2_DATA", "data")
sys.path.insert(0, ROOT + r"\code")
import ner_data as nd

text = ("REPORTABLE IN THE SUPREME COURT OF INDIA CIVIL APPELLATE JURISDICTION Civil Appeal No. 4512 of 2009\n\n"
        "M/s. Ram Kumar & Sons,  ...Appellant(s)\nVersus\nState of U.P. & Ors.  ...Respondent(s)\n"
        "Heard Sri A.K. Mishra, learned counsel. Section 302 of the Indian Penal Code, 1860 applies; "
        "see K.M. Nanavati v. State of Maharashtra, AIR 1962 SC 605. Dated: 12.03.2011. न्यायालय (Hon'ble Justice R. Banumathi)"
        " Later Ram Kumar & Sons again; also Ram Kumar & Sonsx.")
def span(sub, lab, k=0):
    s = -1
    for _ in range(k + 1): s = text.index(sub, s + 1)
    return (s, s + len(sub), lab)
gold = [span("SUPREME COURT OF INDIA", "COURT"), span("4512 of 2009", "CASENO"), span("M/s. Ram Kumar & Sons", "APP"),
        span("State of U.P. & Ors.", "RESP"), span("A.K. Mishra", "A.COUNSEL"), span("Section 302", "STAT"),
        span("Indian Penal Code, 1860", "STAT"), span("K.M. Nanavati v. State of Maharashtra, AIR 1962 SC 605", "PREC"),
        span("12.03.2011", "DATE"), span("R. Banumathi", "JUDGE")]
labels = ["APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT", "AUTH", "COURT", "STAT", "PREC", "DATE", "CASENO"]
tags, tag2id = nd.tag_vocab(labels); id2tag = dict(enumerate(tags)); label2id = {l: i + 1 for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
G = {"d": gold}
for name in ("a1_bpe8000.pkl", "space_bpe8000.pkl"):
    tok = pickle.load(open(ROOT + r"\tokenizers\\" + name, "rb"))
    for split in ("bpe", "fine"):
        ids, wtok, wcs, wce = nd.encode_words(tok, text, split)
        assert tok.decode(ids) == text
        assert all(wcs[i] < wce[i] for i in range(len(wcs))) and all(wce[i] <= wcs[i + 1] for i in range(len(wcs) - 1))
        assert all(text[wcs[w]:wce[w]].startswith(tok.decode([ids[wtok[w]]]).strip()[:1]) for w in range(len(wcs)) if text[wcs[w]:wce[w]].isascii())
        t, wsp, conf = nd.word_targets(wcs, wce, gold, tag2id, label2id)
        rec = nd.spans_from_tags(t, wcs, wce, id2tag)
        cands = {(i, j): (1.0, l) for i, j, l in wsp}
        rec2 = nd.spans_from_candidates(cands, wcs, wce, id2label)
        f1 = nd.strict_scores({"d": rec}, G, labels)["macro_F1"]; f2 = nd.strict_scores({"d": rec2}, G, labels)["macro_F1"]
        print(f"{name:18} {split:4}: {len(ids)} tokens {len(wcs)} words | BIO round-trip {f1:.3f} | span round-trip {f2:.3f} | "
              f"max entity words {max(j - i + 1 for i, j, _ in wsp)}")
ok, start_ok = nd.allowed_transitions(tags)
print("O->I-APP", ok[0][tag2id["I-APP"]], "B-APP->I-APP", ok[tag2id["B-APP"]][tag2id["I-APP"]],
      "B-APP->I-RESP", ok[tag2id["B-APP"]][tag2id["I-RESP"]], "start I-APP", start_ok[tag2id["I-APP"]])
# lenient repair
t = [tag2id["I-APP"], tag2id["I-APP"], 0, tag2id["B-RESP"], tag2id["I-APP"]]
print("repair:", nd.spans_from_tags(t, [0, 5, 10, 15, 20], [4, 9, 14, 19, 24], id2tag))
# propagation
p = nd.propagate([span("M/s. Ram Kumar & Sons", "APP")], text)
print("propagate:", [(text[s:e], l) for s, e, l in p])
print("confusion:", nd.label_confusion({"d": [(gold[0][0], gold[0][1], "AUTH")]}, G))
