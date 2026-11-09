# Phase 2 (L-NER) post-processing learned ONLY from the training annotations (pure Python, no model):
#   1. precision filter  - drop a predicted (string, label) that the training annotators almost never label
#                          (e.g. the bare word "Tribunal" predicted as AUTH)
#   2. gazetteer         - add (string, label) pairs that are (almost) always annotated in the training documents
#   3. year rule         - STAT: include / exclude a following ", 1996" as the training annotations do
#   4. citation rule     - PREC: include / exclude a following law-report citation "(2004) 9 SCC 27", "AIR 1962 SC 605"
# Each rule is switched on only if it improves strict macro-F1 on the validation documents.
import bisect
import re
from collections import Counter, defaultdict


def _word_boundary(t, i, j):
    return (i == 0 or not t[i - 1].isalnum()) and (j == len(t) or not t[j].isalnum())


class TrainStats:
    """how often each string occurs in the training documents, and how often it is annotated with each label"""

    def __init__(self, texts, gold):
        self.texts = [texts[d] for d in gold]
        self.gold = Counter((texts[d][s:e], l) for d in gold for s, e, l in gold[d])
        self.by_string = defaultdict(Counter)
        for (s, l), n in self.gold.items():
            self.by_string[s][l] += n
        self._occ = {}

    def occ(self, s):
        """whole-word occurrences of s in the training documents"""
        if s not in self._occ:
            n = 0
            for t in self.texts:
                i = t.find(s)
                while i >= 0:
                    n += _word_boundary(t, i, i + len(s))
                    i = t.find(s, i + 1)
            self._occ[s] = n
        return self._occ[s]


def precision_filter(pred, texts, st, min_occ=5, max_ratio=0.15):
    out, dropped = {}, Counter()
    for d, spans in pred.items():
        keep = []
        for s, e, l in spans:
            t = texts[d][s:e]
            o = st.occ(t)
            if o >= min_occ and st.gold[(t, l)] / o < max_ratio:
                dropped[(t, l)] += 1
                continue
            keep.append((s, e, l))
        out[d] = keep
    return out, dropped


def gazetteer_entries(st, min_count=3, min_ratio=0.8, min_chars=4):
    out = []
    for s, labs in st.by_string.items():
        l, n = labs.most_common(1)[0]
        if n >= min_count and len(s) >= min_chars and re.search(r"[A-Za-z]", s):
            o = st.occ(s)
            if o and n / o >= min_ratio:
                out.append((s, l))
    return sorted(out, key=lambda x: -len(x[0]))          # longest first


def apply_gazetteer(pred, texts, entries):
    out, added = {}, 0
    for d, spans in pred.items():
        t = texts[d]
        cov = sorted((s, e) for s, e, _ in spans)
        starts = [s for s, _ in cov]
        new = list(spans)
        for string, l in entries:
            i = t.find(string)
            while i >= 0:
                j = i + len(string)
                if _word_boundary(t, i, j):
                    k = bisect.bisect_left(starts, j)
                    if not any(cs < j and ce > i for cs, ce in cov[max(0, k - 30):k]):
                        new.append((i, j, l)); added += 1
                        k2 = bisect.bisect_left(starts, i); cov.insert(k2, (i, j)); starts.insert(k2, i)
                i = t.find(string, i + 1)
        out[d] = sorted(new)
    return out, added


# ---- boundary rules: learn from the training annotations whether a tail is part of the entity ----------------
YEAR_TAIL = re.compile(r",?\s?\d{4}\b")
CITE_TAIL = re.compile(r",?\s?(?:\(\d{4}\)|\[\d{4}\])\s?\d{0,3}\s?[A-Z][A-Za-z.]{0,12}\s?(?:\(\w+\)\s?)?\d{1,5}"
                       r"|,?\s?AIR\s?\d{4}\s?[A-Z][A-Za-z.]{0,12}\s?\d{1,5}")


def learn_tail(texts, gold, label, tail_re):
    """P(the gold entity includes the tail | such a tail exists right after the entity core)"""
    inc = exc = 0
    for d, spans in gold.items():
        t = texts[d]
        for s, e, l in spans:
            if l != label:
                continue
            if tail_re.match(t, e):
                exc += 1                                     # tail follows, annotators left it out
            else:
                txt = t[s:e]
                for m in tail_re.finditer(txt):
                    if m.end() == len(txt) and m.start() > 0:
                        inc += 1; break                      # entity ends with such a tail
    return inc / max(1, inc + exc), inc, exc


def apply_tail(pred, texts, label, tail_re, include):
    out, changed = {}, 0
    for d, spans in pred.items():
        t = texts[d]
        spans = sorted(spans)
        nxt_start = [s for s, _, _ in spans[1:]] + [len(t) + 1]
        new = []
        for (s, e, l), ns in zip(spans, nxt_start):
            if l == label:
                if include:
                    m = tail_re.match(t, e)
                    if m and m.end() <= ns:
                        e = m.end(); changed += 1
                else:
                    txt = t[s:e]
                    for m in tail_re.finditer(txt):
                        if m.end() == len(txt) and m.start() >= 4:
                            e2 = s + m.start()
                            while e2 > s and not t[e2 - 1].isalnum() and t[e2 - 1] not in ").":
                                e2 -= 1
                            if e2 - s >= 4:
                                e = e2; changed += 1
                            break
            new.append((s, e, l))
        out[d] = new
    return out, changed


TITLE_ROLES = ("APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE")


def consistency(pred, texts, title_chars=2500):
    """Document-level label consistency (cf. the "reconciliation" step of Kalamkar et al. 2022): every predicted
    occurrence of the same string gets ONE label - the party/counsel/judge label it has in the cause title (first
    `title_chars` characters) if any, otherwise the majority label; ties without a title label are left alone."""
    out, changed = {}, 0
    for d, spans in pred.items():
        t = texts[d]
        groups = defaultdict(list)
        for s, e, l in spans:
            groups[t[s:e]].append((s, e, l))
        new = []
        for items in groups.values():
            labs = Counter(l for _, _, l in items)
            if len(labs) == 1:
                new += items; continue
            title = Counter(l for s, _, l in items if s < title_chars and l in TITLE_ROLES)
            if title:
                lab = title.most_common(1)[0][0]
            else:
                (lab, n1), (_, n2) = labs.most_common(2)
                if n1 == n2:
                    new += items; continue
            changed += sum(l != lab for _, _, l in items)
            new += [(s, e, lab) for s, e, _ in items]
        out[d] = sorted(new)
    return out, changed


RULES = ("filter", "gazetteer", "stat_year", "prec_cite", "consistency")


def fit(texts, gold):
    """everything the rules need, learned from the training documents"""
    st = TrainStats(texts, gold)
    p_year = learn_tail(texts, gold, "STAT", YEAR_TAIL)
    p_cite = learn_tail(texts, gold, "PREC", CITE_TAIL)
    return dict(stats=st, gazetteer=gazetteer_entries(st), stat_year=p_year, prec_cite=p_cite)


def apply(pred, texts, model, rules):
    log = {}                                # order: repair boundaries first, then filter, then add
    if "stat_year" in rules:
        pred, c = apply_tail(pred, texts, "STAT", YEAR_TAIL, model["stat_year"][0] >= 0.5); log["stat_year_changed"] = c
    if "prec_cite" in rules:
        pred, c = apply_tail(pred, texts, "PREC", CITE_TAIL, model["prec_cite"][0] >= 0.5); log["prec_cite_changed"] = c
    if "consistency" in rules:
        pred, c = consistency(pred, texts); log["consistency_relabelled"] = c
    if "filter" in rules:
        pred, dr = precision_filter(pred, texts, model["stats"]); log["filter_dropped"] = sum(dr.values())
    if "gazetteer" in rules:
        pred, a = apply_gazetteer(pred, texts, model["gazetteer"]); log["gazetteer_added"] = a
    return pred, log
