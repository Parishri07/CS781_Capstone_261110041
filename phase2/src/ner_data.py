# Phase 2 (L-NER) data utilities (pure Python, no torch):
#   gold parsing, words with exact character offsets, word-level BIO tags and span targets,
#   span decoding strategies, document-level label propagation, strict scoring, error analysis, submission.
import bisect
import re
from collections import Counter, defaultdict

# ======================================================================================================
# 1. Gold annotations
# ======================================================================================================
_START = ("start", "start_char", "start_offset", "begin", "s")
_END = ("end", "end_char", "end_offset", "stop", "e")
_LABEL = ("label", "type", "entity_type", "tag", "category")
_TEXT = ("text", "entity_text", "span", "mention", "value")
_IDS = ("id", "doc_id", "document_id", "docid")
_ENTS = ("entities", "annotations", "spans", "labels", "gold")


def _pick(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _entity(item):
    """one annotation in any common shape -> (start, end, label, text or None)"""
    if isinstance(item, (list, tuple)):
        s, e, lab = item[0], item[1], item[2]
        txt = item[3] if len(item) > 3 else None
    elif isinstance(item, dict):
        s, e, lab, txt = _pick(item, _START), _pick(item, _END), _pick(item, _LABEL), _pick(item, _TEXT)
    else:
        raise ValueError(f"unknown annotation format: {item!r}")
    return int(s), int(e), str(lab), (txt if isinstance(txt, str) else None)


def parse_gold(raw):
    """gold json (dict doc -> list, dict doc -> {entities: [...]}, or list of records) -> {doc: [(s,e,lab,txt)]}"""
    out = {}
    if isinstance(raw, list):
        for rec in raw:
            out[str(_pick(rec, _IDS))] = [_entity(x) for x in (_pick(rec, _ENTS) or [])]
        return out
    for doc, v in raw.items():
        if isinstance(v, dict):
            v = _pick(v, _ENTS) or []
        out[str(doc)] = [_entity(x) for x in v]
    return out


def parse_texts(raw):
    """text json -> {doc: text}"""
    if isinstance(raw, list):
        return {str(_pick(r, _IDS)): _pick(r, ("text", "content", "document")) for r in raw}
    return {str(k): (v if isinstance(v, str) else _pick(v, ("text", "content", "document"))) for k, v in raw.items()}


def end_convention(texts, gold):
    """Is `end` exclusive (text[s:e] == entity text) or inclusive (text[s:e+1])? Decided by the data."""
    exc = inc = total = 0
    bad = []
    for doc, ents in gold.items():
        t = texts.get(doc)
        if t is None:
            continue
        for s, e, lab, txt in ents:
            if txt is None:
                continue
            total += 1
            if t[s:e] == txt:
                exc += 1
            elif t[s:e + 1] == txt:
                inc += 1
            elif len(bad) < 5:
                bad.append((doc, s, e, lab, txt, t[s:e + 1]))
    conv = "inclusive" if inc > exc else "exclusive"
    return conv, dict(total_with_text=total, exclusive_match=exc, inclusive_match=inc, examples_no_match=bad)


def to_exclusive(gold, conv):
    """-> {doc: [(s, e_exclusive, label)]} (internal convention: python slice)"""
    add = 1 if conv == "inclusive" else 0
    return {d: [(s, e + add, lab) for s, e, lab, _ in ents] for d, ents in gold.items()}


# ======================================================================================================
# 2. Tokens and words with character offsets
# ======================================================================================================
# The pre-tokenisation decides where an entity can start or end. "bpe" = the split used in pretraining
# (a word keeps its punctuation: "Singh," is one piece). "fine" = punctuation is split off words, so
# "Singh" | "," can be cut exactly. Every piece is still encoded by the same BPE (same vocabulary).
SPLITS = {
    ("space", "bpe"): re.compile(r" ?\S+|\s+(?!\S)|\s+"),
    ("space", "fine"): re.compile(r" ?[^\W_]+| ?_+| ?[^\w\s]|\s+(?!\S)|\s+"),
    ("a1", "bpe"): re.compile(r"\S+|\s+"),
    ("a1", "fine"): re.compile(r"[^\W_]+|_+|[^\w\s]|\s+"),
}


def tokenizer_kind(tok):
    return "space" if getattr(tok, "space_prefix", False) else "a1"


def encode_words(tok, text, split="fine"):
    """text -> ids, wtok, wcs, wce
    ids  : BPE token ids of the whole text (decode(ids) == text)
    words: every pre-token piece that is not pure whitespace. For word w:
           wtok[w] = index of its first non-whitespace token (the token that represents the word),
           [wcs[w], wce[w]) = its characters without surrounding whitespace."""
    pat = SPLITS[(tokenizer_kind(tok), split)]
    ids, wtok, wcs, wce = [], [], [], []
    covered = 0
    for m in pat.finditer(text):
        chunk, base = m.group(), m.start()
        assert base == covered, f"pre-tokeniser skipped characters at {covered}"
        covered = m.end()
        char_of_byte = []
        for ci, ch in enumerate(chunk):
            char_of_byte.extend([ci] * len(ch.encode("utf-8")))
        pos, first, ws, we = 0, None, None, None
        for t in tok.encode_chunk(chunk.encode("utf-8")):
            n = len(tok.itos[t])
            s, e = base + char_of_byte[pos], base + char_of_byte[pos + n - 1] + 1
            pos += n
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if s < e:
                if first is None:
                    first, ws = len(ids), s
                we = e
            ids.append(t)
        if first is not None:
            wtok.append(first); wcs.append(ws); wce.append(we)
    assert covered == len(text), "pre-tokeniser did not cover the whole text"
    return ids, wtok, wcs, wce


# ======================================================================================================
# 3. Targets: BIO tags per word, and (start word, end word, label) for the span head
# ======================================================================================================
IGNORE = -100


def tag_vocab(labels):
    tags = ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]
    return tags, {t: i for i, t in enumerate(tags)}


def word_targets(wcs, wce, spans, tag2id, label2id):
    """gold char spans -> (BIO tag per word, [(first_word, last_word, label_id)], #overlapping spans skipped)"""
    n = len(wcs)
    tags, taken, wspans, conflicts = [0] * n, [False] * n, [], 0
    for s, e, lab in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
        lo = bisect.bisect_right(wce, s)              # first word ending after s
        hi = bisect.bisect_left(wcs, e, lo)           # first word starting at/after e
        if lo >= hi:
            continue
        if any(taken[lo:hi]):
            conflicts += 1
            continue
        for k in range(lo, hi):
            tags[k] = tag2id[("B-" if k == lo else "I-") + lab]
            taken[k] = True
        wspans.append((lo, hi - 1, label2id[lab]))
    return tags, wspans, conflicts


def allowed_transitions(tags):
    """BIO validity: allowed[i][j] = may tag j follow tag i; start_ok[j] = may a sequence start with j."""
    K = len(tags)
    ok = [[True] * K for _ in range(K)]
    start_ok = [not t.startswith("I-") for t in tags]
    for i, a in enumerate(tags):
        for j, b in enumerate(tags):
            if b.startswith("I-") and not (a != "O" and a[2:] == b[2:]):
                ok[i][j] = False                      # I-X only after B-X or I-X
    return ok, start_ok


# ======================================================================================================
# 4. Decoding
# ======================================================================================================
def spans_from_tags(tags, wcs, wce, id2tag):
    """word tags -> [(s, e, label)]. B-X starts a span; I-X continues a span of type X; an I-X after O or
    after another type also starts a span (lenient repair, used with greedy argmax decoding)."""
    out, cur = [], None
    for t, s, e in zip(tags, wcs, wce):
        tag = id2tag[t]
        if tag == "O":
            if cur: out.append(cur); cur = None
            continue
        p, lab = tag.split("-", 1)
        if p == "B" or cur is None or cur[2] != lab:
            if cur: out.append(cur)
            cur = [s, e, lab]
        else:
            cur[1] = e
    if cur: out.append(cur)
    return [tuple(x) for x in out]


def spans_from_candidates(cands, wcs, wce, id2label):
    """span head: cands = {(i, j): (prob, label_id)} (label_id 0 = none) -> non-overlapping spans,
    highest probability first (greedy)."""
    used, out = set(), []
    for (i, j), (p, l) in sorted(cands.items(), key=lambda kv: -kv[1][0]):
        if l == 0 or any(k in used for k in range(i, j + 1)):
            continue
        used.update(range(i, j + 1))
        out.append((wcs[i], wce[j], id2label[l]))
    return sorted(out)


_WORDCH = re.compile(r"\w")


def propagate(spans, text, min_chars=4):
    """Document-level consistency: a string predicted as label L somewhere in the document is also
    labelled L at its other (unlabelled, whole-word) occurrences. Majority label per string."""
    votes = defaultdict(Counter)
    for s, e, l in spans:
        t = text[s:e]
        if len(t) >= min_chars and _WORDCH.search(t):
            votes[t][l] += 1
    covered = sorted((s, e) for s, e, _ in spans)
    starts = [s for s, _ in covered]

    def free(s, e):
        k = bisect.bisect_left(starts, e)
        return not any(cs < e and ce > s for cs, ce in covered[max(0, k - 50):k])

    out = list(spans)
    for t, c in votes.items():
        lab = c.most_common(1)[0][0]
        i = text.find(t)
        while i >= 0:
            j = i + len(t)
            boundary = (i == 0 or not text[i - 1].isalnum()) and (j == len(text) or not text[j].isalnum())
            if boundary and free(i, j):
                out.append((i, j, lab))
                k = bisect.bisect_left(starts, i); covered.insert(k, (i, j)); starts.insert(k, i)
            i = text.find(t, i + 1)
    return sorted(set(out))


def windows(n, size, stride):
    """window start positions covering [0, n); the last one is pulled back to end at n, so every window
    (except in documents shorter than size) has exactly `size` tokens - no padding needed."""
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


# ======================================================================================================
# 5. Strict scoring and error analysis
# ======================================================================================================
def strict_scores(pred, gold, labels):
    """pred/gold: {doc: [(s,e,label)]} -> macro/micro P/R/F1 (exact start, end and label) + per label."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for doc in set(gold) | set(pred):
        g = set(map(tuple, gold.get(doc, []))); p = set(map(tuple, pred.get(doc, [])))
        for x in p & g: tp[x[2]] += 1
        for x in p - g: fp[x[2]] += 1
        for x in g - p: fn[x[2]] += 1
    rows = {}
    for l in labels:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        rows[l] = dict(P=P, R=R, F1=F, tp=tp[l], fp=fp[l], fn=fn[l], support=tp[l] + fn[l])
    used = [l for l in labels if rows[l]["support"] or rows[l]["fp"]]
    mac = lambda k: sum(rows[l][k] for l in used) / max(1, len(used))
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro = 2 * T / (2 * T + FP + FN) if T else 0.0
    return dict(macro_F1=mac("F1"), macro_P=mac("P"), macro_R=mac("R"), micro_F1=micro, per_label=rows)


def error_breakdown(pred, gold):
    """every mistake classified: wrong label on an exact span, boundary error (overlap, same label),
    boundary+label error, missed entity (no overlapping prediction), spurious prediction."""
    c = Counter()
    for doc in set(gold) | set(pred):
        g = set(map(tuple, gold.get(doc, []))); p = set(map(tuple, pred.get(doc, [])))
        c["correct"] += len(p & g)
        ps, used = sorted(p - g), set()
        for s, e, l in sorted(g - p):
            over = [x for x in ps if x[0] < e and x[1] > s]
            if not over:
                c["missed"] += 1; continue
            used.update(over)
            if any(o[0] == s and o[1] == e for o in over): c["wrong_label_exact_span"] += 1
            elif any(o[2] == l for o in over): c["boundary_error_same_label"] += 1
            else: c["boundary_and_label_error"] += 1
        c["spurious"] += len([x for x in ps if x not in used])
    return dict(c)


def label_confusion(pred, gold):
    """exact-span matches with a different label: Counter[(gold label, predicted label)]"""
    c = Counter()
    for doc in gold:
        pm = {(s, e): l for s, e, l in pred.get(doc, [])}
        for s, e, l in gold[doc]:
            if (s, e) in pm and pm[(s, e)] != l:
                c[(l, pm[(s, e)])] += 1
    return c


# ======================================================================================================
# 6. Submission
# ======================================================================================================
def to_submission(pred, conv, texts=None):
    """{doc: [(s, e_exclusive, label)]} -> {doc: [[s, e, label(, text)]]} in the dataset's own end convention"""
    sub = 1 if conv == "inclusive" else 0
    out = {}
    for doc, spans in pred.items():
        rows = []
        for s, e, l in sorted(spans):
            row = [int(s), int(e - sub), l]
            if texts is not None: row.append(texts[doc][s:e])
            rows.append(row)
        out[doc] = rows
    return out
