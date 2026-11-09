# Phase 2 (L-NER) experiment runner: one call = one complete experiment
# (tokenise -> build model -> adapt -> self-test -> train -> choose decoding on validation -> analyse ->
#  predict the test fold -> submission zip + record). Shared by every Phase-2 notebook.
import gc
import json
import math
import os
import time
import zipfile
from collections import Counter

import torch

import ner_data as nd
import ner_model as nm

_DOC_CACHE = {}
SCORES = {}                      # run name -> word-level probabilities (val + test), filled when cfg["keep_scores"]


def _docs(D, arch, split, tag2id, label2id):
    key = (nm.ARCH_INFO[arch][0], split, tuple(D["train_ids"]), tuple(D["val_ids"]))
    if key not in _DOC_CACHE:
        tok = nm.load_tokenizer(arch, D["root"])
        tr, conf = nm.prepare_docs({d: D["texts"][d] for d in D["train_ids"]}, D["gold"], tok, tag2id, label2id, split)
        va, _ = nm.prepare_docs({d: D["texts"][d] for d in D["val_ids"]}, D["gold"], tok, tag2id, label2id, split)
        te, _ = nm.prepare_docs(D["test_texts"], None, tok, tag2id, label2id, split)
        _DOC_CACHE[key] = (tok, tr, va, te, conf)
    return _DOC_CACHE[key]


def build(cfg, D, tags, device):
    bb = nm.Backbone(cfg["arch"], D["root"], bidirectional=cfg["bidirectional"],
                     keep_layers=cfg.get("keep_layers"), init=cfg.get("init", "pretrained"))
    net = nm.NERModel(bb, tags, D["labels"], head=cfg["head"], dropout=cfg["dropout"],
                      class_weights=cfg.get("_class_weights"), max_span_words=cfg.get("_maxw", 32),
                      neg_weight=cfg.get("span_neg_weight", 0.5))
    desc = nm.apply_method(net, cfg)
    return net.to(device), desc


def run_experiment(cfg, D, device, out_root, log=print, teacher=None, keep_net=False):
    """cfg: one experiment's settings; D: data dict (texts, gold, test_texts, labels, train_ids, val_ids,
    conv, root). teacher: a trained NERModel -> knowledge distillation into the model built from cfg."""
    t_start = time.time()
    run_dir = os.path.join(out_root, cfg["run"]); os.makedirs(run_dir, exist_ok=True)
    labels = D["labels"]
    if cfg.get("refit_all"):    # final model: train on all 70 documents (validation scores below are then optimistic)
        D = dict(D, train_ids=sorted(set(D["train_ids"]) | set(D["val_ids"])))
        cfg = dict(cfg, select_last=True, patience=0)
        log(f"REFIT on all {len(D['train_ids'])} documents for {cfg['epochs']} epochs, decoding {cfg.get('decoding')} "
            "(validation documents are now part of training: their scores are NOT an estimate)")
    tags, tag2id = nd.tag_vocab(labels)
    label2id = {l: i + 1 for i, l in enumerate(labels)}
    tok, tr, va, te, conf = _docs(D, cfg["arch"], cfg["split"], tag2id, label2id)
    wd = nm.Windowing(tok, cfg["window"], cfg["header_tokens"], nm.ARCH_INFO[cfg["arch"]][1])
    cfg = dict(cfg)
    cfg["train_stride"] = cfg.get("train_stride") or wd.size // 2
    cfg["eval_stride"] = cfg.get("eval_stride") or (wd.size * 3) // 4
    widths = sorted(j - i + 1 for r in tr.values() for i, j, _ in r["wspans"])
    if cfg["head"] == "span":
        cfg["_maxw"] = cfg.get("max_span_words") or min(48, widths[int(0.995 * (len(widths) - 1))])
        log(f"span head: max {cfg['_maxw']} words per span covers "
            f"{sum(w <= cfg['_maxw'] for w in widths) / len(widths):.2%} of training entities")
    if cfg.get("class_weighting") == "sqrt_inv":
        c = Counter(t for r in tr.values() for t in r["wtags"])
        tot = sum(c.values())
        w = torch.tensor([math.sqrt(tot / max(1, c[i])) for i in range(len(tags))]); cfg["_class_weights"] = w / w[0]
    torch.manual_seed(cfg["seed"])
    net, desc = build(cfg, D, tags, device)
    total, trainable = nm.count_params(net)
    log(f"\n=== {cfg['run']} | {cfg['arch']} | {desc} | head {cfg['head']} | bidirectional {cfg['bidirectional']} | "
        f"params {total:,}, trainable {trainable:,} ({trainable / total:.2%})")
    model = net
    if teacher is not None:
        net = nm.KDWrapper(model, teacher, T=cfg["kd_T"], alpha=cfg["kd_alpha"]).to(device)
    loss0, shape = nm.self_test(net, wd, tr, device, cfg["head"], cfg.get("_maxw"))
    log(f"self-test ok: loss {loss0:.3f}, output {shape}")

    res = nm.train(net, tr, va, D["texts"], {d: D["gold"][d] for d in D["val_ids"]}, labels, wd, cfg, device, log)

    # ---- choose the decoding strategy on the validation documents ----
    t0 = time.time()
    vsc = nm.predict_scores(net, va, wd, cfg["eval_stride"], cfg["eval_batch"], device)
    infer_sec = time.time() - t0
    vgold = {d: D["gold"][d] for d in D["val_ids"]}
    options = {}
    opts = nm.decoding_options(cfg["head"])
    if cfg.get("decoding"):                           # fixed (chosen earlier on validation), e.g. for a refit
        s_, p_ = cfg["decoding"].split("+")[0], cfg["decoding"].endswith("+propagate")
        opts = [(s_, p_)]
    for strat, prop in opts:
        p = nm.decode(vsc, va, model, D["texts"], strat, prop)
        options[f"{strat}{'+propagate' if prop else ''}"] = (nd.strict_scores(p, vgold, labels), p, (strat, prop))
    best_name = max(options, key=lambda k: options[k][0]["macro_F1"])
    VAL, vpred, (strat, prop) = options[best_name]
    log("decoding on validation: " + " | ".join(f"{k} {v[0]['macro_F1']:.4f}" for k, v in options.items())
        + f"  -> {best_name}")
    log(f"VAL strict macro-F1 {VAL['macro_F1']:.4f} | P {VAL['macro_P']:.4f} | R {VAL['macro_R']:.4f} | "
        f"micro-F1 {VAL['micro_F1']:.4f}")
    log(f"{'label':<10} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    for l in labels:
        r = VAL["per_label"][l]; log(f"{l:<10} {r['P']:6.3f} {r['R']:6.3f} {r['F1']:6.3f} {r['support']:8d}")
    json.dump(dict(decoding=best_name, pred={d: [list(x) for x in v] for d, v in vpred.items()},
                   gold={d: [list(x) for x in v] for d, v in vgold.items()}),
              open(os.path.join(run_dir, "val_predictions.json"), "w"))
    ERR = nd.error_breakdown(vpred, vgold)
    CONF = nd.label_confusion(vpred, vgold).most_common(8)
    log(f"errors: {ERR}\nmost confused labels (gold -> predicted, exact span): {CONF}")

    # ---- test fold -> submission ----
    tsc = nm.predict_scores(net, te, wd, cfg["eval_stride"], cfg["eval_batch"], device)
    if cfg.get("keep_scores"):                        # for ensembles: word-level probabilities of this model
        SCORES[cfg["run"]] = dict(val=vsc, test=tsc, va=va, te=te, tags=tags, head=cfg["head"],
                                  tok=nm.ARCH_INFO[cfg["arch"]][0], split=cfg["split"])
    tpred = nm.decode(tsc, te, model, D["test_texts"], strat, prop)
    sub = nd.to_submission(tpred, D["conv"])
    assert set(sub) == set(D["test_texts"])
    for d, rows in sub.items():
        for s, e, l in rows:
            assert l in labels and 0 <= s < e <= len(D["test_texts"][d])
    pj = os.path.join(run_dir, "prediction.json"); json.dump(sub, open(pj, "w"), ensure_ascii=False)
    zp = os.path.join(run_dir, f"submission_{cfg['run']}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"] and set(json.loads(z.read("prediction.json"))) == set(sub)
    n_ent = sum(map(len, sub.values()))
    log(f"test: {n_ent:,} entities in {len(sub)} documents -> {zp}")

    # ---- save ----
    state = nm.trainable_state(model)
    torch.save(state, os.path.join(run_dir, "trainable_weights.pt"))
    trained_mb = sum(v.numel() * v.element_size() for v in state.values()) / 2**20
    if cfg.get("save_full"):
        torch.save({"model": model.state_dict(), "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
                    "tags": tags}, os.path.join(run_dir, "ner_best.pt"))
    rec = dict(run=cfg["run"], arch=cfg["arch"], method=desc, head=cfg["head"], bidirectional=cfg["bidirectional"],
               kd=teacher is not None, cfg={k: v for k, v in cfg.items() if not k.startswith("_")},
               params_total=total, params_trainable=trainable, trainable_share=trainable / total,
               trained_weights_mb=round(trained_mb, 1), val_inference_sec=round(infer_sec, 1),
               decoding=best_name, decoding_options={k: round(v[0]["macro_F1"], 4) for k, v in options.items()},
               val=dict(macro_F1=VAL["macro_F1"], macro_P=VAL["macro_P"], macro_R=VAL["macro_R"],
                        micro_F1=VAL["micro_F1"], per_label=VAL["per_label"]),
               errors=ERR, confusion=[[a, b, n] for (a, b), n in CONF], test_entities=n_ent,
               test_label_counts=dict(Counter(l for rows in sub.values() for _, _, l in rows)),
               overlapping_gold_skipped=conf, zip=os.path.basename(zp), minutes_total=round((time.time() - t_start) / 60, 1),
               **res)
    json.dump(rec, open(os.path.join(run_dir, "record.json"), "w"), indent=1)
    log(f"saved {run_dir}/record.json, trainable_weights.pt ({trained_mb:.1f} MB) | {rec['minutes_total']} min total")
    if keep_net:
        return rec, model
    del net, model; gc.collect(); torch.cuda.empty_cache()
    return rec, None


def summary_table(records, log=print):
    log(f"\n{'run':<34} {'method':<34} {'head':<6} {'trainable':>11} {'val F1':>7} {'P':>6} {'R':>6} {'min':>5}")
    for r in records:
        log(f"{r['run']:<34} {r['method'][:34]:<34} {r['head']:<6} {r['params_trainable']:>11,} "
            f"{r['val']['macro_F1']:7.4f} {r['val']['macro_P']:6.3f} {r['val']['macro_R']:6.3f} {r['minutes_total']:5.1f}")


@torch.no_grad()
def evaluate_model(model, cfg, D, device, log=print):
    """validation macro-F1 and inference time of an already trained model (e.g. a loaded teacher)"""
    labels = D["labels"]
    tags, tag2id = nd.tag_vocab(labels)
    label2id = {l: i + 1 for i, l in enumerate(labels)}
    tok, tr, va, te, _ = _docs(D, cfg["arch"], cfg["split"], tag2id, label2id)
    wd = nm.Windowing(tok, cfg["window"], cfg["header_tokens"], nm.ARCH_INFO[cfg["arch"]][1])
    t0 = time.time()
    sc = nm.predict_scores(model, va, wd, cfg.get("eval_stride") or (wd.size * 3) // 4, cfg["eval_batch"], device)
    sec = time.time() - t0
    vgold = {d: D["gold"][d] for d in D["val_ids"]}
    best = max((nd.strict_scores(nm.decode(sc, va, model, D["texts"], s, p), vgold, labels)["macro_F1"], s, p)
               for s, p in nm.decoding_options(model.head_type))
    total, _ = nm.count_params(model)
    log(f"{cfg['run']}: val macro-F1 {best[0]:.4f} ({best[1]}{'+propagate' if best[2] else ''}) | params {total:,} | "
        f"val inference {sec:.1f}s")
    return dict(val_macro_F1=best[0], params_total=total, val_inference_sec=round(sec, 1))


def run_plan(plan, D, device, out_root, log=print, smoke=False):
    """plan: list of steps dict(cfg=..., keep_as=name or None, teacher=name or None).
    smoke=True : every step runs for 1 epoch; any error stops at once (that is the point of the check).
    smoke=False: a failing step is reported and the remaining steps still run."""
    import traceback
    store, records, failed = {}, [], []
    for step in plan:
        cfg = dict(step["cfg"])
        if smoke:
            cfg.update(run="smoke_" + cfg["run"], epochs=1, patience=0, save_full=False)
        teacher = store.get(step["teacher"]) if step.get("teacher") else None
        if step.get("teacher") and teacher is None:
            failed.append((cfg["run"], "teacher missing")); log(f"SKIP {cfg['run']}: teacher missing"); continue
        try:
            rec, net = run_experiment(cfg, D, device, out_root, log, teacher=teacher,
                                      keep_net=bool(step.get("keep_as")))
            records.append(rec)
            if step.get("keep_as"):
                net.eval(); store[step["keep_as"]] = net
        except Exception as e:
            if smoke:
                raise
            traceback.print_exc(); failed.append((cfg["run"], repr(e)))
        gc.collect(); torch.cuda.empty_cache()
    store.clear(); gc.collect(); torch.cuda.empty_cache()
    return records, failed


# ======================================================================================================
# Ensembles + training-data post-processing (selection on validation, then applied to refit models)
# ======================================================================================================
def ensemble_scores(runs, part):
    """average the word-level tag probabilities of several linear-head models (same tokenizer + split)"""
    first = SCORES[runs[0]]
    for r in runs[1:]:
        assert SCORES[r]["tok"] == first["tok"] and SCORES[r]["split"] == first["split"], "models must share words"
        assert SCORES[r]["head"] == "linear" == first["head"], "ensembles are built from linear-head models"
    return {d: sum(SCORES[r][part][d] for r in runs) / len(runs) for d in first[part]}


def decode_scores(sc, docs, tags, texts, propagate):
    import types
    return nm.decode(sc, docs, types.SimpleNamespace(head_type="linear", tags=tags), texts, "viterbi", propagate)


def select_on_val(names, D, post_model, log=print, max_members=None):
    """every subset of `names` (+/- propagation), then greedy forward selection of the post-processing rules,
    all scored by strict macro-F1 on the validation documents. Returns the chosen configuration."""
    import itertools
    import ner_post as npp
    first = SCORES[names[0]]
    va, tags = first["va"], first["tags"]
    vgold = {d: D["gold"][d] for d in D["val_ids"]}
    F = lambda p: nd.strict_scores(p, vgold, D["labels"])["macro_F1"]
    rows = []
    for k in range(1, (max_members or len(names)) + 1):
        for combo in itertools.combinations(names, k):
            sc = ensemble_scores(list(combo), "val")
            for prop in (False, True):
                p = decode_scores(sc, va, tags, D["texts"], prop)
                rows.append(dict(combo=list(combo), propagate=prop, F1=F(p), _pred=p))
    rows.sort(key=lambda r: -r["F1"])
    log("validation macro-F1 of model combinations (Viterbi decoding):")
    for r in rows[:12]:
        log(f"   {r['F1']:.4f}  {' + '.join(r['combo'])}{'  +propagate' if r['propagate'] else ''}")
    singles = {r["combo"][0]: r["F1"] for r in rows if len(r["combo"]) == 1 and not r["propagate"]}
    best = rows[0]
    rules, cur = [], best["F1"]
    log(f"post-processing on top of the best combination ({cur:.4f}):")
    for rule in npp.RULES:                                   # each rule alone
        p, info = npp.apply(best["_pred"], D["texts"], post_model, (rule,))
        log(f"   + {rule:<10} alone: {F(p):.4f}  {info}")
    remaining = list(npp.RULES)
    while remaining:                                          # greedy forward selection
        trial = {r: F(npp.apply(best["_pred"], D["texts"], post_model, tuple(rules + [r]))[0]) for r in remaining}
        r_best = max(trial, key=trial.get)
        if trial[r_best] <= cur + 1e-4:
            break
        rules.append(r_best); remaining.remove(r_best); cur = trial[r_best]
        log(f"   keep {r_best:<10} -> {cur:.4f}")
    choice = dict(combo=best["combo"], propagate=best["propagate"], rules=rules, val_F1_models=best["F1"],
                  val_F1_final=cur, singles=singles,
                  table=[dict(combo=r["combo"], propagate=r["propagate"], F1=r["F1"]) for r in rows])
    log(f"CHOSEN: {' + '.join(choice['combo'])} | propagate {choice['propagate']} | rules {rules} | val macro-F1 {cur:.4f}")
    return choice


def write_ensemble_zip(runs, propagate, rules, D, post_model, out_dir, name, log=print):
    import ner_post as npp
    first = SCORES[runs[0]]
    sc = ensemble_scores(list(runs), "test")
    pred = decode_scores(sc, first["te"], first["tags"], D["test_texts"], propagate)
    pred, info = npp.apply(pred, D["test_texts"], post_model, tuple(rules)) if rules else (pred, {})
    sub = nd.to_submission(pred, D["conv"])
    assert set(sub) == set(D["test_texts"])
    for d, rows in sub.items():
        for s, e, l in rows:
            assert l in D["labels"] and 0 <= s < e <= len(D["test_texts"][d])
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    pj = os.path.join(out_dir, name, "prediction.json"); json.dump(sub, open(pj, "w"), ensure_ascii=False)
    zp = os.path.join(out_dir, name, f"submission_{name}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pj, arcname="prediction.json")
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ["prediction.json"]
    log(f"{name}: {sum(map(len, sub.values())):,} entities {info} -> {zp}")
    return zp


@torch.no_grad()
def predict_texts(models, cfg, texts, D, device, propagate=True, log=print):
    """pseudo-labels for unlabelled documents: average the word probabilities of `models` (same tokenizer,
    linear heads), Viterbi + optional propagation. Returns {doc: [(s, e, label)]}."""
    import types
    tags, tag2id = nd.tag_vocab(D["labels"])
    tok = nm.load_tokenizer(cfg["arch"], D["root"])
    docs, _ = nm.prepare_docs(texts, None, tok, tag2id, None, cfg["split"])
    wd = nm.Windowing(tok, cfg["window"], cfg["header_tokens"], nm.ARCH_INFO[cfg["arch"]][1])
    stride = cfg.get("eval_stride") or (wd.size * 3) // 4
    t0 = time.time()
    scs = [nm.predict_scores(m, docs, wd, stride, cfg["eval_batch"], device) for m in models]
    sc = {d: sum(s[d] for s in scs) / len(scs) for d in docs}
    pred = nm.decode(sc, docs, types.SimpleNamespace(head_type="linear", tags=tags), texts, "viterbi", propagate)
    n = sum(map(len, pred.values()))
    log(f"pseudo-labelled {len(pred)} documents: {n:,} entities ({n / max(1, len(pred)):.0f} per document) in "
        f"{time.time() - t0:.0f}s | labels {dict(Counter(l for v in pred.values() for *_, l in v).most_common())}")
    return pred


# ======================================================================================================
# Per-label decision biases for macro-F1 (tuned on out-of-fold predictions)
# ======================================================================================================
def viterbi_batch(ems, trans, start, end):
    """BIO-constrained Viterbi for many documents at once (CPU). ems: list of (n_i, K) score tensors."""
    B, K = len(ems), trans.size(0)
    N = max([e.size(0) for e in ems] + [1])
    E = torch.zeros(B, N, K); M = torch.zeros(B, N, dtype=torch.bool)
    for i, e in enumerate(ems):
        E[i, :e.size(0)] = e; M[i, :e.size(0)] = True
    score = start[None] + E[:, 0]
    back = torch.zeros(B, N, K, dtype=torch.int32)
    ident = torch.arange(K, dtype=torch.int32)[None].expand(B, K)
    for t in range(1, N):
        best, idx = (score[:, :, None] + trans[None]).max(1)
        m = M[:, t, None]
        score = torch.where(m, best + E[:, t], score)
        back[:, t] = torch.where(m, idx.to(torch.int32), ident)       # padding: keep the tag
    tag = (score + end[None]).argmax(1)
    out = torch.zeros(B, N, dtype=torch.long)
    for t in range(N - 1, 0, -1):
        out[:, t] = tag
        tag = back[:, t].long().gather(1, tag[:, None]).squeeze(1)
    out[:, 0] = tag
    return [out[i, :e.size(0)].tolist() for i, e in enumerate(ems)]


def decode_bias(sc, docs, tags, texts, bias, propagate=True):
    """add bias[label] to the log-probability of B-label and I-label, then BIO Viterbi (+ propagation)"""
    ok, start_ok = nd.allowed_transitions(tags)
    trans = (~torch.tensor(ok)).float() * nm.NEG
    start = (~torch.tensor(start_ok)).float() * nm.NEG
    end = torch.zeros(len(tags))
    bv = torch.tensor([0.0 if t == "O" else float(bias.get(t[2:], 0.0)) for t in tags])
    ids = list(sc)
    paths = viterbi_batch([sc[d].clamp(min=1e-9).log() + bv for d in ids], trans, start, end)
    id2tag = dict(enumerate(tags))
    pred = {}
    for d, p in zip(ids, paths):
        r = docs[d]
        spans = nd.spans_from_tags(p, r["wcs"], r["wce"], id2tag)
        pred[d] = nd.propagate(spans, texts[d]) if propagate else spans
    return pred


def tune_biases(sc, docs, tags, texts, gold, labels, grid=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
                passes=2, log=print):
    """coordinate ascent: one bias per label, each pass tries every grid value for every label"""
    bias = {l: 0.0 for l in labels}
    F = lambda b: nd.strict_scores(decode_bias(sc, docs, tags, texts, b), gold, labels)["macro_F1"]
    base = best = F(bias)
    for p in range(passes):
        for l in labels:
            trial = {g: (best if g == bias[l] else F(dict(bias, **{l: g}))) for g in grid}
            g = max(trial, key=trial.get)
            if trial[g] > best + 1e-4:
                bias[l], best = g, trial[g]
        log(f"   pass {p + 1}: macro-F1 {best:.4f} (no bias {base:.4f}) | biases { {k: v for k, v in bias.items() if v} }")
    return bias, base, best
