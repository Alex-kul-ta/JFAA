#!/usr/bin/env python3
"""Build EPIC-KITCHENS-100 submission ensembles from exported test.json files."""

from __future__ import annotations

import argparse
from array import array
import gc
import json
import math
from pathlib import Path
import zipfile


N_VERB = 97
N_NOUN = 300
RRF_K = 60.0
EPS = 1.0e-300
ROUND_DIGITS = 6
REPORT_EPOCHS = ["018", "019", "020", "021", "022", "023", "024", "025", "026", "027"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submissions-dir",
        default="outputs/submissions",
        help="Directory containing epoch-XXX/test.json files.",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        default=REPORT_EPOCHS,
        help="Epoch-level submissions to ensemble. Defaults to the report validation table range.",
    )
    parser.add_argument(
        "--quality-prior",
        nargs="+",
        type=float,
        default=None,
        help="Optional weak global prior, one value per epoch. Defaults to a uniform prior.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def softmax_dense(scores: dict[str, float], nclasses: int) -> list[float]:
    vals = [float(scores[str(i)]) for i in range(nclasses)]
    max_v = max(vals)
    vals = [math.exp(v - max_v) for v in vals]
    denom = sum(vals)
    return [v / denom for v in vals]


def softmax_sparse(scores: dict[str, float]) -> list[tuple[str, float]]:
    items = sorted(((str(k), float(v)) for k, v in scores.items()), key=lambda kv: (-kv[1], kv[0]))
    max_v = items[0][1]
    exps = [(k, math.exp(v - max_v)) for k, v in items]
    denom = sum(v for _, v in exps)
    return [(k, v / denom) for k, v in exps]


def top_indices(values: list[float], k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def dense_confidence(probs: list[float]) -> float:
    top2 = sorted(probs, reverse=True)[:2]
    top1 = top2[0]
    margin = top2[0] - top2[1] if len(top2) > 1 else top2[0]
    return max(0.0, min(1.0, 0.65 * top1 + 0.35 * margin))


def sparse_confidence(items: list[tuple[str, float]]) -> float:
    top1 = items[0][1]
    top2 = items[1][1] if len(items) > 1 else 0.0
    margin = top1 - top2
    return max(0.0, min(1.0, 0.65 * top1 + 0.35 * margin))


def metadata(payload: dict) -> dict:
    return {k: payload[k] for k in payload if k != "results"}


def compact_load(paths: list[Path]) -> tuple[dict, list[str], list[dict]]:
    meta = None
    ids = None
    id_to_idx = None
    models = []

    for path in paths:
        print(f"loading {path}", flush=True)
        payload = load_json(path)
        cur_meta = metadata(payload)
        results = payload["results"]
        if meta is None:
            meta = cur_meta
            ids = list(results.keys())
            id_to_idx = {nid: i for i, nid in enumerate(ids)}
        elif cur_meta != meta:
            raise SystemExit(f"metadata mismatch in {path}")
        elif set(results.keys()) != set(id_to_idx):
            raise SystemExit(f"narration id mismatch in {path}")

        model = {
            "verb_probs": [None] * len(ids),
            "noun_probs": [None] * len(ids),
            "action_items": [None] * len(ids),
            "verb_info": [None] * len(ids),
            "noun_info": [None] * len(ids),
            "action_info": [None] * len(ids),
        }
        for count, (narration_id, entry) in enumerate(results.items(), start=1):
            idx = id_to_idx[narration_id]
            verb_probs = softmax_dense(entry["verb"], N_VERB)
            noun_probs = softmax_dense(entry["noun"], N_NOUN)
            action_items = softmax_sparse(entry["action"])

            verb_top5 = tuple(top_indices(verb_probs, 5))
            noun_top5 = tuple(top_indices(noun_probs, 5))
            action_top10 = tuple(k for k, _ in action_items[:10])

            model["verb_probs"][idx] = array("f", verb_probs)
            model["noun_probs"][idx] = array("f", noun_probs)
            model["action_items"][idx] = tuple(action_items)
            model["verb_info"][idx] = (verb_top5[0], verb_top5, dense_confidence(verb_probs))
            model["noun_info"][idx] = (noun_top5[0], noun_top5, dense_confidence(noun_probs))
            model["action_info"][idx] = (action_top10[0], action_top10, sparse_confidence(action_items))
            if count % 4000 == 0:
                print(f"  compacted {count}/{len(results)}", flush=True)

        models.append(model)
        del payload, results
        gc.collect()

    return meta, ids, models


def consensus_scores(infos: list[tuple[object, tuple, float]]) -> list[float]:
    scores = []
    for top1, topk, _ in infos:
        support = sum(1 for other_top1, _, _ in infos if other_top1 == top1) / len(infos)
        topk_set = set(topk)
        overlap = 0.0
        for _, other_topk, _ in infos:
            overlap += len(topk_set.intersection(other_topk)) / max(len(topk), 1)
        overlap /= len(infos)
        scores.append(0.55 * support + 0.45 * overlap)
    return scores


def dynamic_weights(
    infos: list[tuple[object, tuple, float]],
    prior: list[float],
    mode: str,
) -> list[float]:
    conf = [info[2] for info in infos]
    consensus = consensus_scores(infos)

    if mode == "no_prior":
        base = [1.0] * len(infos)
    else:
        base = prior

    if mode == "majority_gate":
        top1s = [info[0] for info in infos]
        counts = {top: top1s.count(top) for top in set(top1s)}
        majority_top, majority_count = max(counts.items(), key=lambda kv: kv[1])
        if majority_count >= 3:
            raw = []
            for i, top in enumerate(top1s):
                if top == majority_top:
                    raw.append(base[i] * (1.0 + conf[i]) * 4.0)
                else:
                    raw.append(base[i] * (0.25 + conf[i]) * 0.25)
            total = sum(raw)
            return [v / total for v in raw]

    raw = []
    for i in range(len(infos)):
        raw.append(base[i] * (0.30 + conf[i]) * (0.45 + consensus[i]))
    total = sum(raw)
    return [v / total for v in raw]


def log_dense(scores: list[float]) -> dict[str, float]:
    return {str(i): round(math.log(max(float(scores[i]), EPS)), ROUND_DIGITS) for i in range(len(scores))}


def validate_payload(payload: dict, zip_path: Path) -> None:
    if payload.get("version") != "0.2":
        raise SystemExit("bad version")
    if payload.get("challenge") != "action_anticipation":
        raise SystemExit("bad challenge")
    results = payload.get("results", {})
    if len(results) != 13092:
        raise SystemExit(f"expected 13092 results, got {len(results)}")
    for narration_id, entry in results.items():
        for key, expected in (("verb", N_VERB), ("noun", N_NOUN), ("action", 100)):
            scores = entry.get(key)
            if not isinstance(scores, dict) or len(scores) != expected:
                raise SystemExit(f"{narration_id}: bad {key} length")
            for score_key, value in scores.items():
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise SystemExit(f"{narration_id}: non-finite {key}[{score_key}]")
    with zipfile.ZipFile(zip_path) as zf:
        if zf.namelist() != ["test.json"]:
            raise SystemExit("zip must contain a flat test.json")


def reference_top_changes(payload: dict, reference_path: Path) -> dict[str, int]:
    ref = load_json(reference_path)["results"]
    changes = {"verb": 0, "noun": 0, "action": 0}
    for narration_id, entry in payload["results"].items():
        ref_entry = ref[narration_id]
        for key in changes:
            new_top = max(entry[key].items(), key=lambda kv: float(kv[1]))[0]
            ref_top = max(ref_entry[key].items(), key=lambda kv: float(kv[1]))[0]
            if new_top != ref_top:
                changes[key] += 1
    return changes


def prob_entry(models: list[dict], idx: int, weights: dict[str, list[float]]) -> dict:
    verb = [0.0] * N_VERB
    noun = [0.0] * N_NOUN
    action = {}

    for model_idx, model in enumerate(models):
        vw = weights["verb"][model_idx]
        nw = weights["noun"][model_idx]
        aw = weights["action"][model_idx]
        for i, value in enumerate(model["verb_probs"][idx]):
            verb[i] += vw * value
        for i, value in enumerate(model["noun_probs"][idx]):
            noun[i] += nw * value
        for key, prob in model["action_items"][idx]:
            action[key] = action.get(key, 0.0) + aw * prob

    action_items = sorted(action.items(), key=lambda kv: (-kv[1], kv[0]))[:100]
    return {
        "verb": log_dense(verb),
        "noun": log_dense(noun),
        "action": {key: round(math.log(max(value, EPS)), ROUND_DIGITS) for key, value in action_items},
    }


def rrf_entry(models: list[dict], idx: int, weights: dict[str, list[float]]) -> dict:
    verb = [0.0] * N_VERB
    noun = [0.0] * N_NOUN
    action = {}

    for model_idx, model in enumerate(models):
        vw = weights["verb"][model_idx]
        nw = weights["noun"][model_idx]
        aw = weights["action"][model_idx]
        verb_ranked = sorted(range(N_VERB), key=lambda i: (-model["verb_probs"][idx][i], i))
        noun_ranked = sorted(range(N_NOUN), key=lambda i: (-model["noun_probs"][idx][i], i))
        for rank, cls_idx in enumerate(verb_ranked, start=1):
            verb[cls_idx] += vw / (RRF_K + rank)
        for rank, cls_idx in enumerate(noun_ranked, start=1):
            noun[cls_idx] += nw / (RRF_K + rank)
        for rank, (key, _) in enumerate(model["action_items"][idx], start=1):
            action[key] = action.get(key, 0.0) + aw / (RRF_K + rank)

    action_items = sorted(action.items(), key=lambda kv: (-kv[1], kv[0]))[:100]
    return {
        "verb": {str(i): round(float(verb[i]), ROUND_DIGITS) for i in range(N_VERB)},
        "noun": {str(i): round(float(noun[i]), ROUND_DIGITS) for i in range(N_NOUN)},
        "action": {key: round(float(value), ROUND_DIGITS) for key, value in action_items},
    }


def make_weights(models: list[dict], idx: int, prior: list[float], mode: str) -> dict[str, list[float]]:
    return {
        "verb": dynamic_weights([m["verb_info"][idx] for m in models], prior, mode),
        "noun": dynamic_weights([m["noun_info"][idx] for m in models], prior, mode),
        "action": dynamic_weights([m["action_info"][idx] for m in models], prior, mode),
    }


def generate(
    mode_name: str,
    fusion: str,
    weight_mode: str,
    out_dir: Path,
    meta: dict,
    ids: list[str],
    models: list[dict],
    prior: list[float],
    reference_path: Path,
    reference_epoch: str,
) -> dict:
    print(f"=== generating {mode_name}: fusion={fusion}, weights={weight_mode} ===", flush=True)
    payload = dict(meta)
    payload["results"] = {}
    weight_sums = {key: [0.0] * len(models) for key in ("verb", "noun", "action")}
    max_weight_sums = {key: 0.0 for key in ("verb", "noun", "action")}
    entry_fn = prob_entry if fusion == "prob" else rrf_entry

    for idx, narration_id in enumerate(ids):
        weights = make_weights(models, idx, prior, weight_mode)
        for key, values in weights.items():
            for i, value in enumerate(values):
                weight_sums[key][i] += value
            max_weight_sums[key] += max(values)
        payload["results"][narration_id] = entry_fn(models, idx, weights)
        if (idx + 1) % 4000 == 0:
            print(f"  wrote entries {idx + 1}/{len(ids)}", flush=True)

    target_dir = out_dir / f"ensemble_{mode_name}"
    target_dir.mkdir(parents=True, exist_ok=True)
    out_json = target_dir / "test.json"
    out_zip = target_dir / "submission.zip"
    with out_json.open("w") as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(out_json, arcname="test.json")
    validate_payload(payload, out_zip)
    changes = reference_top_changes(payload, reference_path)

    summary = {
        "path": str(out_zip),
        "fusion": fusion,
        "weight_mode": weight_mode,
        "avg_weights": {
            key: [value / len(ids) for value in values]
            for key, values in weight_sums.items()
        },
        "avg_max_weight": {
            key: value / len(ids)
            for key, value in max_weight_sums.items()
        },
        "top1_changed_vs_reference": {
            "reference_epoch": reference_epoch,
            "changes": changes,
        },
    }
    with (target_dir / "ensemble_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {out_zip}", flush=True)
    print(f"summary: {summary}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    base = Path(args.submissions_dir).resolve()
    epochs = args.epochs
    quality_prior = args.quality_prior if args.quality_prior is not None else [1.0] * len(epochs)
    if len(quality_prior) != len(epochs):
        raise SystemExit("--quality-prior must have one value per epoch")
    paths = [base / f"epoch-{epoch}" / "test.json" for epoch in epochs]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"missing {path}")
    prior_total = sum(quality_prior)
    prior = [value / prior_total * len(quality_prior) for value in quality_prior]

    meta, ids, models = compact_load(paths)
    reference_epoch = epochs[0]
    reference_path = base / f"epoch-{reference_epoch}" / "test.json"
    modes = [
        ("rank1_dynamic_consensus_conf_prob", "prob", "prior"),
        ("rank2_dynamic_consensus_conf_rrf", "rrf", "prior"),
        ("rank3_dynamic_majority_gate_prob", "prob", "majority_gate"),
        ("rank4_dynamic_consensus_conf_prob_no_prior", "prob", "no_prior"),
    ]
    summaries = []
    for mode_name, fusion, weight_mode in modes:
        summaries.append(
            generate(
                mode_name,
                fusion,
                weight_mode,
                base,
                meta,
                ids,
                models,
                prior,
                reference_path,
                reference_epoch,
            )
        )
        gc.collect()

    with (base / "dynamic_ensemble_summaries.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)
    print(f"wrote {base / 'dynamic_ensemble_summaries.json'}", flush=True)


if __name__ == "__main__":
    main()
