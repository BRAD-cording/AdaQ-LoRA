#!/usr/bin/env python3
"""GPU-free preflight for the safety task group (TruthfulQA + IFEval).

Why this exists
---------------
2026-11-11: all 12 `safety` evaluations burned 1.1-2.7 h each on GPU and then
died at rc=1 in the *metric* stage with

    LookupError: Resource 'punkt_tab' not found.
    Attempted to load 'tokenizers/punkt_tab/english/'

because `lm_eval/tasks/ifeval/instructions_util.py:36 download_nltk_resources()`
calls `nltk.download("punkt_tab")`, which needs `raw.githubusercontent.com`
(unreachable on this host, Errno 104) and then prints a FALSE SUCCESS line
("Downloaded punkt_tab on rank 0").  Every generated prediction was lost
because the process aborted before writing results_*.json.

A `--limit 8` smoke test does NOT catch this: those 8 samples do not exercise
the instructions that call `nltk.word_tokenize` / `count_sentences`.

This preflight runs the exact crash path with no GPU and no model:
  * loads google/IFEval (541 rows) and drives `utils.process_results`
    over EVERY row with a dummy response -> exercises strict+loose metric
    computation for all 25 instruction types;
  * checks the 4 metric keys the aggregation gate expects are present;
  * resolves the two nltk resources that raised LookupError;
  * confirms `truthfulqa_mc1/mc2` tasks can be instantiated from the manager.

Exit 0 = safe to spend GPU hours.  Exit 1 = do not launch.
"""
import sys
import traceback

FAILURES = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    except Exception as exc:  # noqa: BLE001 - preflight must report anything
        FAILURES.append((name, type(exc).__name__, str(exc)[:300]))
        print(f"  FAIL  {name}  {type(exc).__name__}: {str(exc)[:200]}")


def check_nltk_resources():
    import nltk
    found = []
    for res in ("tokenizers/punkt_tab/english/", "tokenizers/punkt/english.pickle"):
        found.append(str(nltk.data.find(res)))
    nltk.word_tokenize("All twelve runs must finish before we ship.")
    return f"nltk {nltk.__version__}, 2 resources resolved"


def check_ifeval_metric_path():
    """Drive ifeval.utils.process_results over all 541 docs with a dummy reply."""
    from datasets import load_dataset
    from lm_eval.tasks.ifeval import utils as ifeval_utils

    ds = load_dataset("google/IFEval", split="train")
    dummy = "This is a short dummy answer used only to exercise the metric path."
    keys = set()
    errors = []
    for row in ds:
        doc = {k: row[k] for k in ("key", "instruction_id_list", "prompt", "kwargs")}
        try:
            out = ifeval_utils.process_results(doc, [dummy])
            keys.update(out.keys())
        except Exception as exc:  # noqa: BLE001
            errors.append((row["key"], type(exc).__name__, str(exc)[:160]))
    if errors:
        raise RuntimeError(
            f"{len(errors)}/{len(ds)} docs failed; first={errors[0]}"
        )
    required = {
        "prompt_level_strict_acc",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_loose_acc",
    }
    missing = required - keys
    if missing:
        raise RuntimeError(f"metric keys missing: {sorted(missing)}; got {sorted(keys)}")
    return f"{len(ds)} docs, keys={sorted(keys)}"


def check_ifeval_aggregation():
    """The agg_inst_level_acc aggregator must flatten the list-of-lists."""
    from datasets import load_dataset
    from lm_eval.tasks.ifeval import utils as ifeval_utils

    ds = load_dataset("google/IFEval", split="train")
    dummy = "One sentence. Two sentences."
    items = []
    for row in ds:
        doc = {k: row[k] for k in ("key", "instruction_id_list", "prompt", "kwargs")}
        items.append(ifeval_utils.process_results(doc, [dummy])["inst_level_strict_acc"])
    acc = ifeval_utils.agg_inst_level_acc(items)
    if not (0.0 <= acc <= 1.0):
        raise RuntimeError(f"inst_level_strict_acc out of range: {acc}")
    return f"inst_level_strict_acc(dummy)={acc:.4f}"


def check_truthfulqa_instantiable():
    from lm_eval.tasks import TaskManager

    tm = TaskManager()
    out = []
    for name in ("truthfulqa_mc1", "truthfulqa_mc2", "ifeval"):
        task = tm.get_task_dict([name])[name] if hasattr(tm, "get_task_dict") else None
        if task is None:
            loaded = tm.load_task_or_group([name])
            if name not in loaded:
                raise RuntimeError(f"{name} not resolvable from TaskManager")
        out.append(name)
    return ", ".join(out)


def main():
    print("=== GPU-free preflight: safety metric path ===")
    check("nltk resources (punkt_tab / punkt)", check_nltk_resources)
    check("ifeval process_results over full 541 docs", check_ifeval_metric_path)
    check("ifeval agg_inst_level_acc", check_ifeval_aggregation)
    check("task manager resolves truthfulqa_mc1/mc2/ifeval", check_truthfulqa_instantiable)
    print("=== summary ===")
    if FAILURES:
        for name, kind, msg in FAILURES:
            print(f"  BLOCKER  {name}: {kind}: {msg}")
        print("VERDICT: BLOCKED - do not launch GPU evaluation")
        return 1
    print("VERDICT: READY - safety metric path is GPU-safe")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
