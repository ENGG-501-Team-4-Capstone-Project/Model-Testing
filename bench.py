"""
Benchmark vision models (via OpenRouter) on extracting fields from lifting-sling tags.
Results are collected into pandas DataFrames.

Setup:
    pip install requests pandas python-dotenv
    echo 'OPENROUTER_API_KEY=sk-or-...' > .env     # next to bench.py
    # images go in images/tags/ (file names must match the keys in ground_truth.json)

Run:
    python bench.py             # run the benchmark
    python bench.py --dry-run   # check images + show prompt, no API calls

Edit MODELS / RUNS / WORKERS below to change the setup.

Outputs (./results): raw_results.json, calls.csv, fields.csv, summary.csv, report.md

From a notebook:
    from bench import run_benchmark
    calls_df, fields_df, summary_df = run_benchmark()
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "nvidia/nemotron-3.5-super-vl:free",
    "deepseek/deepseek-v4.1-flash-1017",
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
]
RUNS = 3  # hits per model per image
WORKERS = 4  # parallel API calls

HERE = Path(__file__).parent
IMAGES_DIR = HERE / "images" / "tags"
RESULTS_DIR = HERE / "results"
TRUTH_PATH = HERE / "answers.json"

load_dotenv(HERE / ".env")


def build_prompt(spec: dict) -> str:
    lines = [f'- "{k}": {v["desc"]}' for k, v in spec["fields"].items()]
    return (
        "You are reading a lifting-sling identification tag in the image.\n"
        "Extract exactly these fields and return ONLY a single JSON object with these keys "
        "(no markdown, no commentary):\n"
        + "\n".join(lines)
        + "\n\nRules: copy text exactly as printed on the tag. For fields marked "
        "'number only', return digits without units or thousands separators. "
        "If a value is not legible or not present, use null."
    )


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def call_model(model, prompt, data_url, api_key, timeout=120, retries=2):
    payload = {
        "model": model,
        "temperature": 0,
        "usage": {"include": True},
        "reasoning": {"enabled": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "sling-tag-benchmark",
    }
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            latency = time.perf_counter() - t0
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2**attempt)
                continue
            if r.status_code != 200:
                return {
                    "error": f"HTTP {r.status_code}: {r.text[:300]}",
                    "latency_s": latency,
                }
            data = r.json()
            if "error" in data:
                return {"error": str(data["error"])[:300], "latency_s": latency}
            usage = data.get("usage", {}) or {}
            return {
                "text": data["choices"][0]["message"].get("content") or "",
                "latency_s": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "cost_usd": usage.get("cost", 0.0) or 0.0,
                "finish_reason": data["choices"][0].get("finish_reason"),
            }
        except requests.RequestException as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2**attempt)
    return {"error": last_err or "unknown error", "latency_s": None}


def parse_json(text: str):
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def normalize(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    s = str(v).lower().strip()
    s = s.replace("ser#", "").replace("s/n", "")
    s = s.replace("\u201d", '"').replace("\u2033", '"').replace("''", '"')
    s = re.sub(r"[,\s]", "", s)
    s = re.sub(
        r"[-\u2010-\u2015\u2212]", "", s
    )  # ignore dashes (-, hyphen, en/em dash, minus)
    s = s.replace('"', "")  # 1/8" -> 1/8
    s = re.sub(r"(?<=\d)(inches|inch|in)$", "", s)  # 1/8in -> 1/8
    if re.match(r"^[\d.]+[a-z]+$", s):  # strip trailing units from numeric values
        s = re.sub(r"(lbs|lb|kgs|kg|tons|ton|mm|ft)$", "", s)
    return s


def values_match(pred, truth) -> bool:
    p, t = normalize(pred), normalize(truth)
    if p == t:
        return True
    try:
        return abs(float(p) - float(t)) < 1e-9
    except ValueError:
        return False


def run_one(model, image_name, spec, run_idx, api_key):
    path = IMAGES_DIR / image_name
    res = call_model(model, build_prompt(spec), image_to_data_url(path), api_key)
    out = {"model": model, "image": image_name, "run": run_idx, **res}
    parsed = parse_json(res.get("text", "")) if "error" not in res else None
    out["json_valid"] = parsed is not None
    out["fields"] = {}
    for key, meta in spec["fields"].items():
        pred = parsed.get(key) if parsed else None
        out["fields"][key] = {
            "expected": meta["value"],
            "predicted": pred,
            "correct": bool(values_match(pred, meta["value"])) if parsed else False,
        }
    return out


def build_dataframes(results, models):
    field_rows = []
    for r in results:
        for k, f in r["fields"].items():
            field_rows.append(
                {
                    "model": r["model"],
                    "image": r["image"],
                    "run": r["run"],
                    "field": k,
                    "expected": f["expected"],
                    "predicted": f["predicted"],
                    "correct": f["correct"],
                }
            )
    fields_df = pd.DataFrame(field_rows)

    calls_df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "image": r["image"],
                "run": r["run"],
                "json_valid": r["json_valid"],
                "error": r.get("error"),
                "latency_s": r.get("latency_s"),
                "prompt_tokens": r.get("prompt_tokens", 0),
                "completion_tokens": r.get("completion_tokens", 0),
                "cost_usd": r.get("cost_usd", 0.0),
                "finish_reason": r.get("finish_reason"),
                "signature": (
                    json.dumps(
                        {k: normalize(v["predicted"]) for k, v in r["fields"].items()},
                        sort_keys=True,
                    )
                    if r["json_valid"]
                    else None
                ),
                "raw_text": r.get("text"),
            }
            for r in results
        ]
    )

    per_call = (
        fields_df.groupby(["model", "image", "run"])["correct"]
        .agg(n_fields="size", n_correct="sum")
        .reset_index()
    )
    per_call["accuracy"] = per_call["n_correct"] / per_call["n_fields"]
    per_call["full_match"] = per_call["n_correct"] == per_call["n_fields"]
    calls_df = calls_df.merge(per_call, on=["model", "image", "run"])

    valid = calls_df[calls_df["json_valid"]]
    grp = valid.groupby(["model", "image"])["signature"]
    n_unique = grp.nunique()
    n_runs = grp.size()
    multi_run = n_runs.index[n_runs.to_numpy() > 1]
    consistent = n_unique.loc[multi_run] == 1
    consistency = pd.Series(
        consistent.groupby(level="model").mean(), name="consistency"
    )

    agg = calls_df.groupby("model").agg(
        n_correct=("n_correct", "sum"),
        n_fields=("n_fields", "sum"),
        full_image_accuracy=("full_match", "mean"),
        json_valid_rate=("json_valid", "mean"),
        errors=("error", lambda s: int(s.notna().sum())),
        avg_latency_s=("latency_s", "mean"),
        p95_latency_s=("latency_s", lambda s: s.quantile(0.95)),
        avg_cost_usd=("cost_usd", "mean"),
        total_cost_usd=("cost_usd", "sum"),
        avg_prompt_tokens=("prompt_tokens", "mean"),
        avg_completion_tokens=("completion_tokens", "mean"),
    )
    agg["field_accuracy"] = agg["n_correct"] / agg["n_fields"]
    agg["cost_per_correct_field_usd"] = agg["total_cost_usd"] / agg["n_correct"].where(
        agg["n_correct"] > 0
    )
    summary_df = agg.drop(columns=["n_correct", "n_fields"]).join(consistency)
    summary_df = summary_df.reindex([m for m in models if m in summary_df.index])
    return calls_df, fields_df, summary_df


def write_report(calls_df, fields_df, summary_df, runs):
    fmt = summary_df.copy()
    for c in (
        "field_accuracy",
        "full_image_accuracy",
        "json_valid_rate",
        "consistency",
    ):
        fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
    by_image = (
        fields_df.pivot_table(
            index="image", columns="model", values="correct", aggfunc="mean"
        )
        * 100
    ).round(1)
    hardest = (
        fields_df.groupby(["image", "field"])["correct"].mean().sort_values().head(10)
        * 100
    ).round(1)
    misses = fields_df[~fields_df["correct"]].head(40)
    return "\n".join(
        [
            "# Benchmark\n",
            f"Calls: {len(calls_df)} | Images: {calls_df['image'].nunique()} | Runs per image per model: {runs}\n",
            "## Overall by model\n```",
            fmt.round(5).to_string(),
            "```\n",
            "## Field accuracy (%) by image\n```",
            by_image.to_string(),
            "```\n",
            "## Hardest fields (% correct)\n```",
            hardest.to_string(),
            "```\n",
            "## Mismatches (first 40)\n```",
            misses.to_string(index=False),
            "```",
        ]
    )


def save_outputs(results, models, runs):
    calls_df, fields_df, summary_df = build_dataframes(results, models)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "raw_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    calls_df.drop(columns=["signature"]).to_csv(RESULTS_DIR / "calls.csv", index=False)
    fields_df.to_csv(RESULTS_DIR / "fields.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "summary.csv")
    (RESULTS_DIR / "report.md").write_text(
        write_report(calls_df, fields_df, summary_df, runs)
    )
    return calls_df, fields_df, summary_df


def check_images(truth):
    missing = [n for n in truth if not (IMAGES_DIR / n).exists()]
    if missing:
        found = (
            sorted(p.name for p in IMAGES_DIR.glob("*")) if IMAGES_DIR.exists() else []
        )
        sys.exit(
            f"Missing images in {IMAGES_DIR}: {missing}\nFiles found there: {found}\n"
            f"Rename them to match, or edit the keys in ground_truth.json."
        )


def run_benchmark(models=MODELS, runs=RUNS, workers=WORKERS):
    """Run the full benchmark and return (calls_df, fields_df, summary_df)."""
    truth = json.loads(TRUTH_PATH.read_text())
    check_images(truth)
    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPEN_ROUTER_API_KEY in .env or your environment")

    jobs = [(m, img, i + 1) for m in models for img in truth for i in range(runs)]
    print(
        f"Running {len(jobs)} calls ({len(models)} models x {len(truth)} images x {runs} runs)"
    )
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(run_one, m, img, truth[img], i, api_key) for m, img, i in jobs
        ]
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            acc = sum(f["correct"] for f in r["fields"].values()) / len(r["fields"])
            status = (
                f"ERROR {r['error'][:60]}"
                if "error" in r
                else f"{acc:.0%} in {r['latency_s']:.1f}s ${r.get('cost_usd', 0):.5f}"
            )
            print(
                f"[{n}/{len(jobs)}] {r['model']} | {r['image']} | run {r['run']} | {status}"
            )
    order = list(truth)
    results.sort(
        key=lambda r: (models.index(r["model"]), order.index(r["image"]), r["run"])
    )
    return save_outputs(results, models, runs)


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark vision models on sling tag field extraction."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="check images and show the prompt, no API calls",
    )
    args = ap.parse_args()

    if args.dry_run:
        truth = json.loads(TRUTH_PATH.read_text())
        check_images(truth)
        first = next(iter(truth))
        print(
            f"{len(truth)} images found. Calls to make: {len(truth) * len(MODELS) * RUNS}"
        )
        print(f"\n--- Prompt for {first} ---\n{build_prompt(truth[first])}")
        return

    _, _, summary_df = run_benchmark()
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n", summary_df.round(4).to_string())
    print(f"\nFull report: {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
