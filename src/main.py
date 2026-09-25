"""1 つの run（= 1 モデル）で SciGym-small を解かせ、評価層 scigym_small の入力ファイルを書く。"""

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import yaml

BUDGET_EXCEEDED = 42


def cli_args():
    """hydra 形式の key=value を読む。hydra / omegaconf は scigym が固定する petab の antlr 版と衝突する"""
    return {k: yaml.safe_load(v) for k, _, v in (a.partition("=") for a in sys.argv[1:])}


def run_instance(cfg, run_dir, instance):
    out = run_dir / "instances" / instance.name
    if (out / "evaluation.json").exists():
        return 0
    args = {
        "instance_dir": str(instance),
        "out_dir": str(out),
        "model": cfg.run_model,
        "base_url": cfg.base_url,
        "max_iterations": 2 if cfg.mode == "sanity" else cfg.max_iterations,
        "eval_debug_rounds": cfg.eval_debug_rounds,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stdout.txt", "a") as log:
        try:
            proc = subprocess.run([sys.executable, "-m", "src.train", json.dumps(args)], stdout=log, stderr=subprocess.STDOUT,
                                  timeout=cfg.instance_timeout)
        except subprocess.TimeoutExpired:
            print(f"[{instance.name}] killed after {cfg.instance_timeout}s", file=log)
            return 1
    if not (out / "evaluation.json").exists():  # 失敗した run の作業ディレクトリは残らないので原因を標準出力へ
        print(f"[{instance.name}] no evaluation.json; log tail:", *(out / "stdout.txt").read_text().splitlines()[-25:], sep="\n  ")
    return proc.returncode


def main():
    cli = cli_args()
    run_id = cli["run"]
    cfg = yaml.safe_load(open("config/config.yaml"))
    cfg.update(cli)
    cfg["run"] = yaml.safe_load(open(f"config/run/{run_id}.yaml"))
    cfg = SimpleNamespace(**cfg, run_model=cfg["run"]["model"])
    run_dir = Path(cfg.results_dir) / run_id
    instances = sorted(p for p in Path(cfg.data_dir).iterdir() if p.is_dir())
    if cfg.mode == "sanity":
        instances = [p for p in instances if p.name == "BIOMD0000000027"]
    elif cfg.mode == "pilot":
        instances = instances[::14]
    stage = cfg.mode.upper()
    # API エラーで evaluation.json が出なかった件は 2 回までやり直す。予算超過（402）は即座に run を止める
    for _ in range(3):
        with ThreadPoolExecutor(cfg.workers) as pool:
            codes = list(pool.map(lambda p: run_instance(cfg, run_dir, p), instances))
        if BUDGET_EXCEEDED in codes:
            print(f"{stage}_VALIDATION: FAIL reason=budget_exceeded")
            sys.exit(1)
    submitted, tokens = {}, {"input_tokens": 0, "output_tokens": 0}
    for instance in instances:
        out = run_dir / "instances" / instance.name
        # Seyval の出力一覧は 1000 ファイルまで。反復ごとのコードは chat_history.yaml に含まれるので削る
        shutil.rmtree(out / "codes", ignore_errors=True)
        (out / "chat_history_readable.txt").unlink(missing_ok=True)
        if not (out / "evaluation.json").exists():
            continue  # 3 試行とも終わらなかった件: 提出無しとして評価層に渡す
        submitted[instance.name] = (out / "final_model.xml").read_text() if (out / "final_model.xml").exists() else None
        for k, v in json.loads((out / "tokens.json").read_text()).items():
            tokens[k] += v
    (run_dir / "eval_inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_inputs" / "scigym_small.json").write_text(json.dumps({"instances": [{
        "id": p.name,
        "reference_sbml": (p / "truth.xml").read_text(),
        "incomplete_sbml": (p / "partial.xml").read_text(),
        "reference_sedml": (p / "truth.sedml").read_text(),
        "submitted_sbml": submitted.get(p.name),
    } for p in instances]}))
    (run_dir / "tokens.json").write_text(json.dumps(tokens))
    n_done = len(submitted)
    if n_done < len(instances):
        print(f"{stage}_VALIDATION: FAIL reason=missing_metrics")
        sys.exit(1)
    print(f"{stage}_VALIDATION_SUMMARY: {json.dumps({'n_instances': n_done, 'n_with_submission': sum(v is not None for v in submitted.values()), **tokens})}")
    print(f"{stage}_VALIDATION: PASS")


if __name__ == "__main__":
    main()
