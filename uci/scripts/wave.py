"""Run the experiment one fold ("wave") at a time.

  python scripts/wave.py prepare K              # task dirs for fold K: 12 datasets x 2 configs
  python scripts/wave.py run K [--parallel 28]  # lock data, run all agents of fold K, kill leftovers
  python scripts/wave.py audit K                # audit every transcript of fold K -> results/audit_fold K.csv
  python scripts/wave.py archive K              # copy kept runs to results/<config>/<dataset>_fK/
  python scripts/wave.py rerun TASK [TASK ...]  # discard a run (integrity breach) and rerun it in a new dir
  python scripts/wave.py status [K]

Options: --configs opus-60 and --datasets energy slump ... restrict a command.
Everything is logged in results/wave_log.txt; verdicts go in results/audit_notes.md.
"""
import argparse
import datetime as dt
import gzip
import json
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONFIGS, DATA, DATASETS, ENV_PY, MAX_LITERALS, PKG, RESULTS, RUNS,  # noqa: E402
                    SETTINGS_TEMPLATE, SNAPSHOT, TASK_MAP, TMP_DIR, read_task_map, render_settings, domain,
                    result_dir, split, task_number)
from adr.agent import prepare_task, run_agent  # noqa: E402

# While agents run, the whole package (README, data, results, scripts) is unreadable: an agent
# must see nothing but its own task directory. Log lines are buffered until it is unlocked.
_buffer, _locked = [], False


def log(msg):
    line = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {msg}"
    print(line, flush=True)
    _buffer.append(line)
    if not _locked:
        RESULTS.mkdir(exist_ok=True)
        with open(RESULTS / "wave_log.txt", "a") as f:
            f.write("\n".join(_buffer) + "\n")
        _buffer.clear()


def lock():
    global _locked
    _locked = True
    os.chmod(PKG, 0o000)


def unlock():
    global _locked
    os.chmod(PKG, 0o755)
    _locked = False


def pids_under(root):
    root = str(Path(root).resolve())
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit() and int(p) != os.getpid():
            try:
                cwd = os.readlink(f"/proc/{p}/cwd")
            except OSError:
                continue
            if cwd == root or cwd.startswith(root + "/"):
                out.append(int(p))
    return out


def kill_under(root):
    """Kill every process whose working directory is under root (never by name pattern)."""
    pids = pids_under(root)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        time.sleep(2)
        pids = pids_under(root)
        if not pids:
            break
    return pids


def select(args):
    configs = args.configs or list(CONFIGS)
    datasets = args.datasets or DATASETS
    return configs, datasets


def cmd_prepare(args):
    k = args.fold
    configs, datasets = select(args)
    tm = read_task_map()
    rows = []
    for name in datasets:
        (Xtr, ytr), _ = split(name, k)
        for c in configs:
            tid = f"t{task_number(c, name, k):04d}"
            wd = RUNS / c / tid
            if wd.exists():
                raise SystemExit(f"{wd} exists: never reuse a task directory")
            prepare_task(wd, Xtr, ytr, MAX_LITERALS, python=str(ENV_PY),
                         budget_min=CONFIGS[c][1], library=SNAPSHOT, domain=domain(name))
            rows.append({"task": tid, "config": c, "dataset": name, "fold": k,
                         "workdir": str(wd), "attempt": 0})
    tm = pd.concat([tm, pd.DataFrame(rows)], ignore_index=True)
    TASK_MAP.parent.mkdir(exist_ok=True)
    tm.to_csv(TASK_MAP, index=False)
    log(f"prepared fold {k}: {len(rows)} tasks ({', '.join(configs)})")


def pending(k, configs, datasets):
    tm = read_task_map()
    tm = tm[(tm.fold == k) & tm.config.isin(configs) & tm.dataset.isin(datasets)]
    # only the latest attempt of a (config, dataset, fold) can be pending: a discarded run's
    # workdir has been moved to results/superseded/ by `rerun`
    tm = tm.sort_values("attempt").groupby(["config", "dataset", "fold"]).tail(1)
    return [r for _, r in tm.iterrows() if not (Path(r.workdir) / "run_summary.json").exists()]


def run_one(r):
    model, budget = CONFIGS[r.config]
    settings = render_settings(r.workdir, f"{r.config}_{r.task}")
    s = run_agent(r.workdir, backend="claude", model=model, budget_min=budget,
                  settings=str(settings), tmpdir=TMP_DIR / f"{r.config}_{r.task}")
    left = kill_under(r.workdir)  # background jobs must not change model.py after the deadline
    return r.task, s, left


def cmd_run(args):
    k = args.fold
    configs, datasets = select(args)
    todo = pending(k, configs, datasets)
    if not todo:
        log(f"fold {k}: nothing to run")
        return
    if not SETTINGS_TEMPLATE.exists():
        raise SystemExit(f"{SETTINGS_TEMPLATE} missing: run scripts/setup.sh and the probe first")
    kill_under(RUNS)
    log(f"fold {k}: locking the package while agents run")
    lock()
    log(f"fold {k}: running {len(todo)} agents, {args.parallel} at a time (data and results locked)")
    try:
        with ThreadPoolExecutor(args.parallel) as ex:
            futs = [ex.submit(run_one, r) for r in todo]
            for f in as_completed(futs):
                tid, s, left = f.result()
                log(f"  {tid}: {s['status']}, {s['minutes']} min, {s['num_turns']} turns"
                    + (f", leftover pids {left}" if left else ""))
    finally:
        kill_under(RUNS)
        unlock()
        log(f"fold {k}: done; leftovers killed, data and results unlocked")


def cmd_audit(args):
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit", PKG / "tools" / "audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.ALLOWED = [str(ENV_PY), "/dev/null", "/usr/bin/env", str(WS_ENV_BIN())]
    k = args.fold
    configs, datasets = select(args)
    tm = read_task_map()
    tm = tm[(tm.fold == k) & tm.config.isin(configs) & tm.dataset.isin(datasets)]
    rows = []
    for _, r in tm.iterrows():
        stream = Path(r.workdir) / "agent_stream.jsonl"
        if not stream.exists():
            rows.append({"task": r.task, "clean": False, "note": "no transcript"})
            continue
        row = audit.audit(r.workdir, stream, task=r.task, own=str(Path(r.workdir).resolve()))
        row.update(config=r.config, dataset=r.dataset, fold=k, attempt=r.attempt)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = RESULTS / f"audit_fold{k}.csv"
    df.to_csv(out, index=False)
    bad = df[~df.clean.astype(bool)]
    log(f"audit fold {k}: {len(df) - len(bad)}/{len(df)} clean; details in {out.name}")
    for _, r in bad.iterrows():
        print(f"  FLAG {r.task} ({r.get('config')}, {r.get('dataset')}): "
              f"web={r.get('web')} network={r.get('network')} outside={r.get('outside')}")


def WS_ENV_BIN():
    return ENV_PY.parent


def cmd_archive(args):
    k = args.fold
    configs, datasets = select(args)
    tm = read_task_map()
    tm = tm[(tm.fold == k) & tm.config.isin(configs) & tm.dataset.isin(datasets)]
    # the kept run of a (config, dataset, fold) is its latest attempt
    tm = tm.sort_values("attempt").groupby(["config", "dataset", "fold"]).tail(1)
    for _, r in tm.iterrows():
        src, dst = Path(r.workdir), result_dir(r.config, r.dataset, k)
        dst.mkdir(parents=True, exist_ok=True)
        for f in ["model.py", "notes.md", "TASK.md", "run_summary.json"]:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        if (src / "agent_stream.jsonl").exists():
            with open(src / "agent_stream.jsonl", "rb") as i, \
                    gzip.open(dst / "agent_stream.jsonl.gz", "wb") as o:
                shutil.copyfileobj(i, o)
        (dst / "task.json").write_text(json.dumps({"task": r.task, "attempt": int(r.attempt)}))
    log(f"archived fold {k}: {len(tm)} runs -> results/<config>/<dataset>_f{k}/")


def cmd_rerun(args):
    tm = read_task_map()
    new = []
    for tid in args.tasks:
        r = tm[tm.task == tid]
        if r.empty:
            raise SystemExit(f"unknown task {tid}")
        r = r.iloc[-1]
        att = int(tm[(tm.config == r.config) & (tm.dataset == r.dataset)
                     & (tm.fold == r.fold)].attempt.max()) + 1
        base = tid.split("r")[0]
        sup = RESULTS / "superseded" / f"{tid}"
        kill_under(r.workdir)
        shutil.move(r.workdir, sup)
        (sup / "REASON.txt").write_text(args.reason + "\n")
        wd = RUNS / r.config / f"{base}r{att}"
        (Xtr, ytr), _ = split(r.dataset, int(r.fold))
        prepare_task(wd, Xtr, ytr, MAX_LITERALS, python=str(ENV_PY),
                     budget_min=CONFIGS[r.config][1], library=SNAPSHOT, domain=domain(r.dataset))
        new.append({"task": f"{base}r{att}", "config": r.config, "dataset": r.dataset,
                    "fold": int(r.fold), "workdir": str(wd), "attempt": att})
        log(f"discarded {tid} ({args.reason}); rerun prepared as {base}r{att}")
    pd.concat([tm, pd.DataFrame(new)], ignore_index=True).to_csv(TASK_MAP, index=False)
    print("now run: python scripts/wave.py run <fold>  (only pending tasks are run)")


def cmd_status(args):
    tm = read_task_map()
    if args.fold:
        tm = tm[tm.fold == args.fold]
    tm["done"] = [(Path(w) / "run_summary.json").exists() for w in tm.workdir]
    print(tm.groupby(["fold", "config"]).done.agg(["sum", "count"]).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "run", "audit", "archive"]:
        p = sub.add_parser(name)
        p.add_argument("fold", type=int)
        p.add_argument("--configs", nargs="+", choices=list(CONFIGS))
        p.add_argument("--datasets", nargs="+", choices=DATASETS)
        if name == "run":
            p.add_argument("--parallel", type=int, default=28)
    p = sub.add_parser("rerun")
    p.add_argument("tasks", nargs="+")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("status")
    p.add_argument("fold", type=int, nargs="?")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "audit": cmd_audit, "archive": cmd_archive,
     "rerun": cmd_rerun, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
