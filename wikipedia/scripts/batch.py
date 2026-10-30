"""Run the agents on the new dataset: 5 folds x the configurations of common.CONFIGS.

  PY scripts/batch.py prepare FOLD             # task dir(s) of one fold (training rows only)
  PY scripts/batch.py run FOLD                 # lock the package, run the pending agents, kill leftovers
  PY scripts/batch.py audit FOLD               # audit the transcripts -> results/audit_fold<k>.csv
  PY scripts/batch.py archive FOLD             # copy kept runs to results/<config>/fold<k>/
  PY scripts/batch.py retire FOLD              # move the fold's task dirs to WS/done and lock them
  PY scripts/batch.py cycle [FOLD ...]         # prepare, run, audit, archive, retire, fold by fold;
                                               # stops at the first audit flag
  PY scripts/batch.py rerun TASK --reason "..."  # discard a run and prepare a rerun
  PY scripts/batch.py status

Folds share rows (the training rows of fold k contain the test rows of fold k'), so at most one
fold is on disk at any time, as in the UCI study: a fold's task dirs are retired before the next
fold is prepared. Test rows are never written to a task directory.
Everything is logged in results/batch_log.txt; verdicts go in results/audit_notes.md.
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
from common import (CONFIGS, DONE, ENV_PY, FOLDS, MAX_LITERALS, PKG, RESULTS, RUNS,  # noqa: E402
                    SETTINGS_TEMPLATE, SNAPSHOT, TASK_MAP, TMP_DIR, read_task_map,
                    render_settings, result_dir, split, task_id, domain)
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
        with open(RESULTS / "batch_log.txt", "a") as f:
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


def current(tm, fold):
    """Latest attempt of every (config, fold) task of this fold."""
    tm = tm[tm.fold == fold]
    return tm.sort_values("attempt").groupby(["config", "fold"]).tail(1)


def write_task(wd, fold, config):
    Xtr, Ytr, _, _ = split(fold)
    prepare_task(wd, Xtr, Ytr, MAX_LITERALS, python=str(ENV_PY),
                 budget_min=CONFIGS[config][1], library=SNAPSHOT, domain=domain())


def cmd_prepare(fold):
    other = [p for p in RUNS.glob("*/t*") if p.is_dir()]
    if other:
        raise SystemExit(f"task dirs still on disk ({other[0]} ...): retire them first")
    tm = read_task_map()
    rows = []
    for c in CONFIGS:
        tid = task_id(c, fold)
        wd = RUNS / c / tid
        if wd.exists() or (DONE / c / tid).exists():
            raise SystemExit(f"{wd} exists or was used: never reuse a task directory")
        write_task(wd, fold, c)
        rows.append({"task": tid, "config": c, "fold": fold, "workdir": str(wd), "attempt": 0})
    tm = pd.concat([tm, pd.DataFrame(rows)], ignore_index=True)
    TASK_MAP.parent.mkdir(exist_ok=True)
    tm.to_csv(TASK_MAP, index=False)
    log(f"prepared fold {fold}: {len(rows)} task(s)")


def run_one(r):
    model, budget = CONFIGS[r.config]
    settings = render_settings(r.workdir, f"{r.config}_{r.task}")
    s = run_agent(r.workdir, backend="claude", model=model, budget_min=budget,
                  settings=str(settings), tmpdir=TMP_DIR / f"{r.config}_{r.task}")
    left = kill_under(r.workdir)  # background jobs must not change model.py after the deadline
    return r.task, s, left


def cmd_run(fold, parallel=2):
    todo = [r for _, r in current(read_task_map(), fold).iterrows()
            if not (Path(r.workdir) / "run_summary.json").exists()]
    if not todo:
        log(f"fold {fold}: nothing to run")
        return
    if not SETTINGS_TEMPLATE.exists():
        raise SystemExit(f"{SETTINGS_TEMPLATE} missing: run scripts/setup.sh and the probe first")
    kill_under(RUNS)
    log(f"fold {fold}: locking the package while {len(todo)} agent(s) run")
    lock()
    try:
        with ThreadPoolExecutor(parallel) as ex:
            futs = [ex.submit(run_one, r) for r in todo]
            for f in as_completed(futs):
                tid, s, left = f.result()
                log(f"  {tid}: {s['status']}, {s['minutes']} min, {s['num_turns']} turns, "
                    f"${s['cost_usd']:.2f}" + (f", leftover pids {left}" if left else ""))
    finally:
        kill_under(RUNS)
        unlock()
        log(f"fold {fold}: done; leftovers killed, package unlocked")


def cmd_audit(fold):
    import importlib.util
    spec = importlib.util.spec_from_file_location("audit", PKG / "tools" / "audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.ALLOWED = [str(ENV_PY), str(ENV_PY.parent), "/dev/null", "/usr/bin/env"]
    rows = []
    for _, r in current(read_task_map(), fold).iterrows():
        stream = Path(r.workdir) / "agent_stream.jsonl"
        if not stream.exists():
            rows.append({"task": r.task, "clean": False, "note": "no transcript"})
            continue
        row = audit.audit(r.workdir, stream, task=r.task, own=str(Path(r.workdir).resolve()))
        row.update(config=r.config, fold=r.fold, attempt=r.attempt)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"audit_fold{fold}.csv", index=False)
    bad = df[~df.clean.astype(bool)]
    log(f"fold {fold} audit: {len(df) - len(bad)}/{len(df)} clean")
    for _, r in bad.iterrows():
        print(f"  FLAG {r.task} ({r.get('config')}): web={r.get('web')} "
              f"network={r.get('network')} outside={r.get('outside')} {r.get('note', '')}")
    return bad.empty


def cmd_archive(fold):
    tm = current(read_task_map(), fold)
    for _, r in tm.iterrows():
        src, dst = Path(r.workdir), result_dir(r.config, int(r.fold))
        dst.mkdir(parents=True, exist_ok=True)
        for f in ["model.py", "notes.md", "TASK.md", "run_summary.json"]:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        if (src / "agent_stream.jsonl").exists():
            with open(src / "agent_stream.jsonl", "rb") as i, \
                    gzip.open(dst / "agent_stream.jsonl.gz", "wb") as o:
                shutil.copyfileobj(i, o)
        (dst / "task.json").write_text(json.dumps({"task": r.task, "attempt": int(r.attempt)}))
    log(f"fold {fold}: archived {len(tm)} run(s) -> results/<config>/fold{fold}/")


def cmd_retire(fold):
    tm = read_task_map()
    for _, r in tm[tm.fold == fold].iterrows():
        src = Path(r.workdir)
        if not src.exists():
            continue
        kill_under(src)
        dst = DONE / r.config / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst)
        os.chmod(dst, 0o000)
    log(f"fold {fold}: task dirs moved to {DONE} and locked")


def cmd_cycle(folds, parallel):
    for k in folds:
        cmd_prepare(k)
        cmd_run(k, parallel)
        if not cmd_audit(k):
            log(f"fold {k}: audit flags; stopping before archiving. Review results/audit_fold{k}.csv")
            return
        cmd_archive(k)
        cmd_retire(k)
    log(f"cycle done: folds {folds}")


def cmd_rerun(tasks, reason):
    tm = read_task_map()
    new = []
    for tid in tasks:
        r = tm[tm.task == tid]
        if r.empty:
            raise SystemExit(f"unknown task {tid}")
        r = r.iloc[-1]
        att = int(tm[(tm.config == r.config) & (tm.fold == r.fold)].attempt.max()) + 1
        sup = RESULTS / "superseded" / tid
        sup.parent.mkdir(parents=True, exist_ok=True)
        src = Path(r.workdir)
        if src.exists():
            kill_under(src)
            shutil.move(str(src), sup)
        else:  # already retired: take the locked task dir from WS/done
            done = DONE / r.config / src.name
            os.chmod(done, 0o755)
            shutil.move(str(done), sup)
        arch = result_dir(r.config, int(r.fold))
        if arch.exists():  # the archived copy of the discarded run goes with it
            shutil.move(str(arch), sup / "archived")
        (sup / "REASON.txt").write_text(reason + "\n")
        base = tid.split("r")[0]
        wd = RUNS / r.config / f"{base}r{att}"
        write_task(wd, int(r.fold), r.config)
        new.append({"task": f"{base}r{att}", "config": r.config, "fold": int(r.fold),
                    "workdir": str(wd), "attempt": att})
        log(f"discarded {tid} ({reason}); rerun prepared as {base}r{att}")
    pd.concat([tm, pd.DataFrame(new)], ignore_index=True).to_csv(TASK_MAP, index=False)


def cmd_status():
    tm = read_task_map()
    if tm.empty:
        print("no tasks")
        return
    tm["done"] = [(Path(w) / "run_summary.json").exists() or
                  (DONE / c / Path(w).name).exists() for w, c in zip(tm.workdir, tm.config)]
    print(tm.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "run", "audit", "archive", "retire"]:
        p = sub.add_parser(name)
        p.add_argument("fold", type=int, choices=range(1, FOLDS + 1))
        if name == "run":
            p.add_argument("--parallel", type=int, default=2)
    p = sub.add_parser("cycle")
    p.add_argument("folds", nargs="*", type=int, default=list(range(1, FOLDS + 1)))
    p.add_argument("--parallel", type=int, default=2)
    p = sub.add_parser("rerun")
    p.add_argument("tasks", nargs="+")
    p.add_argument("--reason", required=True)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "prepare":
        cmd_prepare(a.fold)
    elif a.cmd == "run":
        cmd_run(a.fold, a.parallel)
    elif a.cmd == "audit":
        cmd_audit(a.fold)
    elif a.cmd == "archive":
        cmd_archive(a.fold)
    elif a.cmd == "retire":
        cmd_retire(a.fold)
    elif a.cmd == "cycle":
        cmd_cycle(a.folds, a.parallel)
    elif a.cmd == "rerun":
        cmd_rerun(a.tasks, a.reason)
    else:
        cmd_status()


if __name__ == "__main__":
    main()
