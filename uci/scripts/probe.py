"""Harness probe (README step 2): run one real agent session with exactly the agents' flags and
sandbox settings, and verify isolation from what the tools actually returned.

  PY scripts/probe.py [--model claude-opus-5-5]

Before the session, marker files are planted where an agent must not reach: in $HOME, in another
task's directory, next to the settings files, in /tmp, and as a fake harness transcript
(/tmp/claude-<uid>/.../tasks/*.output). Each check passes only if the agent's command could not
read (or write) that exact file; a filtered view of a directory is not a failure, a readable
marker is. The package itself is locked (chmod 000) during the session, as in the experiment.

PASS requires:
  - exactly the six tools (Bash Read Write Edit Glob Grep); no MCP servers; no skills;
  - network blocked from Bash (curl and Python sockets fail);
  - no marker readable, no write possible outside the task directory, the Read tool refuses a
    file outside the task directory;
  - the agents' Python works and writing inside the task directory works (so the sandbox runs);
  - no identifying words (UCI, CDTree, Yang, benchmark, redo) in the session's initial context.
Writes results/probe_<timestamp>.json. Exit code 0 only if every check passes.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENV_PY, PKG, RESULTS, RUNS, SETTINGS_DIR, SETTINGS_TEMPLATE, TMP_DIR, WS, render_settings  # noqa: E402
from adr.agent import agent_flags  # noqa: E402

HOME = Path.home()
SECRET = "PROBE-MARKER-7f3a"
MARKERS = {  # name: path of a file the agent must not be able to read
    "home": HOME / "adr_probe_marker.txt",
    "other_task": RUNS / "zz_other_task" / "marker.txt",
    "settings": SETTINGS_DIR / "zz_marker.txt",
    "tmp": Path("/tmp") / "adr_probe_marker.txt",
    "harness": Path(f"/tmp/claude-{os.getuid()}") / "zz-probe-project" / "zz-session" / "tasks"
    / "zz_marker.output",
}
ESCAPES = {  # name: path the agent must not be able to create
    "write_ws": WS / "escape_test",
    "write_other_task": RUNS / "zz_other_task" / "escape_test",
    "write_tmp": Path("/tmp") / "adr_escape_test",
}


def checks():
    c = [("net_curl", "curl -sS -m 10 -o /dev/null -w '%{http_code}' https://example.com; echo EXIT=$?",
          lambda o: "EXIT=0" not in o or "000" in o),
         ("net_python", f"{ENV_PY} -c \"import urllib.request as u; u.urlopen('https://example.com', "
          f"timeout=10); print('NET_OK')\"; echo EXIT=$?", lambda o: "NET_OK" not in o)]
    for k, p in MARKERS.items():
        c.append((f"read_{k}", f"cat {p}; echo EXIT=$?", lambda o: SECRET not in o))
    c.append(("read_package", f"cat {PKG}/README.md | head -3; echo EXIT=$?",
              lambda o: "Agentic Distributional" not in o))
    c.append(("find_harness", f"find /tmp -name '*.output' 2>/dev/null | head; echo EXIT=$?",
              lambda o: "zz_marker" not in o))
    for k, p in ESCAPES.items():
        c.append((k, f"touch {p}; echo EXIT=$?", lambda o: True))  # verified on disk below
    c.append(("python_ok", f"{ENV_PY} -c \"import numpy, scipy, pandas; print('PY_OK')\"",
              lambda o: "PY_OK" in o))
    c.append(("write_inside", "touch ./inside_test && echo WRITE_OK", lambda o: "WRITE_OK" in o))
    return c


LEAK = re.compile(r"cdtree|\buci\b|yang|leeuwen|benchmark|/redo\b|redo/", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5-5")
    a = ap.parse_args()
    if not SETTINGS_TEMPLATE.exists():
        raise SystemExit("run scripts/setup.sh first")
    for p in ESCAPES.values():
        p.unlink(missing_ok=True)
    for p in MARKERS.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(SECRET + "\n")
    wd = RUNS / "probe" / dt.datetime.now().strftime("p%Y%m%d%H%M%S")
    wd.mkdir(parents=True)
    settings = render_settings(wd, "probe")
    cl = checks()
    prompt = ("This is a test of your environment. Run each of the following shell commands with "
              "the Bash tool, one tool call per command, exactly as written, in this order. Do not "
              "retry, modify or work around a command that fails; just go on to the next one.\n\n"
              + "\n".join(f"{i + 1}. {c}" for i, (_, c, _) in enumerate(cl)) +
              f"\n\nThen use the Read tool to read the file {HOME}/.bashrc. Finally, reply with the "
              "exact names of all tools you have available, one per line.")
    cmd = ["claude", "-p", prompt] + agent_flags(a.model, settings)
    tmp = TMP_DIR / "probe"
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CLAUDE_CODE_TMPDIR": str(tmp), "TMPDIR": str(tmp)}  # as in run_agent
    os.chmod(PKG, 0o000)  # as during the experiment
    try:
        out = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True, timeout=1200)
    finally:
        os.chmod(PKG, 0o755)
    recs = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    init = next((r for r in recs if r.get("type") == "system" and r.get("subtype") == "init"), {})
    uses, results = {}, {}
    for r in recs:
        msg = r.get("message")
        for c in (msg.get("content") if isinstance(msg, dict) else None) or []:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "tool_use":
                uses[c["id"]] = (c["name"], c.get("input", {}))
            elif c.get("type") == "tool_result":
                cont = c.get("content")
                results[c["tool_use_id"]] = cont if isinstance(cont, str) else json.dumps(cont)
    report = {"model": a.model, "workdir": str(wd), "returncode": out.returncode,
              "stderr": out.stderr[-2000:], "checks": {}, "outputs": {}}
    report["init"] = {k: init.get(k) for k in ("tools", "mcp_servers", "skills", "slash_commands",
                                               "model", "permissionMode")}
    ch = report["checks"]
    ch["tools_exactly_six"] = sorted(init.get("tools", [])) == sorted(
        ["Bash", "Read", "Write", "Edit", "Glob", "Grep"])
    ch["no_mcp_servers"] = not init.get("mcp_servers")
    ch["no_skills"] = not (init.get("skills") or init.get("slash_commands"))
    bash = [(inp.get("command", ""), results.get(i, "")) for i, (n, inp) in uses.items()
            if n == "Bash"]
    for name, command, ok in cl:
        hit = [o for c, o in bash if c.strip() == command.strip()]
        ch[name] = bool(hit) and ok(hit[0])
        report["outputs"][name] = hit[0][:300] if hit else "(command not run)"
    for name, p in ESCAPES.items():  # the decisive test for writes is the file system itself
        ch[name] = ch[name] and not p.exists()
    reads = [results.get(i, "") for i, (n, inp) in uses.items()
             if n == "Read" and ".bashrc" in str(inp)]
    ch["read_tool_confined"] = bool(reads) and not any(
        "alias" in r or "export" in r or "PS1" in r for r in reads)
    report["outputs"]["read_tool"] = reads[0][:300] if reads else "(not run)"
    ch["no_identifying_words"] = not LEAK.search(json.dumps(init))
    report["passed"] = all(ch.values())
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"probe_{wd.name}.json"
    path.write_text(json.dumps(report, indent=1))
    for k, v in ch.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}" + ("" if v else f"   -> {report['outputs'].get(k, '')}"))
    print(f"\n{'ALL CHECKS PASSED' if report['passed'] else 'PROBE FAILED'}; details in {path}")
    for p in list(MARKERS.values()) + list(ESCAPES.values()):
        p.unlink(missing_ok=True)
    shutil.rmtree(MARKERS["harness"].parents[2], ignore_errors=True)
    shutil.rmtree(RUNS / "zz_other_task", ignore_errors=True)
    shutil.rmtree(RUNS / "probe", ignore_errors=True)
    (SETTINGS_DIR / "probe.json").unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
