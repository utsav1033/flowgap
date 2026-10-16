"""
flowgap — find the caller intents your voice agent's flow doesn't handle.

    flowgap init my-agent          # scaffold flow.yaml + transcripts/
    flowgap analyze                # transcripts/ + flow.yaml -> analysis.json
    flowgap report                 # headline number + ranked gaps
    flowgap specs --out specs/     # drafted node spec per gap
    flowgap serve                  # API for the dashboard

Heavy imports (sentence-transformers, UMAP, HDBSCAN) happen inside commands so
`--help` stays instant.
"""

import json
import os
import re
from importlib import resources
from pathlib import Path
from typing import Optional

import typer
from dotenv import find_dotenv, load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    help="Find the caller intents your voice agent's flow doesn't handle, and draft nodes to close them.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(highlight=False)

DEFAULT_ANALYSIS = Path("analysis.json")

AnalysisArg = typer.Argument(DEFAULT_ANALYSIS, help="analysis.json produced by `flowgap analyze`.")


def _load_env():
    # Look for .env starting from where the user runs the command, not the install dir
    load_dotenv(find_dotenv(usecwd=True))


def _fail(msg: str):
    console.print(f"[bold #ff8a3d]error:[/] {msg}")
    raise typer.Exit(1)


def _read_analysis(path: Path) -> dict:
    if not path.exists():
        _fail(f"{path} not found — run `flowgap analyze` first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "gap"


# One accent color; everything else is plain or dim
ACCENT = "#ff8a3d"


def _rate_style(rate: float) -> str:
    """Only high transfer rates get the accent, so the expensive gaps stand out."""
    return f"bold {ACCENT}" if rate >= 0.7 else "" if rate >= 0.3 else "dim"


def _yaml_text(spec: str) -> Text:
    """Comments dim, keys dim, values plain — a calmer read than full syntax colors."""
    out = Text()
    for line in spec.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out.append(line + "\n", style="dim")
            continue
        m = re.match(r"^(\s*(?:- )?)([\w-]+:)(.*)$", line)
        if m:
            out.append(m.group(1))
            out.append(m.group(2), style="dim")
            out.append(m.group(3) + "\n")
        else:
            out.append(line + "\n")
    out.rstrip()
    return out


def _is_keyword_label(g: dict) -> bool:
    return g.get("description", "").startswith("Keyword-derived")


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 1].rstrip() + "…"


def _next_steps(*steps: tuple[str, str]):
    width = max(len(cmd) for cmd, _ in steps)
    for cmd, what in steps:
        console.print(f"  [bold]{cmd.ljust(width)}[/bold]   [dim]{what}[/dim]")
    console.print()


def _title(subtitle: str):
    console.print()
    console.print(Text.assemble(("flowgap", f"bold {ACCENT}"), ("  ·  ", "dim"), (subtitle, "dim")))
    console.print()


def _print_report(analysis: dict, top: Optional[int] = None):
    """Tell the story in order: how much is missed → where it went → how to fix it."""
    m = analysis["metric"]
    total, missed, n = m["total_calls"], m["gap_calls"], m["n_gaps"]
    handled = total - missed
    _title(f"coverage report  ·  {total} calls")

    if not analysis["gaps"]:
        console.print(f"[bold]Your agent handled every recurring request[/bold] across {total} calls.\n")
        return

    console.print(Text.assemble(
        ("Your agent couldn't handle ", "bold"), (f"{m['gap_rate'] * 100:.0f}%", f"bold {ACCENT}"),
        (" of calls.", "bold"),
    ))
    console.print()

    # One bar = every call. Gap bars below use the same scale, so they add up to the orange part.
    width = min(console.width - 4, 64)
    per_char = total / width
    miss_chars = max(1, round(missed / per_char))
    console.print(Text.assemble(
        "  ", ("█" * (width - miss_chars), "grey35"), ("█" * miss_chars, ACCENT),
    ))
    left, right = f"{handled} handled", f"{missed} missed"
    console.print(Text.assemble(
        "  ", (left, "dim"), " " * max(1, width - len(left) - len(right)), (right, ACCENT),
    ))
    console.print()

    gaps = analysis["gaps"][:top] if top else analysis["gaps"]
    console.print(Text.assemble(
        (f"Where the {missed} missed calls went", "bold"),
        (f"  ·  {n} requests with no node in your flow", "dim"),
    ))
    console.print()

    bar_width = max(max(1, round(g["call_count"] / per_char)) for g in gaps)
    grid = Table.grid(padding=(0, 2), expand=True)
    grid.add_column(justify="right", style="dim")
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_column(width=bar_width, no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    for i, g in enumerate(gaps, 1):
        rate = g["transfer_rate"]
        quote = (g.get("sample_turns") or [""])[0]
        grid.add_row(
            str(i),
            Text.assemble((g["intent_name"], "bold"), "\n", (f"“{quote}”", "italic dim")),
            Text("█" * max(1, round(g["call_count"] / per_char)), style=ACCENT),
            Text.assemble(
                (f"{g['call_count']} calls", "bold"), "\n",
                (f"{rate * 100:.0f}% sent to a human", f"{ACCENT}" if rate >= 0.7 else "dim"),
            ),
        )
        grid.add_row()
    console.print(grid)

    if any(_is_keyword_label(g) for g in gaps):
        console.print("[dim]Names are keywords from the calls. Set GEMINI_API_KEY for readable ones.[/dim]\n")

    console.print(Text.assemble(
        ("Fix it", "bold"),
        (f"  ·  {n} node{'s' if n != 1 else ''} drafted, worth up to {missed} calls", "dim"),
    ))
    console.print()
    _next_steps(
        ("flowgap specs --gap 1", "review the fix for the biggest gap"),
        ("flowgap specs -o specs/", "save every drafted node as YAML"),
    )


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Folder to set up (created if missing)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing flow.yaml / .env."),
):
    """Scaffold a project: flow.yaml template, transcripts/ folder and .env."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcripts").mkdir(exist_ok=True)

    flow = directory / "flow.yaml"
    if flow.exists() and not force:
        console.print(f"[dim]skip[/dim]   {flow} (exists)")
    else:
        flow.write_text(resources.files("flowgap").joinpath("templates/flow.yaml").read_text())
        console.print(f"[dim]wrote[/dim] {flow}")

    env = directory / ".env"
    if env.exists() and not force:
        console.print(f"[dim]skip[/dim]   {env} (exists)")
    else:
        env.write_text(
            "# Optional: enables LLM cluster labels and drafted node specs.\n"
            "# Without it FlowGap uses keyword labels and template specs.\n"
            "GEMINI_API_KEY=\n"
        )
        console.print(f"[dim]wrote[/dim] {env}")

    console.print(
        "\nNext:\n"
        f"  1. Describe your agent's nodes in [bold]{flow}[/bold]\n"
        f"  2. Put call transcripts in [bold]{directory / 'transcripts'}[/bold]\n"
        "     (.txt with CALLER:/AGENT: lines, or .json with a \"transcript\" field)\n"
        f"  3. {'cd ' + str(directory) + ' && ' if directory != Path('.') else ''}flowgap analyze"
    )


@app.command()
def analyze(
    transcripts: Path = typer.Argument(Path("transcripts"), help="Folder of call transcripts (.txt or .json)."),
    flow: Path = typer.Option(Path("flow.yaml"), "--flow", "-f", help="Your agent's intended flow."),
    out: Path = typer.Option(DEFAULT_ANALYSIS, "--out", "-o", help="Where to write the analysis."),
    threshold: float = typer.Option(
        0.60, "--threshold", "-t",
        help="Similarity to a flow node needed to count as handled. Lower = fewer gaps.",
    ),
    org: Optional[str] = typer.Option(None, "--org", help="Business name used when drafting node specs."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip Gemini even if GEMINI_API_KEY is set."),
    ground_truth: Optional[Path] = typer.Option(
        None, "--ground-truth", help="Optional labels JSON (only for synthetic/eval data).",
    ),
):
    """Run the pipeline: parse → embed → cluster → label → detect gaps → draft nodes."""
    _load_env()
    if not transcripts.is_dir():
        _fail(f"transcripts folder {transcripts} not found.")
    if not flow.exists():
        _fail(f"flow file {flow} not found — pass --flow or run `flowgap init`.")

    from analyzer import llm
    if no_llm:
        llm.disable()
    if not llm.available():
        console.print("[dim]No GEMINI_API_KEY — using keyword labels and template node specs.[/dim]\n")

    from analyzer.run import run_pipeline
    try:
        analysis = run_pipeline(
            transcripts_dir=transcripts,
            flow_path=flow,
            out_path=out,
            ground_truth_path=ground_truth,
            sim_threshold=threshold,
            org_name=org,
        )
    except ValueError as e:
        _fail(str(e))
    _print_report(analysis)


@app.command()
def report(
    analysis: Path = AnalysisArg,
    top: Optional[int] = typer.Option(None, "--top", "-n", help="Only show the N biggest gaps."),
    as_json: bool = typer.Option(False, "--json", help="Print the metric + gaps as JSON."),
):
    """Show the headline number and the ranked gap list."""
    data = _read_analysis(analysis)
    if as_json:
        gaps = [{k: v for k, v in g.items() if k != "sample_turns"} for g in data["gaps"]]
        print(json.dumps({"metric": data["metric"], "gaps": gaps[:top] if top else gaps}, indent=2, ensure_ascii=False))
        return
    _print_report(data, top)


@app.command()
def specs(
    analysis: Path = AnalysisArg,
    gap: Optional[str] = typer.Option(None, "--gap", "-g", help="Gap rank (1, 2, …) or intent name."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write one <intent>.yaml per gap into this folder."),
):
    """Print or export the drafted node spec for each gap."""
    gaps = _read_analysis(analysis)["gaps"]
    if not gaps:
        console.print("No gaps — nothing to draft.")
        return

    ranked = list(enumerate(gaps, 1))
    gaps = ranked
    if gap is not None:
        if gap.isdigit():
            idx = int(gap) - 1
            if not 0 <= idx < len(ranked):
                _fail(f"gap #{gap} doesn't exist — there are {len(ranked)}.")
            gaps = [ranked[idx]]
        else:
            gaps = [(r, g) for r, g in ranked if g["intent_name"] == gap]
            if not gaps:
                _fail(f"no gap named {gap!r} — see `flowgap report`.")

    if out:
        out.mkdir(parents=True, exist_ok=True)
        for _, g in gaps:
            path = out / f"{_slug(g['intent_name'])}.yaml"
            path.write_text(g["node_spec"].rstrip() + "\n", encoding="utf-8")
            console.print(f"[dim]wrote[/dim] {path}")
        return

    total = len(ranked)
    for rank, g in gaps:
        rate = g["transfer_rate"]
        _title(f"fix for gap {rank} of {total}")
        console.print(Text(g["intent_name"], style="bold"))
        console.print(Text.assemble(
            (f"{g['call_count']} calls your agent couldn't handle", "dim"), ("  ·  ", "dim"),
            (f"{rate * 100:.0f}% sent to a human", ACCENT if rate >= 0.7 else "dim"),
        ))
        console.print("\n[bold]What callers said[/bold]")
        quote_width = max(console.width - 8, 40)
        for t in g.get("sample_turns", [])[:3]:
            console.print(Text(f"  “{_truncate(t, quote_width)}”", style="italic dim"), no_wrap=True)
        console.print()
        spec = _yaml_text(g["node_spec"].rstrip())
        console.print(Panel(spec, title="[bold]Drafted node[/bold] [dim]· review, then add to your flow[/dim]",
                            title_align="left", border_style="dim", box=box.ROUNDED, padding=(0, 1)))

    console.print("\n[bold]Next[/bold]\n")
    steps = [(f"flowgap specs --gap {rank} -o specs/", "save this node as YAML")]
    if rank < total:
        steps.insert(0, (f"flowgap specs --gap {rank + 1}", "next gap"))
    _next_steps(*steps)


@app.command("eval")
def eval_cmd(
    analysis: Path = AnalysisArg,
    ground_truth: Path = typer.Option(..., "--ground-truth", "-g", help="Planted-label JSON for the transcripts."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Also save the results as JSON."),
):
    """Score detected gaps against planted ground truth (synthetic data only)."""
    from eval.evaluate import evaluate, print_eval
    if not ground_truth.exists():
        _fail(f"{ground_truth} not found.")
    result = evaluate(_read_analysis(analysis), ground_truth)
    print_eval(result)
    if out:
        out.write_text(json.dumps(result, indent=2))
        console.print(f"\n[dim]wrote[/dim] {out}")


@app.command()
def serve(
    analysis: Path = AnalysisArg,
    flow: Path = typer.Option(Path("flow.yaml"), "--flow", "-f", help="Intended flow shown in the dashboard."),
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
):
    """Serve an analysis over HTTP for the dashboard (GET /api/analysis, /api/gaps, …)."""
    if not analysis.exists():
        _fail(f"{analysis} not found — run `flowgap analyze` first.")
    # api.main reads these at import time
    os.environ["FLOWGAP_ANALYSIS"] = str(analysis.resolve())
    os.environ["FLOWGAP_FLOW"] = str(flow.resolve())
    os.environ.setdefault("FLOWGAP_EVAL", str(analysis.resolve().with_name("eval_results.json")))
    import uvicorn
    uvicorn.run("api.main:app", host=host, port=port)


def main():
    app()


if __name__ == "__main__":
    main()
