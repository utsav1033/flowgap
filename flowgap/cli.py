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
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer(
    help="Find the caller intents your voice agent's flow doesn't handle, and draft nodes to close them.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DEFAULT_ANALYSIS = Path("analysis.json")

AnalysisArg = typer.Argument(DEFAULT_ANALYSIS, help="analysis.json produced by `flowgap analyze`.")


def _load_env():
    # Look for .env starting from where the user runs the command, not the install dir
    load_dotenv(find_dotenv(usecwd=True))


def _fail(msg: str):
    console.print(f"[red]error:[/red] {msg}")
    raise typer.Exit(1)


def _read_analysis(path: Path) -> dict:
    if not path.exists():
        _fail(f"{path} not found — run `flowgap analyze` first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "gap"


def _print_report(analysis: dict, top: Optional[int] = None):
    m = analysis["metric"]
    console.print(
        f"\n[bold]{m['gap_rate'] * 100:.1f}%[/bold] of calls "
        f"([bold]{m['gap_calls']}[/bold] of {m['total_calls']}) hit an intent your flow doesn't handle "
        f"— {m['n_gaps']} gap{'s' if m['n_gaps'] != 1 else ''} found.\n"
    )
    gaps = analysis["gaps"][:top] if top else analysis["gaps"]
    if not gaps:
        return
    table = Table(show_edge=False, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Gap intent")
    table.add_column("Calls", justify="right")
    table.add_column("Transfer", justify="right")
    table.add_column("Closest node", style="dim")
    table.add_column("Sim", justify="right", style="dim")
    for i, g in enumerate(gaps, 1):
        table.add_row(
            str(i),
            g["intent_name"],
            str(g["call_count"]),
            f"{g['transfer_rate'] * 100:.0f}%",
            g.get("best_flow_match", ""),
            f"{g.get('max_flow_similarity', 0):.2f}",
        )
    console.print(table)
    console.print("\n[dim]See a drafted node:[/dim] flowgap specs --gap 1\n")


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
        console.print(f"[yellow]skip[/yellow]  {flow} (exists)")
    else:
        flow.write_text(resources.files("flowgap").joinpath("templates/flow.yaml").read_text())
        console.print(f"[green]wrote[/green] {flow}")

    env = directory / ".env"
    if env.exists() and not force:
        console.print(f"[yellow]skip[/yellow]  {env} (exists)")
    else:
        env.write_text(
            "# Optional: enables LLM cluster labels and drafted node specs.\n"
            "# Without it FlowGap uses keyword labels and template specs.\n"
            "GEMINI_API_KEY=\n"
        )
        console.print(f"[green]wrote[/green] {env}")

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

    if gap is not None:
        if gap.isdigit():
            idx = int(gap) - 1
            if not 0 <= idx < len(gaps):
                _fail(f"gap #{gap} doesn't exist — there are {len(gaps)}.")
            gaps = [gaps[idx]]
        else:
            gaps = [g for g in gaps if g["intent_name"] == gap]
            if not gaps:
                _fail(f"no gap named {gap!r} — see `flowgap report`.")

    if out:
        out.mkdir(parents=True, exist_ok=True)
        for g in gaps:
            path = out / f"{_slug(g['intent_name'])}.yaml"
            path.write_text(g["node_spec"].rstrip() + "\n", encoding="utf-8")
            console.print(f"[green]wrote[/green] {path}")
        return

    for g in gaps:
        console.rule(f"[bold]{g['intent_name']}[/bold]  {g['call_count']} calls, "
                     f"{g['transfer_rate'] * 100:.0f}% transferred")
        quote_width = max(console.width - 6, 40)
        for t in g.get("sample_turns", [])[:3]:
            t = t if len(t) <= quote_width else t[:quote_width - 1].rstrip() + "…"
            console.print(f"[dim]  “{t}”[/dim]", no_wrap=True)
        console.print()
        console.print(Syntax(g["node_spec"], "yaml", theme="ansi_dark", background_color="default", word_wrap=True))


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
        console.print(f"\n[green]wrote[/green] {out}")


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
