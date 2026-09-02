#!/usr/bin/env python3
"""
scripts/scan_repo.py — Claude-Code Style Terminal CLI for Semantic Drift Detection.

Features:
  - Rich interactive terminal UI with colored panels, spinners, and syntax highlighting.
  - Automatic detection of fine-tuned CodeBERT Joint Encoder (Model B).
  - High-precision joint self-attention drift scoring between code & docstrings.
  - Interactive inspection mode (--interactive / -i) to step through flagged code.
  - Markdown & JSON export capabilities.

Usage Examples:
    python scripts/scan_repo.py .
    python scripts/scan_repo.py semdrift --threshold 0.60
    python scripts/scan_repo.py . --interactive
    python scripts/scan_repo.py . --output markdown --output_file drift_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from typing import List, Dict, Any

# Suppress verbose library warnings for pristine terminal output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer

transformers.logging.set_verbosity_error()

# Ensure Windows terminal uses UTF-8 encoding for Rich icons & progress bars
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Rich terminal formatting imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.theme import Theme
    from rich.prompt import Prompt
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from semdrift.parser.ast_parser import ASTParser, FunctionInfo
from semdrift.parser.formatter import ModelInputFormatter
from semdrift.models.joint_encoder import JointEncoderModel, make_collate_fn, extract_docstring_summary

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "experiments", "v2", "joint_encoder_checkpoint.pt")
FALLBACK_CHECKPOINT = os.path.join(PROJECT_ROOT, "data", "labeled", "joint_encoder_checkpoint.pt")

# Custom theme for subtle, unified cyan/slate aesthetics
custom_theme = Theme({
    "info": "cyan",
    "warning": "cyan",
    "danger": "cyan",
    "success": "cyan",
    "accent": "cyan",
    "muted": "dim white",
})

console = Console(theme=custom_theme, legacy_windows=False) if RICH_AVAILABLE else None


class InferenceDataset(Dataset):
    """Dataset for batching raw (docstring, code, metadata) tuples for inference."""

    def __init__(self, records: List[Dict[str, Any]], clean_docs: bool = True):
        self.records = records
        self.clean_docs = clean_docs

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        raw_doc = rec.get("docstring", "")
        docstring = extract_docstring_summary(raw_doc) if self.clean_docs else raw_doc
        code = rec.get("code", "")
        label_placeholder = 0  # Unused during inference
        meta = rec
        return docstring, code, label_placeholder, meta


def resolve_target_path(path: str) -> str:
    """Resolves target path checking CWD first, then PROJECT_ROOT."""
    if os.path.exists(path):
        return path
    alt_path = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(alt_path):
        return alt_path
    return path


def find_checkpoint(user_path: str | None) -> str:
    """Locates an existing model checkpoint path."""
    if user_path:
        if os.path.exists(user_path):
            return user_path
        alt_user_path = os.path.join(PROJECT_ROOT, user_path)
        if os.path.exists(alt_user_path):
            return alt_user_path
    if os.path.exists(DEFAULT_CHECKPOINT):
        return DEFAULT_CHECKPOINT
    if os.path.exists(FALLBACK_CHECKPOINT):
        return FALLBACK_CHECKPOINT
    raise FileNotFoundError(
        f"Checkpoint file not found at user path '{user_path}', default '{DEFAULT_CHECKPOINT}', "
        f"or fallback '{FALLBACK_CHECKPOINT}'. Please specify a valid checkpoint with --checkpoint."
    )


def render_banner():
    """Renders a subtle, modern minimal header banner."""
    if not RICH_AVAILABLE:
        print("\n" + "=" * 70)
        print(" SemDrift AI — Semantic Drift Detector (Model B Joint Encoder)")
        print("=" * 70 + "\n")
        return

    banner_text = Text()
    banner_text.append("SemDrift ", style="bold cyan")
    banner_text.append("│ ", style="dim cyan")
    banner_text.append("Code & Docstring Semantic Contract Scanner\n", style="bold white")
    banner_text.append("Model: ", style="dim white")
    banner_text.append("CodeBERT Joint-Encoder  ", style="cyan")
    banner_text.append("Device: ", style="dim white")
    banner_text.append(f"{DEFAULT_DEVICE.upper()}  ", style="cyan")
    banner_text.append("Attention: ", style="dim white")
    banner_text.append("Joint Self-Attention", style="cyan")

    console.print(Panel(banner_text, border_style="dim cyan", padding=(0, 2), box=box.ROUNDED))


def extract_functions(target_path: str, skip_test_files: bool = True) -> tuple[List[Dict[str, Any]], int, int]:
    """Extracts function information using ASTParser."""
    resolved_path = resolve_target_path(target_path)
    parser = ASTParser(
        skip_dunder=False,
        skip_private=False,
        skip_test_files=skip_test_files,
    )

    if os.path.isdir(resolved_path):
        base_path = os.path.abspath(resolved_path)
        functions: List[FunctionInfo] = parser.parse_directory(resolved_path)
    elif os.path.isfile(resolved_path):
        base_path = os.path.dirname(os.path.abspath(resolved_path)) or "."
        functions: List[FunctionInfo] = parser.parse_file(resolved_path)
    else:
        raise ValueError(f"Target path does not exist: {target_path} (checked {os.path.abspath(target_path)} and {os.path.abspath(resolved_path)})")

    total_count = len(functions)
    documented_functions = [f for f in functions if f.docstring and f.docstring.strip()]
    undocumented_count = total_count - len(documented_functions)

    formatter = ModelInputFormatter(
        base_path=base_path,
        include_undocumented=False,
        normalise_docstring=False,
    )
    records = formatter.format(documented_functions)

    # Attach file_path, line_number, and raw docstring to records
    for rec in records:
        for fn in documented_functions:
            if rec["code"] == fn.source_code:
                rec["file_path"] = fn.file_path
                rec["line_number"] = fn.line_start
                rec["function_name"] = fn.name
                rec["class_name"] = fn.class_name
                rec["raw_docstring"] = fn.docstring
                break

    return records, total_count, undocumented_count


def run_inference(
    records: List[Dict[str, Any]],
    checkpoint_path: str,
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 16,
    max_length: int = 512,
    doc_max_tokens: int = 96,
    code_truncation: str = "head_tail",
    pooling: str = "cls",
    device: str = DEFAULT_DEVICE,
    clean_docstrings: bool = True,
) -> List[Dict[str, Any]]:
    """Runs model B inference on extracted function records and attaches drift scores."""
    if not records:
        return []

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=40, style="dim white", complete_style="cyan"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            load_task = progress.add_task("Loading CodeBERT Joint Encoder...", total=100)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = JointEncoderModel(model_name=model_name, pooling=pooling, num_labels=2)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            model.eval()
            progress.update(load_task, completed=100)

            dataset = InferenceDataset(records, clean_docs=clean_docstrings)
            collate_fn = make_collate_fn(
                tokenizer,
                max_length=max_length,
                doc_max_tokens=doc_max_tokens,
                truncation_strategy=code_truncation,
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            infer_task = progress.add_task("Analyzing semantic drift via joint self-attention...", total=len(dataloader))
            results = []

            with torch.no_grad():
                for inputs, _, metas in dataloader:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    logits = model(inputs)
                    probs = F.softmax(logits, dim=-1)
                    drift_probs = probs[:, 1].cpu().tolist()

                    for meta, prob in zip(metas, drift_probs):
                        item = dict(meta)
                        item["drift_probability"] = round(prob, 4)
                        item["is_drifted"] = bool(prob >= 0.5)
                        results.append(item)
                    progress.advance(infer_task)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = JointEncoderModel(model_name=model_name, pooling=pooling, num_labels=2)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        dataset = InferenceDataset(records, clean_docs=clean_docstrings)
        collate_fn = make_collate_fn(
            tokenizer,
            max_length=max_length,
            doc_max_tokens=doc_max_tokens,
            truncation_strategy=code_truncation,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        results = []

        with torch.no_grad():
            for inputs, _, metas in dataloader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(inputs)
                probs = F.softmax(logits, dim=-1)
                drift_probs = probs[:, 1].cpu().tolist()

                for meta, prob in zip(metas, drift_probs):
                    item = dict(meta)
                    item["drift_probability"] = round(prob, 4)
                    item["is_drifted"] = bool(prob >= 0.5)
                    results.append(item)

    results.sort(key=lambda x: x["drift_probability"], reverse=True)
    return results


def render_dashboard(target_path: str, total_funcs: int, undoc_funcs: int, flagged: List[Dict[str, Any]], threshold: float):
    """Renders a subtle summary dashboard."""
    if not RICH_AVAILABLE:
        return

    doc_count = total_funcs - undoc_funcs
    flagged_count = len(flagged)
    drift_pct = (flagged_count / doc_count * 100) if doc_count > 0 else 0.0

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim white")
    table.add_column("Value", style="cyan")
    table.add_column("Key2", style="dim white")
    table.add_column("Value2", style="cyan")

    table.add_row("Target Path", f"[dim white]{os.path.abspath(target_path)}[/dim white]", "Threshold", f"{threshold:.2f}")
    table.add_row("Total Functions", str(total_funcs), "Documented", str(doc_count))
    table.add_row("Undocumented", str(undoc_funcs), "Flagged Drift", f"[bold cyan]{flagged_count}[/bold cyan] ({drift_pct:.1f}%)")

    status_msg = f"{flagged_count} SEMANTIC DRIFT(S) DETECTED" if flagged_count > 0 else "ALL DOCSTRINGS ALIGNED WITH CODE"

    panel = Panel(
        table,
        title=f"[cyan bold] {status_msg} [/cyan bold]",
        border_style="dim cyan",
        subtitle="[dim white]Joint Self-Attention Validation Engine[/dim white]",
        box=box.ROUNDED,
    )
    console.print(panel)


def get_confidence_badge(prob: float) -> str:
    """Returns a clean, subtle confidence tag following the cyan/slate theme."""
    pct = prob * 100
    if prob >= 0.90:
        return f"[bold red]CRITICAL DRIFT ({pct:.1f}%)[/bold red]"
    elif prob >= 0.70:
        return f"[bold yellow]HIGH DRIFT ({pct:.1f}%)[/bold yellow]"
    elif prob >= 0.50:
        return f"[cyan]MODERATE DRIFT ({pct:.1f}%)[/cyan]"
    else:
        return f"[dim cyan]ALIGNED ({pct:.1f}%)[/dim cyan]"


def render_flagged_cards(flagged: List[Dict[str, Any]], top_k: int | None):
    """Displays findings as clean, subtle cyan-bordered cards."""
    if not RICH_AVAILABLE:
        return

    display_items = flagged[:top_k] if top_k else flagged

    if not display_items:
        console.print("\n[dim cyan]No function contracts violated the drift threshold.[/dim cyan]\n")
        return

    console.print(f"\n[dim white]Flagged Function Contracts ({len(display_items)} shown):[/dim white]\n")

    for idx, item in enumerate(display_items, 1):
        fn_id = item.get("function_id", "unknown")
        prob = item.get("drift_probability", 0.0)
        file_p = item.get("file_path", "unknown")
        line_n = item.get("line_number", "?")
        raw_doc = item.get("raw_docstring") or item.get("docstring", "")
        code = item.get("code", "")

        badge = get_confidence_badge(prob)

        # Header info
        header_text = Text()
        header_text.append(f"#{idx}  ", style="dim white")
        header_text.append(f"{fn_id}\n", style="bold cyan")
        header_text.append(f"{file_p}:{line_n}", style="dim white")

        # Docstring preview panel
        clean_doc_summary = extract_docstring_summary(raw_doc)
        doc_content = f"[dim white]\"{clean_doc_summary}\"[/dim white]"

        # Syntax highlighted code snippet using clean github-dark theme
        code_syntax = Syntax(code, "python", theme="github-dark", line_numbers=True, word_wrap=True)

        grid = Table(show_header=False, box=None, padding=(0, 0))
        grid.add_column("Content")
        grid.add_row(header_text)
        grid.add_row(Text(""))
        grid.add_row(Text("Docstring Specification:", style="dim white"))
        grid.add_row(Panel(doc_content, border_style="dim cyan", box=box.ROUNDED))
        grid.add_row(Text("Implementation Code:", style="dim white"))
        grid.add_row(Panel(code_syntax, border_style="dim cyan", box=box.ROUNDED))

        card = Panel(
            grid,
            title=f" {badge} ",
            title_align="left",
            border_style="dim cyan",
            padding=(1, 2),
            box=box.ROUNDED,
        )
        console.print(card)
        console.print("")


def interactive_mode(flagged: List[Dict[str, Any]]):
    """Claude Code-style interactive step-through inspection loop."""
    if not RICH_AVAILABLE or not flagged:
        return

    console.print("\n[bold cyan]Entering Interactive Review Mode...[/bold cyan] (Press [bold yellow]Enter[/bold yellow] for next, [bold yellow]q[/bold yellow] to quit)")
    idx = 0
    total = len(flagged)

    while idx < total:
        item = flagged[idx]
        console.clear()
        console.print(f"[dim]Reviewing {idx + 1} of {total} flagged functions[/dim]\n")
        render_flagged_cards([item], top_k=1)

        choice = Prompt.ask(
            "[bold cyan]Action[/bold cyan] ([bold yellow]n[/bold yellow]ext, [bold yellow]p[/bold yellow]rev, [bold yellow]q[/bold yellow]uit)",
            choices=["n", "p", "q", ""],
            default="n",
            show_choices=False,
        )

        if choice in ("n", ""):
            idx += 1
        elif choice == "p":
            idx = max(0, idx - 1)
        elif choice == "q":
            break


def format_markdown_output(
    target_path: str,
    total_funcs: int,
    undoc_funcs: int,
    flagged: List[Dict[str, Any]],
    threshold: float,
    top_k: int | None,
) -> str:
    display_items = flagged[:top_k] if top_k else flagged
    lines = []
    lines.append("# SemDrift — Semantic Drift Detection Report\n")
    lines.append(f"- **Target Path**: `{os.path.abspath(target_path)}`")
    lines.append(f"- **Total Functions Found**: {total_funcs}")
    lines.append(f"- **Documented Functions Analyzed**: {total_funcs - undoc_funcs}")
    lines.append(f"- **Undocumented Functions Skipped**: {undoc_funcs}")
    lines.append(f"- **Drift Threshold**: `{threshold:.2f}`")
    lines.append(f"- **Flagged Functions**: `{len(flagged)}` / `{total_funcs - undoc_funcs}`\n")

    lines.append("## Flagged Functions Ranking\n")
    lines.append("| Rank | Function ID | Location | Drift Confidence | Docstring Summary |")
    lines.append("| :--- | :--- | :--- | :---: | :--- |")

    for idx, item in enumerate(display_items, 1):
        fn_id = item.get("function_id", "unknown")
        prob = item.get("drift_probability", 0.0)
        file_p = item.get("file_path", "unknown")
        line_n = item.get("line_number", "?")
        doc_preview = extract_docstring_summary(item.get("docstring", ""))[:60].replace("|", "\\|")
        lines.append(f"| {idx} | `{fn_id}` | `{file_p}:{line_n}` | **{prob:.4f}** | {doc_preview} |")

    lines.append("\n---\n")
    lines.append("### Detailed Function Code Snippets\n")

    for idx, item in enumerate(display_items, 1):
        fn_id = item.get("function_id", "unknown")
        prob = item.get("drift_probability", 0.0)
        doc = item.get("docstring", "").strip()
        code = item.get("code", "").strip()

        lines.append(f"#### {idx}. `{fn_id}` (Drift Probability: {prob:.4f})\n")
        lines.append("**Docstring:**")
        lines.append("```text")
        lines.append(doc)
        lines.append("```\n")
        lines.append("**Code:**")
        lines.append("```python")
        lines.append(code)
        lines.append("```\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="SemDrift — Claude Code-Style CLI for Scanning Codebases for Semantic Drift."
    )
    parser.add_argument("path", nargs="?", default=".", help="Path to Python file or repository directory to scan")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model_b_checkpoint.pt")
    parser.add_argument("--model_name", default="microsoft/codebert-base", help="HuggingFace transformer model")
    parser.add_argument("--threshold", type=float, default=0.50, help="Drift probability threshold for flagging (0.0 to 1.0)")
    parser.add_argument("--top_k", type=int, default=None, help="Limit output to top-K flagged functions")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--interactive", "-i", action="store_true", default=False, help="Launch interactive step-through mode")
    parser.add_argument("--output", choices=["console", "json", "markdown"], default="console", help="Report format")
    parser.add_argument("--output_file", default=None, help="Optional file path to save output report")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Target device (cuda or cpu)")
    parser.add_argument("--include_test_files", action="store_true", default=False, help="Include test files during parsing")
    parser.add_argument("--no_clean_docstrings", dest="clean_docstrings", action="store_false", default=True, help="Disable docstring cleaning")

    args = parser.parse_args()

    # Render Header Banner
    render_banner()

    # 1. Locate Checkpoint
    checkpoint_path = find_checkpoint(args.checkpoint)

    # 2. Extract Functions
    if RICH_AVAILABLE:
        console.print(f"[dim]Scanning path:[/dim] [cyan]{os.path.abspath(args.path)}[/cyan]...")
    records, total_funcs, undoc_funcs = extract_functions(args.path, skip_test_files=not args.include_test_files)

    if not records:
        if RICH_AVAILABLE:
            console.print("[bold yellow]⚠️ No documented functions found to analyze.[/bold yellow]")
        else:
            print("No documented functions found to analyze.")
        return

    # 3. Run Inference
    results = run_inference(
        records=records,
        checkpoint_path=checkpoint_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        clean_docstrings=args.clean_docstrings,
    )

    # 4. Filter by Threshold
    flagged = [r for r in results if r["drift_probability"] >= args.threshold]

    # 5. Output / Dashboard Rendering
    if args.output == "json":
        report_data = {
            "target_path": os.path.abspath(args.path),
            "total_functions": total_funcs,
            "documented_functions": len(records),
            "undocumented_functions": undoc_funcs,
            "threshold": args.threshold,
            "flagged_count": len(flagged),
            "flagged_functions": flagged[:args.top_k] if args.top_k else flagged,
        }
        report_str = json.dumps(report_data, indent=2)
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(report_str)
            if RICH_AVAILABLE:
                console.print(f"\n[bold green]Report saved to {args.output_file}[/bold green]")
        else:
            print(report_str)

    elif args.output == "markdown":
        report_str = format_markdown_output(
            args.path, total_funcs, undoc_funcs, flagged, args.threshold, args.top_k
        )
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(report_str)
            if RICH_AVAILABLE:
                console.print(f"\n[bold green]Markdown report saved to {args.output_file}[/bold green]")
        else:
            print(report_str)

    else:  # Console mode
        render_dashboard(args.path, total_funcs, undoc_funcs, flagged, args.threshold)
        render_flagged_cards(flagged, args.top_k)

        if args.interactive:
            interactive_mode(flagged)

        if args.output_file:
            # Also write markdown report to file if requested
            md_str = format_markdown_output(
                args.path, total_funcs, undoc_funcs, flagged, args.threshold, args.top_k
            )
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(md_str)
            if RICH_AVAILABLE:
                console.print(f"[bold green]Report also saved to {args.output_file}[/bold green]")


if __name__ == "__main__":
    main()
