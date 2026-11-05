"""TinyGPT CLI: interactive and single-prompt generation.

Improvements for v2:
- temperature, top-k, top-p sampling, repetition penalty, greedy mode
- context-length validation with clear errors
- prompts not encodable are rejected with a message instead of slipping through
- works with both char-level and BPE checkpoints (auto-detected)
- optional --model path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from trainer.generate import generate  # noqa: E402
from trainer.model import TinyGPT  # noqa: E402
from trainer.tokenizer import load_tokenizer  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.prompt import Prompt  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402
from rich.theme import Theme  # noqa: E402

# Palette
ACCENT = "#7aa2f7"
USER = "#7dcfff"
AI = "#9ece6a"
WARM = "#e0af68"
PINK = "#bb9af7"
ERROR = "#f7768e"
MUTED = "#565f89"
TEXT = "#c0caf5"

STREAM_DELAY = 0.045

console = Console(
    theme=Theme(
        {
            "accent": f"bold {ACCENT}",
            "user": f"bold {USER}",
            "ai": f"bold {AI}",
            "warm": WARM,
            "pink": PINK,
            "error": f"bold {ERROR}",
            "muted": MUTED,
            "body": TEXT,
        }
    )
)


class StreamSink:
    """Streaming decorator: slows generation output for the typewriter effect."""

    def __init__(self, delay: float):
        import time

        self._time = time
        self.delay = delay
        self.started = False

    def __call__(self, chunk: str) -> None:
        if not chunk:
            return
        if not self.started:
            stripped = chunk.lstrip()
            if not stripped:
                return
            chunk = stripped
            self.started = True
        console.print(chunk, style=TEXT, end="", markup=False,
                      highlight=False, soft_wrap=True)
        if self.delay:
            self._time.sleep(self.delay)


def print_header(checkpoint_path):
    title = Text()
    title.append("Tiny", style=f"bold {ACCENT}")
    title.append("GPT", style=f"bold {PINK}")
    title.append("   small language model", style=MUTED)
    console.print()
    console.print(title)
    console.print(Text(str(checkpoint_path), style=MUTED))
    console.print(Rule(style=MUTED, characters="─"))


def print_status(device, temperature, tokens, top_k, top_p, rep):
    is_cuda = str(device).startswith("cuda")
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=MUTED)
    grid.add_column()

    grid.add_row("status", Text("model loaded", style=f"bold {AI}"))
    grid.add_row(
        "device", Text("CUDA GPU" if is_cuda else "CPU", style=AI if is_cuda else WARM)
    )
    grid.add_row("temperature", Text(str(temperature), style=WARM))
    grid.add_row("top-k / top-p", Text(f"{top_k} / {top_p or 'off'}", style=WARM))
    grid.add_row("repetition penalty", Text(f"{rep}x", style=WARM))
    grid.add_row("max new tokens", Text(str(tokens), style=PINK))

    console.print(grid)
    console.print(Rule(style=MUTED, characters="─"))


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train it with: uv run python trainer/train.py"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "tokenizer" in checkpoint and checkpoint["tokenizer"]:
        tokenizer = load_tokenizer(checkpoint["tokenizer"])
        if checkpoint.get("stoi") is not None:
            tokenizer.stoi = checkpoint["stoi"]
            tokenizer.itos = checkpoint["itos"]
    elif checkpoint.get("stoi") is not None:
        tokenizer = load_tokenizer({"type": "char", "stoi": checkpoint["stoi"],
                                    "itos": checkpoint["itos"]})
    else:
        raise ValueError("Checkpoint has no vocabulary; cannot decode output.")
    model = TinyGPT(vocab_size=checkpoint["vocab_size"], **checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer, device


def run_prompt(model, tokenizer, device, prompt, args, sink):
    try:
        stream_prompt(prompt, model, tokenizer, device, args, sink)
        console.print(Rule(style=MUTED, characters="·"))
    except ValueError as exc:
        console.print(str(exc), style=f"bold {ERROR}")


def stream_prompt(prompt, model, tokenizer, device, args, sink):
    console.print(Text("tiny ", style=f"bold {AI}"), end="")

    def stream(chunk):
        sink(chunk)

    text, status, _ = generate(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k if not args.greedy else 0,
        top_p=args.top_p if not args.greedy else 0.0,
        repetition_penalty=args.repetition_penalty,
        on_stream=stream,
    )
    # flush the (possibly empty) tail
    if sink.started:
        console.print()
    if status == "max_tokens" and text.strip():
        console.print(Text("\n[length limit reached without <END>]", style=MUTED))


def main():
    tk_help = "max new tokens"
    parser = argparse.ArgumentParser(
        description="TinyGPT: a small local language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Generate from a prompt and exit"
    )
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "model" / "mini_gpt.pt",
                        help="Checkpoint path; default is the v1 baseline")
    parser.add_argument("--tokens", type=int, default=200, help=tk_help)
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature; 0 = greedy decoding")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Sample only from the k most likely tokens; 0 disables")
    parser.add_argument("--top-p", type=float, default=0.0,
                        help="Nucleus sampling threshold in (0, 1); 0 disables")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help=">=1 divides positive logits of already-seen tokens")
    parser.add_argument("--greedy", action="store_true",
                        help="Deterministic decoding (equivalent to --temperature 0)")
    args = parser.parse_args()

    if args.tokens < 1:
        parser.error("--tokens must be at least 1")
    if args.temperature < 0:
        parser.error("--temperature must be 0 or greater")
    if args.top_p < 0 or args.top_p >= 1 and args.top_p != 0:
        parser.error("--top-p must be 0 or in the range (0, 1)")
    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be 1.0 or greater")
    if args.greedy:
        args.temperature = 0.0

    print_header(args.model)

    with console.status(
        f"[{ACCENT}]Loading checkpoint...[/]", spinner="dots", spinner_style=ACCENT
    ):
        try:
            model, tokenizer, device = load_model(args.model)
        except Exception as exc:
            console.print(Text("Model loading failed", style=f"bold {ERROR}"))
            console.print(Text(str(exc), style=MUTED))
            raise SystemExit(1) from exc

    print_status(device, args.temperature, args.tokens, args.top_k, args.top_p,
                 args.repetition_penalty)
    sink = StreamSink(STREAM_DELAY)

    if args.prompt is not None:
        try:
            run_prompt(model, tokenizer, device, args.prompt, args, sink)
        except KeyboardInterrupt:
            console.print()
            console.print("Generation interrupted.", style=WARM)
        return

    console.print(
        Text.assemble(
            ("Type ", MUTED), ("q", f"bold {ERROR}"), (" to quit", MUTED)
        )
    )
    console.print()

    while True:
        try:
            prompt = Prompt.ask(f"[bold {USER}]You[/]").strip()

            if prompt.lower() in {"q", "quit", "e", "exit"}:
                console.print()
                console.print(Text("Thanks for trying TinyGPT.", style=f"bold {PINK}"))
                console.print()
                break

            if not prompt:
                console.print("Enter a prompt first.", style=WARM)
                continue

            run_prompt(model, tokenizer, device, prompt, args, sink)

        except KeyboardInterrupt:
            console.print()
            console.print(
                "Interrupted. Enter another prompt or type 'q' to leave.", style=WARM
            )
        except EOFError:
            console.print()
            console.print("Goodbye.", style=MUTED)
            break


if __name__ == "__main__":
    main()
