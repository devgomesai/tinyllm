import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from trainer.train import MODEL_FILE, TinyGPT  # noqa: E402

from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Palette
ACCENT = "#7aa2f7"    # soft blue
USER = "#7dcfff"      # cyan
AI = "#9ece6a"        # green
WARM = "#e0af68"      # amber
PINK = "#bb9af7"      # violet
ERROR = "#f7768e"     # rose
MUTED = "#565f89"     # slate
TEXT = "#c0caf5"      # soft white

console = Console(
    theme=Theme({
        "accent": f"bold {ACCENT}",
        "user": f"bold {USER}",
        "ai": f"bold {AI}",
        "warm": WARM,
        "pink": PINK,
        "error": f"bold {ERROR}",
        "muted": MUTED,
        "body": TEXT,
    })
)


def print_header():
    title = Text()
    title.append("Tiny", style=f"bold {ACCENT}")
    title.append("GPT", style=f"bold {PINK}")
    title.append("   character-level transformer", style=MUTED)
    console.print()
    console.print(title)
    console.print(Rule(style=MUTED, characters="─"))


def print_status(device, temperature, tokens):
    is_cuda = str(device).startswith("cuda")
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style=MUTED)
    grid.add_column()

    grid.add_row("status", Text("model loaded", style=f"bold {AI}"))
    grid.add_row("device", Text("CUDA GPU" if is_cuda else "CPU", style=AI if is_cuda else WARM))
    grid.add_row("temperature", Text(str(temperature), style=WARM))
    grid.add_row("max characters", Text(str(tokens), style=PINK))

    console.print(grid)
    console.print(Rule(style=MUTED, characters="─"))


def print_help():
    console.print(
        Text.assemble(
            ("Type ", MUTED),
            ("q", f"bold {ERROR}"),
            (" or ", MUTED),
            ("quit", f"bold {ERROR}"),
            (", ", MUTED),
            ("e", f"bold {ERROR}"),
            (" or ", MUTED),
            ("exit", f"bold {ERROR}"),
            (" to leave", MUTED),
        )
    )
    console.print()


def generate(prompt, model, stoi, itos, device, tokens, temperature):
    console.print(Text("tiny ", style=f"bold {AI}"), end="")

    stream_response(prompt, model, stoi, itos, device, tokens, temperature)

    console.print(Rule(style=MUTED, characters="·"))


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_FILE}\n"
            "Train it once with: uv run python trainer/train.py"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
    model = TinyGPT(vocab_size=checkpoint["vocab_size"], **checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint["stoi"], checkpoint["itos"], device


@torch.inference_mode()
def stream_response(prompt, model, stoi, itos, device, max_new_tokens, temperature):
    encoded = [stoi[ch] for ch in prompt if ch in stoi]
    if not encoded:
        console.print(
            "Prompt has no characters in the training vocabulary.",
            style=WARM,
            markup=False,
        )
        return
    context = torch.tensor([encoded], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(context[:, -model.block_size:])
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
        console.print(
            itos[next_token.item()],
            end="",
            style=TEXT,
            markup=False,
            highlight=False,
        )
    console.print()


def main():
    parser = argparse.ArgumentParser(
        description="TinyGPT: a local character-level language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, default=None,
                        help="Generate from a prompt and exit")
    parser.add_argument("--tokens", type=int, default=300,
                        help="Maximum new characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature; lower is more predictable")
    args = parser.parse_args()

    if args.tokens < 1:
        parser.error("--tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")

    print_header()

    with console.status(f"[{ACCENT}]Loading checkpoint...[/]", spinner="dots", spinner_style=ACCENT):
        try:
            model, stoi, itos, device = load_model()
        except Exception as exc:
            console.print(Text("Model loading failed", style=f"bold {ERROR}"))
            console.print(Text(str(exc), style=MUTED))
            raise SystemExit(1) from exc

    print_status(device, args.temperature, args.tokens)

    if args.prompt is not None:
        try:
            generate(args.prompt, model, stoi, itos, device, args.tokens, args.temperature)
        except KeyboardInterrupt:
            console.print()
            console.print("Generation interrupted.", style=WARM)
        return

    print_help()

    while True:
        try:
            prompt = Prompt.ask(f"[bold {USER}]you[/]").strip()

            if prompt.lower() in {"q", "quit", "e", "exit"}:
                console.print()
                console.print(Text("Thanks for trying TinyGPT.", style=f"bold {PINK}"))
                console.print()
                break

            if not prompt:
                console.print("Enter a prompt first.", style=WARM)
                continue

            generate(prompt, model, stoi, itos, device, args.tokens, args.temperature)

        except KeyboardInterrupt:
            console.print()
            console.print("Interrupted. Enter another prompt or type 'q' to leave.", style=WARM)
        except EOFError:
            console.print()
            console.print("Goodbye.", style=MUTED)
            break


if __name__ == "__main__":
    main()