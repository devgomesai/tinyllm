"""Shared sampling/generation logic for TinyGPT checkpoints.

Works with both tokenizers. Used by the CLI (tests/testllm.py) and the
evaluation framework (tests/evaluate.py).
"""

from __future__ import annotations

import torch

STOP_MARKERS = ("<END>", "\nUser:", "\nTinyLLM:")


def marker_hold_len(text: str) -> int:
    """Length of the trailing text that could still become a stop marker."""
    longest = max(len(m) for m in STOP_MARKERS)
    for length in range(min(longest - 1, len(text)), 0, -1):
        tail = text[-length:]
        if any(marker.startswith(tail) for marker in STOP_MARKERS):
            return length
    return 0


@torch.inference_mode()
def sample_next(logits, temperature=1.0, top_k=0, top_p=0.0):
    """One sampling step from last-position logits [1, vocab]. Returns a token."""
    next_logits = logits[:, -1, :]
    if temperature <= 0:
        return torch.argmax(next_logits, dim=-1, keepdim=True)
    next_logits = next_logits / temperature
    if top_k and top_k > 0:
        values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        next_logits = next_logits.masked_fill(
            next_logits < values[:, [-1]], float("-inf")
        )
    if top_p and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # drop every token whose predecessors alone already exceed p
        remove = (cum - probs) >= top_p
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        next_logits = torch.full_like(next_logits, float("-inf"))
        next_logits.scatter_(1, sorted_idx, sorted_logits)
    probs = torch.softmax(next_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 10,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    on_stream=None,
    stop_on_marker: bool = True,
):
    """Generate a reply for 'User: prompt\\nTinyLLM:'.

    Returns (text, status, ids). status: 'end' (<END> reached), 'marker'
    (a conversation header formed inside the answer), or 'max_tokens'.

    on_stream(text_chunk) is called with incremental decoded text that is safe
    to display (holdback prevents streaming partial markers).
    """
    prompt_text = f"User: {prompt}\nTinyLLM:"
    ids = tokenizer.encode(prompt_text)
    if not ids:
        raise ValueError("Prompt cannot be encoded: no known tokens.")
    if len(ids) > model.block_size:
        raise ValueError(
            f"Prompt exceeds the model context: {len(ids)} tokens > "
            f"{model.block_size}. Use a shorter prompt."
        )
    context = torch.tensor([ids], dtype=torch.long, device=device)

    def apply_repetition_penalty(logits, recent_ids):
        if repetition_penalty == 1.0 or not recent_ids:
            return logits
        seen = sorted(set(recent_ids))
        idxs = torch.tensor(seen, dtype=torch.long, device=device)
        vals = logits[0, idxs]
        penalized = torch.where(
            vals > 0, vals / repetition_penalty, vals * repetition_penalty
        )
        logits = logits.clone()
        logits[0, idxs] = penalized
        return logits

    generated_ids: list[int] = []
    emitted = 0  # streaming position in the decoded generated text
    status = "max_tokens"

    for step in range(max_new_tokens):
        window = context[:, -model.block_size:]
        logits = model(window)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = apply_repetition_penalty(logits, generated_ids[-256:])
        if tokenizer.pad_id is not None:
            logits[0, tokenizer.pad_id] = float("-inf")
        next_token = sample_next(logits, temperature, top_k, top_p)
        tid = int(next_token.item())

        if tokenizer.end_id is not None and tid == tokenizer.end_id:
            status = "end"
            break

        generated_ids.append(tid)
        text = tokenizer.decode(generated_ids)

        cut_at = None
        if stop_on_marker:
            for marker in STOP_MARKERS:
                pos = text.find(marker)
                if pos != -1 and (cut_at is None or pos < cut_at):
                    cut_at = pos
                    status = "end" if marker == "<END>" else "marker"

        if cut_at is not None:
            text = text[:cut_at].rstrip()
            if on_stream is not None and len(text) > emitted:
                on_stream(text[emitted:])
            return text, status, generated_ids

        if on_stream is not None:
            safe_end = len(text) - marker_hold_len(text)
            if safe_end > emitted:
                on_stream(text[emitted:safe_end])
                emitted = safe_end

        context = torch.cat((context, next_token), dim=1)

    final_text = tokenizer.decode(generated_ids)
    if stop_on_marker:
        for marker in STOP_MARKERS:
            pos = final_text.find(marker)
            if pos != -1:
                final_text = final_text[:pos]
                status = "end" if marker == "<END>" else "marker"
                break
    if on_stream is not None and len(final_text) > emitted:
        on_stream(final_text[emitted:])
    return final_text.rstrip(), status, generated_ids
