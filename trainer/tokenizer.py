"""Tokenizers for TinyGPT: character-level and a small BPE built from scratch.

Both tokenizers expose the same interface:

    encode(text) -> list[int]
    decode(ids) -> str
    vocab_size, pad_id, end_id, name, state() -> dict

The checkpoint stores state() so a checkpoint is self-describing.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


class CharTokenizer:
    """One integer per character. Unknown characters are reported, not guessed."""

    name = "char"

    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)
        self.pad_id = None  # char model never pads
        self.end_id = None  # not meaningful for char level
        self.unknowns = 0

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, text: str) -> list[int]:
        out = []
        for ch in text:
            if ch in self.stoi:
                out.append(self.stoi[ch])
            else:
                self.unknowns += 1
        return out

    def decode(self, ids) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)

    def state(self) -> dict:
        return {
            "type": "char",
            "stoi": self.stoi,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        return cls(state["stoi"], state["itos"])

    def encode_answer_parts(self, prompt_part: str, answer_part: str) -> tuple[list[int], list[bool]]:
        """Full-text encode with an answer-only mask for a formatted example.

        prompt_part ends right after 'TinyLLM:'; answer_part starts immediately
        after it (includes the leading space and trailing newline before <END>).
        """
        p_ids = self.encode(prompt_part)
        a_ids = self.encode(answer_part)
        return p_ids + a_ids, [False] * len(p_ids) + [True] * len(a_ids)


# Pre-token pattern: letters | digits | punctuation runs | whitespace runs.
# Decode = plain concatenation, so pretokenization is lossless.
_PRETOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+|[^\sa-zA-Z0-9]+|\s+")


def _pretokens(text: str) -> list[str]:
    return _PRETOKEN_RE.findall(text)


class BPETokenizer:
    """Byte-pair-encoding tokenizer trained on the corpus, with specials.

    Special tokens: <END> (0) ends every answer; <PAD> (1) pads batches.
    """

    name = "bpe"

    def __init__(self, specials: list[str], base_chars: list[str],
                 merges: list[tuple[str, str]]):
        self.specials = specials
        self.base_chars = base_chars
        self.merges = merges
        self._build_tables()

    def _build_tables(self) -> None:
        self.symbol_to_id: dict[str, int] = {}
        for s in self.specials:
            self.symbol_to_id[s] = len(self.symbol_to_id)
        for ch in self.base_chars:
            if ch not in self.symbol_to_id:
                self.symbol_to_id[ch] = len(self.symbol_to_id)
        for a, b in self.merges:
            self.symbol_to_id[a + b] = len(self.symbol_to_id)
        self.id_to_symbol = {i: s for s, i in self.symbol_to_id.items()}
        self.vocab_size = len(self.symbol_to_id)
        self.end_id = self.special_id("<END>")
        self.pad_id = self.special_id("<PAD>") if "<PAD>" in self.specials else None
        self.special_ids = {self.symbol_to_id[s] for s in self.specials}
        self._merge_rank = {pair: r for r, pair in enumerate(self.merges)}
        self.unknowns = 0

    def special_id(self, name: str) -> int:
        return self.symbol_to_id[name]

    @classmethod
    def build(cls, text: str, n_merges: int = 400, seed: int = 42) -> "BPETokenizer":
        specials = ["<END>", "<PAD>"]
        plain = text.replace("<END>", " ")
        # keep the <END> letters in the base alphabet so that mixed-case text
        # round-trips even when the chars occur only inside the marker
        base_chars = sorted(set(plain) | set("<END>"))
        # Pretokenize and count each piece once-weighted by occurrences.
        piece_counts: Counter[tuple[str, ...]] = Counter()
        for piece in _pretokens(plain):
            piece_counts[tuple(piece)] += 1

        merges: list[tuple[str, str]] = []
        weights = dict(piece_counts)
        for _ in range(n_merges):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols, w in weights.items():
                if len(symbols) < 2:
                    continue
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += w
            if not pair_counts:
                break
            best, _max = pair_counts.most_common(1)[0]
            merges.append(best)
            new_weights: dict[tuple[str, ...], int] = {}
            for symbols, w in weights.items():
                syms = list(symbols)
                i = 0
                out = []
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == best[0] and syms[i + 1] == best[1]:
                        out.append(syms[i] + syms[i + 1])
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                new_weights[tuple(out)] = w
            weights = new_weights
        return cls(specials, base_chars, merges)

    # ------------------------------------------------------------------
    def _encode_piece(self, piece: tuple[str, ...] | str) -> list[int]:
        symbols = list(piece)
        if len(symbols) == 1 and symbols[0] == "":
            return []
        # repeatedly merge the lowest-rank adjacent pair
        while len(symbols) >= 2:
            best_pair, best_rank = None, None
            for a, b in zip(symbols, symbols[1:]):
                r = self._merge_rank.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best_pair, best_rank = (a, b), r
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            out = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        ids = []
        for s in symbols:
            if s in self.symbol_to_id:
                ids.append(self.symbol_to_id[s])
            else:
                self.unknowns += 1
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text, mapping literal '<END>' occurrences to the special id."""
        parts = re.split(r"(<END>)", text)
        ids: list[int] = []
        for part in parts:
            if part == "<END>":
                ids.append(self.end_id)
            elif part:
                for piece in _pretokens(part):
                    ids.extend(self._encode_piece(piece))
        return ids

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            s = self.id_to_symbol.get(int(i))
            if s is not None:
                out.append(s)
        return "".join(out)

    def encode_answer_parts(self, prompt_part: str, answer_part: str) -> tuple[list[int], list[bool]]:
        """Encode a formatted example with an answer-only mask.

        answer_part already contains the text up to and including the literal
        '<END>'; tokens over this span are marked True. The <END> special id
        is counted as part of the answer (matching the original char-level
        training where '<END>' was included in the loss).
        """
        p_ids = self.encode(prompt_part)
        p_ids_clean = [i for i in p_ids]
        # answer part: encode after the last 'TinyLLM:' boundary by encoding
        # the answer text alone. This forces a token boundary exactly where
        # generation begins.
        a_ids = self.encode(answer_part)
        return p_ids_clean + a_ids, [False] * len(p_ids_clean) + [True] * len(a_ids)

    def state(self) -> dict:
        return {
            "type": "bpe",
            "specials": self.specials,
            "base_chars": self.base_chars,
            "merges": [list(p) for p in self.merges],
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        return cls(state["specials"], state["base_chars"],
                   [tuple(p) for p in state["merges"]])


def load_tokenizer(state: dict):
    t = state["type"] if isinstance(state, dict) else state
    if t == "char":
        return CharTokenizer.from_state(state)
    if t == "bpe":
        return BPETokenizer.from_state(state)
    raise ValueError(f"Unknown tokenizer type: {t}")
