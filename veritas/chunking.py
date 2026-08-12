import re
from typing import List, Tuple
from veritas.schemas import Chunk

# A "unit" is the smallest indivisible piece of text: a markdown heading, a list
# item, a table row, or a sentence inside a prose paragraph. Chunks are built by
# packing whole units, so every chunk is an exact slice of the source document.
_BLOCK_LINE = re.compile(r'^\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\||>)')
_SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+(?=["\'(\[`]?[A-Z0-9])')

# Sentence breaks are suppressed after these, which end in a period but not a sentence.
_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.", "Fig.", "No.", "Sec.", "Ref.",
    "Dr.", "Mr.", "Mrs.", "Ms.", "Inc.", "Ltd.", "Corp.", "U.S.", "Rev.", "al.",
)


def iter_units(text: str) -> List[Tuple[int, int]]:
    """Character spans of every atomic unit in `text`, in document order.

    Spans are half-open and never overlap; ``text[start:end]`` is the unit verbatim.
    """
    units: List[Tuple[int, int]] = []
    offset = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            offset += len(line)
            continue

        line_start = offset + (len(line) - len(line.lstrip()))
        line_end = line_start + len(stripped)

        if _BLOCK_LINE.match(line):
            # Headings, list items and table rows are structural: never split them.
            units.append((line_start, line_end))
        else:
            units.extend(_split_sentences(text, line_start, line_end))

        offset += len(line)

    return units


def _split_sentences(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Sentence spans within ``text[start:end]``, merging false abbreviation breaks."""
    segment = text[start:end]
    boundaries = [0]
    for match in _SENTENCE_BREAK.finditer(segment):
        preceding = segment[boundaries[-1]:match.start()].rstrip()
        if any(preceding.endswith(abbr) for abbr in _ABBREVIATIONS):
            continue
        boundaries.append(match.end())
    boundaries.append(len(segment))

    spans = []
    for i in range(len(boundaries) - 1):
        piece = segment[boundaries[i]:boundaries[i + 1]]
        piece_start = start + boundaries[i] + (len(piece) - len(piece.lstrip()))
        piece_end = start + boundaries[i] + len(piece.rstrip())
        if piece_end > piece_start:
            spans.append((piece_start, piece_end))
    return spans


def estimate_tokens(text: str) -> int:
    """Rough token count, assuming ~1.3 tokens per whitespace-delimited word."""
    return int(len(text.split()) * 1.3)


def chunk_document(
    doc_id: str,
    doc_title: str,
    text: str,
    target_tokens: int = 256,
    overlap_sentences: int = 1,
) -> List[Chunk]:
    """Pack atomic units into ~`target_tokens` chunks with `overlap_sentences` carry-over.

    Every chunk records the exact character span it was cut from, so
    ``source_text[chunk.char_start:chunk.char_end] == chunk.text``.
    """
    units = iter_units(text)
    if not units:
        return []

    chunks: List[Chunk] = []
    window: List[int] = []  # indices into `units`
    window_tokens = 0

    def flush() -> None:
        nonlocal chunks
        char_start = units[window[0]][0]
        char_end = units[window[-1]][1]
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::c{len(chunks):02d}",
            doc_id=doc_id,
            doc_title=doc_title,
            text=text[char_start:char_end],
            char_start=char_start,
            char_end=char_end,
            sent_range=(window[0], window[-1]),
        ))

    for idx, (u_start, u_end) in enumerate(units):
        unit_tokens = estimate_tokens(text[u_start:u_end])

        if window and window_tokens + unit_tokens > target_tokens:
            flush()
            carry = window[-overlap_sentences:] if overlap_sentences > 0 else []
            window = list(carry)
            window_tokens = sum(estimate_tokens(text[units[i][0]:units[i][1]]) for i in window)

        window.append(idx)
        window_tokens += unit_tokens

    if window:
        flush()

    return chunks
