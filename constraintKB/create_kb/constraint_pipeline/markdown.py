from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^]]*\]\([^)]+\)")
REFERENCE_TAIL_RE = re.compile(r"^(?:FIG(?:URE)?|TABLE)\s+E\d+", re.IGNORECASE)
EXCLUDED_SECTION_RE = re.compile(
    r"^(?:references?|bibliography|acknowledg(?:e?ments?)?|"
    r"credit authorship contribution statement|declaration of competing interest|"
    r"conflicts? of interest)$",
    re.IGNORECASE,
)


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_block_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", html.unescape(value))
    return collapse_whitespace(value)


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^\w.-]+", "-", value, flags=re.UNICODE).strip("-._")
    return value or "document"


@dataclass(slots=True)
class Block:
    document_id: str
    block_id: str
    kind: str
    text: str
    source_path: str
    line_start: int
    line_end: int
    section_path: list[str]
    excluded_reason: str | None = None
    duplicate_of: str | None = None
    table_index: int | None = None
    table_row_index: int | None = None

    @property
    def eligible(self) -> bool:
        return (
            self.excluded_reason is None
            and self.duplicate_of is None
            and bool(self.text)
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["eligible"] = self.eligible
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Block":
        fields = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        return cls(**fields)


@dataclass(slots=True)
class Document:
    document_id: str
    source_path: str
    source_sha256: str
    title: str
    blocks: list[Block]
    nonempty_source_lines: int
    covered_nonempty_source_lines: int

    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        exclusions: dict[str, int] = {}
        for block in self.blocks:
            kinds[block.kind] = kinds.get(block.kind, 0) + 1
            if block.excluded_reason:
                exclusions[block.excluded_reason] = exclusions.get(block.excluded_reason, 0) + 1
        return {
            "blocks": len(self.blocks),
            "eligible_blocks": sum(block.eligible for block in self.blocks),
            "duplicate_blocks": sum(block.duplicate_of is not None for block in self.blocks),
            "block_kinds": kinds,
            "exclusions": exclusions,
            "nonempty_source_lines": self.nonempty_source_lines,
            "covered_nonempty_source_lines": self.covered_nonempty_source_lines,
            "source_line_coverage": (
                self.covered_nonempty_source_lines / self.nonempty_source_lines
                if self.nonempty_source_lines
                else 1.0
            ),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "title": self.title,
            "stats": self.stats(),
        }


@dataclass(slots=True)
class Chunk:
    document_id: str
    chunk_id: str
    primary_block_ids: list[str]
    context_block_ids: list[str]
    text: str
    char_count: int

    @property
    def all_block_ids(self) -> list[str]:
        return self.context_block_ids + self.primary_block_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "primary_block_ids": self.primary_block_ids,
            "context_block_ids": self.context_block_ids,
            "all_block_ids": self.all_block_ids,
            "text": self.text,
            "char_count": self.char_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(
            document_id=value["document_id"],
            chunk_id=value["chunk_id"],
            primary_block_ids=list(value["primary_block_ids"]),
            context_block_ids=list(value["context_block_ids"]),
            text=value["text"],
            char_count=int(value["char_count"]),
        )


@dataclass(slots=True)
class _Cell:
    text: str
    rowspan: int = 1
    colspan: int = 1


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_Cell]] = []
        self._row: list[_Cell] | None = None
        self._cell_parts: list[str] | None = None
        self._rowspan = 1
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            attr_map = dict(attrs)
            self._rowspan = _positive_int(attr_map.get("rowspan"), 1)
            self._colspan = _positive_int(attr_map.get("colspan"), 1)
            self._cell_parts = []
        elif tag in {"br", "p", "li"} and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_parts is not None and self._row is not None:
            self._row.append(
                _Cell(
                    text=normalize_block_text("".join(self._cell_parts)),
                    rowspan=self._rowspan,
                    colspan=self._colspan,
                )
            )
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return max(parsed, 1)


def _expand_rows(rows: list[list[_Cell]]) -> list[list[str]]:
    expanded: list[list[str]] = []
    pending: dict[int, tuple[int, str]] = {}
    for raw_row in rows:
        values: dict[int, str] = {column: text for column, (_, text) in pending.items()}
        next_pending: dict[int, tuple[int, str]] = {
            column: (remaining - 1, text)
            for column, (remaining, text) in pending.items()
            if remaining - 1 > 0
        }
        column = 0
        for cell in raw_row:
            while column in values:
                column += 1
            for offset in range(cell.colspan):
                target = column + offset
                values[target] = cell.text
                if cell.rowspan > 1:
                    next_pending[target] = (cell.rowspan - 1, cell.text)
            column += cell.colspan
        pending = next_pending
        width = max(values, default=-1) + 1
        expanded.append([values.get(index, "") for index in range(width)])
    width = max((len(row) for row in expanded), default=0)
    return [row + [""] * (width - len(row)) for row in expanded]


def _looks_like_header(row: list[str]) -> bool:
    normalized = {collapse_whitespace(cell).lower() for cell in row if cell}
    header_terms = {
        "rank",
        "variable",
        "diagnosis",
        "syndrome",
        "test",
        "parameter",
        "clinical phenotype",
        "chromosomal abnormalities",
    }
    return bool(normalized & header_terms)


def table_rows(table_html: str) -> list[str]:
    parser = _TableParser()
    parser.feed(table_html)
    rows = _expand_rows(parser.rows)
    if not rows:
        text = normalize_block_text(re.sub(r"<[^>]+>", " ", table_html))
        return [text] if text else []
    has_header = _looks_like_header(rows[0])
    headers = rows[0] if has_header else [f"column_{index + 1}" for index in range(len(rows[0]))]
    output: list[str] = []
    start = 1 if has_header else 0
    for row_index, row in enumerate(rows[start:], start=1):
        pairs: list[str] = []
        for column, value in enumerate(row):
            if not value:
                continue
            header_value = headers[column] if column < len(headers) and headers[column] else f"column_{column + 1}"
            pairs.append(f"{header_value} = {value}")
        if pairs:
            output.append(f"TABLE ROW {row_index}: " + " | ".join(pairs))
    return output


def _section_exclusion(section_path: list[str]) -> str | None:
    for section in section_path:
        if EXCLUDED_SECTION_RE.match(collapse_whitespace(section)):
            return f"excluded_section:{collapse_whitespace(section)}"
    return None


def _block_id(document_id: str, index: int, kind: str, text: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{text}".encode("utf-8")).hexdigest()[:10]
    return f"{document_id}:B{index:05d}:{digest}"


def parse_markdown(path: Path) -> Document:
    path = path.resolve()
    raw = path.read_text(encoding="utf-8")
    source_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    document_id = slugify(f"{path.parent.name}-{path.stem}")
    lines = raw.splitlines()
    nonempty_lines = {index for index, line in enumerate(lines, 1) if line.strip()}
    covered_lines: set[int] = set()
    blocks: list[Block] = []
    section_stack: list[tuple[int, str]] = []
    title = path.stem
    paragraph: list[tuple[int, str]] = []
    table_index = 0

    def current_sections() -> list[str]:
        return [name for _, name in section_stack]

    def append_block(
        kind: str,
        text: str,
        line_start: int,
        line_end: int,
        *,
        current_table_index: int | None = None,
        table_row_index: int | None = None,
        excluded_reason: str | None = None,
    ) -> None:
        normalized = normalize_block_text(text)
        if not normalized:
            return
        block_index = len(blocks) + 1
        block = Block(
            document_id=document_id,
            block_id=_block_id(document_id, block_index, kind, normalized),
            kind=kind,
            text=normalized,
            source_path=str(path),
            line_start=line_start,
            line_end=line_end,
            section_path=current_sections(),
            excluded_reason=excluded_reason or _section_exclusion(current_sections()),
            table_index=current_table_index,
            table_row_index=table_row_index,
        )
        blocks.append(block)
        if line_start > 0:
            covered_lines.update(
                number for number in range(line_start, line_end + 1) if number in nonempty_lines
            )

    def flush_paragraph() -> None:
        if not paragraph:
            return
        append_block(
            "paragraph",
            "\n".join(value for _, value in paragraph),
            paragraph[0][0],
            paragraph[-1][0],
        )
        paragraph.clear()

    index = 0
    while index < len(lines):
        line_number = index + 1
        line = lines[index]

        in_references = any(
            collapse_whitespace(name).lower() in {"reference", "references", "bibliography"}
            for _, name in section_stack
        )
        lookahead = " ".join(lines[index : min(index + 3, len(lines))])
        if in_references and REFERENCE_TAIL_RE.search(collapse_whitespace(lookahead)):
            flush_paragraph()
            section_stack = [(2, "SUPPLEMENTARY MATERIAL")]

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading = normalize_block_text(heading_match.group(2))
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, heading))
            if level == 1 and title == path.stem:
                title = heading
            append_block("heading", heading, line_number, line_number)
            index += 1
            continue

        if "<table" in line.lower():
            flush_paragraph()
            table_start = line_number
            table_parts = [line]
            while "</table>" not in table_parts[-1].lower() and index + 1 < len(lines):
                index += 1
                table_parts.append(lines[index])
            table_end = index + 1
            table_index += 1
            raw_table = "\n".join(table_parts)
            parsed_rows = table_rows(raw_table)
            if parsed_rows:
                for row_index, row_text in enumerate(parsed_rows, 1):
                    append_block(
                        "table_row",
                        row_text,
                        table_start,
                        table_end,
                        current_table_index=table_index,
                        table_row_index=row_index,
                    )
            else:
                append_block(
                    "table",
                    raw_table,
                    table_start,
                    table_end,
                    current_table_index=table_index,
                )
            index += 1
            continue

        if MARKDOWN_IMAGE_RE.search(line):
            flush_paragraph()
            residual = normalize_block_text(MARKDOWN_IMAGE_RE.sub(" ", line))
            if residual:
                append_block("paragraph", residual, line_number, line_number)
            else:
                covered_lines.add(line_number)
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append((line_number, line))
        index += 1

    flush_paragraph()

    seen: dict[str, str] = {}
    for block in blocks:
        if block.kind not in {"paragraph", "table_row", "table"} or len(block.text) < 80:
            continue
        duplicate_key = hashlib.sha256(block.text.casefold().encode("utf-8")).hexdigest()
        if duplicate_key in seen:
            block.duplicate_of = seen[duplicate_key]
        else:
            seen[duplicate_key] = block.block_id

    return Document(
        document_id=document_id,
        source_path=str(path),
        source_sha256=source_hash,
        title=title,
        blocks=blocks,
        nonempty_source_lines=len(nonempty_lines),
        covered_nonempty_source_lines=len(covered_lines),
    )


def format_block(block: Block, primary: bool) -> str:
    section = " > ".join(block.section_path) or "ROOT"
    role = "PRIMARY" if primary else "CONTEXT"
    return (
        f"[BLOCK id={block.block_id} role={role} type={block.kind} "
        f"lines={block.line_start}-{block.line_end} section={section}]\n{block.text}\n[/BLOCK]"
    )


def build_chunks(
    document: Document,
    *,
    max_chars: int = 12_000,
    overlap_blocks: int = 2,
) -> list[Chunk]:
    if max_chars < 1_000:
        raise ValueError("max_chars must be at least 1000")
    if overlap_blocks < 0:
        raise ValueError("overlap_blocks cannot be negative")
    eligible = [block for block in document.blocks if block.eligible]
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_chars = 0
    for block in eligible:
        block_chars = len(format_block(block, primary=True)) + 2
        if current and current_chars + block_chars > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(block)
        current_chars += block_chars
    if current:
        groups.append(current)

    chunks: list[Chunk] = []
    eligible_index = {block.block_id: index for index, block in enumerate(eligible)}
    heading_indices: dict[str, list[int]] = {}
    first_table_indices: dict[int, int] = {}
    for eligible_position, block in enumerate(eligible):
        if block.kind == "heading":
            heading_indices.setdefault(block.text, []).append(eligible_position)
        if block.table_index is not None:
            first_table_indices.setdefault(block.table_index, eligible_position)
    for chunk_index, primary_blocks in enumerate(groups, 1):
        start = eligible_index[primary_blocks[0].block_id]
        context_positions = set(range(max(0, start - overlap_blocks), start))
        primary_ids = {block.block_id for block in primary_blocks}
        for primary_block in primary_blocks:
            primary_position = eligible_index[primary_block.block_id]
            for section_name in primary_block.section_path:
                candidates = heading_indices.get(section_name, [])
                prior = [position for position in candidates if position <= primary_position]
                if prior:
                    context_positions.add(prior[-1])
            if primary_block.table_index is not None:
                table_start = first_table_indices[primary_block.table_index]
                context_positions.update(range(max(0, table_start - 2), table_start))
        context_blocks = [
            eligible[position]
            for position in sorted(context_positions)
            if eligible[position].block_id not in primary_ids
        ]
        rendered = [format_block(block, primary=False) for block in context_blocks]
        rendered.extend(format_block(block, primary=True) for block in primary_blocks)
        text = "\n\n".join(rendered)
        chunks.append(
            Chunk(
                document_id=document.document_id,
                chunk_id=f"{document.document_id}:C{chunk_index:04d}",
                primary_block_ids=[block.block_id for block in primary_blocks],
                context_block_ids=[block.block_id for block in context_blocks],
                text=text,
                char_count=len(text),
            )
        )
    return chunks


def block_map(blocks: Iterable[Block]) -> dict[str, Block]:
    return {block.block_id: block for block in blocks}
