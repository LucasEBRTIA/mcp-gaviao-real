"""Page range parsing, token estimation, and budget truncation helpers."""

from __future__ import annotations


def parse_page_spec(spec: str, total_pages: int) -> list[int]:
    """Parse 1-based page specifications like '1-5', '1,3,8-12', or 'all'."""
    if total_pages <= 0:
        raise ValueError("Documento não possui páginas válidas.")

    cleaned = (spec or "").strip().lower()
    if not cleaned:
        raise ValueError("Especificação de páginas vazia.")
    if cleaned == "all":
        return list(range(1, total_pages + 1))

    selected: set[int] = set()
    for part in cleaned.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"Segmento vazio na especificação de páginas: '{spec}'")
        if "-" in token:
            pieces = token.split("-", 1)
            if not pieces[0].strip().isdigit() or not pieces[1].strip().isdigit():
                raise ValueError(f"Intervalo de páginas inválido: '{token}'")
            start = int(pieces[0].strip())
            end = int(pieces[1].strip())
            if start > end:
                raise ValueError(f"Intervalo invertido '{token}': página inicial maior que final.")
            if start < 1 or end > total_pages:
                raise ValueError(
                    f"Intervalo '{token}' fora dos limites do documento (1..{total_pages})."
                )
            selected.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Número de página inválido: '{token}'")
            page_num = int(token)
            if page_num < 1 or page_num > total_pages:
                raise ValueError(
                    f"Página {page_num} fora dos limites do documento (1..{total_pages})."
                )
            selected.add(page_num)

    return sorted(selected)


def estimate_tokens(text: str) -> int:
    """Fast deterministic token estimate (~4 characters per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate_pages_by_budget(
    pages: list[tuple[int, str]],
    max_tokens: int,
) -> tuple[list[tuple[int, str]], bool, int | None]:
    """Keep pages fitting within max_tokens while always keeping at least the first page."""
    if not pages:
        return [], False, None
    if max_tokens <= 0:
        raise ValueError("max_tokens deve ser maior que zero.")

    kept: list[tuple[int, str]] = []
    used_tokens = 0

    for idx, (page_num, text) in enumerate(pages):
        page_tokens = estimate_tokens(text)
        if idx == 0 and page_tokens > max_tokens:
            max_chars = max_tokens * 4
            notice = f"\n\n<!-- [TRUNCADO: página {page_num} excedeu max_tokens={max_tokens}] -->"
            clipped = text[: max(100, max_chars - len(notice))] + notice
            kept.append((page_num, clipped))
            next_p = pages[1][0] if len(pages) > 1 else None
            return kept, True, next_p

        if used_tokens + page_tokens > max_tokens:
            return kept, True, page_num

        kept.append((page_num, text))
        used_tokens += page_tokens

    return kept, False, None
