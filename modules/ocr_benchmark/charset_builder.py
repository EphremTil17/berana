import json
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from modules.ocr_benchmark.dataset import read_manifest
from modules.ocr_benchmark.metrics import normalize_ethiopic_text
from schemas.ocr_coverage import CharsetTierConfig, CoverageTier, EthiopicCharsetConfig
from utils.logger import get_logger

logger = get_logger("OCRBenchmarkCharsetBuilder")


ETHIOPIC_UNICODE_BLOCKS: tuple[tuple[int, int], ...] = (
    (0x1200, 0x137F),  # Ethiopic
    (0x2D80, 0x2DDF),  # Ethiopic Extended
    (0xAB01, 0xAB2F),  # Ethiopic Extended-A
)

DEFAULT_IGNORED_CHARS = [
    " ",
    "\t",
    "\n",
    "\r",
    ",",
    ".",
    ":",
    ";",
    "!",
    "?",
    "'",
    '"',
    "-",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "/",
]


def _ethiopic_assigned_chars() -> list[str]:
    chars: list[str] = []
    for start, end in ETHIOPIC_UNICODE_BLOCKS:
        for codepoint in range(start, end + 1):
            ch = chr(codepoint)
            name = unicodedata.name(ch, "")
            if not name.startswith("ETHIOPIC"):
                continue
            chars.append(ch)
    return sorted(set(chars), key=ord)


def _is_ethiopic_codepoint(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in ETHIOPIC_UNICODE_BLOCKS)


def _ethiopic_chars_from_wiktionary(url: str) -> list[str]:
    request = Request(
        url=url,
        headers={"User-Agent": "berana-ocr-benchmark-charset-builder/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="ignore")

    chars = {ch for ch in html if _is_ethiopic_codepoint(ord(ch))}
    return sorted(chars, key=ord)


def _split_ethiopic_chars(chars: list[str]) -> tuple[list[str], list[str]]:
    letters: list[str] = []
    symbols: list[str] = []
    for ch in chars:
        category = unicodedata.category(ch)
        if category in {"Lo", "Lm", "Mn", "Mc"}:
            letters.append(ch)
        elif category in {"Po", "No"}:
            symbols.append(ch)
        else:
            symbols.append(ch)
    return letters, symbols


def _build_family_grid(chars: list[str]) -> dict[str, str]:
    family_bins: dict[int, list[str]] = {}
    for ch in chars:
        codepoint = ord(ch)
        base = codepoint - (codepoint % 8)
        family_bins.setdefault(base, []).append(ch)

    grid: dict[str, str] = {}
    for base in sorted(family_bins.keys()):
        forms = sorted(family_bins[base], key=ord)
        slots = ["_"] * 8
        for ch in forms:
            order = ord(ch) - base
            if 0 <= order <= 7:
                slots[order] = ch
        grid[f"{base:04X}"] = "".join(slots)
    return grid


def _manifest_char_counts(manifest_path: Path, doc_stem: str | None = None) -> Counter:
    rows = read_manifest(manifest_path)
    if doc_stem:
        rows = [row for row in rows if row.doc_stem == doc_stem]
    counts: Counter = Counter()
    for row in rows:
        text = str(row.gt_text or "").strip()
        if not text:
            continue
        normalized = normalize_ethiopic_text(text)
        counts.update(normalized)
    return counts


def _unique_chars(chars: list[str]) -> list[str]:
    return sorted(set(chars), key=ord)


def _load_ethiopic_inventory(source: str, wiktionary_url: str) -> list[str]:
    if source == "wiktionary":
        chars = _ethiopic_chars_from_wiktionary(wiktionary_url)
        if not chars:
            raise ValueError(
                f"No Ethiopic characters found from Wiktionary source: {wiktionary_url}"
            )
        logger.info("Loaded %d Ethiopic characters from Wiktionary source.", len(chars))
        return chars
    if source == "unicode":
        chars = _ethiopic_assigned_chars()
        logger.info("Loaded %d Ethiopic characters from Unicode block ranges.", len(chars))
        return chars
    raise ValueError("Unsupported source. Use 'unicode' or 'wiktionary'.")


def _build_tier_charlists(
    *,
    letters: list[str],
    symbols: list[str],
    manifest_path: Path | None,
    doc_stem: str | None,
    high_min_count: int,
    medium_min_count: int,
    declaration_only: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    high_chars: list[str] = []
    medium_chars: list[str] = []
    rare_chars: list[str] = []
    optional_chars: list[str] = []

    if declaration_only:
        logger.info("Declaration-only mode enabled: tiers will not include explicit char lists.")
    elif manifest_path is not None:
        counts = _manifest_char_counts(manifest_path, doc_stem=doc_stem)
        for ch in letters:
            seen = int(counts.get(ch, 0))
            if seen >= high_min_count:
                high_chars.append(ch)
            elif seen >= medium_min_count:
                medium_chars.append(ch)
            elif seen > 0:
                rare_chars.append(ch)
            else:
                optional_chars.append(ch)
    else:
        optional_chars.extend(letters)

    optional_chars.extend(symbols)
    if declaration_only:
        return [], [], [], []
    return high_chars, medium_chars, rare_chars, optional_chars


def generate_unicode_charset_config(
    *,
    output_path: Path,
    manifest_path: Path | None = None,
    doc_stem: str | None = None,
    high_min_count: int = 20,
    medium_min_count: int = 10,
    rare_min_count: int = 5,
    source: str = "unicode",
    wiktionary_url: str = "https://en.wiktionary.org/wiki/Appendix:Unicode/Ethiopic",
    declaration_only: bool = True,
) -> EthiopicCharsetConfig:
    """Generate Ethiopic charset configuration from Unicode/Wiktionary with optional tiering."""
    all_ethiopic = _load_ethiopic_inventory(source=source, wiktionary_url=wiktionary_url)

    letters, symbols = _split_ethiopic_chars(all_ethiopic)
    family_grid = _build_family_grid(all_ethiopic)
    high_chars, medium_chars, rare_chars, optional_chars = _build_tier_charlists(
        letters=letters,
        symbols=symbols,
        manifest_path=manifest_path,
        doc_stem=doc_stem,
        high_min_count=high_min_count,
        medium_min_count=medium_min_count,
        declaration_only=declaration_only,
    )

    cfg = EthiopicCharsetConfig(
        schema_version="1.0",
        name="Ethiopic Charset (Unicode-Derived)",
        description=(
            "Automatically generated from Ethiopic Unicode inventory "
            f"(source={source}). "
            "Tiering is data-driven when manifest is provided."
        ),
        allowed_scripts=[
            "Ethiopic",
            "Ethiopic Extended",
            "Ethiopic Extended-A",
        ],
        unicode_blocks=[
            "U+1200..U+137F",
            "U+2D80..U+2DDF",
            "U+AB01..U+AB2F",
        ],
        tiers={
            CoverageTier.HIGH: CharsetTierConfig(
                min_count=high_min_count, chars=_unique_chars(high_chars)
            ),
            CoverageTier.MEDIUM: CharsetTierConfig(
                min_count=medium_min_count, chars=_unique_chars(medium_chars)
            ),
            CoverageTier.RARE: CharsetTierConfig(
                min_count=rare_min_count, chars=_unique_chars(rare_chars)
            ),
            CoverageTier.OPTIONAL: CharsetTierConfig(
                min_count=0, chars=_unique_chars(optional_chars)
            ),
        },
        ignored_chars=DEFAULT_IGNORED_CHARS,
        normalization_profile="ethiopic_v1",
        family_grid=family_grid,
        order_labels={
            "0": "geez",
            "1": "kaib",
            "2": "salis",
            "3": "rabi",
            "4": "hamis",
            "5": "sadis",
            "6": "sab",
            "7": "additional",
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            cfg.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    logger.info(
        "Generated Unicode charset config at %s | high=%d medium=%d rare=%d optional=%d",
        output_path,
        len(cfg.tiers[CoverageTier.HIGH].chars),
        len(cfg.tiers[CoverageTier.MEDIUM].chars),
        len(cfg.tiers[CoverageTier.RARE].chars),
        len(cfg.tiers[CoverageTier.OPTIONAL].chars),
    )
    return cfg
