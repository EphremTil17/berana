import re
import unicodedata

import editdistance

# Known equivalent characters mapping (visually identical or phonetically merged variants)
# Ethiopian keyboards often output distinct codepoints that render indistinguishably
# or are treated equivalently in standard Ge'ez/Amharic.
ETHIOPIC_EQUIVALENTS = {
    "ሐ": "ሀ",  # Ha equivalents
    "ሓ": "ሀ",
    "ኅ": "ሀ",
    "ኃ": "ሀ",
    "ሠ": "ሰ",  # Sa equivalents
    "ጸ": "ፀ",  # Tse equivalents
    "ዐ": "አ",  # A equivalents
    "ኣ": "አ",
}


def normalize_ethiopic_text(text: str, strict_punctuation: bool = True) -> str:
    """Normalize Ethiopic text for objective CER/WER evaluation.

    This function:
    1. Applies Unicode NFC normalization.
    2. Trims leading/trailing whitespace.
    3. Collapses internal contiguous whitespace.
    4. Maps known visually identical Ethiopic character equivalents to canonical forms.
    """
    if not text:
        return ""

    # 1. Unicode Normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Trim surrounding whitespace
    text = text.strip()

    # 3. Normalize punctuation to spaces to approximate JiWER-style punctuation handling.
    if strict_punctuation:
        text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)

    # 4. Collapse internal spaces
    text = re.sub(r"\s+", " ", text)

    # 5. Map known confusing Ethiopic equivalents
    for variant, canonical in ETHIOPIC_EQUIVALENTS.items():
        text = text.replace(variant, canonical)

    return text


def calculate_cer_wer(pred: str, gt: str) -> tuple[float, float, bool]:
    """Calculate Character Error Rate and Word Error Rate.

    Args:
        pred: The recognized text output from the OCR model (already normalized).
        gt: The ground truth text (already normalized).

    Returns:
        tuple containing (cer, wer, exact_match)
    """
    if not gt:
        cer = 1.0 if pred else 0.0
        wer = 1.0 if pred else 0.0
        return cer, wer, pred == gt

    # CER
    cer = editdistance.eval(pred, gt) / max(len(gt), 1)

    # WER
    pred_words = pred.split()
    gt_words = gt.split()
    wer = editdistance.eval(pred_words, gt_words) / max(len(gt_words), 1)

    # exact_match
    exact_match = pred == gt

    return float(cer), float(wer), exact_match


def align_chars_levenshtein(pred: str, gt: str) -> list[tuple[str | None, str | None]]:
    """Return deterministic global alignment pairs using Levenshtein DP backtrace."""
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion from pred (gt insertion)
                dp[i][j - 1] + 1,  # insertion into pred (gt deletion)
                dp[i - 1][j - 1] + cost,  # match/substitute
            )

    pairs: list[tuple[str | None, str | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                pairs.append((pred[i - 1], gt[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((pred[i - 1], None))
            i -= 1
            continue
        pairs.append((None, gt[j - 1]))
        j -= 1

    pairs.reverse()
    return pairs


def build_char_confusion_counts(
    aligned_pairs: list[tuple[str | None, str | None]],
) -> dict[str, dict[str, int]]:
    """
    Build confusion counts where keys are predicted chars and values are gt-char counters.
    Uses <INS> and <DEL> for insertion/deletion channels.
    """
    matrix: dict[str, dict[str, int]] = {}
    for pred_ch, gt_ch in aligned_pairs:
        pred_key = pred_ch if pred_ch is not None else "<DEL>"
        gt_key = gt_ch if gt_ch is not None else "<INS>"
        matrix.setdefault(pred_key, {})
        matrix[pred_key][gt_key] = matrix[pred_key].get(gt_key, 0) + 1
    return matrix
