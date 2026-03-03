from modules.ocr_benchmark.metrics import calculate_cer_wer, normalize_ethiopic_text


def test_ethiopic_nfc_normalization():
    # Evaluates normalization: composed vs decomposed handling via NFC + internal whitespace collapse.
    text = "ሀሁሂ  ሃሄ "
    norm = normalize_ethiopic_text(text)
    assert norm == "ሀሁሂ ሃሄ"


def test_cer_wer_calculation():
    gt = "ሀሁ ሂሃ ሄህሆ"  # len gt chars = 9 (including spaces)
    pred = "ሀሁ ሂሃ ሄሆ"  # missed one char 'ህ'

    cer, wer, exact = calculate_cer_wer(pred, gt)

    assert exact is False
    assert cer > 0.0
    assert wer > 0.0


def test_exact_match():
    gt = "ሀሁ"
    pred = "ሀሁ"
    cer, wer, exact = calculate_cer_wer(pred, gt)

    assert exact is True
    assert cer == 0.0
    assert wer == 0.0
