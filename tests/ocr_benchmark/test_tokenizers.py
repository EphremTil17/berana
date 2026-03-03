import pytest
from transformers import XLMRobertaTokenizer


def test_ethiopic_xlmr_tokenizer_roundtrip():
    """
    Hard-halt preflight validation to ensure the base multilingual tokenizer
    does not fragment or <unk> Ethiopic characters in a way that would destroy
    TrOCR finetuning validity before it even begins.
    """
    try:
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    except Exception as e:
        pytest.skip(f"Could not load xlm-roberta-base (no internet or cache): {e}")

    # High-density Ge'ez/Amharic sample with punctuation
    text = "በስመ አብ ወወልድ ወመንፈስ ቅዱስ፥ አሐዱ አምላክ አሜን።"

    input_ids = tokenizer.encode(text, add_special_tokens=False)

    # Check for Unknown token (ID 3 in XLM-R)
    unk_token_id = tokenizer.unk_token_id
    if unk_token_id in input_ids:
        # We raise ValueError instead of assert so it's a hard fail
        raise ValueError("CRITICAL: Tokenizer produced <unk> tokens for valid Ethiopic text.")

    decoded = tokenizer.decode(input_ids)

    # Clean standard spaces (XLM-R SPM uses literal block space)
    decoded_clean = decoded.replace(" ", "").replace(" ", "").strip()
    text_clean = text.replace(" ", "").replace(" ", "").strip()

    assert decoded_clean == text_clean, (
        f"Roundtrip failed!\nOriginal: {text_clean}\nDecoded:  {decoded_clean}"
    )
