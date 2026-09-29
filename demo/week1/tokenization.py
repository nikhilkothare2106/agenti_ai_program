
import tiktoken

STRINGS = {
    "S1": "The quarterly revenue increased by 12.5% year-over-year.",
    "S2": "こんにちは、今日の天気はどうですか?",
    "S3": "def calculate_total(items): return sum(i.price * i.qty for i in items)",
    "S4": "Sub-word tokenization doesn't always split on whitespace—e.g., 'unbelievably' or 'ChatGPT-4o-mini'.",
}

# Counts I obtained; the script checks your run against them.
EXPECTED = {"S1": 14, "S2": 13, "S3": 17, "S4": 29}


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")

    for name, text in STRINGS.items():
        ids = enc.encode(text)

        # Decode each token id individually. Byte-level BPE can produce tokens
        # that are only part of a multi-byte character; decode() would show
        # those as "�", so we also keep the raw bytes for inspection.
        decoded = [enc.decode([i]) for i in ids]
        raw_bytes = [enc.decode_single_token_bytes(i) for i in ids]

        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))

        print(f"{name}: {text}")
        print(f"  tokens        : {len(ids)}")
        print(f"  chars / bytes : {n_chars} / {n_bytes}")
        print(f"  chars per tok : {n_chars / len(ids):.2f}")
        print(f"  bytes per tok : {n_bytes / len(ids):.2f}")
        print(f"  token ids     : {ids}")
        print(f"  decoded       : {decoded}")
        if any("\ufffd" in d for d in decoded):
            print(f"  raw bytes     : {raw_bytes}  (some tokens split a multi-byte character)")

        assert enc.decode(ids) == text, "round-trip failed"

        status = "OK" if len(ids) == EXPECTED[name] else f"MISMATCH (expected {EXPECTED[name]})"
        print(f"  check vs. reference count: {status}\n")


if __name__ == "__main__":
    main()



# ## Short note: tiktoken and tokenization

# **Token:** a chunk of text (word, sub-word, punctuation) mapped to an integer ID. Models read tokens, not characters, and limits and pricing are counted in tokens.

# **tiktoken:** OpenAI's fast open-source library for converting text to token IDs and back.

# **cl100k_base:** one specific tokenizer with a vocabulary of about 100k tokens, used by GPT-3.5 and GPT-4. Different tokenizers give different counts for the same text.

# **How it splits (BPE):** Byte Pair Encoding starts from raw bytes and merges the most frequent adjacent pairs, learned from a large corpus. A regex first cuts text into chunks (words with their leading space, digit runs of up to 3, punctuation), and merges never cross those chunks. It works on UTF-8 bytes, so any text can be encoded.

# **Functions used:**
# - `tiktoken.get_encoding("cl100k_base")` loads the tokenizer.
# - `enc.encode(text)` turns text into a list of token IDs. Its length is the token count.
# - `enc.decode(ids)` turns IDs back into text (a lossless round-trip).
# - `enc.decode([i])` decodes one token to see its text piece.
# - `enc.decode_single_token_bytes(i)` returns a token's raw bytes.

# **Our results:** S1 = 14, S2 = 13, S3 = 17, S4 = 29 tokens.

# **Key takeaways:**
# - The count is deterministic: the same text and tokenizer always give the same IDs.
# - English prose gets about 4 chars/token. S2 (Japanese) got only 1.38 because each character is 3 bytes and few were merged in training.
# - S4 (3.41 chars/token) fragmented because rare words and punctuation-heavy text (`unbelievably`, `ChatGPT-4o-mini`) split into small pieces.