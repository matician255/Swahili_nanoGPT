
"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""
import regex as re # type: ignore
from .base import Tokenizer, get_stats, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
  def __init__(self, pattern=None):
    """
    - pattern: optional string to override the default (GPT-4 split pattern)
    - special_tokens: str -> int dictionary of special tokens
    example: {'<|endoftext|>': 100257}
   """
    super().__init__()
    self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
    self.compile_pattern = re.compile(self.pattern)
    self.special_tokens = {}
    self.inverse_special_tokens = {}

  def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # split the text into text chunk
    text_chunks = re.findall(self.compile_pattern, text) # ('hello!' ) --->("hello", "!", " ")

    # input text processing in chunks
    ids = [list(ch.encode('utf-8')) for ch in text_chunks]
    # ids = [[72, 101, 108, 108, 111],  # "Hello"
            #[33],                        # "!"
            #[32]]                       # " "

    #iteratively merge the common pairs to create new tokens
    merges = {} # (int, int)
    vocab = {idx: bytes([idx]) for idx in range(256)}

    for i in range(num_merges):
      stats = {}
      # passing in stats will update it in place, adding up counts
      for chunk_ids in ids:
          get_stats(chunk_ids, stats)
      # find the pair with the highest count
      pair = max(stats, key=stats.get)
      # mint a new token: assign it the next available id
      idx = 256 + i
      # replace all occurrences of pair in ids with idx
      ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
        # save the merge
      merges[pair] = idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
      # print
      if verbose:
        print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

    # save class variables
    self.merges = merges # used in encode()
    self.vocab = vocab   # used in decode()

  def register_special_tokens(self, special_tokens):
    """
    - special_tokens: str -> int dictionary of special tokens
    example: {'<|mwisho|>': 100257}
    """
    self.special_tokens = special_tokens # for encoding
    # for decoding
    self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

  def decode(self, ids):
    part_bytes = []
    for idx in ids:
      if idx in self.vocab:
        part_bytes.append(self.vocab[idx])
      elif idx in self.inverse_special_tokens:
        part_bytes.append(self.inverse_special_tokens[idx])
      else:
        raise ValueError(f"Invalid token id: {idx}")
    text_byte = b''.join(part_bytes)
    text = text_byte.decode('utf-8', errors='replace')
    return text

  def _encode_chunk(self, text_bytes):
    # return the token ids
    # let's begin. first, convert all bytes to integers in range 0..255
    ids = list(text_bytes)
    while len(ids) >= 2:
      # find the pair with the lowest merge index
      stats = get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      # subtle: if there are no more merges available, the key will
      # result in an inf for every single pair, and the min will be
      # just the first pair in the list, arbitrarily
      # we can detect this terminating case by a membership check
      if pair not in self.merges:
        break # nothing else can be merged anymore
      # otherwise let's merge the best pair (lowest merge index)
      idx = self.merges[pair]
      ids = merge(ids, pair, idx)
    return ids

  def encode_ordinary(self, text):
    """Encoding that ignores any special tokens."""
    # split text into chunks of text by categories defined in regex pattern
    text_chunk = re.findall(self.compile_pattern, text)
    ids = []
    # all chunks of text are encoded separately, then results are joined
    for chunk in text_chunk:
      chunk_bytes = chunk.encode('utf-8') # raw bytes
      chunk_ids = self._encode_chunk(chunk_bytes)
      ids.extend(chunk_ids)
    return ids


  def encode(self, text, allowed_special='none_raise'):
    special = None
    if allowed_special == "all":
      special = self.special_tokens
    elif allowed_special == "none":
      special = {}
    elif allowed_special == "none_raise":
      special = {}
      assert all(token not in text for token in self.special_tokens)
    elif isinstance(allowed_special, set):
      special = {v: k for k, v in self.special_tokens.items() if k in allowed_special}
    else:
      raise ValueError(f"allowed_special={allowed_special} not understood")
    if not special:
      return self.encode_ordinary(text)

    special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
    special_chunks = re.split(special_pattern, text)

    ids = []
    for part in special_chunks:
      if part in special:
        ids.append(special[part])
      else:
        ids.extend(self.encode_ordinary(part))

    return ids
