from base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
  def __init__(self):
    super().__init__()

  def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # text processing
    text_bytes = text.encode('utf-8') # raw bytes
    ids = list(text_bytes) # turn bits --> int

    # merge common pairs to create new tokens
    merges = {} # (int, int)
    vocab = {idx: bytes([idx]) for idx in range(256)} # int --> bytes

    for i in range(num_merges):
      stats = get_stats(ids)
      pair = max(stats, key=stats.get)
      #create a new token and assign it to available id
      idx = 256 + i
      # replace the occurence of pair in ids with idx
      ids = merge(ids, pair, idx)
      # save the merge
      merges[pair] = idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
      # print
      if verbose:
        print(f'merge {i + 1}/{num_merges}: {pair} -->{idx} ({vocab[idx]} had {stats[pair]} occurrences)')

    # save class variables
    self.merges = merges # used in encode()
    self.vocab = vocab #used in decode()

  def encode(self, text):
    # given text string is converted to intergers
    text_bytes = text.encode('utf-8')
    ids = list(text_bytes)
    while len(ids) >= 2:
      stats = get_stats(ids) # get the most recurring pair of ids
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf'))) # get the pair in merges with min freq
      # subtle: if there are no more merges available, the key will
      # result in an inf for every single pair, and the min will be
      # just the first pair in the list, arbitrarily
      # we can detect this terminating case by a membership check
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      ids = merge(ids,pair, idx) # replace the pair with new token
    return ids

  def decode(self, ids):
    text_bytes = b''.join(self.vocab[idx] for idx in ids) # int --> bytes
    text = text_bytes.decode('utf-8', errors='replace') #bytes --> text
    return text
