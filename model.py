from dataclasses import dataclass
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import inspect
import tiktoken



class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # key, query, value projections for all heads but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANO_SCALE_INIT = 1
    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    # a mask
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
    .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size() # batch_size, sequence length, embedding dimensionality(n_embd)
    # calculate query , key, value for all heads in
    # nh is the number of heads and hs is the head size
    # C = nh * hs
    qkv = self.c_attn(x) # (B, T, 3*C)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #(B, nh, T, T)
    # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1)
    # y = att @ v # (B, nh, T, T) * (B, nh, T, hs)--> (B, nh, T, hs)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # Re assemble head outputs side by side
    # output projection
    y = self.c_proj(y)
    return y


class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANO_SCALE_INIT = 1

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x



class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.ln_1  = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


@dataclass
class GPTConfig:
  block_size: int = 512 # max sequence length
  vocab_size: int = 50257 # number of tokens
  n_head: int = 6 # number of heads
  n_layers: int = 6 # number of layers
  n_embd: int = 384 # embedding dimension


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size, config.n_embd),
        wpe = nn.Embedding(config.block_size, config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
        ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # weight sharing
    self.transformer.wte.weight = self.lm_head.weight

    # init params
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANO_SCALE_INIT'):
        std *= (2 * self.config.n_layers) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


  def forward(self, idx, targets=None):
    #idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f"Cannot forward, sequence of length {T}"
    # forward the token and position embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T,)
    pos_emb = self.transformer.wpe(pos) # shape (T, n_embd)
    tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
    x = tok_emb + pos_emb # shape (B, T, n_embd)
    # forward the blocks in the transformer
    for block in self.transformer.h:
      x = block(x)
    # Forward the final layernorm and classifier
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # shape (B, T, vocab_size)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return logits, loss

  def configure_optimizers(self, weight_decay, learning_rate, device):
    # start with all the candidate parameters that require grad
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create option groups, any parameters that iS 2D will be weight decayed , all biases and layernorm dont
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {num_decay_params}")
    print(f"num non-decayed parameter tensors: {num_nodecay_params}")
    # create Adam optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    used_fused = fused_available and 'cuda' in device
    print(f'using fused AdamW: {used_fused}')
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=used_fused)
    return optimizer



  @classmethod
  def from_pretrained(cls, model_type):
    """loads a pretrained GPT-2 model weights from huggingface"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("Loading weights from pretrained gpt: %s" % model_type)

    # n_layers, n_head and n_embd are determined from model_type
    config_args = {
        'gpt2': dict(n_layers=12, n_head=12, n_embd=768), # 124M params
        'gpt2-medium': dict(n_layers=24, n_head=16, n_embd=1024), # 350M params
        'gpt2-large': dict(n_layers=20, n_head=20, n_embd=1200), # 774M params
        'gpt2-xl': dict(n_layers=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257 # for GPT model checkpoints
    config_args['block_size'] = 1024 # for GPT model checkpoints
    #create from-scratch initialized minGPT model
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict() # Grabs all parameter names in the model.
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Filters out any parameters ending in .attn.bias (i.e.skipping those)

    # init a huggingface/transformer model
    model_hf = GPT2LMHeadModel.from_pretrained(model_type) # Downloads and instantiates the pre-trained HF GPT-2 model
    sd_hf = model_hf.state_dict() # Extracts its parameters into a state dict.

    # copy while ensuring all parameters are aligned and match in names and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    # open ai checkpoints use a "Conv1D" module, but we only want to use
    # This means that we have to transpose these weights when we import them
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}" # Ensures same number of “copyable” parameters in both models.
    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        # special treatment for the Conv1D weights we need to transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        # vanilla copy over the other parameters
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])
    return model

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
      """
      Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
      the sequence max_new_tokens times, feeding the predictions back into the model each time.
      Most likely you'll want to make sure to be in model.eval() mode of operation for this.
      """
      for _ in range(max_new_tokens):
          # if the sequence context is growing too long we must crop it at block_size
          idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
          # forward the model to get the logits for the index in the sequence
          logits, _ = self(idx_cond)
          # pluck the logits at the final step and scale by desired temperature
          logits = logits[:, -1, :] / temperature
          # optionally crop the logits to only the top k options
          if top_k is not None:
              v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
              logits[logits < v[:, [-1]]] = -float('Inf')
          # apply softmax to convert logits to (normalized) probabilities
          probs = F.softmax(logits, dim=-1)
          # sample from the distribution
          idx_next = torch.multinomial(probs, num_samples=1)
          # append sampled index to the running sequence and continue
          idx = torch.cat((idx, idx_next), dim=1)

      return idx



class DataLoaderLite:
  def __init__(self, B, T):
    self.B = B
    self.T = T

    # At init load tokens from disk and store them in memory
    with open('/content/drive/MyDrive/swahil_stories.txt', 'r', encoding='utf-8') as f:
      text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    print(f"loaded {len(self.tokens)} tokens")
    print(f"1 epoch = {len(self.tokens) // (B * T)} steps")

    # State
    self.current_position = 0

  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position:self.current_position + B * T + 1]
    x = (buf[:-1]).view(B, T)
    y = (buf[1:]).view(B, T)
    # advances the position in the tensor
    self.current_position += B * T
    # if loading the next batch would be out of bounds, reset
    if self.current_position + (B * T + 1) > len(self.tokens):
      self.current_position = 0
    return x, y



