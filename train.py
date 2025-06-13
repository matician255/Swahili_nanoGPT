# AUTO DETECT THE DEVICE
import time
import inspect
import tiktoken
import math
import torch

from model import GPT, GPTConfig, DataLoaderLite
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
elif hasattr(torch.backend, "mps") and torch.backends.mps.is_available():
  device = 'mps'
print('using device:', {device})

# Set the seed to 42
torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)


# gradient accumulation
total_batch_size = 57344
B = 8 # micro batch size
T = 1024
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B*T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B=8, T=512)

# auto mixed precision 
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained('gpt2')
model.to(device)
model = torch.compile(model)


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
  # 1) linear warmup for warmup iteration steps
  if it < warmup_steps:
    return max_lr * (it+1) / warmup_steps
  # 2) if it < lr_decay_iters, return min learning rate
  if it > max_steps:
    return min_lr
  # 3) if in between use cosine decay down to min learning rate
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff start at 1 and goes to 0
  return min_lr + coeff * (max_lr - min_lr)
  

#optimization
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
  t0 = time.time()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device) 
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    # backward pass
    loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  # determine the learning rate for this iteration
  lr = get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  torch.cuda.synchronize() # makes sure we wait for the GPU to finish th workload before we check time taken
  t1 = time.time()
  dt = (t1 - t0) # time difference in seconds
  token_processed = train_loader.B * train_loader.T * grad_accum_steps
  token_per_second =  token_processed / dt
  print(f"step{step},| loss: {loss_accum.item():.6f},| lr:{lr:.4e}, norm: {norm:.4f},| {dt * 1000:.2f}ms, |tok/sec: {token_per_second:.2f}")