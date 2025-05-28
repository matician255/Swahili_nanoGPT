from model import BigramLanguageModel


#hyperparameters setup--------------------------------
batch_size = 64
block_size = 256 
n_embd = 384
n_heads = 6
n_layer = 6
eval_iters = 200
eval_interval = 100
learning_rate = 3e-4
max_iters = 5000
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#-----------------------------------------------------

torch.manual_seed(1338)


# here are the unique characters in the dataset
chars = sorted(list(set(cleaned_swahili)))
vocab_size = len(chars)

#create mapping from characters to intergers and viceversa
stoi = {ch:i for i, ch in enumerate(chars)}  
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #takes string returns list intergers
decode = lambda l: ''.join([itos[i] for i in l]) #takes list of intergers returns a string

#train and test split
data = torch.tensor(encode(cleaned_swahili), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    """
    move your input (x) and target (y) tensors to the same device — 
    either the CPU or the GPU — so that model computations can run without error"""
    x, y = x.to(device), y.to(device) 
    return x, y


@torch.no_grad() # tells python not to track gradients in this function 
def estimate_loss(): 
    out = {}
    model.eval()
    for split in ['train', 'val']: # It loops over both training and validation data.
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split) #  grabbing a fresh mini-batch for every iteration
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # resets the model to train mode
    return out




model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True) # clear the gradients
    loss.backward() #compute new gradients
    optimizer.step() #update the model weights