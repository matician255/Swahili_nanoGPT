*This is an implemented tokenizer that mimics the GPT-4 tokenizer called tiktonen, with this tokenizer you can train you dataset, save and load your vocabulary, encode and decode at inference settings givig you more flexibility to train your own vocabulary from scratch*

sample  
```py
# demo
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ['basic', 'regex']):
  tokenizer = TokenizerClass()
  tokenizer.train(text, 512, verbose=True)
```

After training you can load your model and encode your dataset ready to be fed in the transformer and can decode the output

```py
reg_tokenizer = RegexTokenizer()
reg_tokenizer.load("/content/models/regex.model")

# 2. Iterate over tok.vocab and print each token ID + its decoded form
from pprint import pprint

print(f"Total vocab size: {len(reg_tokenizer.vocab)} tokens\n")

text1 = "we ni mtoto mzuri"
print(f"Input text: '{text1}'") # Print the input text

ids = reg_tokenizer.encode(text1)
print(f"Encoded IDs: {ids}") # Print the encoded IDs

dec = reg_tokenizer.decode(ids)
print(f"Decoded text: '{dec}'") # Print the decoded text using print()
```