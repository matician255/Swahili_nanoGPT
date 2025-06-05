#-------------training our tokenizer-----------------------------
import os 
import time
from .regex import RegexTokenizer
from .basic import BasicTokenizer

text = open('swahili.txt', "r", encoding='utf-8').read()

os.makedirs('models', exist_ok=True)

t0 = time.time()

for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ['basic', 'regex']):
  tokenizer = TokenizerClass()
  tokenizer.train(text, 512, verbose=True)
  prefix = os.path.join('models', name)
  tokenizer.save(prefix)

t1 = time.time()
print(f'training took {t1-t0:.2f} seconds')