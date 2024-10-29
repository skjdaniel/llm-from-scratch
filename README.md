[llm-from-scratch](https://github.com/skjdaniel/llm-from-scratch/blob/master/llm-from-scratch.ipynb): Build GPT-2 (small) from 
scratch and train on a (very) small dataset. (It's not really from scratch: I use the `gpt2` (BPE) tokenizer from `tiktoken`.) 
Use the "trained" model and top-k sampling for text generation. 

[load-gpt-weights-from-hf](https://github.com/skjdaniel/llm-from-scratch/blob/master/load-gpt-weights-from-hf.ipynb): Build GPT-2 (small) 
from scratch and load the weights from Hugging Face. Generate text using the same text-generation function as in 
[llm-from-scratch](https://github.com/skjdaniel/llm-from-scratch/blob/master/llm-from-scratch.ipynb). 
A lot of the code is recycled from that notebook.
