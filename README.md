### Notebooks
(more detail below)

- [1-llm-from-scratch](https://github.com/skjdaniel/llm-from-scratch/blob/master/1-llm-from-scratch.ipynb): Build GPT-2 (small) from 
scratch and train on a (very) small dataset. 
Use the "trained" model and top-k sampling for text generation. 

- [2-load-gpt-weights-from-hf](https://github.com/skjdaniel/llm-from-scratch/blob/master/2-load-gpt-weights-from-hf.ipynb): Build GPT-2 (small) 
from scratch and load the weights from Hugging Face. Generate text using the same text-generation function as in notebook 1 above. (A lot of the code is recycled from that notebook.)

- [3-convert-gpt2-to-llama2](https://github.com/skjdaniel/llm-from-scratch/blob/master/3-convert-gpt2-to-llama2.ipynb): Construct Llama 2 (7B). Instantiate a toy model. (I don't train the model, load model weights from somewhere else, or use the model for text generation or question answering.)

___

### Notebook contents

[1-llm-from-scratch](https://github.com/skjdaniel/llm-from-scratch/blob/master/1-llm-from-scratch.ipynb): Build GPT-2 (small) from 
scratch and train on a (very) small dataset. 
Use the "trained" model and top-k sampling for text generation. 

- Imports.
- `MultiHeadAttention` class for masked MHA using `einsum` and `rearrange` from `einops`. 
- Model config, including the number of attention heads, the number of transformer blocks, and vocab size. Here we use a shorter context length (256 instead of the original 1024).
- `ApproxGELU` class. The feedforward network of the transformer blocks use this approximation to GELU activation.
- `TransformerBlock` class with pre-layernorm configuration.
- `GPTVerdict` class. GPT-2 model to be trained on a text called "The Verdict".
- Instantiate the model. The model has ~162 million parameters (the original GPT-2 has ~124 million because of weight tying).
- Define functions to convert from text to token indexes and from token indexes to text, and to greedily generate token indexes.
- Greedily generate a bit of text from the untrained model, just to check if everything's working.
- Load the text to train on (in this case a short story called The Verdict).
- Define the PyTorch dataset and a `create_dataloader` function.
- Print out a trial batch in order to get a feel for how the dataloader works.
- Define the train and val datasets and dataloaders.
- Functions to evaluate the model (returning train and val losses) and to generate and print sample text.
- Functions to train the model and plot losses.
- Train the model or, if model already trained, load saved weights.
- Function to generate text using top-k sampling. Print out some text generated using different values of k and different temperatures.

[2-load-gpt-weights-from-hf](https://github.com/skjdaniel/llm-from-scratch/blob/master/2-load-gpt-weights-from-hf.ipynb): Build GPT-2 (small) 
from scratch and load the weights from Hugging Face. Generate text using the same text-generation function as in notebook 1 above. (A lot of the code is recycled from that notebook.)

- Imports, including `GPT2Model` from `transformers`.
- `MultiHeadAttention` class.
- Approximate GELU activation for use in transformer block.
- Transformer block.
- GPT model.
- Model configuration. This is the same as in the llm-from-scratch notebook, except for the `context_length` (which is now 1024 instead of 256), and the bias for the queries, keys, and values (now set to `True`).
- Instantiate the model.
- Load GPT-2 (small) from Hugging Face.
- Define functions to load the weights from the Hugging Face model into our model.
- Load the weights into our model.
- Define functions to generate text samples using top-k sampling.
- Generate text samples with various values for k and for the temperature.

[3-convert-gpt2-to-llama2](https://github.com/skjdaniel/llm-from-scratch/blob/master/3-convert-gpt2-to-llama2.ipynb): Construct Llama 2 (7B). Instantiate a toy model. (I don't train the model, load model weights from somewhere else, or use the model for text generation or question answering.)

- Imports.
- Define functions to implement RoPE.
- 'MultiHeadAttention` class incorporating RoPE.
- Transformer block with new multi-head attention, RMSNorm, and SwiGLU activation.
- Model configuration (GPT-2 (small), LLama 2 (7B), and Llama 2 (toy)).
- `Llama2Model` class.
- Instantiate toy Llama 2 model and count trainable parameters.
