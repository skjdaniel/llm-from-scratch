# LLMs from scratch

Sebastian Raschka has a book called "Build a Large Language Model (From Scratch)". I don't have this book; I'm working through parts of the corresponding Github repo (https://github.com/rasbt/LLMs-from-scratch). 

I try to simplify the code in Raschka's repo by 1) using `einops` and `einsum` (especially for attention mechanisms) and 2) omitting details that I don't want to focus on. I often use different variable, function, and class names.

There may be (hopefully small) mistakes or inconsistencies that I've missed or haven't bothered to correct.

### Notebooks (brief contents)

- [1-llm-from-scratch](https://github.com/skjdaniel/llm-from-scratch/blob/master/1-llm-from-scratch.ipynb): Build GPT-2 (small) from 
scratch and train on a (very) small dataset. 
Use the "trained" model and top-k sampling for text generation. 

- [2-load-gpt-weights-from-hf](https://github.com/skjdaniel/llm-from-scratch/blob/master/2-load-gpt-weights-from-hf.ipynb): Build GPT-2 (small) 
from scratch and load the weights from Hugging Face. Generate text using the same text-generation function as in notebook 1 above. (A lot of the code is recycled from that notebook.)

- [3-convert-gpt2-to-llama2](https://github.com/skjdaniel/llm-from-scratch/blob/master/3-convert-gpt2-to-llama2.ipynb): Construct Llama 2 (7B). Instantiate a toy model. (I don't train the model, load model weights from somewhere else, or use the model for text generation or question answering.)

- [4-from-llama2-to llama3](https://github.com/skjdaniel/llm-from-scratch/blob/master/4-from-llama2-to-llama3.ipynb): Convert Llama 2 7B to Llama 3 8B. Instantiate a toy Llama 3 model. (I don't train the model or load model weights from elsewhere.)
___

### Notebooks (more detailed contents)

**[1-llm-from-scratch](https://github.com/skjdaniel/llm-from-scratch/blob/master/1-llm-from-scratch.ipynb)**

Build GPT-2 (small) from  scratch and train on a (very) small dataset. 
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

**[2-load-gpt-weights-from-hf](https://github.com/skjdaniel/llm-from-scratch/blob/master/2-load-gpt-weights-from-hf.ipynb)** 

Build GPT-2 (small) from scratch and load the weights from Hugging Face. Generate text using the same text-generation function as in notebook 1 above. (A lot of the code is recycled from that notebook.)

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

**[3-convert-gpt2-to-llama2](https://github.com/skjdaniel/llm-from-scratch/blob/master/3-convert-gpt2-to-llama2.ipynb)** 

Construct Llama 2 (7B). Instantiate a toy model. (I don't train the model, load model weights from somewhere else, or use the model for text generation or question answering.)

I don't train the model, load model weights from somewhere else, or use the model for text generation or question answering.

Key differences between GPT-2 (small) and Llama 2:
- Llama 2 uses rotary position embeddings (RoPE). (RoPE applies rotations to the query and key vectors in the self-attention mechanism. GPT adds positional embeddings to the inputs.)
- Llama 2 uses gated SiLU (gated Sigmoid Linear Unit = SwiGLU) activation inside the MLP of the transformer block (instead of the approximate GELU used by GPT2).
- Llama 2 uses RMSNorm (rather than LayerNorm).
- Llama 2 uses 16-bit precision (rather than 32-bit precision, to save memory).
- LLama 2 uses `bias=False` in all linear transformations.
- LLama 2 doesn't use dropout.
- For training and text generation, Llama 2 uses Google's SentencePiece tokenizer (rather than OpenAI's Tiktoken) (not relevant for this notebook).

In this notebook:
- Imports.
- Define functions to implement RoPE.
- 'MultiHeadAttention` class incorporating RoPE.
- Transformer block with new multi-head attention, RMSNorm, and SwiGLU activation.
- Model configuration (GPT-2 (small), LLama 2 (7B), and Llama 2 (toy)).
- `Llama2Model` class.
- Instantiate toy Llama 2 model and count trainable parameters.

**[4-from-llama2-to llama3](https://github.com/skjdaniel/llm-from-scratch/blob/master/4-from-llama2-to-llama3.ipynb)** 

Convert Llama 2 7B to Llama 3 8B. Instantiate a toy Llama 3 model. 

I don't train the model or load model weights from elsewhere.

Raschka defines a `SharedBuffer` class so that we can reuse the `mask`, `sin`, and `cos` tensors in the transformer blocks. I don't implement this here.

Differences between Llama 2 7B and Llama 3 8B:
- Different RoPE parameters (`theta_base` is now 500,000 rather than 10,000, and `context_window` is now 8,192 rather than 4,096)
- Llama 3 uses grouped-query attention (GQA) rather than multi-head attention (MHA).
- Some parameters are different. The context length has doubled (as mentioned above). The hidden dimension of the MLP in the transformer block is a bit larger. The vocab size is much larger.  

In this notebook:
- Imports.
- Implement RoPE (same as in Llama 2; only the `theta_base` and `context_window` are different).
- RoPE parameters (comparing Llama 2 and Llama 3).
- `GroupedQueryAttention` class.
- `MultiHeadAttention` class from Llama 2 for comparison.
- Illustration of some differences betweeh GQA and MHA.
- Transformer block. 
- Llama 3 model class.
- Configuration for Llama 3 8B, Llama 2 7B (for comparison), and a toy Llama 3 model.
- Instantiate the toy Llama 3 model.
