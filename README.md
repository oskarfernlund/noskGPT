![alt text](https://github.com/oskarfernlund/noskGPT/blob/master/assets/logo.png)

noskGPT is a nano-scale (1,170,625 parameter) generative pretrained transformer which generates Shakespearian dialogue. It has been pre-trained on an abridged corpus of William Shakespeare's plays (see `data/shakespeare.txt`) and uses simple character-level tokenisation. The scaled dot-product attention mechanisms and transformer architecture have been written from scratch using `pytorch`, and are inspired by the original transformer architecture outlined in [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

## Architecture

The intention of noskGPT is to be small and simple enough to train and run on a typical laptop CPU.

Summary of noskGPT architecture:

- Vocabulary size: 65 characters
- Block size / context length: 256 characters
- Embedding dimensionality: 128
- Attention heads per transformer block: 4
- Layers / transformer blocks: 5
- Dropout probability (training): 0.1

## Installation

To install noskGPT's dependencies, you will need `python 3.10` and `poetry`.

```
poetry use <path to python 3.10 executable>
poetry install
```

If you do not have `poetry` or do not wish to use it, the dependencies in the `pyproject.toml` file can be installed manually using a package manager like `pip` or `conda`.

## Command Line Interface

![alt text](https://github.com/oskarfernlund/noskGPT/blob/master/assets/cli3.png)

noskGPT has a simple command line interface which can be invoked as follows:

```
python noskgpt.py --max-chars=1000
```

The optional `--max-chars` flag specifies the number of characters noskGPT will generate in response to a given prompt, and has a default value of 1000. An example prompt / response interaction with noskGPT is shown below:

![alt text](https://github.com/oskarfernlund/noskGPT/blob/master/assets/cli4.png)

The generated text definitely isn't perfect -- the overall tone feels very Shakespearian but noskGPT tends to make up words (largely a consequence of the character-level tokenisation) and the overall narrative isn't very clear. However, I think this is pretty impressive for such a small and simple model!

## Training

