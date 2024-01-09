![alt text](https://github.com/oskarfernlund/noskGPT/blob/master/assets/logo.png)

noskGPT is a nano-scale (1,170,625 parameter) generative pretrained transformer which generates Shakespearian dialogue. It has been pre-trained on an abridged corpus of William Shakespeare's plays (see the `shakespeare.txt` file in the `data/` directory) and uses simple character-level tokenisation. The scaled dot-product attention mechanisms and transformer architecture have been written from scratch using `pytorch`, and are inspired by the original transformer architecture outlined in [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

## Architecture :house:

The intention of noskGPT is to be small and simple enough to train and run on a typical laptop CPU.

Summary of noskGPT architecture:

- Vocabulary size: 65 characters
- Block size / context length: 256 characters
- Embedding dimensionality: 128
- Attention heads per transformer block: 4
- Layers / transformer blocks: 5
- Dropout probability (training): 0.1

The model weights can be found in the `weights.pth` file in the root directory.

## Installation :minidisc:

Clone this repository using `git clone https://github.com/oskarfernlund/noskGPT.git` and navigate to the root directory using `cd noskGPT`.

### Poetry
To install noskGPT's dependencies, you will need `python 3.10` and `poetry`.

```
poetry use <path to python 3.10 executable>
poetry install
```

### Conda / Pip

If you do not have `poetry` or do not wish to use it, the following steps can be taken using `conda` and `pip`:

1. Create a new `conda` environment using `conda create -n noskgpt python=3.10`
2. Activate the environment using `conda activate noskgpt`
3. Install the dependencies using `pip install -r requirements.txt`

## Command Line Interface :computer:

![alt text](https://github.com/oskarfernlund/noskGPT/blob/master/assets/cli_logo.png)

noskGPT has a simple command line interface which can be invoked as follows from the root directory:

```
python noskgpt.py --max-chars=1000
```

The optional `--max-chars` flag specifies the number of characters noskGPT will generate in response to a given prompt, and has a default value of 1000. An example prompt / response interaction with noskGPT is shown below:

![alt text](https://github.com/oskarfernlund/noskGPT/blob/master/assets/cli_prompt.png)

The generated text definitely isn't perfect -- the overall tone feels very Shakespearian but noskGPT tends to make up words (largely a consequence of the character-level tokenisation) and the overall narrative isn't very clear. However, I think this is pretty impressive for such a small and simple model!

## Training :hourglass:

Training noskGPT took me about 3-4 hours on my M2 MacBook CPU.

Summary of noskGPT training details:

- batch size: 64
- learning rate: 0.001
- training epochs: 10,000

The final validation loss I was able to achieve using the architecure and training details above was around 1.52. If you have more patience than me or access to more compute resources, I would encourage you to try scaling up the model (or using more sophisticated tokenisation) to see if you can do better :) To convert the training notebook from `.py` to `.ipynb`, run the following command from the root directory:

```
jupytext --to notebook training.py
```

## Acknowledgements :heart:

Thank you very much to Andrej Karpathy for your wonderful tutorial on generative language models :sparkles:
