"""
Command-line executable script for generating text with noskGPT.
"""

from argparse import ArgumentParser

import torch

from noskgpt.model import LanguageModel
from noskgpt.tokeniser import CharToIntTokeniser


BLOCK_SIZE = 256
NUM_EMBEDDINGS = 128
NUM_HEADS = 4
NUM_LAYERS = 5
DROPOUT = 0.1


if __name__ == "__main__":

    # Argument parser
    parser = ArgumentParser()
    parser.add_argument(
        "--max-chars",
        "-mc",
        type=int,
        default=1000,
        help="Number of characters noskGPT will generate when prompted"
    )
    args = parser.parse_args()

    # Print logo
    print("\n")
    print("                  _     ____ ____ _____ ")
    print("  _ __   ___  ___| | __/ ___|  _ \\_   _|")
    print(" | '_ \\ / _ \\/ __| |/ / |  _| |_) || |  ")
    print(" | | | | (_) \\__ \\   <| |_| |  __/ | |  ")
    print(" |_| |_|\\___/|___/_|\\_\\\\____|_|    |_|  ")
    print("                                        ")

    # Load corpus
    with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
        corpus = f.read()
    vocab_size = len("".join(sorted(list(set(corpus)))))
    tokeniser = CharToIntTokeniser(corpus)

    # Load model
    model = LanguageModel(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        num_embeddings=NUM_EMBEDDINGS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    model.load_state_dict(torch.load("weights.pth"))
    model.eval()

    # Generate text
    print("\nWelcome to the noskGPT interactive console :)")
    print("Enter <exit> into the prompt at any point to quit")
    print("\n>> prompt: ", end="")
    prompt = str(input())
    print("\n...\n")
    while prompt != "exit":
        model.generate(
            prompt=prompt,
            tokeniser=tokeniser,
            max_new_tokens=args.max_chars
        )
        print("\n>> prompt: ", end="")
        prompt = str(input())
        print("\n...\n")
