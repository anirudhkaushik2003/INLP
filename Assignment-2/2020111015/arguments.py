import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="type of model to use", default="f", choices=["f", "r"] )
    return parser.parse_args()