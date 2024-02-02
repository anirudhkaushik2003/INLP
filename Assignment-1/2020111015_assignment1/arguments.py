import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='N-gram Language Model')
    parser.add_argument('--lm_type', type=str, default='LM1',
                        help='Language Model type', choices=['g', 'i'])
    parser.add_argument('--corpus_path', type=str, default='Pride and Prejudice - Jane Austen.txt',
                        help='Path to corpus file')
    args = parser.parse_args()
    return args

