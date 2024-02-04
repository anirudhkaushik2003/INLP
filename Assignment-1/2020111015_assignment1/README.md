# INLP Assignment 1
### Anirudh Kaushik 2020111015

## Instruction on how to run the code
### 1. Language Model
 - `python language_mode.py --lm_type <type> --corpus_path <path_to_corpus>`
 - Example: `python language_mode.py --lm_type i --corpus_path "data/corpus.txt"`
### 2. Generation
 - `python generator.py --lm_type <type> --corpus_path <path_to_corpus> --k <k>`
 - Example: `python generator.py --lm_type i --corpus_path "data/corpus.txt" --k 3`
 - `lm_type` can be `i` for interpolation or `g` for good-turing
### 3. Evaluation
 1. In the file `language_mode.py`, you can use the `evaluate` function to evaluate the language model.
 2. Example usage:
 ```python
    if __name__ == "__main__":
        model = N_Gram_Model()
        model.sentence_probability()
        # model.evaluate() # <<<--- uncomment this line in the code
```
 3. before running the `language_model.py` file, delete the saved models in the `models` directory so that it doesn't use a pre-trained model, otherwise it will be equivalent to testing twice (train split will be counted as test split).
 - `python language_mode.py --lm_type <type> --corpus_path <path_to_corpus>`
 - Example: `python language_mode.py --lm_type i --corpus_path "data/corpus.txt"`

### 4. Tokenizer
 1. `python tokenizer.py`
 2. The terminal will prompt you to input a sentence, you cannot input a multiline sentence (i.e. a sentence with a newline character in it). as the python input function does not support multiline input. for multi-line input, pass a file to the tokenizer in the `tokenizer.py` file.
 3. Example usage:
 ```python
     if __name__ == "__main__": 
        file = open("data/corpus.txt", "r") 
        T = Tokenizer(file.read()) # <- for file based input
 ```

## Directory Structure
 ```
2020111015_assignment1
├──README.md
├──language_model.py
├──generator.py
├──tokenizer.py
├──arguments.py <- contains the argument parser
├──models <- contains saved models
    ├── 2020111015_LM1.pkl
    ├── 2020111015_LM2.pkl
    ├── 2020111015_LM3.pkl
    └── 2020111015_LM4.pkl
├──perplexity <- contains saved perplexity values
    ├── 2020111015_LM1_train-perplexity.pkl
    ├── 2020111015_LM1_test-perplexity.pkl
    ├── 2020111015_LM2_train-perplexity.pkl
    ├── 2020111015_LM2_test-perplexity.pkl
    ├── 2020111015_LM3_train-perplexity.pkl
    ├── 2020111015_LM3_test-perplexity.pkl
    ├── 2020111015_LM4_train-perplexity.pkl
    └── 2020111015_LM4_test-perplexity.pkl
```
    
    
