import pickle
import numpy as np
from language_model import N_Gram_Model

if __name__ == "__main__":
    # Load the model
    model = N_Gram_Model()
    model.generate(model.k)