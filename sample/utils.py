import ast
import pandas as pd
import numpy as np

def parse_features( embeddings):
    X = embeddings.to_numpy()
    if isinstance(X[0], str):
        X_features = []
        for x in X:
            emb_list = ast.literal_eval(x)
            X_features.append(np.array(emb_list))
        features = np.array(X_features)
    elif isinstance(X[0], list):
        Y = np.array([ np.array(x) for x in X])
        features = Y
    elif type(X[0]).__module__== np.__name__:
        features = X
    else:
        raise NotImplementedError

    return features