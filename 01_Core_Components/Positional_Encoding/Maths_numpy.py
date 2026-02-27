import numpy as np

def positional_encoding_optimized(seq_len, d_model):
    # Create a matrix of positions (seq_len, 1)
    position = np.arange(seq_len)[:, np.newaxis]
    
    # Create the divisor term using log space for numerical stability
    # (d_model,)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    
    # Use broadcasting to fill the even and odd indices
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe