import torch
import torch.nn as nn

vocab_size = 30
embedding_dim = 8

token_embedding = nn.Embedding(vocab_size, embedding_dim)

input_ids = torch.tensor(
    [7, 25, 21, 23, 19, 28, 9, 25, 13, 23, 1, 17, 20, 24, 22, 13],
    dtype=torch.long
)

embedded_input = token_embedding(input_ids)
print("Input shape:", input_ids.shape)
print("Embedded shape:", embedded_input.shape)
print(embedded_input)