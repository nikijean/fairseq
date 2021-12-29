import pdb

import torch.nn as nn
import pdb
class SplitEmbedding(nn.Module):
    def __init__(self, embeddings_one, embeddings_two):
        super().__init__()
        self.embeddings_one = embeddings_one
        self.embeddings_two = embeddings_two
        self.embeddings_one_vocab_size = self.embeddings_one.num_embeddings

    def forward(self, indices):
        #only batch size of 1 is supported for now.
        assert indices.shape[0] == 1
        indices.squeeze()
        t_result = indices - (self.embeddings_one_vocab_size - 1)
        t_result[t_result < 0] = 0
        emb_two = self.embeddings_two(t_result)
        emb_two[t_result == 0, :] = 0
        indices[indices >= self.embeddings_one_vocab_size] = self.embeddings_one.padding_idx
        emb_one = self.embeddings_one(indices)
        return emb_one + emb_two
