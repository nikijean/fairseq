
import torch.nn as nn


class SplitEmbedding(nn.Module):
    def __init__(self, embeddings_one, num_extra_embeddings):
        super().__init__()
        self.embeddings_one = embeddings_one
        self.embeddings_two = self.create_extra_embedding(num_extra_embeddings + 1)
        self.embeddings_one_vocab_size = self.embeddings_one.num_embeddings

    def create_extra_embedding(self, num_embeddings, other_idx=0):
        embedding_dim = self.embeddings_one.embedding_dim
        m = nn.Embedding(num_embeddings, embedding_dim)
        m.weight = nn.parameter.Parameter(m.weight.type_as(self.embeddings_one.weight))
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[other_idx], 0)
        return m

    def forward(self, indices):
        t_result = indices - (self.embeddings_one_vocab_size - 1)
        t_result[t_result < 0] = 0
        emb_two = self.embeddings_two(t_result)
        emb_two[t_result == 0, :] = 0
        indices[indices >= self.embeddings_one_vocab_size] = self.embeddings_one.padding_idx
        emb_one = self.embeddings_one(indices)
        return emb_one + emb_two
