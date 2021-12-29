import torch
from fairseq.data.encoders import subword_nmt_bpe as bpe
import pdb
from fairseq.data.dictionary import *
import torch.nn as nn
from fairseq.fairseq.modules.split_embedding import *


def Embedding(num_embeddings, embedding_dim, other_idx=0):
    m = nn.Embedding(num_embeddings, embedding_dim)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[other_idx], 0)
    return m

def build_embedding( dictionary, embed_dim):
    num_embeddings = len(dictionary)
    emb = Embedding(num_embeddings, embed_dim)

    return emb

def main():
    en2de2 = torch.hub.load('nikijean/fairseq:main', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')
    initial_dict_length = len(en2de2.src_dict)
    new_bpe = bpe.SubwordNMTBPE(en2de2.cfg['bpe'], glossaries=["snorflblab", "barflbla"])
    en2de2.bpe = new_bpe
    #pdb.set_trace()
    print(en2de2.bpe.encode("I am a snorflblab barflbla"))
    print(en2de2.encode("I am a snorflblab barflbla", True))

    cfg = en2de2.models[0].cfg
    #sub_d = SubDictionary(en2de2.src_dict, 32768)

    num_embeddings = len(en2de2.src_dict) - initial_dict_length + 1 #plus one for the 0 embedding
    new_embeddings = Embedding(num_embeddings, 1024)
        #en2de2.models[0].build_embedding(cfg, sub_d, 1024)

    t = torch.LongTensor([0, 1])
    t2 = en2de2.encode("I am a snorflblab barflbla")

    split_embeddings = SplitEmbedding(en2de2.models[0].encoder.embed_tokens, new_embeddings)
    emb = split_embeddings(t2)
    pdb.set_trace()

if __name__ == "__main__":
    main()
