import torch
from fairseq.data.encoders import subword_nmt_bpe as bpe
import pdb
from fairseq.data.dictionary import *
import torch.nn as nn
from fairseq.modules.split_embedding import *


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
    #mbart.CC25
    #translation_model = torch.hub.load('nikijean/fairseq:main', 'mbart.CC25', tokenizer='moses', bpe='subword_nmt')

    translation_model = torch.hub.load('nikijean/fairseq:main', 'transformer.wmt16.en-de', tokenizer='moses', bpe='subword_nmt')
    initial_dict_length = len(translation_model.src_dict)
    new_bpe = bpe.SubwordNMTBPE(translation_model.cfg['bpe'], glossaries=["snorflblab", "barflbla"])
    translation_model.bpe = new_bpe
    pdb.set_trace()
    #TODO: we actually need to update the src_dict from file, rather than doing it this way.
    #setting the second argument to True forces the word to be addesd to the dictionary.
    encoded_result = translation_model.encode("I am a snorflblab barflbla", True) #True adds new word to src_dict if it's not there


    num_embeddings = len(translation_model.src_dict) - initial_dict_length + 1 #plus one for the 0 embedding
    #TODO: second argument to Embedding should not be hard-coded
    new_embeddings = Embedding(num_embeddings, 1024)

    split_embeddings = SplitEmbedding(translation_model.models[0].encoder.embed_tokens, new_embeddings)
    translation_model.models[0].encoder.embed_tokens = split_embeddings
    result = translation_model.translate("Hello world")

if __name__ == "__main__":
    main()
