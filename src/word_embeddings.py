import torch
import torch.nn as nn
import torch.nn.functional as F

class word_embeddings(nn.Module):
    """
    Word Embeddings module

    Stores word embeddings using the PyTorch embedding object
    """
    def __init__(self, id2word,word2id,id2counts,vocab_size, word_emb_size, sparse=True):
        super().__init__()
        self.id2word = id2word
        self.word2id = word2id
        self.id2counts = id2counts
        self.vocab_size = vocab_size
        self.word_emb_size = word_emb_size
        self.on_cuda = False
        self.isSparse = sparse
        self.embeddings = torch.nn.Embedding(vocab_size, word_emb_size,sparse=sparse)
        self.embeddings.weight.data.uniform_(-1/word_emb_size,1/word_emb_size)

    def forward(self,indices):
        return self.embeddings(indices)

    def save_embeddings_w2v_format(self,file_name):
        file = open(file_name,"w", encoding="utf-8")
        file.write(str(self.vocab_size) + " " + str(self.word_emb_size) + "\n")
        for word in self.word2id:
            file.write(word + " ")
            if self.on_cuda==True:
                embedding = self.forward(torch.tensor([self.word2id[word]]).cuda()).squeeze()
            else:
                embedding = self.forward(torch.tensor([self.word2id[word]])).squeeze()
            embedding = embedding.tolist()
            for entry in embedding:
                file.write(str((entry))+ " ")
            file.write("\n")
        file.close()
        return
