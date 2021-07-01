import torch
import sys
from word_embeddings import *

word_emb = torch.load(sys.argv[1], map_location=torch.device('cpu'))
word_emb.on_cuda = False
word_emb.save_embeddings_w2v_format(sys.argv[2])


