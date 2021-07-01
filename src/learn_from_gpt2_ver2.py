import pickle
import random
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import os

import math

from transformers import GPT2Tokenizer, GPT2Model

from word_embeddings import *


def ll_loss(scores):
    """
        Clamped logistic loss
    """
    scores = torch.clamp(scores, min=-20, max=20)
    new_scores = torch.mean(torch.log(1 +  torch.exp(-scores)))
    return new_scores

def rejection_probs(t,id2counts):
    total_counts = 0
    rej_prob = []
    for counts in id2counts:
        total_counts+=counts
    for i in range(len(id2counts)):
        freq_scaled = id2counts[i]/total_counts
        rej_prob.append(1 - min(1,math.sqrt(t/freq_scaled) + t/freq_scaled))

    return rej_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for training word embeddings from GPT2 Model')

    parser.add_argument('--location_dataset', action='store', type=str, default=None,
                        help='Location where dataset and the vocabulary are stored [default:None]')

    parser.add_argument('--model_folder', action='store', type=str, default=None,
                        help='Location where the model and vectors are stored [default:None]')

    parser.add_argument('--pretrained_gpt2_model', action='store', type=str, default="gpt2",
                        help='Name of the pretrained GPT2 model  [default:gpt2]')

    parser.add_argument('--gpu_id', action='store', type=int, default=-1,
                        help='GPU ID for training. -1 implies training on CPU [default:-1]')

    parser.add_argument('--word_emb_size', action='store', type=int, default=768,
                        help='word embedding dimensions - must be equal to the embedding dimensions of the loaded GPT2 model [default:768]')

    parser.add_argument('--t', action='store', type=float, default=5e-6,
                        help='threshold parameter for subsampling. [default:5e-6]')

    parser.add_argument('--algo', action='store', type=str, default="SparseAdam",
                        help='gradient descent algorithm used for training the model. [default:SparseAdam]')

    parser.add_argument('--lr', action='store', type=float, default=0.001,
                        help='learning rate used for gradient descent. [default:0.001]')

    parser.add_argument('--lr_update', action='store', type=float, default=0.99,
                        help='multiplicative factor by which the learning rate for SGD is updated after each epoch. [default:0.99]')

    parser.add_argument('--momentum', action='store', type=float, default=0.0,
                        help='Momentum value for SGD training [default:0.0]')

    parser.add_argument('--weight_decay', action='store', type=float, default=0.00,
                        help='weight decay for regularization. [default:0.00]')

    parser.add_argument('--batch_size', action='store', type=int, default=128,
                        help='Batch size used during training. [default:128]')

    parser.add_argument('--update_lr_every', action='store', type=int, default=500,
                        help='Update learning rate for every [ ] batches. [default:500]')

    parser.add_argument('--MAX_LEN', action='store', type=int, default=200,
                        help='Maximum length of a sentence after using the GPT2 tokenizer. Sentences with more than this length are pruned. [default:200]')

    parser.add_argument('--num_negatives', action='store', type=int, default=10,
                        help='Number of negatives sampled for each true target. [default:10]')

    parser.add_argument('--print_loss_every', action='store', type=int, default=500,
                        help='Print loss for every [ ] batches. [default:500]')

    parser.add_argument('--save_model_every', action='store', type=int, default=10000,
                        help='Save model for every [ ] batches. [default:10000]')

    parser.add_argument('--num_epochs', action='store', type=int, default=5,
                        help='Number of epochs for training. [default:5]')

    parser.add_argument('--num_workers', action='store', type=int, default=15,
                        help='Number of workers for loading data. [default:15]')

    parser.set_defaults()
    args = parser.parse_args()

    if args.algo == "SparseAdam":
        args.num_workers=1

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s')

    logging.info("Loading pretrained GPT2 model")
    model = GPT2Model.from_pretrained(args.pretrained_gpt2_model)
    model.half()
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_gpt2_model)
    tokenizer.add_special_tokens({'pad_token' : "pad"})

    if args.gpu_id == -1:
        use_gpu = False
    else:
        use_gpu = True
        torch.cuda.set_device(args.gpu_id)
        model.to('cuda')

    model.eval()
    logging.info("GPT2 model loaded")

    logging.info("Loading vocabulary and dataset")

    id2word = pickle.load(open(args.location_dataset + "/id2word.p","rb"))
    id2counts = pickle.load(open(args.location_dataset + "/id2counts.p","rb"))
    word_counts = pickle.load(open(args.location_dataset + "/word_counts.p","rb"))
    word2id = pickle.load(open(args.location_dataset + "/word2id.p","rb"))

    lines,words_locs,num_words = pickle.load(open(args.location_dataset + "/dataset.p","rb"))

    logging.info("Dataset and vocabulary loaded")

    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    logging.info("Creating the word embeddings model")

    if args.algo=="Adam":
        embedding_model = word_embeddings(id2word,word2id,id2counts,len(word2id),args.word_emb_size, sparse=False)
    else:
        embedding_model = word_embeddings(id2word,word2id,id2counts,len(word2id),args.word_emb_size, sparse=True)

    if use_gpu==True:
        embedding_model.to('cuda')
        embedding_model.on_cuda = True

    rej_prob = rejection_probs(args.t,id2counts)

    logging.info("word embeddings model created")

    logging.info("Initializing the optimizer")
    if args.algo=="SGD":
        optimizer = optim.SGD(embedding_model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.lr_update)
    if args.algo=="Adam":
        optimizer = optim.Adam(embedding_model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, amsgrad=True)
    if args.algo=="SparseAdam":
        optimizer = optim.SparseAdam(embedding_model.parameters(), lr=args.lr)

    logging.info("Optimizer initialized.")

    logging.info("Creating Dataloader.")

    line_indices = torch.tensor(range(len(lines)))
    num_samples_per_epoch = int(len(lines))

    sent_buffer_dataloader = torch.utils.data.DataLoader(line_indices, batch_size=2**18, shuffle=True,num_workers=1)

    logging.info("Dataloader created.")

    logging.info("Starting training")

    running_loss = 0.0
    word_indices = torch.LongTensor(range(len(id2counts)))
    if args.algo=="SGD":
        current_lr = args.lr
    total_words = int(torch.sum(torch.FloatTensor(num_words)))
    for i in range(args.num_epochs):
        total_words_done = 0

        logging.info("Epoch " + str(i) + " starting")
        for j, sent_buffer_indices in enumerate(sent_buffer_dataloader):
            num_words_buffer = torch.FloatTensor(num_words)[sent_buffer_indices]
            num_samples = int(len(sent_buffer_indices))

            negative_sampler = list(torch.utils.data.WeightedRandomSampler(torch.sqrt(torch.FloatTensor(id2counts)), num_samples=args.num_negatives*num_samples, replacement=True))
            neg_sample_index = 0
            sent_dataloader = torch.utils.data.DataLoader(sent_buffer_indices, shuffle=True, batch_size=args.batch_size,num_workers=args.num_workers)

            total_num_examples_batch = 0
            for k, data in enumerate((sent_dataloader)):
                optimizer.zero_grad()
                loss = 0.0
                sent_indices = data

                targets = []
                targets_sent_id = []
                sentences = []
                total_words_enumerated = 0
                for sent_id, index in enumerate(sent_indices):

                    sentence = lines[index]

                    word_count = num_words[index]

                    sentence = sentence.split()

                    for word in sentence:
                        if word in word2id:
                            total_words_enumerated+=1
                            if random.uniform(0, 1) <  rej_prob[word2id[word]]:
                                continue
                            targets.append(word2id[word])
                            targets_sent_id.append(sent_id)

                    ## No masking at all in this version
                    # sentence[word_location] = '[MASK]'
                    sentence = " ".join(sentence)
                    sentences.append(sentence)

                targets = torch.LongTensor(targets).cuda()

                negatives = [negative_sampler[i%len(negative_sampler)] for i in range(neg_sample_index,neg_sample_index + args.num_negatives*targets.shape[0])]
                neg_sample_index += args.num_negatives*targets.shape[0]

                negatives  = torch.LongTensor(negatives).cuda()
                negatives = negatives.view(args.num_negatives,-1)

                tokenized_sentences = []
                attention_masks = []
                max_length = 0
                for sentence in sentences:
                    tokenized_sentence = tokenizer.tokenize(sentence)
                    if len(tokenized_sentence) > args.MAX_LEN:
                        tokenized_sentence = tokenized_sentence[:args.MAX_LEN-1]
                    max_length = max(max_length,len(tokenized_sentence))
                    tokenized_sentences.append(tokenized_sentence)
                for tokenized_sentence in tokenized_sentences:
                    attention_mask = [1]*len(tokenized_sentence) + [0]*(max_length-len(tokenized_sentence))
                    attention_masks.append(attention_mask)
                num_examples = targets.shape[0]
                # Padding the sentences
                sent_lengths = torch.tensor([len(sentence) for sentence in tokenized_sentences]).view(-1,1).cuda()
                tokenized_sentences = [sentence + (max_length - len(sentence))*['pad'] for sentence in tokenized_sentences]
                indexed_tokens = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences]

                with torch.no_grad():
                    # See the models docstrings for the detail of the inputs
                    tokens_tensor = torch.tensor(indexed_tokens).cuda()
                    attention_masks = torch.tensor(attention_masks).cuda()
                    outputs = model(tokens_tensor,attention_mask = attention_masks)
                    # Transformers models always output tuples.
                    # See the models docstrings for the detail of all the outputs
                    # In our case, the first element is the hidden state of the last layer of the OpenAIGPT2 model
                    encoded_layers = outputs[0]
                    mask = torch.arange(encoded_layers.shape[1]).unsqueeze(0).repeat(encoded_layers.shape[0],1).cuda()
                    mask = (mask < sent_lengths)
                    mask = mask.unsqueeze(2).repeat(1,1,args.word_emb_size).float().cuda()
                    encoded_layers = encoded_layers*mask
                    encoded_layers = torch.sum(encoded_layers,1)/sent_lengths.float()
                    sent_repr = []
                    for id in targets_sent_id:
                        sent_repr.append(encoded_layers[id])
                    #print(sent_repr[1].shape)
                    #print(sent_repr[2].shape)
                    #print(sent_repr[-1].shape)
                    sent_repr = torch.stack(sent_repr)

                target_embeddings = embedding_model(targets)
                similarity_scores = torch.bmm(sent_repr.unsqueeze(1), target_embeddings.unsqueeze(2)).squeeze()
                loss += ll_loss(similarity_scores)
                for neg_ind in range(args.num_negatives):
                    fake_target_embeddings = embedding_model(negatives[neg_ind])
                    neg_similarity_scores = torch.bmm(sent_repr.unsqueeze(1), -fake_target_embeddings.unsqueeze(2)).squeeze()
                    loss += ll_loss(neg_similarity_scores)

                running_loss += loss.tolist()*num_examples
                loss.backward()
                optimizer.step()
                if args.algo=="SGD":
                    scheduler.step()
                    current_lr = current_lr*args.lr_update

                total_num_examples_batch+=num_examples
                total_words_done+=total_words_enumerated
                print(str(total_words_done*100/total_words) + "% done ",end="\r")

                if (k+1)%args.update_lr_every == 0:
                    if args.algo=="SGD":
                        scheduler.step()
                        current_lr = current_lr *args.lr_update


                if (k+1)%args.print_loss_every == 0:
                    logging.info("running loss " + str(running_loss/total_num_examples_batch))
                    if args.algo=="SGD":
                        logging.info("Current Learning Rate " + str(current_lr))
                    running_loss= 0.0
                    total_num_examples_batch = 0

                if (k+1)%args.save_model_every == 0:
                    torch.save(embedding_model, args.model_folder+"/model_last_ckpt.bin")


        torch.save(embedding_model, args.model_folder+"/model_epoch"+str(i+1)+".bin")
        embedding_model.save_embeddings_w2v_format(args.model_folder + "/vectors_epoch" + str(i+1) + ".txt")
        logging.info("Epoch " + str(i) + " is finished.")

    logging.info("Training finished.")
    logging.info("Saving model and vectors.")
    torch.save(embedding_model, args.model_folder+"/model_final.bin")
    embedding_model.save_embeddings_w2v_format(args.model_folder + "/vectors_final.txt")
    logging.info("Models and vectors saved in the folder " + args.model_folder)
