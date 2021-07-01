import pickle
import argparse
import os
import logging

def construct_vocab(file_location, min_count, vocab_size):
    """ Constructing vocabulary for the dataset

        Keyword arguments:
        file_location : Location of the dataset(The dataset is assumed to be in a tokenized format)
        min_count : minimum count of each word to be included in the vocabulary
        vocab_size: Maximum vocabulary size allowed

        Output:
        id2word,word2id,id2counts,word_counts (self-explainable)
    """
    word_counts = dict()

    file = open(file_location,"r", encoding="utf-8")


    for i,line in enumerate(file):
        if i % 10000 == 0:
            print(str(i) + " lines processed.", end="\r")
        for word in line.split():
            word = word.lower()
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    new_word_counts = dict()
    for word in word_counts:
        if word_counts[word] >= min_count:
            new_word_counts[word] = word_counts[word]

    sorted_word_counts = sorted(list(new_word_counts.items()),key=lambda count: count[1],reverse=True)
    if len(sorted_word_counts) > vocab_size:
        sorted_word_counts = sorted_word_counts[:vocab_size]

    word_counts = dict()
    word2id = dict()
    id2word = []
    id2counts = []
    i = 0

    for word_count_pair in sorted_word_counts:
        word_counts[word_count_pair[0]] = word_count_pair[1]
        word2id[word_count_pair[0]] = i
        id2word.append(word_count_pair[0])
        id2counts.append(word_count_pair[1])
        i+=1
    file.close()
    return id2word,word2id,id2counts,word_counts

def construct_dataset(file_location,word2id):
    """ Constructing vocabulary for the dataset
        Keyword arguments:
        file_location : Location of the dataset(The dataset is assumed to be in a tokenized format)
        word2id : dictionary which maps words to ids

        Output:
        lines: Dataset stored in an array format. Each entry represents a sentence.
        words_locs: Location of words in the vocabulary(word2id) for each line in lines
        num_words: Number of words in the vocabulary(word2id) for each line in lines
    """
    lines = []
    words_locs = []
    num_words = []
    file = open(file_location,"r", encoding="utf-8")
    for i,line in enumerate(file):
        if i % 10000 == 0:
            print(str(i) + " lines processed.", end="\r")
        line = line.lower()
        words_loc = []
        words_in_vocab = 0
        for j,word in enumerate(line.split()):
            if word in word2id:
                words_loc.append(j)
                words_in_vocab += 1
        if words_in_vocab==0:
            continue
        lines.append(line[:-1])
        words_locs.append(words_loc)
        num_words.append(words_in_vocab)
    file.close()
    return lines, words_locs, num_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for creating the vocabulary and saving the dataset')

    parser.add_argument('--dataset_location', action='store', type=str, default="wiki_text",
                        help='Address of the dataset [default:wiki_text]')

    parser.add_argument('--min_count', action='store', type=int, default=10,
                        help='min occurrence count for each word in the dictionary[default:10]')

    parser.add_argument('--max_vocab_size', action='store', type=int, default=200000,
                        help='Maximum size of the vocabulary [default:200000]')

    parser.add_argument('--location_save_vocab_dataset', action='store', type=str, default="training_datset/",
                        help='Address of the dataset [default:training_datset/]')

    parser.set_defaults()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s')

    logging.info("Creating vocabulary")
    id2word,word2id,id2counts, word_counts = construct_vocab(args.dataset_location, args.min_count, args.max_vocab_size)

    if not os.path.isdir(args.location_save_vocab_dataset):
        os.makedirs(args.location_save_vocab_dataset)
    logging.info("Vocabulary created")

    pickle.dump(id2word,open(args.location_save_vocab_dataset +"/id2word.p","wb"))
    pickle.dump(id2counts,open(args.location_save_vocab_dataset +"/id2counts.p","wb"))
    pickle.dump(word_counts,open(args.location_save_vocab_dataset +"/word_counts.p","wb"))
    pickle.dump(word2id,open(args.location_save_vocab_dataset +"/word2id.p","wb"))
    logging.info("Vocabulary saved")

    logging.info("Creating dataset")

    lines, words_locs, num_words = construct_dataset(args.dataset_location,word2id)

    logging.info("Dataset created")
    with open(args.location_save_vocab_dataset + "/dataset.p","wb") as f:
        pickle.dump([lines,words_locs,num_words],f)

    logging.info("Dataset saved")
    exit()
