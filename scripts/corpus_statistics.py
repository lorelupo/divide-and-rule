"""
Compute corpus statistics based on the preprocessed file and the document
containing the indices of the headlines.

For example:

python scripts/corpus_statistics.py --corpus=data/en-ru/voita_opensubs/context_aware/standard/train.en --headlines=data/en-ru/voita_opensubs/context_aware/standard/train.en-ru.en.heads

Returns:

Number of documents: 1500000
Number of sentences: 6000000
Document length:  
        mean: 4.0
        std: 0.0
        max: 4
Sentence length: 
        mean: 8.9
        std: 5.1
        max: 78

Other examples:

python scripts/corpus_statistics.py --corpus=data/en-de/nei/standard/tmp/train.en --headlines=data/en-de/nei/standard/train.en-de.en.heads

"""

import numpy as np
import argparse

def load_doc_heads(doc_heads_path):
    heads = []
    with open(doc_heads_path) as infile:
        for line in infile:
            heads.append(int(line.strip()))
    return np.array(heads)

def main(corpus, headlines):
    # read corpus
    with open(corpus, 'r') as c:
        lines = c.readlines()
    heads = load_doc_heads(headlines)
    # compute length of each sentence in the corpus
    sentlens = np.array([len(l.split()) for l in lines])
    # compute length of each document in the corpus
    doclens = heads[1:] - heads[:-1]
    print(doclens)
    # print statistics
    print("Number of documents: {}".format(len(heads)))
    print("Number of sentences: {}".format(len(sentlens)))
    print("Document length (in sentences):  \n\tmean: {}\n\tstd: {}\n\tmax: {}".format(np.mean(doclens), np.std(doclens), max(doclens)))
    print("Sentence length (in tokens): \n\tmean: {}\n\tstd: {}\n\tmax: {}".format(np.mean(sentlens), np.std(sentlens), max(sentlens)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description='Compute corpus statistics based on the preprocessed file'
                'and the document containing the indices of the headlines.',
    )
    parser.add_argument('--corpus', required=True, metavar='PATH',
                        help="Corpus file to analyse.")
    parser.add_argument('--headlines', required=False, metavar='PATH',
                        help="Headlines file corresponding to corpus.")
    args = parser.parse_args()

    main(args.corpus, args.headlines)
