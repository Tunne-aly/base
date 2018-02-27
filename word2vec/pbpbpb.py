import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from os import listdir, sep
from os.path import isfile
import pickle
import re
import csv

torch.manual_seed(1)
rgx = re.compile(r"\d")
splitter = re.compile(r"[?!.]")


def main(path=None, rebuild=False, out="w2v", skipgram_n=5):
    sentences = read_sentences(path, rebuild, out)
    skipgrams = read_skipgrams(sentences, rebuild, out, skipgram_n)
    i = 0
    word_to_ix = {"<UNK>": i}
    i += 1
    for s in skipgrams:
        if s[0] not in word_to_ix:
            word_to_ix[s[0]] = i
            i += 1


def read_skipgrams(sentences, rebuild, out, n):
    f_name = "{}.s".format(out)
    skipgrams = []
    if rebuild or not isfile(f_name):
        for s in sentences:
            skipgrams.extend(get_skipgrams(s[1], n))
        with open(f_name, 'wb') as outp:
            pickle.dump(skipgrams, outp)
    else:
        with open(f_name, 'rb') as inp:
            skipgrams = pickle.load(inp)
    return skipgrams


def read_sentences(path, rebuild, out):
    f_name = "{}.sgm".format(out)
    sentences = []
    if rebuild or not isfile(f_name):
        for p in listdir(path):
            if p.endswith(".csv"):
                sentences.extend(get_sentences(sep.join((path, p))))
        with open(f_name, 'wb') as outp:
            pickle.dump(sentences, outp)
    else:
        with open(f_name, 'rb') as inp:
            sentences = pickle.load(inp)
    return sentences


def get_skipgrams(s, n):
    for i in range(len(s)):
        gram = ["" for j in range(2 * n)]
        for j in range(1, n + 1):
            if i - j >= 0:
                gram[n - j] = s[i - j]
            if i + j < len(s):
                gram[n + j - 1] = s[i + j]
        yield s[i], gram


def get_sentences(csv_path):
    with open(csv_path) as infile:
        reader = csv.reader(infile)
        for line in reader:
            for para in line[1].splitlines():
                for s in splitter.split(para):
                    if s:
                        yield (line[0], [rgx.sub("#", w).lower() for w in s.split()])


if __name__ == "__main__":
    main("/home/saska/Documents/rev")