import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from os import listdir, sep
import re
import csv

torch.manual_seed(1)
rgx = re.compile(r"\d")


def main(path):
    sentences = []
    for p in listdir(path):
        if p.endswith(".csv"):
            sentences.extend(get_sentences(sep.join((path, p))))
    skipgrams = []
    for s in sentences:
        skipgrams.extend(get_skipgrams(s[1], 2))
    for i in range(19):
        print(skipgrams[i])


def get_skipgrams(s, n):
    for i in range(len(s)):
        gram = ["" for j in range(2 * n)]
        for j in range(1, n + 1):
            if i - j >= 0:
                gram[n - j] = s[i - j]
            if i + j < len(s):
                gram[n + j - 1] = s[i + j]
        yield gram


def get_sentences(csv_path):
    with open(csv_path) as infile:
        reader = csv.reader(infile)
        for line in reader:
            for para in line[1].splitlines():
                for s in para.split("."):
                    if s:
                        yield (line[0], [rgx.sub("#", w) for w in s.split()])


def hmm(i):
    while i > 0:
        yield [i, i - 1]
        i -= 1


if __name__ == "__main__":
    #l = []
    #l.extend(hmm(10))
    #print(l)
    main("/home/saska/Documents/rev")