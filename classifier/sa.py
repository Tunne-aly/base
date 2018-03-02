from gensim.models import Word2Vec, KeyedVectors
from os import listdir, sep
from os.path import isfile
import sys
import csv
import re
from random import shuffle
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce

rgx = re.compile(r"\d")
splitter = re.compile(r"[?!.]")


class Entwork(nn.Module):
    def __init__(self):
        super(Entwork, self).__init__()
        self.cv1 = nn.Conv2d(1, 6, (3, 20))
        self.cv2 = nn.Conv2d(6, 16, (3, 20))
        self.fc1 = nn.Linear(384, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.cv1(x)), (3, 3))
        x = F.max_pool2d(F.relu(self.cv2(x)), (3, 3))
        x = x.view(-1, reduce(lambda z, y: z * y, x.size()[1:], 1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


def main(data_path=None, sentences=None, relearn=False, embeddings_path="embeddins.bin", block_length=20):
    if not sentences:
        print("Reading sentences from file...")
        sentences = get_sentences(data_path)
        print("Read {} sentences".format(len(sentences)))
    if isfile(embeddings_path) and not relearn:
        print("Retreiving embeddings from {}".format(embeddings_path))
        embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
    else:
        print("Calculating new embeddings...")
        model = Word2Vec([t[1] for t in sentences], size=300, window=10, workers=2)
        print("Done")
        embeddings = model.wv
        del model
        embeddings.save_word2vec_format(embeddings_path, binary=True)
        print("Embeddings stored to {}".format(embeddings_path))
    print("Building sentece tensors...")
    shuffle(sentences)
    test_data_loader = DataLoader(
        [(int(s[0]) - 1, get_sentence_tensor(s[1], embeddings, block_length)) for s in sentences[:300]],
        batch_size=100, shuffle=False)
    train_data_loader = DataLoader(
        [(int(s[0]) - 1, get_sentence_tensor(s[1], embeddings, block_length)) for s in sentences[300:]],
        batch_size=256, shuffle=True)
    del sentences
    ent = Entwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ent.parameters(), lr=0.001, momentum=0.9)
    print("Training...")
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            labels, inputs = data
            inputs, labels = Variable(inputs), Variable(labels.long())
            optimizer.zero_grad()
            outs = ent(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
        print("epoch {} loss: {}".format(epoch, running_loss))
    print("Evaluating model accuracy...")
    c = 0
    for i, data in enumerate(test_data_loader):
        labels, inputs = data
        outs = ent(Variable(inputs))
        _, preds = torch.max(outs.data, 1)
        for i in range(len(labels)):
            if labels[i] == preds[i]:
                c += 1
            print("Actual: {}, predicted: {}".format(labels[i], preds[i]))
    print("Accuracy: {}".format(c / 300))


def get_sentence_tensor(sentence, embeddings, block_length):
    return torch.stack([torch.stack([get_embedding(i, sentence, embeddings) for i in range(block_length)])])


def get_embedding(idx, sentence, embeddings):
    return (torch.from_numpy(embeddings[sentence[idx]])
            if len(sentence) > idx and sentence[idx] in embeddings
            else torch.zeros(300))


def get_sentences(data_path):
    ls = []
    for p in listdir(data_path):
        if p.endswith(".csv"):
            for sent in read_sentences("{}{}{}".format(data_path, sep, p)):
                ls.append(sent)
    return ls


def read_sentences(file_path):
    with open(file_path) as infile:
        reader = csv.reader(infile)
        for line in reader:
            for para in line[1].splitlines():
                for s in splitter.split(para):
                    if s:
                        yield line[0], [rgx.sub("#", w).lower() for w in s.split()]


def calc_length(frac, data_path=None, sentences=None):
    if not sentences:
        sentences = get_sentences(data_path)
    d = {}
    c = 0
    for s in sentences:
        l = len(s)
        c += 1
        if l not in d:
            d[l] = 1
        else:
            d[l] = d[l] + 1
    i = 1
    tot = 0
    while tot < frac * c:
        if i in d:
            tot += d[i]
        i += 1
    print("{}% of sentences are shorter than {}".format(tot * 100 / c, i + 1))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify file path")
    else:
        main(sys.argv[1])
