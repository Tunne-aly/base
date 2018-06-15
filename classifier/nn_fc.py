import csv
from importlib.util import find_spec
from argparse import ArgumentParser, Namespace
from datetime import datetime
from random import shuffle
from os import getpid
from time import sleep
import torch
from torch import nn, load, Tensor, max, save, cat, transpose
from torch.cuda import empty_cache
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

now = datetime.now


def _argparse() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-b", "--binary_classification", action="store_true",
                            help="Change dataset lables to binary.")
    arg_parser.add_argument("-s", "--batch_size", type=int, default=None,
                            help="Change batch size for traning and result evaluation.")
    arg_parser.add_argument("-e", "--epochs", type=int, default=20,
                            help="Set number of epochs for training.")
    arg_parser.add_argument("-c", "--early_cancel", action="store_false",
                            help="don't cancel training on apparent convergence.")
    arg_parser.add_argument("-t", "--testset_size", type=int, default=200,
                            help="Specify testset size")
    arg_parser.add_argument("-d", "--dataset_size", type=int, default=None,
                            help="specify dataset size")
    arg_parser.add_argument("-n", "--good_bad_neutral", action="store_true",
                            help="Change data set to good-bad-neutral. Superseded by '-b'")
    arg_parser.add_argument("-a", "--auto_testset", action="store_true",
                            help="automatically set testset size to 10%% of dataset size")
    arg_parser.add_argument("-m", "--in_memory", action="store_true",
                            help="store all data in RAM during processing")
    arg_parser.add_argument("-g", "--no-cuda", action="store_false",
                            help="That's no gouda.")
    arg_parser.add_argument("-y", "--hidden_layers", type=int, nargs='*', default=[100],
                            help="number of neurons on hidden layers")
    return arg_parser


class SentenceLoader(Dataset):
    def __init__(self, metadata, block_length, in_memory=False):
        self.block_length = block_length
        self.in_memory = in_memory
        if in_memory:
            self.metadata = [[v[0], self.load_tensor(v[1]), v[2], v[3]] for v in metadata]
        else:
            self.metadata = metadata

    def load_tensor(self, t_path):
        ten = load(t_path)
        if ten.size()[1] < self.block_length:
            ten = F.pad(ten, (0, 0, 0, self.block_length - ten.size()[1])).data
        elif ten.size()[1] > self.block_length:
            ten = ten.resize_(1, self.block_length, 300)
        return ten

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        ls = self.metadata[idx]
        ten = ls[1]
        if not self.in_memory:
            ten = self.load_tensor(ls[1])
        return {'lable': ls[0], 'ten': ten,
                'file': ls[2], 'row': ls[3]}


class Entwork(nn.Module):
    def __init__(self, hiddens, output_class_count=2):
        super(Entwork, self).__init__()
        in_size = 232 * 300
        self.fcl = nn.ModuleList()
        for size in hiddens:
            self.fcl.append(nn.Linear(in_size, size))
            in_size = size
        self.fc = nn.Linear(in_size, output_class_count)

    def forward(self, x):
        x = x.view(-1, 232 * 300)
        for cfc in self.fcl:
            x = cfc(x)
            x = torch.nn.functional.relu(x)
        return self.fc(x)


def log(s, log_file):
    if log_file:
        log_file.write("{}\n".format(s))
        log_file.flush()
    print(s)


def hits(preds, labels):
    return len([i for i in zip(labels, preds) if i[0] == i[1][0]])


def hits2(preds, labels):
    return len([i for i in zip(labels, preds) if i[0] in i[1]])


def train(net: Entwork, data_loader: DataLoader, weights, log_file=None, epochs=20,
          device=torch.device("cpu"), stop_on_converge=True, test_loader: DataLoader = None):
    min_seen = 100000
    gap = 0.001
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.RMSprop(net.parameters())
    log("Training...", log_file)
    for epoch in range(epochs):
        acc = 0
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            labels, inputs = data["lable"].to(device), data["ten"].to(device)
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outs = net(inputs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
            acc += (outs.max(1)[1] == labels).float().sum()
        tr_res = ""
        t_loss = None
        if test_loader:
            t_running = 0.0
            t_acc = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    labels, inputs = data["lable"].to(device), data["ten"].to(device)
                    inputs, labels = Variable(inputs), Variable(labels)
                    outs = net(inputs)
                    loss = criterion(outs, labels)
                    t_running+= loss.data.item()
                    t_acc += (outs.max(1)[1] == labels).float().sum()
            t_loss = t_running / len(test_loader)
            t_acc = t_acc / len(test_loader.dataset)
            tr_res = ". Training loss: {:.4f}, acc: {:.4f}".format(t_loss, t_acc)
        loss = running_loss / len(data_loader)
        log("epoch {} loss: {:.4f}, acc: {:.4f}{}".format(epoch, loss, acc / len(data_loader.dataset), tr_res), log_file)
        if stop_on_converge and epoch > 10:
            if t_loss and t_loss + gap > min_seen > t_loss - gap:
                log("loss seems to have converged. Stopping", log_file)
                break
            elif loss + gap > min_seen > loss - gap:
                log("loss seems to have converged. Stopping", log_file)
                break
        if t_loss:
            min_seen = min(min_seen, t_loss)
        else:
            min_seen = min(min_seen, loss)


def test(net, data_loader: DataLoader, log_file=None, device=torch.device("cpu"), binary=False):
    with torch.no_grad():
        res = []
        log("Evaluating model accuracy...", log_file)
        c = 0
        t = 0
        w = 0
        for i, data in enumerate(data_loader):
            labels, inputs = data["lable"].to(device), data["ten"].to(device)
            fps, rs = data["file"], data["row"]
            outs = net(Variable(inputs))
            preds = list(map(arg_2_max, outs.data))
            for j in range(len(labels)):
                t += 1
                if labels[j] == preds[j][0]:
                    c += 1
                elif abs(labels[j] in preds[j]) == 1:
                    w += 1
                res.append("Actual: {}, predicted: {} ... {}:{}".format(labels[j], preds[j], fps[j], rs[j]))
        log("Accuracy: {} correct out of {}. fraction: {:.4f}".format(c, t, c / t), log_file)
        if not binary:
            log("{} out of {} accurate at 2. fraction: {:.4f}".
                format(w + c, len(data_loader.dataset), (w + c) / len(data_loader.dataset)),
                log_file)
        log("\n".join(res), log_file)


def check_training(net, data_loader: DataLoader, log_file=None, device=torch.device("cpu"), binary=False):
    with torch.no_grad():
        log("evaluating prediction accuracy with training set", log_file)
        acc = 0
        obacc = 0
        for i, data in enumerate(data_loader):
            labels, inputs = data["lable"].to(device), data["ten"].to(device)
            outs = net(Variable(inputs))
            preds = list(map(arg_2_max, outs.data))
            acc += hits(preds, labels)
            obacc += hits2(preds, labels)
        log("Accuracy on training set: {} correct out of {}. fraction: {}"
            .format(acc, len(data_loader.dataset), acc / len(data_loader.dataset)), log_file)
        if not binary:
            log("{} out of {} accurate at 2. fraction: {}".
                format(obacc, len(data_loader.dataset), obacc / len(data_loader.dataset)),
                log_file)


def load_metadata(fname):
    with open(fname, 'r') as in_file:
        in_csv = csv.reader(in_file)
        for row in in_csv:
            yield [int(row[0])] + row[1:]


def make_binary(metadata):
    for row in metadata:
        if row[0] < 2:
            row[0] = 0
            yield row
        elif row[0] > 2:
            row[0] = 1
            yield row


def make_trinary(metadata):
    for row in metadata:
        if row[0] < 2:
            row[0] = 0
            yield row
        elif row[0] == 2:
            row[0] = 1
            yield row
        else:
            row[0] = 2
            yield row


def make_weights(metadata, binary=True, trinary=False):
    l = [0.0, 0.0] if binary else [0.0, 0.0, 0.0, 0.0, 0.0]
    l = [0.0, 0.0, 0.0] if trinary else l
    for row in metadata:
        l[row[0]] += 1
    for i, v in enumerate(l):
        l[i] = len(metadata) / v
    return tuple(l)


def arg_2_max(ten):
    v1 = ten[0]
    i1 = 0
    v2 = ten[1]
    i2 = 1
    if v2 > v1:
        v1, i1, v2, i2 = v2, i2, v1, i1
    for i in range(2, len(ten)):
        if ten[i] > v2:
            v2 = ten[i]
            i2 = i
            if v2 > v1:
                v1, i1, v2, i2 = v2, i2, v1, i1
    return i1, i2


def run_cycle(logfile, batch_size=250, dataset_size=None,
              testset_size=200, block_length=232, model_output=True,
              epochs=20, binary_classification=True, stop_on_converge=True,
              gbn=False, autoset=True, in_memory=False, cuda=True, hiddens=None, desc="model"):
    if not hiddens:
        hiddens = [100]
    device = torch.device("cpu")
    if torch.cuda.is_available() and cuda:
        device = torch.device("cuda:0")
        log("Using GPU", logfile)
    metadata = list(load_metadata('metadata.txt'))
    log("Loaded metadata", logfile)
    if not dataset_size:
        dataset_size = len(metadata)
    if autoset:
        testset_size = dataset_size // 10
        log("set testset size to {}".format(testset_size), logfile)
    if binary_classification:
        metadata = list(make_binary(metadata))
    elif gbn:
        metadata = list(make_trinary(metadata))
    shuffle(metadata)
    test_loader = DataLoader(SentenceLoader(metadata[:testset_size], block_length, in_memory),
                             shuffle=False, batch_size=batch_size, num_workers=4)
    train_meta = metadata[testset_size:dataset_size] if dataset_size else metadata[testset_size:]
    train_loader = DataLoader(SentenceLoader(train_meta, block_length, in_memory),
                              shuffle=True, batch_size=batch_size, num_workers=4)
    weights = Tensor(make_weights(train_meta, binary=binary_classification, trinary=gbn)).to(device)
    log("Calculated weights: {}".format(weights), logfile)
    if binary_classification:
        net = Entwork(hiddens)
    elif gbn:
        net = Entwork(hiddens, output_class_count=3)
    else:
        net = Entwork(hiddens, output_class_count=5)
    net = net.to(device)
    if find_spec("psutil"):
        import psutil
        rss = psutil.Process(getpid()).memory_info().rss / 1024 ** 3
        log("Current resident set size is {0:.2f} Gigs".format(rss), logfile)
    log("Started training at {}".format(now()), logfile)
    train(net, train_loader, weights, log_file=logfile, epochs=epochs,
          device=device, stop_on_converge=stop_on_converge, test_loader=test_loader)
    log("Done training at {}".format(now()), logfile)
    if model_output:
        name = "model_{}.bin".format(desc)
        with open(name, 'wb') as out_file:
            save(net.state_dict(), out_file)
            log("Stored model state dictionary to {}".format(name), logfile)
    check_training(net, train_loader, log_file=logfile, device=device, binary=binary_classification)
    test(net, test_loader, log_file=logfile, device=device, binary=binary_classification)


def main(args: Namespace):
    desc = "{}_({})_{}_{}".format(
        "binary" if args.binary_classification else ("gbn" if args.good_bad_neutral else "labelled"),
        "-".join(map(str, args.hidden_layers)),
        "auto" if args.auto_testset else args.testset_size, args.epochs)
    with open("{}.log".format(desc), 'w') as logfile:
        log("Started working on {}".format(desc), logfile)
        batch_size = args.batch_size
        if not batch_size:
            batch_size = 100 if (args.hidden_layers[0] > 1000) or \
                                (args.hidden_layers[0] > 7000 and len(args.hidden_layers) > 2) else 500
        log("batch size: {}".format(batch_size), logfile)
        run_cycle(logfile, batch_size=batch_size,
                  binary_classification=args.binary_classification,
                  epochs=args.epochs,
                  stop_on_converge=args.early_cancel,
                  testset_size=args.testset_size,
                  dataset_size=args.dataset_size,
                  gbn=args.good_bad_neutral,
                  autoset=args.auto_testset,
                  in_memory=args.in_memory,
                  cuda=args.no_cuda,
                  hiddens=args.hidden_layers,
                  desc=desc)


if __name__ == '__main__':
    main(_argparse().parse_args())
