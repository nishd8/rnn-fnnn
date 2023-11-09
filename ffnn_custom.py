import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html


class FFNN(nn.Module):
    def __init__(self, input_dim, d1,d2):
        super(FFNN, self).__init__()
        self.W1 = nn.Linear(input_dim, d1)
        # The rectified linear unit; one valid choice of activation function
        self.activation1 = nn.ReLU()
        self.W2 = nn.Linear(d1, d2)
        self.activation2 = nn.LeakyReLU()
        self.output_dim = 5
        self.W3 = nn.Linear(d2, self.output_dim)

        # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.softmax = nn.LogSoftmax()
        # The cross-entropy/negative log likelihood loss taught in class
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        a = self.W1(input_vector)
        b = self.activation1(a)
        c = self.W2(b)
        d = self.activation2(c)
        e = self.W3(d)
        predicted_vector = self.softmax(e)

        return predicted_vector


# Returns:
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    
    d = []
    for elt in data:
        d.append((elt["text"].split(), int(elt["stars"]-1)))

    return d

class CSVLogger:
    def __init__(self,path,header) -> None:
        f = open(path,'w')
        f.write(header)
        f.close()    
        self.path = path
        self.csv = []

    def log(self,line):
        self.csv.append(line)
    
    def save(self):
        f = open(self.path,'a')
        f.write("\n".join(self.csv))
        f.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d1", "--dim1",
                        required=True, help="array of hidden_dim")

    parser.add_argument("-d2", "--dim2",
                        required=True, help="array of hidden_dim")
    parser.add_argument("-e", "--epochs",
                        required=True, help="array of num of epochs to train")
    parser.add_argument("--train_data", required=True,
                        help="path to training data")
    parser.add_argument("--val_data", required=True,
                        help="path to validation data")
    parser.add_argument("--test_data", default="to fill",
                        help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    training_logger = CSVLogger("results/ffnn_custom_logs.csv",f"Dim1,Dim2,Epoch,Training Accuracy,Training Time,Validation Accuracy,Validation Time\n") 
    test_logger = CSVLogger("results/ffnn_custom,result.csv","Dim1,Dim2,Total Epochs, Test Accuracy, Test Time\n")
    # load data
    print("========== Loading data ==========")
    # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    train_data = load_data(args.train_data)
    valid_data = load_data(args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    
    model = FFNN(input_dim=len(vocab), d1=int(args.dim1),d2=int(args.dim2))
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
    for epoch in range(int(args.epochs)):
        print(f"{epoch+1}:")
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        # Good practice to shuffle order of training data
        random.shuffle(train_data)
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index *
                                                    minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(
                    predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        tr_acc= correct / total
        tr_time = time.time() - start_time
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(
            epoch + 1, tr_acc))
        print("Training time for this epoch: {}".format(tr_time))

        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index *
                                                    minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(
                    predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
        
        val_acc= correct / total
        val_time = time.time() - start_time

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(
            epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(
            time.time() - start_time))
        training_logger.log(",".join([str(args.dim1),str(args.dim2),str(epoch+1),str(tr_acc),str(tr_time),str(val_acc),str(val_time)]))

    if args.test_data != 'to fill':
        print("=====Loading Test Data ====")
        test_data = load_data(args.test_data)
        print("=====Vectorizing Test Data ====")
        test_data = convert_to_vector_representation(test_data, word2index)
        predictions = []

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            minibatch_size = 16
            N = len(test_data)
            start_time = time.time()

            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = test_data[minibatch_index *
                                                        minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(
                        predicted_vector.view(1, -1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
        
        test_acc= correct / total
        test_time = time.time() - start_time

        test_logger.log(','.join([str(args.dim1),str(args.dim2),str(args.epochs),str(test_acc),str(test_time)]))


    training_logger.save()
    test_logger.save()
    # write out to results/test.out
