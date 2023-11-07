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
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        # The rectified linear unit; one valid choice of activation function
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.softmax = nn.LogSoftmax()
        # The cross-entropy/negative log likelihood loss taught in class
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        first_hidden_layer_rep = self.W1(input_vector)

        activation_op = self.activation(first_hidden_layer_rep)

        # [to fill] obtain output layer representation
        output_layer_rep = self.W2(activation_op)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output_layer_rep)

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

def custom_print(dimension,epoch,str):
    print(str)
    f = open(f'logs/ffnn/d_{dimension}_e_{epoch}.txt','a')
    f.write(f"{str}\n")
    f.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim",
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
    hidden_dimensions = args.hidden_dim.split(',')
    epochs_array = args.epochs.split(',')
    for dimension in hidden_dimensions:
        model = FFNN(input_dim=len(vocab), h=int(dimension))
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for epochs in epochs_array:
            custom_print(dimension,epochs,f"========== Initializing Model with {dimension} hidden dimensions and training for {epochs} ==========")
            for epoch in range(int(epochs)):
                custom_print(dimension,epochs,f"========== Training for {epoch} epoch ==========")
                model.train()
                optimizer.zero_grad()
                loss = None
                correct = 0
                total = 0
                start_time = time.time()
                custom_print(dimension,epochs,"Training started for epoch {}".format(epoch + 1))
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
                custom_print(dimension,epochs,"Training completed for epoch {}".format(epoch + 1))
                custom_print(dimension,epochs,"Training accuracy for epoch {}: {}".format(
                    epoch + 1, correct / total))
                custom_print(dimension,epochs,"Training time for this epoch: {}".format(time.time() - start_time))

                loss = None
                correct = 0
                total = 0
                start_time = time.time()
                custom_print(dimension,epochs,"Validation started for epoch {}".format(epoch + 1))
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
                custom_print(dimension,epochs,"Validation completed for epoch {}".format(epoch + 1))
                custom_print(dimension,epochs,"Validation accuracy for epoch {}: {}".format(
                    epoch + 1, correct / total))
                custom_print(dimension,epochs,"Validation time for this epoch: {}".format(
                    time.time() - start_time))

            if args.test_data != 'to fill':
                custom_print(dimension,epochs,"=====Loading Test Data ====")
                test_data = load_data(args.test_data)
                custom_print(dimension,epochs,"=====Vectorizing Test Data ====")
                test_data = convert_to_vector_representation(test_data, word2index)
                predictions = []

                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    for input_vector, _ in tqdm(test_data):
                        predicted_vector = model(input_vector)
                        predicted_label = torch.argmax(predicted_vector).item()
                        predictions.append(predicted_label)

                # Write predictions to an external file
                file_name = f"results/ffnn/test_d_{dimension}_e_{epoch}.out"
                with open(file_name, 'w') as f:
                    for prediction in predictions:
                        f.write(str(prediction) + '\n')

                custom_print(dimension,epoch,f"Test predictions written to {file_name}")

    # write out to results/test.out
