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
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        hidden, _ = self.rnn(inputs)
        # [to fill] obtain output layer representations
        output = self.W(hidden)
        # [to fill] sum over output 
        output_sum = torch.sum(output, dim=0)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output_sum)
        
        return predicted_vector

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
    
    training_logger = CSVLogger("results/rnn_logs.csv",f"Dimension,Total Epochs,Current Epoch,Training Accuracy,Training Time, Validation Accuracy, Validation Time\n") 
    test_logger = CSVLogger("results/rnn_result.csv","Dimension,Total Epochs, Test Accuracy,Test Time\n")
    hidden_dimensions = args.hidden_dim.split(',')
    epochs_array = args.epochs.split(',')

    print("========== Loading data ==========")
    train_data = load_data(args.train_data) 
    valid_data = load_data(args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    for dimension in hidden_dimensions:
    
        model = RNN(50, int(dimension))  # Fill in parameters
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))    
        stopping_condition = False
        epoch = 0

        last_train_accuracy = 0
        last_validation_accuracy = 0
        for epochs in epochs_array:
            while not stopping_condition:
                start_time = time.time()
                random.shuffle(train_data)
                model.train()
                # You will need further code to operationalize training, ffnn.py may be helpful
                print("Training started for epoch {}".format(epoch + 1))
                train_data = train_data
                correct = 0
                total = 0
                minibatch_size = 16
                N = len(train_data)

                loss_total = 0
                loss_count = 0
                for minibatch_index in tqdm(range(N // minibatch_size)):
                    optimizer.zero_grad()
                    loss = None
                    for example_index in range(minibatch_size):
                        input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                        input_words = " ".join(input_words)

                        # Remove punctuation
                        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                        # Look up word embedding dictionary
                        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                        # Transform the input into required shape
                        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                        output = model(vectors)

                        # Get loss
                        example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                        # Get predicted label
                        predicted_label = torch.argmax(output)

                        correct += int(predicted_label == gold_label)
                        # print(predicted_label, gold_label)
                        total += 1
                        if loss is None:
                            loss = example_loss
                        else:
                            loss += example_loss

                    loss = loss / minibatch_size
                    loss_total += loss.data
                    loss_count += 1
                    loss.backward()
                    optimizer.step()
                print(loss_total/loss_count)
                print("Training completed for epoch {}".format(epoch + 1))
                print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
                trainning_accuracy = correct/total
                training_time = time.time() - start_time

                start_time = time.time()

                model.eval()
                correct = 0
                total = 0
                random.shuffle(valid_data)
                print("Validation started for epoch {}".format(epoch + 1))
                valid_data = valid_data

                for input_words, gold_label in tqdm(valid_data):
                    input_words = " ".join(input_words)
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                            in input_words]

                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                    output = model(vectors)
                    predicted_label = torch.argmax(output)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    # print(predicted_label, gold_label)
                print("Validation completed for epoch {}".format(epoch + 1))
                print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
                validation_accuracy = correct/total
                val_time = time.time() - start_time

                training_logger.log(",".join([str(dimension),str(epochs),str(epoch+1),str(trainning_accuracy),str(training_time),str(validation_accuracy),str(val_time)]))

                if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
                    stopping_condition=True
                    print("Training done to avoid overfitting!")
                    print(f"Best validation accuracy is: {last_validation_accuracy}")
                else:
                    last_validation_accuracy = validation_accuracy
                    last_train_accuracy = trainning_accuracy

                epoch += 1

            if args.test_data != 'to fill':
                print("=====Loading Test Data ====")
                test_data = load_data(args.test_data)
                start_time = time.time()
                model.eval()
                correct = 0
                total = 0
                random.shuffle(test_data)
                print(f"Testing started for {dimension} dimensions & {epochs} epochs")
                

                for input_words, gold_label in tqdm(test_data):
                    input_words = " ".join(input_words)
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                            in input_words]

                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                    output = model(vectors)
                    predicted_label = torch.argmax(output)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    # print(predicted_label, gold_label)
                test_accuracy = correct / total
                test_time = time.time() - start_time
                print(f"Testing accuracy for {dimension} dimensions & {epochs} epochs is {test_accuracy}")
                test_logger.log(','.join([str(dimension),str(epochs),str(test_accuracy),str(test_time)]))

    test_logger.save()
    training_logger.save()
    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
