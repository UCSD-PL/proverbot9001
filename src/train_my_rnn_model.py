import pandas as pd
import torch
import torch.nn as nn
import coq2vec
import sys

vectorizer = coq2vec.CoqTermRNNVectorizer()
vectorizer.load_weights("coq2vec/term2vec-weights-59.dat")

all_categories = [0,1,2,3]

n_categories = 4

n_iters = 100
print_every = 1
plot_every = 10
learning_rate = 0.005
#plot_every = 1

class zhannRNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(zhannRNN, self).__init__()

        n_hidden_one = 1024
        n_hidden_two = 256
        n_hidden_three = 64
        #n_hidden_four = 128
        #n_hidden_five = 64
        #n_hidden_six = 32
        #n_hidden_seven = 16
        #n_hidden_eight = 8

        self.i2h1 = nn.Linear(input_size, n_hidden_one)
        self.i2h2 = nn.Linear(n_hidden_one, n_hidden_two)
        self.i2h3 = nn.Linear(n_hidden_two, n_hidden_three)
        #self.i2h4 = nn.Linear(n_hidden_three, n_hidden_four)
        #self.i2h5 = nn.Linear(n_hidden_four, n_hidden_five)
        #self.i2h6 = nn.Linear(n_hidden_five, n_hidden_six)
        #self.i2h7 = nn.Linear(n_hidden_six, n_hidden_seven)
        #self.i2h8 = nn.Linear(n_hidden_seven, n_hidden_eight)
        self.h2o = nn.Linear(n_hidden_three, output_size)
        self.tanlayer = nn.Tanh()

    def forward(self, input):
        #combined = torch.cat((input, hidden), 1)
        hidden_one = self.i2h1(input)
        hidden_one = self.tanlayer(hidden_one)
        hidden_two = self.i2h2(hidden_one)
        hidden_two = self.tanlayer(hidden_two)
        hidden_three = self.i2h3(hidden_two)
        hidden_three = self.tanlayer(hidden_three)
        #hidden_four = self.i2h4(hidden_three)
        #hidden_four = self.tanlayer(hidden_four)
        #hidden_five = self.i2h5(hidden_four)
        #hidden_five = self.tanlayer(hidden_five)
        #hidden_six = self.i2h6(hidden_five)
        #hidden_six = self.tanlayer(hidden_six)
        #hidden_seven = self.i2h7(hidden_six)
        #hidden_seven = self.tanlayer(hidden_seven)
        #hidden_eight = self.i2h8(hidden_seven)
        output = self.h2o(hidden_three)
        return output

    #def initHidden(self):
    #    return torch.zeros(1, self.hidden_size)

