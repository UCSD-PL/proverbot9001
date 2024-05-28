import pandas as pd
import torch
import torch.nn as nn
import coq2vec
import sys
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

vectorizer = coq2vec.CoqTermRNNVectorizer()
vectorizer.load_weights("coq2vec/term2vec-weights-59.dat")

#datadf = pd.read_csv(predictor)
all_categories = [0,1,2,3]

n_categories = 4

n_iters = 2
print_every = 1
plot_every = 10
learning_rate = 0.005
#plot_every = 1

class zhannRNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(zhannRNN, self).__init__()

        n_hidden_one = 2048
        n_hidden_two = 512
        n_hidden_three = 64
        #n_hidden_four = 128
        #n_hidden_five = 64
        #n_hidden_six = 32
        #n_hidden_seven = 16
        #n_hidden_eight = 8

        #self.i2h1 = nn.Linear(input_size, n_hidden_one)
        #self.i2h2 = nn.Linear(n_hidden_one, n_hidden_two)
        #self.i2h3 = nn.Linear(n_hidden_one, n_hidden_three)
        #self.i2h4 = nn.Linear(n_hidden_three, n_hidden_four)
        #self.i2h5 = nn.Linear(n_hidden_four, n_hidden_five)
        #self.i2h6 = nn.Linear(n_hidden_five, n_hidden_six)
        #self.i2h7 = nn.Linear(n_hidden_six, n_hidden_seven)
        #self.i2h8 = nn.Linear(n_hidden_seven, n_hidden_eight)
        #self.h2o = nn.Linear(n_hidden_three, output_size)
        #self.tanlayer = nn.Tanh()
        self.rnn1 = torch.nn.RNN(input_size, n_hidden_one, nonlinearity='tanh', batch_first=True)
        self.linear1 = torch.nn.Linear(n_hidden_one, output_size)


    def forward(self, x):
        #combined = torch.cat((input, hidden), 1)
        #hidden_one = self.i2h1(input)
        #hidden_one = self.tanlayer(hidden_one)
        #hidden_two = self.i2h2(hidden_one)
        #hidden_two = self.tanlayer(hidden_two)
        #hidden_three = self.i2h3(hidden_one)
        #hidden_three = self.tanlayer(hidden_three)
        #hidden_four = self.i2h4(hidden_three)
        #hidden_four = self.tanlayer(hidden_four)
        #hidden_five = self.i2h5(hidden_four)
        #hidden_five = self.tanlayer(hidden_five)
        #hidden_six = self.i2h6(hidden_five)
        #hidden_six = self.tanlayer(hidden_six)
        #hidden_seven = self.i2h7(hidden_six)
        #hidden_seven = self.tanlayer(hidden_seven)
        #hidden_eight = self.i2h8(hidden_seven)
        #output = self.h2o(hidden_three)
        #return output
        h = self.rnn1(x)[0]
        x = self.linear1(h)
        return x

    #def initHidden(self):
    #    return torch.zeros(1, self.hidden_size)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


all_tensors = torch.load('encoded_goal_tactics.pt')
all_correct = torch.load('encoded_correct.pt')

n_letters = len(all_tensors[0])
print("n letters", flush=True)
print(n_letters)

#criterion = nn.NLLLoss()
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()

rnn = zhannRNN(n_letters, n_categories)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

def train(category_tensor, line_tensor):
    category_tensor = category_tensor
    line_tensor = line_tensor
    rnn.zero_grad()

    output = rnn(line_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    #for p in rnn.parameters():
    #    p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

lines_stacks = []
correct_stacks = []


for k in range(round(0.90*len(all_tensors)/1024)):
    lines_stacks.append(torch.stack(all_tensors[(k*1024):((k+1)*1024)], dim=0))
    correct_stacks.append(torch.stack(all_correct[(k*1024):((k+1)*1024)], dim=0))

print("about to train",flush=True)
for iter in range(1, n_iters + 1):
    current_loss = 0
    #for k in range(round(0.75*len(all_tensors))):
    for k in range(len(lines_stacks)):
        category_tensor = correct_stacks[k]
        line_tensor = lines_stacks[k]
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss
        if (k % 10 == 0):
            print("k")
            print(k,flush=True)
        k = k + 1

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        print("loss")
        print(current_loss/(round(0.90*len(all_tensors))),flush=True)
        #print("tensors")
        #print(category_tensor)
        #print(output)
print("saving", flush=True)
torch.save(rnn.state_dict(), sys.argv[1])
print("saved", flush=True)
print("test", flush=True)
current_loss = 0
