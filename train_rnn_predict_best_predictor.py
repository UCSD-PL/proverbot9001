import pandas as pd
import torch
import torch.nn as nn
import coq2vec
import sys

vectorizer = coq2vec.CoqTermRNNVectorizer()
vectorizer.load_weights("coq2vec/term2vec-weights-59.dat")

predictor = sys.argv[1]

datadf = pd.read_csv(predictor)
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

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


all_tensors = []
all_correct = []
for idx, row in datadf.iterrows():
    rowdict = dict(row)
    encoded_tensor = vectorizer.term_to_vector(rowdict['goal'])
    all_tensors.append(encoded_tensor)
    all_correct.append(torch.tensor([float(rowdict['0']), float(rowdict['1']), float(rowdict['2']), float(rowdict['3'])], dtype=torch.float))

n_letters = len(all_tensors[0])
print("n letters")
print(n_letters)


#criterion = nn.NLLLoss()
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()

rnn = zhannRNN(n_letters, n_categories)

def train(category_tensor, line_tensor):
    #category_tensor = category_tensor.unsqueeze(0)
    line_tensor = line_tensor.unsqueeze(0)
    #print(line_tensor.size())
    rnn.zero_grad()
    print("size")
    print((line_tensor.size()[0])

    for i in range(line_tensor.size()[0]):
        output = rnn(line_tensor[i])
        loss = criterion(output, category_tensor)
        loss.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

for iter in range(1, n_iters + 1):
    current_loss = 0
    for k in range(len(all_tensors)):
        category_tensor = all_correct[k]
        line_tensor = all_tensors[k]
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        print("loss")
        print(current_loss)
        print("tensors")
        print(category_tensor)
        print(output)
torch.save(rnn.state_dict(), sys.argv[2])
#test_tens = all_tensors[4]
#test_model = zhannRNN(n_letters, n_categories)
#test_model.load_state_dict(torch.load(sys.argv[2]))
#print("try it")
#print(test_model(test_tens))
    # Add current loss avg to list of losses
    #if iter % plot_every == 0:
    #    all_losses.append(current_loss / plot_every)
    #    current_loss = 0
