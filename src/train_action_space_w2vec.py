import fasttext 
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--datafile", type=str)
args = parser.parse_args()

model = fasttext.train_unsupervised(args.datafile, model='cbow', lr = 0.1,epoch = 10000)

print(model.get_word_vector("hello"))