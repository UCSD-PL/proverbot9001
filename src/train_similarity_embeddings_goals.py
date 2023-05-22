import argparse
from collections import defaultdict
import gc
import itertools
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import wandb
import dataloader
from coq_serapy.contexts import truncate_tactic_context, FullContext, ProofContext, Obligation
from coq_serapy import load_commands, kill_comments, get_hyp_type, get_indexed_vars_dict, get_stem, split_tactic, contextSurjective
import pickle
import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
import random

from losses import *
import numpy as np
from typing import List
import re

import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('--wandb_log', action= 'store_true')
args = parser.parse_args()
if args.wandb_log :
	wandb.init(project="Proverbot", entity="avarghese", name="training tactic classifier")



symbols_regexp = (r',|(?::>)|(?::(?!=))|(?::=)|\)|\(|;|@\{|~|\+{1,2}|\*{1,2}|&&|\|\||'
				  r'(?<!\\)/(?!\\)|/\\|\\/|(?<![<*+-/|&])=(?!>)|%|(?<!<)-(?!>)|'
				  r'<-|->|<=|>=|<>|\^|\[|\]|(?<!\|)\}|\{(?!\|)|\.(?=$|\s+)')
def get_symbols(string: str) -> List[str]:
	return [word for word in re.sub(
		r'(' + symbols_regexp + ')',
		r' \1 ', string).split()
			if word.strip() != '']


class Lang:
	def __init__(self, name):
		self.name = name
		self.char2index = {}
		self.char2count = {}
		self.index2char = {0 : "PAD"}
		self.n_chars = 1 #Skipping 0

	def addSentence(self, sentence):
		for character in list(sentence):
			self.addChar(character)

	def addChar(self, char):
		if char not in self.char2index:
			self.char2index[char] = self.n_chars
			self.char2count[char] = 1
			self.index2char[self.n_chars] = char
			self.n_chars += 1
		else:
			self.char2count[char] += 1


def indexesFromSymbols(lang, sentence, ignore_missing = False):
	if ignore_missing :
		to_ret = []
		for word in list(sentence) :
			if word in lang.char2index :
				to_ret.append(lang.char2index[word])
		return to_ret
	else :
		return [lang.char2index[word] for word in list(sentence)]
	




class EncoderRNN(nn.Module):
	def __init__(self, language_size, emedding_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(language_size, emedding_size)
		self.lstm = nn.LSTM(emedding_size, hidden_size, batch_first=True)

	def forward(self, input):
		embedded = self.embedding(input)
		# print("Emvedded size insize Encoder", embedded.shape)
		output, (hidden, cell) = self.lstm(embedded)
		# print("HIdden size insize Encoder", hidden.shape)
		return torch.squeeze(hidden)

	def initHidden(self,device):
		return torch.zeros(1, 1, self.hidden_size, device=device)
	
	def initCell(self,device):
		return torch.zeros(1, 1, self.hidden_size, device=device)


class ObligationEmbedding(nn.Module) :
	def __init__(self, language_size, embedding_size,num_classes, max_symbols, hidden_size = 500, device = 'cuda' ):
		super(ObligationEmbedding, self).__init__()
		self.embedding_size = embedding_size
		self.encoder = EncoderRNN(language_size, embedding_size, hidden_size)
		self.device = device

		self.lin1 = nn.Linear(hidden_size + hidden_size,hidden_size)
		self.lin2 = nn.Linear(hidden_size,hidden_size)
		self.linfinal = nn.Linear(hidden_size,hidden_size)
		self.map_to_classs = nn.Linear(hidden_size, num_classes)
		self.relu = nn.LeakyReLU()
		self.max_symbols = max_symbols


	def network_pass(self,x) :
		x = self.relu(self.lin1(x))
		x = self.relu(self.lin2(x))
		x = self.linfinal(x)
		# x = torch.unsqueeze(x,1)
		x = nn.functional.normalize(x,dim=1)
		return x
			
	
	def forward(self, states, training=False ) :
		goals, hyps = list(zip(*states))
		num_hyp_per_obligation = []
		for hyp_list in hyps :
			num_hyp_per_obligation.append(len(hyp_list))

		temp_goals = []
		for i in goals :
			temp_goals.append(trim_or_pad(i, self.max_symbols))
		
		# print(temp_goals)
		goals = np.array(temp_goals)
		# print("Goals np array shape", goals.shape)

		goals = torch.tensor(goals, dtype=torch.int).to(self.device)
		# print("Goals shape", goals.shape)
		goals = self.encoder(goals)
		# print("Goals encoded shape", goals.shape)

		


		embeddings = self.network_pass(goals)
		# if training :
		# 	embeddings = self.map_to_classs(embeddings)
		return embeddings
	


def trim_or_pad(array, max_len, pad = 0) :
	return_array = None
	if len(array) >= max_len :
		return_array = array[:max_len]
	else :
		return_array = np.pad(array,  (0,max_len - len(array)), constant_values = pad )
	assert len(return_array) == max_len, str(len(array)) + " " + str(len(return_array)) + " " + str(max_len)
	return return_array

if __name__ == "__main__" :
	
	if True :
		pass
		# data = dataloader.scraped_tactics_from_file("data/compcert-scrape.txt", filter_spec="default", max_term_length=30)
		# tactic_context_mapping = {}
		# lang = Lang("termlan")

		# hyp_terms = []
		# goal_terms = []
		# print("Scraping Tactic :")
		# for tactic in tqdm.tqdm(data) :
		# 	# assert len(tactic.context.fg_goals) == 1, str(tactic.tactic) + 
		# 	try :
		# 		goal_symbols = get_symbols(tactic.context.fg_goals[0].goal)
		# 	except Exception as e:
		# 		print("Error for fetching symbols for goal", e)
		# 		print(tactic.tactic)
		# 		print(len(tactic.context.fg_goals),len(tactic.context.bg_goals),len(tactic.context.shelved_goals), len(tactic.context.given_up_goals))
		# 		continue

		# 	lang.addSentence(goal_symbols)
		# 	goal_terms.append(tuple(goal_symbols))



		# 	hyp_symbols = []

		# 	for hyp in tactic.context.fg_goals[0].hypotheses :
		# 		curr_hyp_symbols = get_symbols(hyp)
		# 		hyp_symbols.append(curr_hyp_symbols)
		# 		lang.addSentence(curr_hyp_symbols)

		# 		hyp_terms.append(tuple(curr_hyp_symbols))
		# 		# if len(curr_hyp_symbols) < 5 :
		# 		# 	print(hyp)
		

		# 	current_obl = [goal_symbols, hyp_symbols]

		# 	tactic_class,tactic_args = split_tactic(tactic.tactic)
		# 	if tactic_class in tactic_context_mapping :
		# 		tactic_context_mapping[tactic_class].append(current_obl)
		# 	else :
		# 		tactic_context_mapping[tactic_class] = [current_obl]


		# with open("data/similarity_embeddings_file_data","wb") as f :
		#  	pickle.dump((tactic_context_mapping,lang),f)

		

	# -------------------------------------------------------------------------------------
	# with open("data/similarity_embeddings_file_data","rb") as f :
	# 	tactic_context_mapping,lang = pickle.load(f)
	
	
	# key_length = { i: len(tactic_context_mapping[i]) for i in tactic_context_mapping}
	# for i in key_length :
	# 	if key_length[i] < 2 :
	# 		print("Deleting ", i)
	# 		assert len(tactic_context_mapping[i]) < 2
	# 		del tactic_context_mapping[i]
	

	MAX_SYMBOLS = 200
	
	# keys = list(tactic_context_mapping.keys())
	# data = {}
	# for tactic,contexts in tqdm.tqdm(tactic_context_mapping.items()) :
	# 	for context in contexts :
	# 		goal, hyps = context
	# 		goal_ind = tuple(indexesFromSymbols(lang,goal))
	# 		# print(type(goal_ind))
	# 		# print(goal_ind)
	# 		hyps_ind = []
	# 		for hyp in hyps :
	# 			hyp_ind = tuple(indexesFromSymbols(lang,hyp))
	# 			hyps_ind.append(hyp_ind)
	# 		if len(hyps_ind) == 0 :
	# 			hyps_ind.append( tuple(indexesFromSymbols(lang,":")) )
			
	# 		context_index = (goal_ind, tuple(hyps_ind))
	# 		data[context_index] = keys.index(tactic)
	# 		# print(context_index, data[context_index])


	# data_mapping = {}
	# for context_index, tactic in tqdm.tqdm(data.items()) :
	# 	goal, hyps = context_index
	# 	list_goal = list(goal)
	# 	list_hyps = []
	# 	for hyp in hyps :
	# 		list_hyps.append(list(hyp))
	# 	context_index = [list_goal,list_hyps]
	# 	if tactic in data_mapping :
	# 		data_mapping[tactic].append(context_index)
	# 	else :
	# 		data_mapping[tactic] = [context_index]
	# del data

	# ones = []
	# for tactic in data_mapping :
	# 	if len(data_mapping[tactic]) < 2 :
	# 		ones.append(tactic)
	
	# for item in ones :
	# 	assert len(data_mapping[item]) < 2
	# 	del data_mapping[item]

	# keys = list(data_mapping.keys())

	# print("saving")
	# with open("data/similarity_embeddings_file_preprocessed_data", "wb") as f:
	# 	pickle.dump((data_mapping,lang,keys),f)
	
	# quit()

	print("loading data")
	with open("data/similarity_embeddings_file_preprocessed_data", "rb") as f:
		data_mapping,lang, keys = pickle.load(f)
	
	torch.cuda.empty_cache()
	EMBEDDING_SIZE = 200
	HIDDEN_SIZE = 200
	device = "cuda"
	model = ObligationEmbedding(lang.n_chars, EMBEDDING_SIZE, len(keys), max_symbols = MAX_SYMBOLS, hidden_size=HIDDEN_SIZE, device=device).to(device)
	optimizer = optim.SGD(model.parameters(), lr=0.01)
	criterion = SupConLoss(contrast_mode = "one")
	# criterion = nn.CrossEntropyLoss()
	nepochs = 1000
	minibatch_size = 50



	for _ in range(nepochs) :

		# x_train_batch, y_train_batch = list(zip(*random.sample(data, minibatch_size)))
		x_train_batch = []
		y_train_batch = []
		for tactic_index in data_mapping :
			x_train_batch += random.sample(data_mapping[tactic_index],2)
			y_train_batch += [tactic_index,tactic_index]


		y_train_batch = torch.tensor(y_train_batch).to(device)
		optimizer.zero_grad()
		gc.collect()
		features = model(x_train_batch, training = True)
		gc.collect()
		# print(list(torch.sum(features**2, axis=2).detach().cpu().numpy()))
		features = torch.unsqueeze(features,1)
		# print("Features size",features.shape)
		loss = criterion(features,y_train_batch)
		loss.backward()
		optimizer.step()
		
		if args.wandb_log :
			wandb.log({"Loss":loss})
		if _%100 == 0 :
			print(loss)


	print(X_test.shape)
	y_test_pred = []
	batch_size = min(minibatch_size,len(X_test))
	for X_sample in np.array_split(X_test, len(X_test)//batch_size) :
		curr_pred = model(X_sample).detach().cpu().numpy()
		curr_pred =  np.argmax(curr_pred, axis = 1)
		y_test_pred += list(curr_pred)

	y_test_pred = np.array(y_test_pred)
	print(y_test_pred.shape)
	print(classification_report(Y_test,y_test_pred))

	

