import argparse
import itertools

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

from losses import SupervisedContrastiveLoss
import numpy as np
from typing import List
import re

import matplotlib.pyplot as plt



parser = argparse.ArgumentParser()
parser.add_argument('--wandb_log', action= 'store_true')
args = parser.parse_args()
if args.wandb_log :
	wandb.init(project="Proverbot", entity="avarghese")



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
	def __init__(self, language_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(language_size, hidden_size)
		self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

	def forward(self, input):
		embedded = self.embedding(input)
		print("Emvedded size insize Encoder", embedded.shape)
		output, (hidden, cell) = self.lstm(embedded)
		print("HIdden size insize Encoder", hidden.shape)
		return torch.squeeze(hidden)

	def initHidden(self,device):
		return torch.zeros(1, 1, self.hidden_size, device=device)
	
	def initCell(self,device):
		return torch.zeros(1, 1, self.hidden_size, device=device)


class ObligationEmbedding(nn.Module) :
	def __init__(self, language_size, embedding_size, hidden_size = 500, device = 'cuda'):
		super(ObligationEmbedding, self).__init__()
		self.embedding_size = embedding_size
		self.encoder = EncoderRNN(language_size, embedding_size)
		self.device = device

		self.lin1 = nn.Linear(hidden_size + hidden_size,hidden_size)
		self.lin2 = nn.Linear(hidden_size,hidden_size)
		self.linfinal = nn.Linear(hidden_size,hidden_size)
		self.relu = nn.LeakyReLU()


	def network_pass(self,x) :
		x = self.relu(self.lin1(x))
		x = self.relu(self.lin2(x))
		x = self.linfinal(x)
		# x = torch.unsqueeze(x,1)
		x = nn.functional.normalize(x,p=2,dim=1)
		return x
	
	def forward(self, states ) :
		goals, hyps = list(zip(*states))
		num_hyp_per_obligation = []
		for hyp_list in hyps :
			num_hyp_per_obligation.append(len(hyp_list))

		goals = torch.tensor(np.array(goals), dtype=torch.int).to(self.device)
		print("Goals shape", goals.shape)
		goals = self.encoder(goals)
		print("Goals encoded shape", goals.shape)
		hyps = np.concatenate(hyps, axis=0)
		hyps = torch.tensor(hyps, dtype=torch.int).to(self.device)
		print("Hyps shape", hyps.shape)
		hyps = self.encoder(hyps)
		print("Hyps encoded shape", hyps.shape)


		encoded_sum_hyp = torch.zeros_like(goals).to(self.device)
		for i in range(len(num_hyp_per_obligation)) :
			encoded_sum_hyp[i,:] = (torch.sum(hyps[ sum(num_hyp_per_obligation[:i]) : sum(num_hyp_per_obligation[:i]) + num_hyp_per_obligation[i] ], dim=0 ))

		concatenated_tensor= torch.cat( (goals, encoded_sum_hyp) , dim = 1 )
		embeddings = self.network_pass(concatenated_tensor)
		
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

		# -----------------------------------------------------------------------------------
		# with open("data/similarity_embeddings_file_data","wb") as f :
		# 	pickle.dump((goal_terms,hyp_terms),f)

		# with open("data/similarity_embeddings_file_data","rb") as f :
		# 	goal_terms,hyp_terms = pickle.load(f)

		# goal_terms = list(set(goal_terms))
		# freq_data = list(map(len, goal_terms))

		# print(len(goal_terms))
		# plt.hist(freq_data, bins=100, range=(0,1000))
		# plt.savefig("output/freq_plt.png")

		# print("Maximum length : ",max(freq_data))
		# print("Minimum length : ", min(freq_data))

		# goal_terms = [i[1:700] for i in goal_terms]
		# goal_terms = list(set(goal_terms))
		# print("After compressing length",len(goal_terms))

		# def stack_padding(l):
		# 	return np.column_stack((itertools.zip_longest(*l, fillvalue=0)))
		# goal_terms = stack_padding(goal_terms)
		# goal_terms = np.array(goal_terms)
		


		# def trim_lists(list_of_lists):
		# 	shortest_length = 1
		# 	largest_length = len(list_of_lists[0])
		# 	print("Shortest Length : ", shortest_length)
		# 	print("Largest Length : ", largest_length)
		# 	modified_lists = list_of_lists[:,:shortest_length]
		# 	unique_lists = np.unique(modified_lists, axis=0)
		
		# 	while shortest_length < largest_length:
		# 		print(shortest_length)
		# 		shortest_length += 1
		# 		modified_lists = list_of_lists[:,:shortest_length]
		# 		unique_lists = np.unique(modified_lists, axis=0)
		# 		if len(modified_lists) == len(unique_lists) :
		# 			break
		
		# 	return [list(lst) for lst in unique_lists]

		# unique_set = trim_lists(goal_terms)
		# print(len(unique_set[0]))
		# ---------------------------------------------------------------------------------

	with open("data/similarity_embeddings_file_data","rb") as f :
		tactic_context_mapping,lang = pickle.load(f)
	
	EMBEDDING_SIZE = 500
	MAX_SYMBOLS = 256
	device = "cuda"
	model = ObligationEmbedding(lang.n_chars, EMBEDDING_SIZE, hidden_size=500, device=device).to(device)
	
	keys = list(tactic_context_mapping.keys())
	data = []
	for tactic,contexts in tqdm.tqdm(tactic_context_mapping.items()) :
		for context in contexts :
			goal, hyps = context
			goal_ind = trim_or_pad(indexesFromSymbols(lang,goal),MAX_SYMBOLS)
			hyps_ind = []
			for hyp in hyps :
				hyp_ind = trim_or_pad(indexesFromSymbols(lang,hyp), MAX_SYMBOLS)
				hyps_ind.append(hyp_ind)
			if len(hyps_ind) == 0 :
				hyps_ind.append(trim_or_pad(indexesFromSymbols(lang,":"), MAX_SYMBOLS))
			
			context_index = [goal_ind, hyps_ind]
			data.append((context_index,keys.index(tactic)))
	random.shuffle(data)

	



	optimizer = optim.SGD(model.parameters(), lr=0.01)
	criterion = SupervisedContrastiveLoss()
	nepochs = 500
	minibatch_size = 50

	for _ in range(nepochs) :
		x_train_batch, y_train_batch = list(zip(*random.sample(data, minibatch_size)))
		y_train_batch = torch.tensor(y_train_batch).to(device)
		optimizer.zero_grad()
		features = model(x_train_batch)
		# print(list(torch.sum(features**2, axis=2).detach().cpu().numpy()))
		#torch.unsqueeze(features,1)
		print("Features size",features.shape)
		loss = criterion(features,y_train_batch)
		loss.backward()
		optimizer.step()
		
		if args.wandb_log :
			wandb.log({"Loss":loss})
		if _%100 == 0 :
			print(loss)
		quit()
	

