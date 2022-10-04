import torch
from torch import nn
from torch import optim
from train_encoder import EncoderRNN, DecoderRNN, Lang, tensorFromSentence, EOS_token, SOS_token
from tokenizer import get_symbols, get_words,tokenizers
from search_file import loadPredictorByFile
from search_strategies import completed_proof
from train_encoder import EncoderRNN, DecoderRNN, Lang, tensorFromSentence, EOS_token
from tokenizer import get_symbols, get_words,tokenizers
import pickle
import gym
import fasttext
import os, sys
import dataloader, tqdm
from pathlib_revised import Path2
import coq_serapy as serapi_instance
from coq_serapy import load_commands, kill_comments, get_hyp_type, get_indexed_vars_dict, get_stem, split_tactic, contextSurjective
from coq_serapy.contexts import truncate_tactic_context, FullContext, ProofContext
from models.components import DNNScorer
import numpy as np


def get_state_vector_from_text(state_text, language_model, device, state_model) :
	state_sentence = get_symbols(state_text)
	state_tensor = tensorFromSentence(language_model,state_sentence,device, ignore_missing = True)
	with torch.no_grad() :
		state_model_hidden = state_model.initHidden(device)
		state_model_cell = state_model.initCell(device)
		input_length = state_tensor.size(0)
		for ei in range(input_length):
			_, state_model_hidden,state_model_cell = state_model(state_tensor[ei], state_model_hidden,state_model_cell)

		
		state= state_model_hidden
	state = state.cpu().detach().numpy().flatten()
	# state = np.append(state,[self.num_commands]).astype("float") 
	return state

def sentence_from_tensor(lang, tensor) :
	symbols = []
	for i in tensor :
		symbols.append(lang.index2char[i])
	
	text = " ".join(symbols)

	return text




def get_text_from_state_vector(state_vector,language_model,device,decoder_model) :
	output = []
	with torch.no_grad() :
		state_vector = torch.tensor(state_vector, dtype=torch.float32)
		state_vector = torch.reshape(state_vector, (1,1,-1))
		decoder_cell = decoder_model.initCell(device)
		decoder_input = torch.tensor([[SOS_token]], device=device)

		decoder_hidden = state_vector
		target_length = 256
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_cell = decoder_model(decoder_input, decoder_hidden, decoder_cell)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()  # detach from history as input
			output.append(topi.item())
			if decoder_input.item() == EOS_token:
				break
	
	return sentence_from_tensor(language_model,output)

def is_hyp_token(arg, obligation) :
	
	if arg in obligation.goal and arg in obligation.hypotheses :
		print("arg in both")
		quit()
	elif arg in obligation.goal :
		return False
	elif arg in obligation.hypotheses :
		return True
	
	# print("Arg nowhere")
	return False

scraped_tactics = dataloader.scraped_tactics_from_file("/home/avarghese_umass_edu/work/rl/proverbot9001/CompCert/common/Globalenvs.v.scrape", "all", 30, None)
predictor = loadPredictorByFile(Path2("data/polyarg-weights.dat"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_model =  torch.load("data/encoder_symbols.model", map_location=torch.device(device))
decoder_model =  torch.load("data/decoder_symbols.model", map_location=torch.device(device))

with open("data/encoder_language_symbols.pkl","rb") as f:
	language_model = pickle.load(f)

proof_start = False

with open("tactics.txt", "r") as f :
	whole_file = f.read()
	list_of_tactic_classes = whole_file.split("\n")
	for i in range(len(list_of_tactic_classes)) :
		list_of_tactic_classes[i] = list_of_tactic_classes[i].strip().rstrip(".")

index = 0
num_zero_vectors = 0
for tactic in tqdm.tqdm(scraped_tactics) : 
	index += 1
	if index == 10 :
		break
	if proof_start :
		if tactic.tactic.lstrip().rstrip().lower() == "qed." :
			proof_start = False
		elif  len(tactic.context.fg_goals) != 0 :
			current_context = ProofContext(tactic.context.fg_goals,tactic.context.bg_goals,tactic.context.shelved_goals,tactic.context.given_up_goals)
			action = tactic.tactic.lstrip().rstrip().rstrip(".")
			state = tactic.context.fg_goals[0].goal.lstrip().rstrip()
			print("State Text :",state)
			state_vector = get_state_vector_from_text(state, language_model, device, state_model)
			
			return_text = get_text_from_state_vector(state_vector,language_model,device,decoder_model)
			print("Decoded Text :", return_text)

	else :
		if  tactic.tactic.lstrip().rstrip().lower() == "proof."   :
			proof_start = True