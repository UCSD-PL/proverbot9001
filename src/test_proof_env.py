from collections import deque
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time, argparse, random
from pathlib_revised import Path2
import dataloader
import coq_serapy as serapi_instance
from coq_serapy import load_commands, kill_comments, get_hyp_type, get_indexed_vars_dict, get_stem, split_tactic, contextSurjective
from coq_serapy.contexts import truncate_tactic_context, FullContext
from search_file import loadPredictorByFile
from search_strategies import completed_proof
from train_encoder import EncoderRNN, DecoderRNN, Lang, tensorFromSentence, EOS_token
from tokenizer import get_symbols, get_words,tokenizers
import pickle
import gym
import fasttext
import os, sys
from util import nostderr, unwrap, eprint, mybarfmt
from collections import defaultdict
from coqproofenv import ProofEnv, FastProofEnv
import tqdm
from train_rl_tabular_maxvval import Agent_model, get_epsilon_greedy



parser = argparse.ArgumentParser()
parser.add_argument("--proof_file", type=Path2)
parser.add_argument("--max-tuples", default=None, type=int)
parser.add_argument("--tokenizer",
						choices=list(tokenizers.keys()), type=str,
						default=list(tokenizers.keys())[0])
parser.add_argument("--num-keywords", default=100, type=int)
parser.add_argument("--lineend", action="store_true")
parser.add_argument('--wandb_log', action= 'store_true')
parser.add_argument('--weightsfile', default = "data/polyarg-weights.dat", type=Path2)
parser.add_argument("--max_term_length", type=int, default=256)
parser.add_argument("--max_attempts", type=int, default=5)
parser.add_argument('--prelude', default=".")
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--update_every', type=int, default=200)
parser.add_argument('--use_fast_check',  action= 'store_true')
parser.add_argument('--lambda', dest='l', type=float, default = 0.5)
parser.add_argument('--lr', type=float, default = 0.01)

args = parser.parse_args()


MAX_STATE_SYMBOLS = 30
FINAL_REWARD = 5






def get_vvals(next_state_names, agent_model):
	to_ret = []
	for each_state in next_state_names :
		to_ret.append(agent_model.get_vval(each_state))
	return to_ret


def get_available_actions_with_next_state_vectors(env, predictor) :
	# print(len( env.coq.proof_context.fg_goals),  env.coq.proof_context.fg_goals)
	# print(completed_proof(env.coq))
	relevant_lemmas = env.coq.local_lemmas[:-1]
	print(env.coq.proof_context)
	full_context_before = FullContext(relevant_lemmas, env.coq.prev_tactics,  env.coq.proof_context)
	predictions = predictor.predictKTactics(
		truncate_tactic_context(full_context_before.as_tcontext(),
								args.max_term_length), args.max_attempts)
	
	next_states = []
	list_of_pred = []
	print("Available actions", [_.prediction for _ in predictions])
	if args.use_fast_check :
		all_available_pred =  [_.prediction.lstrip().rstrip() for _ in predictions]
		all_next_states = env.check_next_states(all_available_pred)
		for next_state_ind in range(len(all_next_states)) :
			curr_next_state = all_next_states[next_state_ind]
			if len(curr_next_state) == 0 :
				continue
			else :
				next_states.append(preprocess_state(curr_next_state))
				list_of_pred.append( predictions[next_state_ind] )
	else :
		for prediction_idx, prediction in enumerate(predictions):
			curr_pred = prediction.prediction.strip()
			state_vec = env.check_next_state(curr_pred)
			if len(state_vec) == 0 :
				continue
			else :
				list_of_pred.append( prediction )
				next_states.append(preprocess_state(state_vec))

	
	return next_states, list_of_pred


def select_action(agent_model, env, predictor, epsilon) :

	next_states, predictions = get_available_actions_with_next_state_vectors(env, predictor)
	
	if len(next_states) == 0 or len(predictions) == 0 :
		print("No predictions available for the current state in select action")
		return None, None, {"vval" : -5}

	for i in range(len(next_states)) :
		if type(next_states[i]) == str and next_states[i] == "fin" :
			action_idx = i
			return predictions[action_idx],[],{"vval" : FINAL_REWARD}
		# else :
		# 	print("Passed",  type(next_states[i]) )

	vvals = get_vvals(next_states, agent_model)
	print(vvals)
	
	print("Current Options : ", [predictions[i].prediction.lstrip().rstrip() + " : " + str(vvals[i]) for i in range(len(vvals)) ])
	env.curr_proof_tactics.append("Current Options : " + " ".join([predictions[i].prediction.lstrip().rstrip() + " : " + str(vvals[i]) for i in range(len(vvals)) ]) )
	action_idx = get_epsilon_greedy([i for i in vvals],[pred.certainty for pred in predictions], epsilon)

	return predictions[action_idx],next_states[action_idx],{"vval" : vvals[action_idx]}



def preprocess_state(state_text_vec) :
	#make this cleaner in association with select action and get all states, especially checking if state is string part
	if type(state_text_vec) == str :
		return state_text_vec
	if len(state_text_vec) < MAX_STATE_SYMBOLS :
		state_text_vec = np.pad(state_text_vec, pad_width= ((0,MAX_STATE_SYMBOLS - len(state_text_vec)),) )
	else :
		state_text_vec = np.array(state_text_vec)
		state_text_vec = state_text_vec[:MAX_STATE_SYMBOLS]
	return state_text_vec




proof_files = [args.proof_file.path]
env = ProofEnv(proof_files, args.prelude, args.wandb_log, state_type = "text")

s,_ = env.reset()
env.run_to_proof("Lemma store_init_data_neutral:")
s = env.get_state_vector_from_text( env.coq.proof_context.fg_goals[0].goal.lstrip().rstrip() )
agent_model = Agent_model()
curr_epsilon = 0.1
predictor = loadPredictorByFile(args.weightsfile)

input("Start Everything?")
while True:

		print("Starting evaluation of 1 pass of the main timestep T loop")
		print(env.coq.proof_context)
		curr_state_name = env.coq.proof_context.fg_goals[0].goal.strip()
		print("State Name : ", curr_state_name)
		prediction,_,agent_info = select_action(agent_model, env, predictor, curr_epsilon)
		if prediction == None :
			print("No actions available")
			s_next,episode_r, done, info = env.admit_and_skip_proof()
		else :
			print("Selected action :" +  prediction.prediction  + "; Take step <press Enter>?")
			s_next,episode_r, done, info = env.step(prediction.prediction)

		print("Step taken")

		if done :
			break