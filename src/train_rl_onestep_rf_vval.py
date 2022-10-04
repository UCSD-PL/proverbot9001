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
from coqproofenv import *



np.random.seed(2)






class Memory :
	def __init__(self) :
		self.mem = []
		self.num_items = 0
				
	def add(self,s,r) :
		self.mem.append([s,r])
		self.num_items += 1
	
	def clear(self) :
		self.mem = []
		self.num_items = 0
	
	def sample_random_minibatch(self,n = None) :
		if n :
			mem_batch = random.sample(self.mem,n)
		else :
			mem_batch = list(self.mem)
			random.shuffle(mem_batch)
		return mem_batch






def get_epsilon_greedy(vvals,epsilon):
	coin = np.random.rand()
	if coin < epsilon :
		return np.argmax(vvals)
	else :
		return np.random.randint(low = 0, high=len(vvals))



def get_vvals(next_states, agent_model):
	vvals = agent_model.predict(next_states)
	return vvals


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




def get_available_actions_with_next_state_vectors(env, predictor) :
	# print(len( env.coq.proof_context.fg_goals),  env.coq.proof_context.fg_goals)
	# print(completed_proof(env.coq))
	relevant_lemmas = env.coq.local_lemmas[:-1]
	full_context_before = FullContext(relevant_lemmas, env.coq.prev_tactics,  env.coq.proof_context)
	predictions = predictor.predictKTactics(
		truncate_tactic_context(full_context_before.as_tcontext(),
								args.max_term_length), args.max_attempts)
	
	next_states = []
	list_of_pred = []
	print("Available actions", [_.prediction for _ in predictions])
	for prediction_idx, prediction in enumerate(predictions):
		curr_pred = prediction.prediction.lstrip().rstrip()
		if curr_pred in env.restrictions[ env.coq.proof_context.fg_goals[0].goal ] :
			continue
		state_vec = env.check_next_state(curr_pred)
		if len(state_vec) == 0 :
			continue
		else :
			list_of_pred.append( prediction )
			next_states.append(state_vec)

	
	return next_states, list_of_pred


def select_action(agent_model, env, predictor, epsilon) :

	next_states, predictions = get_available_actions_with_next_state_vectors(env, predictor)

	if len(next_states) == 0 or len(predictions) == 0 :
		print("No predictions available for the current state in select action")
		return None, None, {"vval" : -5}

	if "fin" in next_states :
		action_idx = next_states.index("fin")
		return predictions[action_idx],[],{"vval" : FINAL_REWARD}

	vvals = get_vvals(next_states, agent_model)
	
	print("Current Options : ", [predictions[i].prediction.lstrip().rstrip() + " : " + str(vvals[i]) for i in range(len(vvals)) ])
	action_idx = get_epsilon_greedy([i.item() for i in vvals],epsilon)

	return predictions[action_idx],next_states[action_idx],{"vval" : vvals[action_idx]}


def select_random_action(env, predictor) :
	print("Selecting Completely Random Action")
	next_states, predictions = get_available_actions_with_next_state_vectors(env, predictor)

	if len(next_states) == 0 or len(predictions) == 0 :
		print("No predictions available for the current state in completely random action")
		return None, None, None

	if "fin" in next_states :
		action_idx = next_states.index("fin")
		return predictions[action_idx], []

	action_idx = np.random.choice(range(len(next_states)))
	return predictions[action_idx],next_states[action_idx]


def RF_QLearning(T_max, gamma, batch_size, args) :

	env = FastProofEnv(args.proof_file.path, args.prelude, args.wandb_log)
	predictor = loadPredictorByFile(args.weightsfile)
	tactic_space_model = fasttext.train_unsupervised(args.proof_file.path, model='cbow', lr = 0.1,epoch = 1000)
	memory = Memory()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	s = env.reset()

	# with open("data/Rf_agent_model.pkl", 'rb') as f:
	# 	# agent_model = RandomForestRegressor()
	# 	agent_model = pickle.load(f)

	agent_model = RandomForestRegressor()

	temp_x = []
	temp_y = []
	for _ in range(10) :
		prediction, next_state = select_random_action(env, predictor)
		s_next,episode_r, done, info = env.step(prediction.prediction)
		if len(next_state) == 0 :
			continue
		temp_x.append(next_state)
		temp_y.append(episode_r)

	print(np.array(temp_x).shape, np.array(temp_y).shape)
	print(temp_x)
	agent_model.fit(temp_x, temp_y)
	s,info = env.reset()
	print("Setup Done")


	
	T = 0
	episode_r = 0
	curr_epsilon = 0.1
	epsilon_increment = 10/T_max
	perf_queue = deque(maxlen=20)

	state_rewards = []
	states_in_this_proof = []
	agent_vval = []
	live_rewards = []
	while T <= T_max :
		print(T)
		# a = get_epsilon_greedy( agent_model.get_qvals(torch.tensor(s)), curr_epsilon )
		# print(s)

		info = {"state_change": False}
		print("Entering the loop")
		while info["state_change"] == False :
			prediction,_,agent_info = select_action(agent_model, env, predictor, curr_epsilon)
			if prediction == None :
				print("Exiting the loop")
				s_next,episode_r, done, info = env.abort_and_finish_proof()
				break
			print("Selected action :" +  prediction.prediction  + "; Take step <press Enter>?")
			s_next,episode_r, done, info = env.step(prediction.prediction)

		print("Step taken")
		if not done :
			episode_r += prediction.certainty

		if args.wandb_log :
			wandb.log({"T" : T})
			wandb.log({"Num Proofs encountered" : env.num_proofs, "Num Proofs Solved" : env.num_proofs_solved})

			
		if prediction == None :
			state_rewards[-1] = episode_r
			live_rewards.append(episode_r)
		else :
			state_rewards.append(episode_r)
			states_in_this_proof.append(s)
			agent_vval.append(agent_info["vval"])
			live_rewards.append(episode_r)

		

		if done :
			total_curr_rewards = 0
			true_vvals = []
			for i in range(len(state_rewards) - 1, -1, -1) :
				total_curr_rewards = state_rewards[i] + total_curr_rewards*gamma
				memory.add(states_in_this_proof[i], total_curr_rewards)
				true_vvals.insert(0,total_curr_rewards)
			

			
			for i in range(len(state_rewards)):	
				if args.wandb_log :
					wandb.log({"Agent Surprise factor" : abs(agent_vval[i] - true_vvals[i]) })
					wandb.log({"Live Rewards": live_rewards[i]})
			
			perf_queue.append(total_curr_rewards)
			state_rewards = []
			states_in_this_proof = []
			agent_vval = []
			live_rewards = []
			if args.wandb_log :
				wandb.log({"Episode Total Rewards":total_curr_rewards})
				wandb.log({"Exploration Factor":curr_epsilon})
				wandb.log({"Gain" : sum(perf_queue)/len(perf_queue)})
				
			
			mem_batch = memory.sample_random_minibatch()
			states, rewards = zip(*mem_batch)
			print(states)
			agent_model.fit(states, rewards)
		
			if curr_epsilon < 1 :
				curr_epsilon += epsilon_increment
				

		s = s_next
		T += 1




if __name__ == "__main__" :
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


	args = parser.parse_args()

	if args.wandb_log :
		if args.run_name :
			wandb.init(project="Proverbot", entity="avarghese", name=args.run_name)
		else :
			wandb.init(project="Proverbot", entity="avarghese")


	total_num_steps = 20000
	gamma = 1
	batch_size = 100
	FINAL_REWARD = 5

	RF_QLearning(T_max= total_num_steps, gamma = gamma, args = args, batch_size = batch_size)

		