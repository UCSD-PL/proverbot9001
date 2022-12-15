from collections import deque
import json
from sklearn import preprocessing
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
from multiprocessing import Pool
from mpire import WorkerPool
import copy

np.random.seed(2)


def eprint(*args, **kwargs):
	if "guard" not in kwargs or kwargs["guard"]:
		print(*args, file=sys.stderr,
			  **{i: kwargs[i] for i in kwargs if i != 'guard'})
		sys.stderr.flush()



class Agent_model :
	def __init__(self) :
		self.memory = defaultdict(lambda : 0)

	def get_vval(self,s) :
		return self.memory[s]

	def train(self,s,g) :
		self.memory[s] = max(self.memory[s],g)

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


class Memory_max :
	def __init__(self) :
		self.mem = []
		self.num_items = 0
		self.index_dict = defaultdict(lambda : None)
		self.add_called = 0
				
	def add(self,s_vector ,s_text, G) :
		# print("Who shaoll say ....", s_text, G)
		# print("debug",(s_text in self.index_dict, s_text, self.index_dict))
		self.add_called += 1
		if s_text in self.index_dict :
			i = self.index_dict[s_text]
			self.mem[i][1] = max(G, self.mem[i][1]) #Index 1 is the reward, 0 is the vector
		
		else :
			self.index_dict[s_text] = self.num_items
			self.mem.append([s_vector,G])
			self.num_items += 1
		if args.wandb_log :
			wandb.log({"Num times add called" : self.add_called})
			
			wandb.log({ "debug" : (s_text in self.index_dict, s_text, self.index_dict)})

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






def get_epsilon_greedy(vvals,probs= None, epsilon = 1, no_random_if_solved = False):
	# eprint(probs)
	vvals = np.array(vvals)
	if no_random_if_solved  or epsilon == 1:
		if FINAL_REWARD in vvals :
			return np.argmax(vvals)
	probs = np.array(probs)
	probs = probs/np.sum(probs)
	# eprint(sum(probs),probs)
	coin = np.random.rand()
	if coin < epsilon :
		return np.random.choice(np.flatnonzero(np.isclose(vvals, vvals.max())))

	else :
		coin = np.random.rand()
		if coin < epsilon :
			return np.random.choice( range(0, len(vvals)), p = None)
		else :
			return np.random.choice( range(0, len(vvals)), p = probs)



def get_vvals(next_state_names, agent_model):
	to_ret = []
	for each_state in next_state_names :
		to_ret.append(agent_model.get_vval(each_state))
	return to_ret

def is_hyp_token(arg, obligation) :
	
	if arg in obligation.goal and arg in obligation.hypotheses :
		print("Error arg in both")
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
		return None, None, {"vval" : 0}

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



def train_step( agent_model ,optimizer, loss_object,memory,batch_size = 200, device="cpu") :
	minibatch = memory.sample_random_minibatch(batch_size)
	states, rewards = zip(*minibatch)
	
	
	state_tensor = torch.tensor(states).to(device)
	reward_tensor = torch.tensor(rewards).to(device)


	optimizer.zero_grad()
	predicted_rewards = agent_model(state_tensor)

	loss = loss_object(predicted_rewards,reward_tensor)
	loss.backward()
	optimizer.step()
	return loss.item()


def NN_VLearning(T_max, gamma, args) :

	with open("compcert_projs_splits.json",'r') as f :
		data = json.load(f)
	# proof_files = data[0]["test_files"]
	proof_files = [args.proof_file.path]

	if args.use_fast_check :
		env = FastProofEnv(proof_files, args.prelude, args.wandb_log, state_type = "text", num_check_engines = args.max_attempts)
	else :
		env = ProofEnv(proof_files, args.prelude, args.wandb_log, state_type = "text")
	predictor = loadPredictorByFile(args.weightsfile)
	memory = Memory_max()

	s,_ = env.reset()

	# with open("data/Rf_agent_model.pkl", 'rb') as f:
	# 	# agent_model = RandomForestRegressor()
	# 	agent_model = pickle.load(f)

	agent_model = Agent_model()

	
	T = 0
	episode_r = 0
	curr_epsilon = 0.1
	epsilon_increment = 1.25/T_max
	perf_queue = deque(maxlen=20)

	state_rewards = []
	states_in_this_proof = []
	state_names = []
	agent_vval = []
	live_rewards = []
	while T <= T_max :
		print(T)
		# a = get_epsilon_greedy( agent_model.get_qvals(torch.tensor(s)), curr_epsilon )
		# print(s)

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
		# if not done :
		# 	episode_r += prediction.certainty

		if args.wandb_log :
			wandb.log({"T" : T})
			wandb.log({"Num Proofs encountered" : env.num_proofs, "Num Proofs Solved" : env.num_proofs_solved})

		

		if prediction == None :
			if len(state_rewards) > 0 : # For proofs which don't fail in the first step itself bcz if it fails in the first step there is nothing to do.
				state_rewards[-1] = episode_r
				live_rewards.append(episode_r)
		else :
			state_rewards.append(episode_r)
			states_in_this_proof.append(preprocess_state(s))
			state_names.append(curr_state_name)
			agent_vval.append(agent_info["vval"])
			live_rewards.append(episode_r)

		

		if done :
			total_curr_rewards = 0
			true_vvals = []
			for i in range(len(state_rewards) - 1, -1, -1) :
				total_curr_rewards = state_rewards[i] + total_curr_rewards*gamma
				memory.add(states_in_this_proof[i], state_names[i], total_curr_rewards)
				if args.wandb_log :
					wandb.log({"Total_curr_reward" : total_curr_rewards })
				agent_model.train(state_names[i],total_curr_rewards)
				true_vvals.insert(0,total_curr_rewards)
					
			for i in range(len(state_rewards)):	
				if args.wandb_log :
					wandb.log({"Agent Surprise factor" : abs(agent_vval[i] - true_vvals[i]) })
					wandb.log({"Live Rewards": live_rewards[i]})
			
			perf_queue.append(total_curr_rewards)
			state_rewards = []
			states_in_this_proof = []
			agent_vval = []
			state_names = []
			live_rewards = []
			if args.wandb_log :
				wandb.log({"Episode Total Rewards":total_curr_rewards})
				wandb.log({"Exploration Factor":curr_epsilon})
				wandb.log({"Gain" : sum(perf_queue)/len(perf_queue)})
				wandb.log({"Unique states encountered" : memory.num_items})
				wandb.log({"Proof times" : env.proof_time})





		if curr_epsilon < 1 :
			curr_epsilon += epsilon_increment
		
		
		if T%100 == 0 :
			with open("data/agent_state_vval.pkl" , 'wb') as f :
				pickle.dump( dict(agent_model.memory),f)
		

		s = s_next
		T += 1
	
	print("Finished Training")




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
	parser.add_argument('--nepochs', type=int, default=200)
	parser.add_argument('--batch_size', type=int, default=200)
	parser.add_argument('--update_every', type=int, default=200)
	parser.add_argument('--use_fast_check',  action= 'store_true')
	parser.add_argument('--lambda', dest='l', type=float, default = 0.5)
	parser.add_argument('--lr', type=float, default = 0.01)


	args = parser.parse_args()

	if args.wandb_log :
		if args.run_name :
			wandb.init(project="Proverbot", entity="avarghese", name=args.run_name)
		else :
			wandb.init(project="Proverbot", entity="avarghese")


	total_num_steps = 10000
	gamma = 1
	FINAL_REWARD = 1
	MAX_STATE_SYMBOLS = 30
	NN_VLearning(T_max= total_num_steps, gamma = gamma, args = args)

		