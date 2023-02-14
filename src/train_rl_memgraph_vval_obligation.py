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
from coq_serapy.contexts import truncate_tactic_context, FullContext, ProofContext, Obligation
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


class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first = True)

	def forward(self, input, hidden, cell):
		embedded = self.embedding(input)
		output, (hidden, cell) = self.lstm(embedded, (hidden,cell))
		return output, hidden, cell

	def initHidden(self,device,batch_size):
		return torch.zeros(1, batch_size, self.hidden_size, device=device)
	
	def initCell(self,device,batch_size):
		return torch.zeros(1, batch_size, self.hidden_size, device=device)


class RegressorNN(nn.Module) :
	def __init__(self, input_size,output_size,hidden_size=200) :
		super(RegressorNN,self).__init__()
		self.lin1 = nn.Linear(input_size,hidden_size)
		self.lin2 = nn.Linear(hidden_size,hidden_size)

		self.linfinal = nn.Linear(hidden_size,output_size)
		self.softmax = nn.Softmax()
		self.relu = nn.LeakyReLU()
		# self.apply(self.init_weights)
		
	def forward(self,x) :
		x = self.relu(self.lin1(x))
		x = self.relu(self.lin2(x))
		x = self.linfinal(x)
		return x		

class Agent_model(nn.Module):
	def __init__(self, dictionary_size, hidden_size, output_size, device):
		super(Agent_model, self).__init__()
		self.encoder = EncoderRNN(dictionary_size,hidden_size)
		self.regressor = RegressorNN(hidden_size*2,output_size)
		self.device = device
		self.sigmoid  = nn.Sigmoid()

	
	def forward(self, states) : #debug the return shape
		# print("len states",len(states))
		goals, hyps = list(zip(*states))
		num_hyp_per_obligation = []
		for hyp_list in hyps :
			num_hyp_per_obligation.append(len(hyp_list))

		# print(goals)
		# for i in goals :
		# 	print(len(i))
		# print("num hpy",num_hyp_per_obligation)
		goals = torch.tensor(np.array(goals)).to(self.device)
		# print("Goals shape",goals.shape)
		hyps = np.concatenate(hyps, axis=0)
		hyps = torch.tensor(hyps).to(self.device)
		# print("hyps shape", hyps.shape)
		goal_encoder_output, goal_encoder_hidden, goal_encoder_cell = self.encoder(goals, self.encoder.initHidden(self.device,goals.shape[0]), self.encoder.initCell(self.device, goals.shape[0]))
		# print("Geh pre", goal_encoder_hidden.shape)
		goal_encoder_hidden = torch.squeeze(goal_encoder_hidden, dim=0)
		hyp_encoder_output, hyp_encoder_hidden, hyp_encoder_cell = self.encoder(hyps, self.encoder.initHidden(self.device,hyps.shape[0]), self.encoder.initCell(self.device, hyps.shape[0]))
		# print("heh pre", hyp_encoder_hidden.shape)
		hyp_encoder_hidden = torch.squeeze(hyp_encoder_hidden,dim=0)
		# print("HEH",hyp_encoder_hidden.shape)
		# print("Geh", goal_encoder_hidden.shape)
		encoded_sum_hyp = torch.zeros_like(goal_encoder_hidden).to(self.device)
		for i in range(len(num_hyp_per_obligation)) :
			encoded_sum_hyp[i,:] = (torch.sum(hyp_encoder_hidden[ sum(num_hyp_per_obligation[:i]) : num_hyp_per_obligation[i] ] ))
		
		# print("encoded sum hyp", encoded_sum_hyp.shape)
		concatenated_tensor= torch.cat( (goal_encoder_hidden, encoded_sum_hyp) , dim = 1 )

		# print("concat shape",concatenated_tensor.shape)
		
		regressor_output = self.regressor(concatenated_tensor)
		softmaxed = self.sigmoid(regressor_output)
		# print("final state shape",softmaxed.shape)
		return softmaxed

class Memory_graph :
	def __init__(self) :
		self.mem = []
		self.num_items = 0
		self.index_dict = {}
		self.add_called = 0
		self.reward_map = { -1 : [], 0 : [], 1 : []}
		self.forward_index_dict = {}
		self.backward_index_dict = {}
				
	def add(self, s_contexts, rewards, next_state_contexts) :
		self.add_called += 1
		if args.wandb_log :
			wandb.log({"Num times add called" : self.add_called})
		# print("Who shaoll say ....", s_text, G)
		# print("debug",(s_text in self.index_dict, s_text, self.index_dict))
		
		indexes = []
		next_state_vectors_indexes = []
		end_reward = rewards[-1]
		for i in range(len(s_contexts)) :
			s_vector = preprocess_statevec(context_to_state_vec(s_contexts[i]))
			s_text = context_to_name(s_contexts[i])
			print("index", i, "State text ~=", s_text)
			if s_text in self.index_dict :
				j = self.index_dict[s_text]
				indexes.append(j)
				print("Found", i, "in memory with index", j, "and vval", self.mem[j][1])
			else :
				self.index_dict[s_text] = self.num_items
				self.forward_index_dict[self.num_items] = set()
				self.backward_index_dict[self.num_items] = set()
				self.mem.append([s_vector,0])
				self.reward_map[0].append(self.num_items)
				indexes.append(self.num_items)
				print("Added", i, "in memory with index", self.num_items, "and vval", 0)
				self.num_items += 1

			curr_next_state_vectors_indexes = []
			print("Looping over next states")
			for j in range(len(next_state_contexts[i])) :
				curr_next_state_vector =  preprocess_statevec(context_to_state_vec(next_state_contexts[i][j]))
				curr_next_state_vector_text = context_to_name(next_state_contexts[i][j])
				print("Next state index", j, "State text ~=", curr_next_state_vector_text)
				if curr_next_state_vector_text in self.index_dict :
					k = self.index_dict[curr_next_state_vector_text]
					curr_next_state_vectors_indexes.append(k)
					print("Found", j, "in memory with index", k, "and vval", self.mem[k][1])
				else :
					self.index_dict[curr_next_state_vector_text] = self.num_items
					self.forward_index_dict[self.num_items] = set()
					self.backward_index_dict[self.num_items] = set()
					self.mem.append([curr_next_state_vector,0])
					self.reward_map[0].append(self.num_items)
					curr_next_state_vectors_indexes.append(self.num_items)
					print("Added", i, "in memory with index", self.num_items, "and vval", 0)
					self.num_items += 1
			next_state_vectors_indexes.append(curr_next_state_vectors_indexes)
			print("Finished looping over next states")

		assert len(indexes) == len(s_contexts) == len(next_state_vectors_indexes)
		for i in range(1,len(indexes)) :
			prev_ind = indexes[i-1]
			now_ind = indexes[i]
			self.forward_index_dict[prev_ind].add(now_ind)
			self.backward_index_dict[now_ind].add(prev_ind)

			for j in range(len(next_state_vectors_indexes[i-1])) :
				curr_next_state_vector_ind = next_state_vectors_indexes[i-1][j]
				self.forward_index_dict[prev_ind].add(curr_next_state_vector_ind)
				self.backward_index_dict[curr_next_state_vector_ind].add(prev_ind)
		
		for ind in indexes :
			print(ind, self.forward_index_dict[ind],self.backward_index_dict[ind])
		print("^^^^^")
		last_ind = indexes[-1]
		curr_last_ind_vval = self.mem[last_ind][1]
		self.mem[last_ind][1] = end_reward
		self.reward_map[curr_last_ind_vval].remove(last_ind)
		self.reward_map[end_reward].append(last_ind)

		
		if curr_last_ind_vval != 0 and curr_last_ind_vval != end_reward :
			print(curr_last_ind_vval,end_reward)
			raise ValueError("Last Vval Changed when not 0")

		self.curr_visited_while_fixing = []
		print("Starting fix graph")
		self.fix_graph(last_ind)
		print("Finished fix graph")


	def fix_graph(self, ind) :
		print(ind, self.forward_index_dict[ind],self.backward_index_dict[ind])
		# if ind in self.curr_visited_while_fixing :
		# 	return
		# 	raise ValueError("Loops in Proof graph")
		self.curr_visited_while_fixing.append(ind)

		forward_vvals = []
		for forward_ind in self.forward_index_dict[ind] :
			forward_vvals.append( self.mem[forward_ind][1] )
		
		if len(forward_vvals) > 0 :
			currvval = self.mem[ind][1] 
			maxforwardvval = max(forward_vvals)
			self.reward_map[currvval].remove(ind)
			self.reward_map[maxforwardvval].append(ind)
			self.mem[ind][1] = maxforwardvval

		for backward_ind in self.backward_index_dict[ind] :
			self.fix_graph(backward_ind)

	def change_value(self, index, new_value) :
		print("Updating index", index, "with Old Value -", self.mem[index][1], "to New Value -", new_value)
		assert self.mem[index][1] != -new_value
		if new_value == 0 :
			assert self.mem[index][1] == 0

		self.mem[index][1] = new_value

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
	
	def sample_random_nonzero_minibatch(self, n = None) :
		numpymem = np.array(self.mem)
		positive_sample = list(numpymem[self.reward_map[1] ])
		negative_sample = list(numpymem[self.reward_map[-1] ])
		mem_batch = positive_sample + negative_sample
		if n :
			mem_batch = random.sample(mem_batch,n)
		else :
			random.shuffle(mem_batch)
		return mem_batch

	def sample_balanced(self) :
		# print(len(self.reward_map[-1]), len(self.reward_map[1]))
		sample_size = min(len(self.reward_map[-1]), len(self.reward_map[1]))
		numpymem = np.array(self.mem)
		positive_sample = list(numpymem[ random.sample(self.reward_map[1], sample_size) ])
		negative_sample = list(numpymem[  random.sample(self.reward_map[-1], sample_size)  ])
		return_list = positive_sample + negative_sample
		random.shuffle(return_list)
		return return_list

	@property
	def num_nonzero_items(self):
		return  len(self.reward_map[-1]) +  len(self.reward_map[1])




def context_to_state_vec(proof_context) :

	# print(type(proof_context))
	assert type(proof_context) == ProofContext
	goal_state_sentence = get_symbols( proof_context.fg_goals[0].goal.strip())
	goal_indexes = get_index_from_sentence(goal_state_sentence)
	all_hyp_indexes = []
	for hyp in proof_context.fg_goals[0].hypotheses :
		hyp_sentence =  get_symbols( hyp.strip())
		hyp_indexes =  get_index_from_sentence(hyp_sentence)
		all_hyp_indexes.append(hyp_indexes)

	all_hyp_indexes.append( get_index_from_sentence(":"))
	return [goal_indexes, all_hyp_indexes]



def get_epsilon_greedy(vvals,probs= None, epsilon = 1):
	# eprint(probs)
	probs = np.array(probs)
	probs = probs/np.sum(probs)
	# eprint(sum(probs),probs)
	coin = np.random.rand()
	if coin < epsilon :
		return np.argmax(vvals)
	else :
		coin = np.random.rand()
		if coin >= epsilon  and 0.4 < epsilon < 0.6: #
			return np.random.choice( range(0, len(vvals)), p = probs)
		else :
			return np.random.choice( range(0, len(vvals)), p = None)



def get_vvals(next_states, agent_model):
	# print(next_states)
	next_state_vectors = []
	next_state_texts = []
	for i in next_states :
		# print(i)
		next_state_vectors.append(preprocess_statevec(context_to_state_vec(i)))
		next_state_texts.append(context_to_name(i))


	with torch.no_grad() :
		vvals = agent_model(next_state_vectors)
	
	vvals = vvals.detach().cpu().numpy().flatten()
	# for i in range(len(next_states)) :
	# 	if next_state_texts[i] in memory.index_dict :
	# 		j = memory.index_dict[ next_state_texts[i]]
	# 		if memory.mem[j][1] == 1 :
	# 			vvals[i] = 1
	return vvals
	

def get_storedvvals(next_states,memory) :
	next_state_texts = []
	stored_vvals = []
	for i in next_states :
		next_state_texts.append(context_to_name(i))
	for i in range(len(next_states)) :
		if next_state_texts[i] in memory.index_dict :
			j = memory.index_dict[ next_state_texts[i]]
			stored_vvals.append(memory.mem[j][1])
		else :
			stored_vvals.append(0)
	return stored_vvals

def swap_with_stored_vvals(network_vvals, stored_vvals) :
	assert len(network_vvals) == len(stored_vvals)
	final_vvals = []
	for i in range(len(network_vvals)) :
		if stored_vvals[i] in (-1,1) :
			final_vvals.append(stored_vvals[i])
		else :
			final_vvals.append(network_vvals[i])
	return final_vvals

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


def is_tactics_repeating(self,context, cutoff = 4) :
	tactics_used = context.prev_tactics
	if tactics_used[-cutoff:].count(tactics_used[-1]) == cutoff :
		return True
	return False

def repeating_actions(action, tactics_used, cutoff = 6) :
	if len(tactics_used) < cutoff :
		return False
	if tactics_used[-cutoff:].count(tactics_used[-1]) == cutoff :
		return True
	return False



def get_available_actions_with_next_state_vectors(env, predictor, tactics_used) :
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
		result = env.check_next_states(all_available_pred)
		# print(result)
		# quit()
		all_next_states, all_next_infos = result
		# print(all_next_states)
		# print(all_next_infos)
		for next_state_ind in range(len(all_next_states)) :
			curr_next_state = all_next_states[next_state_ind]
			if curr_next_state == None or repeating_actions(predictions[next_state_ind], tactics_used):
				continue
			else :
				next_states.append(curr_next_state)
				list_of_pred.append( predictions[next_state_ind] )
	else :
		for prediction_idx, prediction in enumerate(predictions):
			curr_pred = prediction.prediction.strip()
			state_vec = env.check_next_state(curr_pred)
			if state_vec == None :
				continue
			else :
				list_of_pred.append( prediction )
				next_states.append(state_vec)

	
	return next_states, list_of_pred

def obligation_to_vec(obligation) :
	# print(type(proof_context))
	assert type(obligation) == Obligation
	goal_state_sentence = get_symbols(obligation.goal.strip())
	goal_indexes = get_index_from_sentence(goal_state_sentence)
	all_hyp_indexes = []
	for hyp in obligation.hypotheses :
		hyp_sentence =  get_symbols( hyp.strip())
		hyp_indexes =  get_index_from_sentence(hyp_sentence)
		all_hyp_indexes.append(hyp_indexes)

	all_hyp_indexes.append( get_index_from_sentence(":"))
	return [goal_indexes, all_hyp_indexes]

def context_scorer(proof_context,agent_model) :
	score = 1
	for obl  in proof_context.fg_goals :
		vec = preprocess_statevec(obligation_to_vec(obl))
		score *= agent_model(vec).detach().cpu().item()
	for obl  in proof_context.bg_goals :
		vec = preprocess_statevec(obligation_to_vec(obl))
		score *= agent_model(vec).detach().cpu().item()
	for obl  in proof_context.shelved_goals :
		vec = preprocess_statevec(obligation_to_vec(obl))
		score *= agent_model(vec).detach().cpu().item()
	for obl  in proof_context.given_up_goals :
		vec = preprocess_statevec(obligation_to_vec(obl))
		score *= agent_model(vec).detach().cpu().item()
	return score

def select_action(agent_model, memory, env, predictor, epsilon, tactics_used, device = "cpu") :
	a = time.time()
	next_states, predictions = get_available_actions_with_next_state_vectors(env, predictor, tactics_used)
	b = time.time()
	print("Time for running Get_vailable_actions_with... within select action", b - a)
	if len(next_states) == 0 or len(predictions) == 0 :
		print("No predictions available for the current state in select action")
		return None, None, {"vval" : NEGATIVE_REWARD, "next_state_vectors" : []}

	for i in range(len(next_states)) :
		if type(next_states[i]) == str and next_states[i] == "fin" :
			action_idx = i
			return predictions[action_idx],[],{"vval" : FINAL_REWARD, "next_state_vectors" : []}
		# else :
		# 	print("Passed",  type(next_states[i]) )

	a = time.time()
	network_vvals = get_vvals(next_states, agent_model)
	stored_vvals = get_storedvvals(next_states, memory)
	vvals = swap_with_stored_vvals(network_vvals, stored_vvals)
	b = time.time()
	print("Time for running get vvals within select action", b - a)
	print(network_vvals)
	print(stored_vvals)
	print(vvals)
	
	print("Current Options : ", [predictions[i].prediction.lstrip().rstrip() + " : " + str(network_vvals[i]) + "("+ str(stored_vvals[i]) +")" for i in range(len(vvals)) ])
	env.curr_proof_tactics.append("Current Options : " + " ".join([predictions[i].prediction.lstrip().rstrip() + " : " + str(network_vvals[i]) + "("+ str(stored_vvals[i]) +")" for i in range(len(vvals)) ]) )
	action_idx = get_epsilon_greedy([i for i in vvals],[pred.certainty for pred in predictions], epsilon)

	return predictions[action_idx],next_states[action_idx],{"vval" : vvals[action_idx], "next_state_vectors" : next_states}



def preprocess_statevec(state) :
	# if type(state)== str and state == "fin" :
	# 	return state
	assert len(state) == 2
	goal,hyps = state

	assert type(goal) == type(hyps) == list

	hyps_processed = []
	for hyp in hyps :
		if len(hyp) < MAX_STATE_SYMBOLS :
			curr_hyp_processed = np.pad(hyp, pad_width= ((0,MAX_STATE_SYMBOLS - len(hyp)),) )
		else :
			curr_hyp_processed = np.array(hyp)
			curr_hyp_processed = curr_hyp_processed[:MAX_STATE_SYMBOLS]
		hyps_processed.append(curr_hyp_processed)
	if len(goal) < MAX_STATE_SYMBOLS :
		goal_processed = np.pad(goal, pad_width= ((0,MAX_STATE_SYMBOLS - len(goal)),) )
	else :
		goal_processed = np.array(goal)
		goal_processed = goal_processed[:MAX_STATE_SYMBOLS]
	
	assert len(goal_processed) == MAX_STATE_SYMBOLS
	return [goal_processed,hyps_processed]



def train_step( agent_model ,optimizer, loss_object,memory,batch_size = 200, device="cpu") :
	minibatch = memory.sample_random_nonzero_minibatch(batch_size)
	# minibatch = memory.sample_balanced()
	states, rewards = zip(*minibatch)
	
	# print(states)
	# state_tensor = torch.tensor(states).to(device)
	rewards = np.array(rewards)
	rewards[rewards == -1] = 0
	reward_tensor = torch.tensor(rewards,dtype = torch.float).to(device)

	optimizer.zero_grad()
	predicted_rewards = agent_model(states).squeeze()

	# print("Shapes - predicted, reward_tensor", predicted_rewards.shape, reward_tensor.shape)
	# quit()

	loss = loss_object(predicted_rewards,reward_tensor)
	loss.backward()
	optimizer.step()
	return loss.item()

def context_to_name(proof_context) :
	list_of_obligation_names = []
	for obl  in proof_context.fg_goals :
		curr_name = obl.goal.strip() + " ".join(sorted(obl.hypotheses))
		list_of_obligation_names.append(curr_name)
	
	for obl  in proof_context.bg_goals :
		curr_name = obl.goal.strip() + " ".join(sorted(obl.hypotheses))
		list_of_obligation_names.append(curr_name)

	for obl  in proof_context.shelved_goals :
		curr_name = obl.goal.strip() + " ".join(sorted(obl.hypotheses))
		list_of_obligation_names.append(curr_name)
	
	for obl  in proof_context.given_up_goals :
		curr_name = obl.goal.strip() + " ".join(sorted(obl.hypotheses))
		list_of_obligation_names.append(curr_name)

	list_of_obligation_names.sort()
	
	return " ".join(list_of_obligation_names)

def get_index_from_sentence(sentence) :
	return indexesFromSentence(language_model, sentence, ignore_missing = True)


def NN_VLearning(T_max, gamma, args) :
	device = "cuda"
	with open("compcert_projs_splits.json",'r') as f :
		data = json.load(f)
	# proof_files = data[0]["test_files"]
	proof_files = [args.proof_file.path]

	if args.use_fast_check :
		env = FastProofEnv(proof_files, args.prelude, args.wandb_log, state_type = "proof_context", max_proof_len = args.max_proof_len, num_check_engines = args.max_attempts, info_on_check = True)
	else :
		env = ProofEnv(proof_files, args.prelude, args.wandb_log, state_type = "proof_context",  max_proof_len = args.max_proof_len, info_on_check=True)
	predictor = loadPredictorByFile(args.weightsfile)
	
	if args.resume :
		print("Resuming")
		with open("data/memory_graph_obligation.pkl","rb") as f:
			memory = pickle.load(f)
		agent_model = torch.load("data/agent_model_obligation.torch")
	else :
		memory = Memory_graph()
		agent_model = Agent_model( env.language_model.n_chars, 400,1, device).to(device)

	s,_ = env.reset()
	
	
	loss_object = nn.BCELoss()
	optimizer = optim.SGD(agent_model.parameters(),lr = 0.09)

	
	

	
	T = 0
	episode_r = 0
	curr_epsilon = 0.1
	epsilon_increment = 1.15/T_max
	perf_queue = deque(maxlen=20)

	state_rewards = []
	states_in_this_proof = []
	agent_vval = []
	live_rewards = []
	a = time.time()
	time_outside_env = []
	time_for_select_action = []
	tactics_used = []
	next_state_vectors = []
	while T <= T_max :
		print(T)
		# a = get_epsilon_greedy( agent_model.get_qvals(torch.tensor(s)), curr_epsilon )
		# print(s)

		print("Starting evaluation of 1 pass of the main timestep T loop")
		# print(env.coq.proof_context)
		# curr_state_name = env.coq.proof_context.fg_goals[0].goal.strip() + " ".join(sorted(env.coq.proof_context.fg_goals[0].hypotheses))
		curr_state_name = context_to_name(env.coq.proof_context)
		print("State Name : ", "<start_name>" + curr_state_name + "<end_name>")
		c = time.time()
		prediction,_,agent_info = select_action(agent_model, memory, env, predictor, curr_epsilon, tactics_used, device = device)
		
		d = time.time()
		print("Time for select action", d - c)
		time_for_select_action.append(d -c)

		b = time.time()
		time_outside_env.append(b - a)
		if prediction == None :
			print("No actions available")
			s_next,episode_r, done, info = env.admit_and_skip_proof()
		else :
			print("Selected action :" +  prediction.prediction  + "; Take step <press Enter>?")
			s_next,episode_r, done, info = env.step(prediction.prediction)
			tactics_used.append(prediction.prediction)

		print("Step taken")
		a = time.time()
		# if not done :
		# 	episode_r += prediction.certainty

		if args.wandb_log :
			wandb.log({"T" : T})
			wandb.log({"Num Proofs encountered" : env.num_proofs, "Num Proofs Solved" : env.num_proofs_solved})

		


		state_rewards.append(episode_r)
		states_in_this_proof.append(s)
		agent_vval.append(agent_info["vval"])
		live_rewards.append(episode_r)
		next_state_vectors.append(agent_info["next_state_vectors"])

		

		if done :
			print("Current Proof time", env.proof_time)
			print("Current calculated proof time", env.proof_time_calculated) 
			print("Current calculated proof time outside", sum(time_outside_env)) 
			print("Current total calculated proof time", sum(time_outside_env) +  env.proof_time_calculated)
			print("Time for select action in this proof", sum(time_for_select_action))
			time_outside_env = []
			time_for_select_action = []
			total_curr_rewards = live_rewards[-1]
			tactics_used = []
			

			if len(states_in_this_proof) > 0 :
				memory.add(states_in_this_proof, state_rewards, next_state_vectors)
				true_vval = max(state_rewards)

			for i in range(len(state_rewards)):	
				if args.wandb_log :
					wandb.log({"Agent Surprise factor" : abs( (agent_vval[i]*2 - 1) - true_vval) })
					wandb.log({"Live Rewards": live_rewards[i]})
			
			perf_queue.append(total_curr_rewards)
			state_rewards = []
			states_in_this_proof = []
			agent_vval = []
			live_rewards = []
			next_state_vectors = []
			if args.wandb_log :
				wandb.log({"Episode Total Rewards":total_curr_rewards})
				wandb.log({"Exploration Factor":curr_epsilon})
				wandb.log({"Gain" : sum(perf_queue)/len(perf_queue)})
				wandb.log({"Unique states encountered" : memory.num_items})
				wandb.log({"Proof times" : env.proof_time})


		if memory.num_nonzero_items >= args.batch_size and T% args.update_every == 0:
			a = time.time()
			for _ in range(args.nepochs) :
				loss = train_step( agent_model ,optimizer, loss_object,memory,batch_size = args.batch_size, device = device)
				if args.wandb_log :
					wandb.log({"Loss":loss})
			b = time.time()
			print("Time to train", b-a)
			env.curr_proof_tactics.append("\n=========================== Network Trained ================================")

		if curr_epsilon < 1 :
			curr_epsilon += epsilon_increment
		
		
		if T%500 == 0 and T > 0:
			torch.save(agent_model,"data/agent_model_obligation.torch")
			with open("data/memory_graph_obligation.pkl","wb") as f:
				pickle.dump(memory,f)
			# with open("data/memory_graphval_obligation.pkl","wb") as f:
			# 	pickle.dump(memory, f)
		

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
	parser.add_argument('--resume', action= 'store_true')
	parser.add_argument('--weightsfile', default = "data/polyarg-weights.dat", type=Path2)
	parser.add_argument("--max_term_length", type=int, default=256)
	parser.add_argument("--max_attempts", type=int, default=7)
	parser.add_argument("--max_proof_len", type=int, default=50)
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
	NEGATIVE_REWARD = -1
	MAX_STATE_SYMBOLS = 40
	with open("data/encoder_language_symbols.pkl","rb") as f:
		language_model = pickle.load(f)
	NN_VLearning(T_max= total_num_steps, gamma = gamma, args = args)

		