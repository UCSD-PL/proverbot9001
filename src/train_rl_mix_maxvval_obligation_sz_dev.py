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
from pathlib import Path
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
		# x = self.relu(self.lin2(x))
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

class Memory_max :
	def __init__(self) :
		self.mem = []
		self.num_items = 0
		self.index_dict = {}
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



def get_vvals(next_states, next_state_texts, agent_model, memory, device, replace_vval=True):
	# print(next_states)
	with torch.no_grad() :
		vvals = agent_model(next_states)
	
	vvals = vvals.detach().cpu().numpy()

	for i in range(len(next_states)) :
		if next_state_texts[i] in memory.index_dict :
			j = memory.index_dict[ next_state_texts[i]]
			if replace_vval:
				if memory.mem[j][1] > 0.5 :
					vvals[i] = memory.mem[j][1]
	return vvals
	

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
	next_state_texts = []
	print("Available actions", [_.prediction for _ in predictions])
	if args.use_fast_check :
		all_available_pred =  [_.prediction.lstrip().rstrip() for _ in predictions]
		result = env.check_next_states(all_available_pred)
		# print(result)
		# quit()
		all_next_states, all_next_infos = result
		print(all_next_states)
		print(all_next_infos)
		for next_state_ind in range(len(all_next_states)) :
			curr_next_state = all_next_states[next_state_ind]
			if len(curr_next_state) == 0 or repeating_actions(predictions[next_state_ind], tactics_used):
				continue
			else :
				curr_next_state_text = all_next_infos[next_state_ind]["state_text"] 
				next_states.append(preprocess_state(curr_next_state))
				list_of_pred.append( predictions[next_state_ind] )
				next_state_texts.append(curr_next_state_text)
	else :
		for prediction_idx, prediction in enumerate(predictions):
			curr_pred = prediction.prediction.strip()
			state_vec = env.check_next_state(curr_pred)
			if len(state_vec) == 0 :
				continue
			else :
				list_of_pred.append( prediction )
				next_states.append(preprocess_state(state_vec))

	
	return next_states, list_of_pred, next_state_texts


def select_action(agent_model, memory, env, predictor, epsilon, tactics_used, device = "cpu",replace_vval=False) :
	a = time.time()
	next_states, predictions, next_state_texts = get_available_actions_with_next_state_vectors(env, predictor, tactics_used)
	b = time.time()
	print("Time for running Get_vailable_actions_with... within select action", b - a)
	if len(next_states) == 0 or len(predictions) == 0 :
		print("No predictions available for the current state in select action")
		return None, None, {"vval" : -5}

	for i in range(len(next_states)) :
		if type(next_states[i]) == str and next_states[i] == "fin" :
			action_idx = i
			return predictions[action_idx],[],{"vval" : FINAL_REWARD}
		# else :
		# 	print("Passed",  type(next_states[i]) )

	a = time.time()
	vvals = get_vvals(next_states, next_state_texts, agent_model, memory, device,replace_vval=False)
	b = time.time()
	print("Time for running get vvals within select action", b - a)
	print(vvals)
	
	print("Current Options : ", [predictions[i].prediction.lstrip().rstrip() + " : " + str(vvals[i]) for i in range(len(vvals)) ])
	env.curr_proof_tactics.append("Current Options : " + " ".join([predictions[i].prediction.lstrip().rstrip() + " : " + str(vvals[i]) for i in range(len(vvals)) ]) )
	action_idx = get_epsilon_greedy([i for i in vvals],[pred.certainty for pred in predictions], epsilon)

	return predictions[action_idx],next_states[action_idx],{"vval" : vvals[action_idx]}



def preprocess_state(state) :
	if type(state)== str and state == "fin" :
		return state

	goal,hyps = state
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
	
	return [goal_processed,hyps_processed]



def train_step( agent_model ,optimizer, loss_object,memory,batch_size = 200, device="cpu") :
	minibatch = memory.sample_random_minibatch(batch_size)
	states, rewards = zip(*minibatch)
	
	# print(states)
	# state_tensor = torch.tensor(states).to(device)
	reward_tensor = torch.tensor(rewards,dtype = torch.float).to(device)


	optimizer.zero_grad()
	predicted_rewards = agent_model(states).squeeze()

	# print("Shapes - predicted, reward_tensor", predicted_rewards.shape, reward_tensor.shape)
	# quit()

	loss = loss_object(predicted_rewards,reward_tensor)
	loss.backward()
	optimizer.step()
	return loss.item()


def NN_VLearning(T_max, gamma, args) :

	with open("compcert_projs_splits.json",'r') as f :
		data = json.load(f)
	# proof_files = data[0]["test_files"]
	proof_files = [Path(args.proof_file)]

	if args.use_fast_check :
		env = FastProofEnv(proof_files, args.prelude, args.wandb_log, state_type = args.state_type, max_proof_len = args.max_proof_len, num_check_engines = args.max_attempts, info_on_check = True)
	else :
		env = ProofEnv(proof_files, args.prelude, args.wandb_log, state_type = args.state_type,  max_proof_len = args.max_proof_len, info_on_check=True)
	predictor = loadPredictorByFile(args.weightsfile)
	
	memory = Memory_max()
	s,_ = env.reset()
	device = "cuda"
	agent_model = Agent_model( env.language_model.n_chars, 500,1, device).to(device)
	loss_object = nn.BCELoss()
	optimizer = optim.SGD(agent_model.parameters(),lr = 0.01)


	

	
	T = 0
	episode_r = 0
	curr_epsilon = 0.01
	epsilon_increment = 1.15/T_max
	perf_queue = deque(maxlen=20)

	state_rewards = []
	states_in_this_proof = []
	state_names = []
	agent_vval = []
	live_rewards = []
	a = time.time()
	time_outside_env = []
	time_for_select_action = []
	tactics_used = []
	while T <= T_max :
		print(T)
		# a = get_epsilon_greedy( agent_model.get_qvals(torch.tensor(s)), curr_epsilon )
		# print(s)

		print("Starting evaluation of 1 pass of the main timestep T loop")
		# print(env.coq.proof_context)
		curr_state_name = env.coq.proof_context.fg_goals[0].goal.strip()
		print("State Name : ", curr_state_name)
		c = time.time()
		prediction,_,agent_info = select_action(agent_model, memory, env, predictor, curr_epsilon, tactics_used, device = device,replace_vval=args.replace_vval)
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
			print("Current Proof time", env.proof_time)
			print("Current calculated proof time", env.proof_time_calculated) 
			print("Current calculated proof time outside", sum(time_outside_env)) 
			print("Current total calculated proof time", sum(time_outside_env) +  env.proof_time_calculated)
			print("Time for select action in this proof", sum(time_for_select_action))
			time_outside_env = []
			time_for_select_action = []
			total_curr_rewards = 0
			true_vvals = []
			tactics_used = []
			for i in range(len(state_rewards) - 1, -1, -1) :
				total_curr_rewards = state_rewards[i] + total_curr_rewards*gamma
				memory.add(states_in_this_proof[i], state_names[i], total_curr_rewards)
				if args.wandb_log :
					wandb.log({"Total_curr_reward" : total_curr_rewards })
				# agent_model.train(state_names[i],total_curr_rewards)
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


		if memory.num_items >= args.batch_size and T% args.update_every == 0:
			a = time.time()
			for _ in range(args.nepochs) :
				loss = train_step( agent_model ,optimizer, loss_object,memory,batch_size = args.batch_size, device = device)
				if args.wandb_log :
					wandb.log({"Loss":loss})
			b = time.time()
			print("Time to train", b-a)

		if curr_epsilon < 1 :
			curr_epsilon += epsilon_increment
		
		
		if T%1000 == 0 :
			torch.save(agent_model,"data/nn_mixval_agent.model")
			with open("data/memory_mix_maxval.pkl","wb") as f:
				pickle.dump(memory, f)
		

		s = s_next
		T += 1
	
	print("Finished Training")




if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("--proof_file", type=str)
	parser.add_argument("--max-tuples", default=None, type=int)
	parser.add_argument("--tokenizer",
							choices=list(tokenizers.keys()), type=str,
							default=list(tokenizers.keys())[0])
	parser.add_argument("--num-keywords", default=100, type=int)
	parser.add_argument("--lineend", action="store_true")
	parser.add_argument('--wandb_log', action= 'store_true')
	parser.add_argument('--weightsfile', default = "data/polyarg-weights.dat", type=str)
	parser.add_argument("--max_term_length", type=int, default=256)
	parser.add_argument("--max_attempts", type=int, default=7)
	parser.add_argument("--max_proof_len", type=int, default=30)
	parser.add_argument('--prelude', default=".")
	parser.add_argument('--run_name', type=str, default=None)
	parser.add_argument('--nepochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=50)
	parser.add_argument('--update_every', type=int, default=100)
	parser.add_argument('--use_fast_check',  action= 'store_true')
	parser.add_argument('--lambda', dest='l', type=float, default = 0.5)
	parser.add_argument('--lr', type=float, default = 0.01)
	parser.add_argument('--replace_vval', action="store_true", default=False)
	parser.add_argument('--state_type',type=str,default='index')


	args = parser.parse_args()

	if args.wandb_log :
		if args.run_name :
			wandb.init(project="Proverbot", entity="dylanzhang", name=args.run_name)
		else :
			wandb.init(project="Proverbot", entity="dylanzhang")


	total_num_steps = 10000
	gamma = 1
	FINAL_REWARD = 1
	MAX_STATE_SYMBOLS = 40
	NN_VLearning(T_max= total_num_steps, gamma = gamma, args = args)

		