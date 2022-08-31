import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time, argparse, random
from pathlib_revised import Path2
import dataloader
import coq_serapy as serapi_instance
from coq_serapy import load_commands, kill_comments
from search_file import completed_proof, loadPredictorByFile, truncate_tactic_context, FullContext
from train_encoder import EncoderRNN, DecoderRNN, Lang, tensorFromSentence, EOS_token
from tokenizer import get_symbols, get_words,tokenizers
import pickle
import gym
import fasttext
import os, sys


# scraped_tactics = dataloader.scraped_tactics_from_file(str(args.scrape_file), args.max_tuples)

# #print(type(scraped_tactics), len(scraped_tactics))

# for tactics in scraped_tactics :
#     #print("Tactic", tactics.tactic.strip())
#     #print("Relavant Lemmas : ", tactics.relevant_lemmas)
#     #print("previous_tactics : ", tactics.prev_tactics)
#     #print("Proof context : ")
#     #print("    Foreground goals :" )
#     for i in tactics.context.fg_goals :
#         #print("           Hypothesis : ", i.hypotheses)
#         #print("           Goals : ", i.goal)
#     #print("    Background goals :" )
#     for i in tactics.context.bg_goals :
#         #print("           Hypothesis : ", i.hypotheses)
#         #print("           Goals : ", i.goal)
#     #print("    Shelved goals :" )
#     for i in tactics.context.shelved_goals :
#         #print("           Hypothesis : ", i.hypotheses)
#         #print("           Goals : ", i.goal)
#     #print("    Given up goals :" )
#     for i in tactics.context.given_up_goals :
#         #print("           Hypothesis : ", i.hypotheses)
#         #print("           Goals : ", i.goal)       
#     #print("The tactic : ", tactics.tactic)
#     #print()
#     #print()


class ProofEnv(gym.Env) :
	def __init__(self, proof_file, prelude, time_per_command=100):
		self.action_space = None
		self.observation_space = None
		self.prelude= prelude
		self.max_num_proofs = 5


		self.proof_file = proof_file
		self.commands = load_commands(proof_file, progress_bar=True)
		self.proof_line_num = 0
		self.n_proofs = 0

		self.coq_running = False
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.time_per_command= time_per_command
		self.load_state_model()
		self.load_language_model()


	def goto_next_proof(self):
		# self.num_commands = 0
		while self.proof_line_num < len(self.commands) :
			#print("==-== ",self.commands[self.proof_line_num].lstrip().rstrip(), "==-=====-", self.proof_line_num)
			if self.commands[self.proof_line_num].lstrip().rstrip() == "Proof." :
				print("Found Proof : ", kill_comments(self.commands[self.proof_line_num - 1].lstrip().rstrip()))
				self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
				self.proof_line_num += 1
				break

			self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
			self.proof_line_num += 1

		
		if self.proof_line_num >= len(self.commands) or self.n_proofs > self.max_num_proofs:
			#print("File done")
			self.reset_to_start_of_file()
			return self.goto_next_proof()

		return self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip())

	def clear_proof_context(self) :
		while self.coq.proof_context != None :
			self.coq.cancel_last()

	def solve_curr_from_file(self) :
		self.clear_proof_context()
		while self.commands[self.proof_line_num].lstrip().rstrip() != "Proof.":
			self.proof_line_num -= 1

		self.proof_line_num -= 1
		while self.coq.proof_context != None :
			self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
			self.proof_line_num += 1
		
		#print(self.commands[self.proof_line_num-1])
		#print(self.commands[self.proof_line_num])
		assert self.commands[self.proof_line_num] != "Qed."
		


	def reset_to_start_of_file(self) :
		if self.coq_running :
			self.coq.kill()
		self.coq = serapi_instance.SerapiInstance(['sertop', '--implicit'],"Globalenvs", prelude = self.prelude)
		self.coq.verbose = 0
		self.coq.quiet = True
		self.proof_line_num = 0
		self.scraped_tactic_index = 0
		self.n_proofs = 0
		self.coq_running = True


	def load_state_model(self) :
		self.state_model =  torch.load("data/encoder_symbols.model", map_location=torch.device(self.device))
	def load_language_model(self) :
		with open("data/encoder_language_symbols.pkl","rb") as f:
			self.language_model = pickle.load(f)


	def get_state_vector_from_text(self,state_text) :
		state_sentence = get_symbols(state_text)
		print("State text :", state_text)
		state_tensor = tensorFromSentence(self.language_model,state_sentence,self.device, ignore_missing = True)
		with torch.no_grad() :
			state_model_hidden = self.state_model.initHidden(self.device)
			state_model_cell = self.state_model.initCell(self.device)
			input_length = state_tensor.size(0)
			for ei in range(input_length):
				_, state_model_hidden,state_model_cell = self.state_model(state_tensor[ei], state_model_hidden,state_model_cell)

			
			state= state_model_hidden
		state = state.cpu().detach().numpy().flatten()
		# state = np.append(state,[self.num_commands]).astype("float") 
		return state


	def step(self, action):
		done = True
		action_to_be_taken = self.commands[self.proof_line_num].lstrip().rstrip()


		print("This : ",action_to_be_taken,self.proof_line_num)
		if action != action_to_be_taken :
			r = -1
		else :
			r = 1

		try:
			self.coq.run_stmt(action_to_be_taken, timeout= self.time_per_command)
			self.proof_line_num += 1
		except (serapi_instance.TimeoutError, serapi_instance.ParseError,
				serapi_instance.CoqExn, serapi_instance.OverflowError,
				serapi_instance.ParseError,
				RecursionError,
				serapi_instance.UnrecognizedError) as e:
			#print("One of known errors", e)
			r = -1
			quit()
		except serapi_instance.CoqAnomaly:
			#print("Coq Anomaly")
			self.kill()
			quit()
		except :
			#print("Some error")
			self.kill()
			quit()
		else :
			
			
			if self.coq.proof_context == None :
				print("No context")
				quit()

			print(completed_proof(self.coq))
			while not (completed_proof(self.coq)) and len(self.coq.proof_context.fg_goals) == 0 :
				print("Running - ",self.commands[self.proof_line_num].lstrip().rstrip())
				self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
				self.proof_line_num += 1
				print(completed_proof(self.coq))
			

			if completed_proof(self.coq) :
				self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
				self.proof_line_num += 1
				print("Current proof fin")
				self.n_proofs += 1
				self.goto_next_proof()
				done = True
			
		next_state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal)
		return next_state, r, done, {'next_true_action' : self.commands[self.proof_line_num].lstrip().rstrip()}


	def reset(self):
		self.reset_to_start_of_file()
		state = self.goto_next_proof()
		return state, {'next_true_action' : self.commands[self.proof_line_num].lstrip().rstrip()}


class Memory :
	def __init__(self) :
		self.mem = []
		self.num_items = 0
				
	def add(self,s,a,r,sn) :
		self.mem.append([s,a,r,sn])
		self.num_items += 1
	
	def clear(self) :
		self.mem = []
		self.num_items = 0
	
	def sample_random_minibatch(self,n) :
		mem_batch = random.sample(self.mem,n)
		return mem_batch

class Agent_model(nn.Module) :
	def __init__(self,input_size,output_size) :
		super(Agent_model,self).__init__()
		self.lin1 = nn.Linear(input_size,10)
		self.lin2 = nn.Linear(10,output_size)
		self.relu = nn.ReLU()
		self.apply(self.init_weights)



	def forward(self,x) :
		x = self.relu(self.lin1(x))
		x = self.lin2(x)
		return x

	def init_weights(self,m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.01)


def get_epsilon_greedy(qvals,epsilon):
	coin = np.random.rand()
	if coin < epsilon :
		return np.argmax(qvals)
	else :
		return np.random.randint(low = 0, high=len(qvals))



def get_qvals(state_actions, agent_model, grad_tensor = False):
	final_state_tensor = torch.tensor(np.array(state_actions))
	if not grad_tensor :
		with torch.no_grad() :
				qvals = agent_model(final_state_tensor).detach().cpu().numpy()
	else :
		qvals = agent_model(final_state_tensor)
	return qvals

def get_state_action(s, tactic_space_model, env, predictor) :
	relevant_lemmas = env.coq.local_lemmas[:-1]
	full_context_before = FullContext(relevant_lemmas, env.coq.prev_tactics,  env.coq.proof_context)
	predictions = predictor.predictKTactics(
		truncate_tactic_context(full_context_before.as_tcontext(),
								args.max_term_length), args.max_attempts)
	

	state_action = []
	for prediction_idx, prediction in enumerate(predictions):
		tactic_vec = tactic_space_model.get_word_vector(prediction.prediction)
		final_state_vec = np.concatenate((s,tactic_vec))
		state_action.append(final_state_vec)

	return state_action, predictions


def select_action(s, agent_model, true_action, tactic_space_model, env, predictor, epsilon) :

	state_action, predictions = get_state_action(s, tactic_space_model, env, predictor)
	state_true_action = np.concatenate((s,tactic_space_model.get_word_vector(true_action)))
	state_action.append(state_true_action)
	qvals = get_qvals(state_action, agent_model)
	action_idx = get_epsilon_greedy([i.item() for i in qvals],epsilon)

	if action_idx == len(state_action) - 1 :
		return true_action, state_action

	return predictions[action_idx].prediction,state_action

	


def Bandit(T_max, batch_size, args) :
	
	env = ProofEnv(args.proof_file.path, args.prelude)
	predictor = loadPredictorByFile(args.weightsfile)
	tactic_space_model = fasttext.train_unsupervised(args.proof_file.path, model='cbow', lr = 0.1,epoch = 10)
	memory = Memory()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	s,info = env.reset()
	

	num_states = s.shape[0]
	num_actions = 1
	tactic_space_dim = tactic_space_model.get_word_vector("auto.").shape[0]

	agent_model = Agent_model(num_states + tactic_space_dim,num_actions).to(device)
	optimizer = optim.Adam(agent_model.parameters(), lr = 0.001) #Try Adam
	optimizer.zero_grad()
	
	T = 0
	episode_r = 0
	curr_epsilon = 0.2
	update_every = batch_size

	while T <= T_max :
		# a = get_epsilon_greedy( agent_model.get_qvals(torch.tensor(s)), curr_epsilon )
		# #print(s)
		true_action = info["next_true_action"]
		prediction, state_action = select_action(s, agent_model, true_action, tactic_space_model, env, predictor, curr_epsilon)
		#print("Selected action :" +  prediction  + "; Take step <press Enter>?")
		s_next,episode_r, done, info = env.step(prediction)
		#print("Step taken")
		memory.add(state_action,prediction,episode_r, s_next)
		
		s = s_next
		T += 1

		if args.wandb_log :                
			wandb.log({"True Rewards" : episode_r})
			wandb.log({"Exploration Factor:":curr_epsilon})


		if T <= batch_size or T%update_every != 0 :
			continue
		
		
		mem_batch = memory.sample_random_minibatch(batch_size)
		y = []
		curr_q = []
		for state_action, action, reward, nxt_state in mem_batch :    
			curr_q.append(get_qvals(state_action,agent_model, grad_tensor = True))
			y.append(reward)

				
	
		y = torch.tensor(y).to(device) #dont forget to send to device
		curr_q = torch.stack(curr_q).to(device) #dont forget to send to device
		loss =  torch.sum((y - curr_q)**2)

		
		loss.backward()
		s = s_next

		optimizer.step()
		optimizer.zero_grad()
		if curr_epsilon < 1 :
			curr_epsilon += 0.01

	

	

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
	parser.add_argument("--max_attempts", type=int, default=10)
	parser.add_argument('--prelude', default=".")


	args = parser.parse_args()

	if args.wandb_log :
		wandb.init(project="Proverbot", entity="avarghese")


	total_num_steps = 10000
	gamma = 1
	batch_size = 200
	

	Bandit(T_max= total_num_steps, args = args, batch_size = batch_size)

		