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
from coq_serapy import load_commands, kill_comments
from search_file import completed_proof, loadPredictorByFile, truncate_tactic_context, FullContext
from train_encoder import EncoderRNN, DecoderRNN, Lang, tensorFromSentence, EOS_token
from tokenizer import get_symbols, get_words,tokenizers
import pickle
import gym
import fasttext
import os, sys

np.random.seed(1)

# scraped_tactics = dataloader.scraped_tactics_from_file(str(args.scrape_file), args.max_tuples)

# print(type(scraped_tactics), len(scraped_tactics))

# for tactics in scraped_tactics :
#     print("Tactic", tactics.tactic.strip())
#     print("Relavant Lemmas : ", tactics.relevant_lemmas)
#     print("previous_tactics : ", tactics.prev_tactics)
#     print("Proof context : ")
#     print("    Foreground goals :" )
#     for i in tactics.context.fg_goals :
#         print("           Hypothesis : ", i.hypotheses)
#         print("           Goals : ", i.goal)
#     print("    Background goals :" )
#     for i in tactics.context.bg_goals :
#         print("           Hypothesis : ", i.hypotheses)
#         print("           Goals : ", i.goal)
#     print("    Shelved goals :" )
#     for i in tactics.context.shelved_goals :
#         print("           Hypothesis : ", i.hypotheses)
#         print("           Goals : ", i.goal)
#     print("    Given up goals :" )
#     for i in tactics.context.given_up_goals :
#         print("           Hypothesis : ", i.hypotheses)
#         print("           Goals : ", i.goal)       
#     print("The tactic : ", tactics.tactic)
#     print()
#     print()


class ProofEnv(gym.Env) :
	def __init__(self, proof_file, prelude, time_per_command=100):
		self.action_space = None
		self.observation_space = None
		self.prelude= prelude

		self.proof_file = proof_file
		self.commands = load_commands(proof_file, progress_bar=True)
		self.proof_line_num = 0

		self.coq_running = False
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.in_agent_proof_mode= False
		self.in_file_proof_mode= True  #Likely wont be using but just for sake of testing

		self.time_per_command= time_per_command
		self.load_state_model()
		self.load_language_model()
		self.max_attempts = 15
		self.test_file = open("output/output_test_file.txt","w")
		self.curr_proof_tactics = []
		self.max_num_proofs = 15
		self.num_proofs = 0


	def goto_next_proof(self):
		print("Going to next Proof")
		assert self.in_agent_proof_mode == False
		assert self.in_file_proof_mode == True

		self.num_commands = 0
		print("Before : ",self.coq.proof_context)
		while self.proof_line_num < len(self.commands) :# and  self.num_proofs <= self.max_num_proofs :
			not_function = kill_comments(self.commands[self.proof_line_num - 1]).lstrip().rstrip().split()[0].lower() != "function"
			if self.commands[self.proof_line_num].lstrip().rstrip() == "Proof." and not_function:
				print(self.commands[self.proof_line_num - 1].lstrip().rstrip().split()[0].lower())
				print("Found Proof : ", kill_comments(self.commands[self.proof_line_num - 1].lstrip().rstrip()))
				self.curr_proof_tactics = [ "\n", self.commands[self.proof_line_num - 1].lstrip().rstrip(), "Proof."]
				
				self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
				self.proof_line_num += 1
				self.in_agent_proof_mode= True
				self.in_file_proof_mode = False
				self.num_proofs  += 1
				break
			
			
			self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
			self.proof_line_num += 1
		
		print("After : ",self.coq.proof_context)
		if self.proof_line_num >= len(self.commands) : #or self.num_proofs >= self.max_num_proofs :
			print("File Finished")
			self.test_file.write("\n ----------------------------------------------------- \n")
			self.reset_to_start_of_file()
			
			return self.goto_next_proof()

		return self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip())


	def navigate_file_end_of_current_proof(self) :
		assert self.in_agent_proof_mode == False
		assert self.in_file_proof_mode == False
		while self.proof_line_num < len(self.commands)  and self.commands[self.proof_line_num].lstrip().rstrip().lower() != "qed." :
			self.proof_line_num += 1
		self.proof_line_num += 1
	
	def clear_proof_context(self) :
		while self.coq.proof_context != None :
			self.coq.cancel_last()

	def solve_curr_from_file(self) :
		print("Starting to solve from current file")
		print(self.proof_line_num)
		self.clear_proof_context()
		print( self.commands[self.proof_line_num].lstrip().rstrip() )
		while self.commands[self.proof_line_num].lstrip().rstrip() != "Proof.":
			self.proof_line_num -= 1
			print( self.commands[self.proof_line_num].lstrip().rstrip() )

		print(self.proof_line_num)
		self.proof_line_num -= 1
		print(self.commands[self.proof_line_num].lstrip().rstrip())
		self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
		self.proof_line_num += 1
		while self.coq.proof_context != None :
			self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
			self.proof_line_num += 1
		

		assert self.commands[self.proof_line_num] != "Qed."

		print("Done solvinf from File", self.proof_line_num)

		


	def reset_to_start_of_file(self) :
		if self.coq_running :
			self.coq.kill()
		self.coq = serapi_instance.SerapiInstance(['sertop', '--implicit'],None, prelude = self.prelude)
		self.coq.verbose = 3
		self.coq.quiet = True
		self.proof_line_num = 0
		self.coq_running = True
		self.num_proofs = 0
		self.in_agent_proof_mode= False
		self.in_file_proof_mode= True 


	def load_state_model(self) :
		self.state_model =  torch.load("data/encoder_symbols.model", map_location=torch.device(self.device))
	def load_language_model(self) :
		with open("data/encoder_language_symbols.pkl","rb") as f:
			self.language_model = pickle.load(f)


	def get_state_vector_from_text(self,state_text) :
		state_sentence = get_symbols(state_text)
		print(state_text)#,state_sentence)
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
		done = False
		# prediction = self.get_pred(action)
		prediction = action
		
		
		try:
			self.coq.run_stmt(prediction, timeout= self.time_per_command)

		except (serapi_instance.TimeoutError, serapi_instance.ParseError,
				serapi_instance.CoqExn, serapi_instance.OverflowError,
				serapi_instance.ParseError,
				RecursionError,
				serapi_instance.UnrecognizedError) as e:
			print("One of known errors", e)
			r = -1
		except serapi_instance.CoqAnomaly:
			print("Coq Anomaly")
			self.kill()
			quit()
		except :
			print("Some error")
			self.kill()
			quit()
		else :
			r = 0.1
			self.num_commands += 1
			self.curr_proof_tactics.append(prediction)

			if len(self.coq.proof_context.fg_goals) > 1 :
				print(self.coq.proof_context.fg_goals,self.coq.proof_context.bg_goals)
				print("Running {")
				self.coq.run_stmt( "{", timeout= self.time_per_command)
			if len(self.coq.proof_context.fg_goals) == 0 and not  completed_proof(self.coq):
				print(self.coq.proof_context.fg_goals,self.coq.proof_context.bg_goals)
				print("Running }")
				print(completed_proof(self.coq))
				self.coq.run_stmt( "}", timeout= self.time_per_command)
				print(completed_proof(self.coq))
			
			if completed_proof(self.coq) :
				self.coq.run_stmt( "Qed.", timeout= self.time_per_command)
				r = 40
				print("Current proof fin with Good rewards")
				self.test_file.write("\n".join(self.curr_proof_tactics) )
				self.test_file.flush()

				self.in_agent_proof_mode= False
				self.in_file_proof_mode = False
				self.navigate_file_end_of_current_proof()
				self.in_agent_proof_mode= False
				self.in_file_proof_mode = True
				self.goto_next_proof()
				done = True
			if self.coq.proof_context == None :
				print("No context")
				quit()
				self.goto_next_proof()
				print("Went to next proof")
			
			if self.num_commands > self.max_attempts :
				r = -20
				self.in_agent_proof_mode= False
				self.in_file_proof_mode = True
				print("Too many attempts, aborting training on current proof")
				self.coq.run_stmt("Abort.", timeout= self.time_per_command)
				
				self.solve_curr_from_file()
				self.goto_next_proof()
				done = True   
		
		


		next_state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal)
		return next_state, r, done, {}


	def reset(self):
		self.reset_to_start_of_file()
		state = self.goto_next_proof()
		return state


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


def get_epsilon_greedy(qvals,epsilon):
	coin = np.random.rand()
	if coin < epsilon :
		return np.argmax(qvals)
	else :
		return np.random.randint(low = 0, high=len(qvals))



def get_qvals(state_actions, agent_model):
	qvals = agent_model.predict(state_actions)
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


def select_action(s, agent_model, tactic_space_model, env, predictor, epsilon) :

	state_action, predictions = get_state_action(s, tactic_space_model, env, predictor)
	qvals = get_qvals(state_action, agent_model)
	action_idx = get_epsilon_greedy([i.item() for i in qvals],epsilon)

	return predictions[action_idx],state_action[action_idx],{"qval" : qvals[action_idx]}

def select_random_action(s, tactic_space_model, env, predictor) :
	state_action, predictions = get_state_action(s, tactic_space_model, env, predictor)
	action_idx = np.random.choice(range(len(state_action)))
	return predictions[action_idx].prediction,state_action[action_idx]


def RF_QLearning(T_max, gamma, batch_size, args) :

	env = ProofEnv(args.proof_file.path, args.prelude)
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
		prediction, state_action = select_random_action(s, tactic_space_model, env, predictor)
		s_next,episode_r, done, info = env.step(prediction)
		temp_x.append(state_action)
		temp_y.append(episode_r)
	agent_model.fit(temp_x, temp_y)
	s = env.reset()
	print("Setup Done")


	
	T = 0
	episode_r = 0
	curr_epsilon = 0.2
	update_every = 20
	perf_queue = deque(maxlen=update_every)

	state_rewards = []
	states_in_this_proof = []
	agent_qval = []
	live_rewards = []
	while T <= T_max :
		# a = get_epsilon_greedy( agent_model.get_qvals(torch.tensor(s)), curr_epsilon )
		# print(s)
		
		prediction,state_action,agent_info = select_action(s, agent_model, tactic_space_model, env, predictor, curr_epsilon)
		print("Selected action :" +  prediction.prediction  + "; Take step <press Enter>?")
		s_next,episode_r, done, _ = env.step(prediction.prediction)
		print("Step taken")
		episode_r += prediction.certainty

		state_rewards.append(episode_r)
		states_in_this_proof.append(state_action)
		agent_qval.append(agent_info["qval"])
		live_rewards.append(episode_r)



		if done :
			total_curr_rewards = 0
			true_qvals = []
			for i in range(len(state_rewards) - 1, -1, -1) :
				total_curr_rewards = state_rewards[i] + total_curr_rewards*gamma
				memory.add(states_in_this_proof[i], total_curr_rewards)
				true_qvals.insert(0,total_curr_rewards)
			

			
			for i in range(len(state_rewards)):	
				if args.wandb_log :
					wandb.log({"Agent Surprise factor" : abs(agent_qval[i] - true_qvals[i]) })
					wandb.log({"Live Rewards": live_rewards[i]})
			
			perf_queue.append(total_curr_rewards)
			state_rewards = []
			states_in_this_proof = []
			agent_qval = []
			live_rewards = []
			if args.wandb_log :
				wandb.log({"Episode Total Rewards":total_curr_rewards})
				wandb.log({"Exploration Factor":curr_epsilon})
				wandb.log({"Gain" : sum(perf_queue)/len(perf_queue)})
				wandb.log({"T" : T})
			
			mem_batch = memory.sample_random_minibatch()
			state_actions, rewards = zip(*mem_batch)
			agent_model.fit(state_actions, rewards)
		
			if curr_epsilon < 1 :
				curr_epsilon += 0.005
				

		s = s_next
		T += 1
		
		# if T <= batch_size  or T%update_every != 0 :
		# 	continue
		
		
		
	

	

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
	parser.add_argument("--max_attempts", type=int, default=7)
	parser.add_argument('--prelude', default=".")
	parser.add_argument('--run_name', type=str, default=None)


	args = parser.parse_args()

	if args.wandb_log :
		if args.run_name :
			wandb.init(project="Proverbot", entity="avarghese", name=args.run_name)
		else :
			wandb.init(project="Proverbot", entity="avarghese")


	total_num_steps = 100000
	gamma = 1
	batch_size = 100
	

	RF_QLearning(T_max= total_num_steps, gamma = gamma, args = args, batch_size = batch_size)

		