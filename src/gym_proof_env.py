# Gym interface for coqproofenv

from collections import deque
import multiprocessing
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
from coq_serapy import load_commands, kill_comments, get_hyp_type, get_indexed_vars_dict, get_stem, split_tactic, contextSurjective, summarizeContext
from coq_serapy import ending_proof, possibly_starting_proof
from coq_serapy.contexts import truncate_tactic_context, FullContext
from search_file import loadPredictorByFile
from search_strategies import completed_proof
from train_encoder import EncoderRNN, DecoderRNN, Lang, indexesFromSentence, tensorFromSentence, EOS_token
from tokenizer import get_symbols, get_words,tokenizers
import pickle
import gym
import fasttext
import os, sys
from util import nostderr, unwrap, eprint, mybarfmt
from collections import defaultdict
from mpire import WorkerPool
from multiprocessing import Pool, Pipe, Process
import itertools
import io
import coq2vec
import gymnasium as gym

# 1. Currently do not support 'vectorization' of environments 
# 3. Observations need to be encoded to push into memory and train networks
# 4. 

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


class ProofEnv(gym.Env):
	def __init__(self, proof_files, prelude, wandb = False, time_per_command=100, max_proof_len = 50, write_solved_proofs = True, 
				state_type = "index", info_on_check = False,
				weightsfile=None):
		self.action_space = None
		self.observation_space = None
		self.prelude= prelude
		self.proof_files = proof_files
		self.proof_file_index = 0
		self.proof_line_num = 0
		self.wandb_log = wandb
		self.coq_running = False
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.in_agent_proof_mode= False
		self.in_file_proof_mode= True  #Likely wont be using but just for sake of testing
		# self.state_type = state_type #text, index, vector
		self.write_solved_proofs = write_solved_proofs
		self.info_on_check = info_on_check
		self.weightsfile = weightsfile
		self.max_attempts = max_attempts
		
		# TODO: see if we can put predictor in the envionment?
		self.time_per_command= time_per_command
		# TODO: REMOVE this 
		# if state_type == "vector" :
		# 	self.load_state_model()
		self.load_language_model()
		self.max_proof_len = max_proof_len
		self.test_file =  "output/output_test_file.txt"
		with open(self.test_file,"w") as f :
			f.write("")
		self.proof_lines_file = "output/output_proof_lines.txt"
		with open(self.proof_lines_file,"w") as f :
			f.write("")
		self.context_file =  "output/output_context_file.txt"
		with open(self.context_file,"w") as f :
			f.write("")
		
		# if "vector" in state_type :		
		# 	self.termvectorizer = coq2vec.CoqTermRNNVectorizer()
		# 	self.termvectorizer.load_weights("data/term2vec-weights-59.dat")
		# 	num_hyps_encoded = 4
		# 	self.obligationvectorizer = coq2vec.CoqContextVectorizer(self.termvectorizer, num_hyps_encoded)
		self.curr_proof_tactics = []
		self.max_num_proofs = 15
		self.num_proofs = 0
		# self.restrictions = defaultdict(lambda: [])
		self.load_list_tactic_classes()

	def test_file_write(self, write_str) :
		if self.write_solved_proofs :
			with open(self.test_file,"a") as f :
				f.write(write_str)
				f.flush()

	def context_file_write(self, write_str) :
		if self.write_solved_proofs :
			with open(self.context_file,"a") as f :
				f.write(write_str)
				f.flush()
	def prooflines_file_write(self, write_str) :
		if self.write_solved_proofs :
			with open(self.proof_lines_file,"a") as f :
				f.write(write_str)
				f.flush()

	def load_list_tactic_classes(self) :
		with open("tactics.txt", "r") as f :
			whole_file = f.read()
			self.list_of_tactic_classes = whole_file.split("\n")
			for i in range(len(self.list_of_tactic_classes)) :
				self.list_of_tactic_classes[i] = self.list_of_tactic_classes[i].strip().rstrip(".")

	def run_to_proof(self, proof_contains) :
		print("Running to proof",proof_contains)
		while self.proof_line_num < len(self.commands) :# and  self.num_proofs <= self.max_num_proofs :
			if proof_contains in self.commands[self.proof_line_num] :
				print("Found Proof : ", kill_comments(self.commands[self.proof_line_num].lstrip().rstrip()))
				self.curr_proof_tactics = [ "\n", "(" + str(self.num_proofs + 1) + ") ",  self.commands[self.proof_line_num - 1].lstrip().rstrip(), "Proof."]
				
				self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
				self.proof_line_num += 1
				self.in_agent_proof_mode= True
				self.in_file_proof_mode = False
				self.num_proofs  += 1
				self.proof_contexts_in_path.append(self.coq.proof_context)
				break
			else:	
				not_function = kill_comments(self.commands[self.proof_line_num - 1]).lstrip().rstrip().split()[0].lower() != "function"
				if self.commands[self.proof_line_num].lstrip().rstrip() == "Proof." and not_function:
					self.num_proofs  += 1
				self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
				self.proof_line_num += 1

	def goto_next_proof(self):
		print("Going to next Proof")
		assert self.in_agent_proof_mode == False
		assert self.in_file_proof_mode == True
		self.end_proof_time = time.time()
		self.num_commands = 0
		self.proof_contexts_in_path = []
		print("Before : ",self.coq.proof_context)
		while self.proof_line_num < len(self.commands) :# and  self.num_proofs <= self.max_num_proofs :
			not_function = kill_comments(self.commands[self.proof_line_num - 1]).lstrip().rstrip().split()[0].lower() != "function"
			if self.commands[self.proof_line_num].lstrip().rstrip() == "Proof." and not_function:
				print(self.commands[self.proof_line_num - 1].lstrip().rstrip().split()[0].lower())
				print("Found Proof : ", kill_comments(self.commands[self.proof_line_num - 1].lstrip().rstrip()))
				self.curr_proof_tactics = [ "\n", "(" + str(self.num_proofs + 1) + ") ",  self.commands[self.proof_line_num - 1].lstrip().rstrip(), "Proof."]
				
				self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
				self.proof_line_num += 1
				self.in_agent_proof_mode= True
				self.in_file_proof_mode = False
				self.num_proofs  += 1
				self.proof_contexts_in_path.append(self.coq.proof_context)
				break
			else :	
				self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
				self.proof_line_num += 1
		
		
		if self.proof_line_num >= len(self.commands) : #or self.num_proofs >= self.max_num_proofs :
			print("File Finished")
			# if self.use_test :
			self.test_file_write("\n ----------------------------------------------------- \n")
			self.reset_to_start_of_file()
			
			return self.goto_next_proof()

		print("Context After finding proof : ",self.coq.proof_context)
		self.proof_time = self.end_proof_time - self.start_proof_time
		self.start_proof_time = time.time()
		self.proof_time_calculated = sum(self.debug_time)
		self.debug_time = []
		return None #self.get_state_vector( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip())


	def navigate_file_end_of_current_proof(self) :
		# This is wrong - Not all proofs end with QED. Also make this section cleaner.
		print("Navigating file to the end of current proof without running")
		assert self.in_agent_proof_mode == False
		assert self.in_file_proof_mode == False
		while self.proof_line_num < len(self.commands)  and not ending_proof(self.commands[self.proof_line_num]) :
			print("Navigating ", self.commands[self.proof_line_num])
			self.proof_line_num += 1
		print("Navigating finished", self.commands[self.proof_line_num], ending_proof(self.commands[self.proof_line_num]))
		self.proof_line_num += 1
		# print("Navigated to :", self.commands[self.proof_line_num] )
	
	def clear_coq_proof_context(self) :
		while self.coq.proof_context != None :
			self.coq.cancel_last()

	def solve_curr_from_file(self) :
		raise Exception("Solving from File called. Don't solve from File.")
		print("Starting to solve from current file")
		print("Proof Line Num : ", self.proof_line_num )
		self.clear_coq_proof_context()
		print( "Currently on :", self.commands[self.proof_line_num].lstrip().rstrip() )
		print("Finding the Previous Proof. statement")
		while self.commands[self.proof_line_num].lstrip().rstrip() != "Proof.":
			self.proof_line_num -= 1
			print( self.commands[self.proof_line_num].lstrip().rstrip() )
		print("Context After finding Proof. statement : ",self.coq.proof_context, " << should be None")
		print(self.proof_line_num)
		self.proof_line_num -= 1
		print("Found the proof section for - ",self.commands[self.proof_line_num].lstrip().rstrip())
		self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
		print("Context After running the proof statement for the above: ",self.coq.proof_context)
		self.proof_line_num += 1
		while self.coq.proof_context != None :
			self.context_file_write("Running - "+ self.commands[self.proof_line_num].lstrip().rstrip() + "\n") 
			self.coq.run_stmt(self.commands[self.proof_line_num].lstrip().rstrip(), timeout= self.time_per_command)
			self.context_file_write( str(self.coq.proof_context) + '\n')
			self.proof_line_num += 1
	
		assert ending_proof(self.commands[self.proof_line_num - 1])

		print("Done solving from File", self.proof_line_num)


	def reset_to_start_of_file(self) :
		print("Reseting to Start of next file")
		if self.coq_running :
			self.coq.kill()
		self.coq = serapi_instance.SerapiInstance(['sertop', '--implicit'],None, prelude = self.prelude)
		self.coq.verbose = 3
		self.coq.quiet = True
		self.proof_line_num = 0
		self.coq_running = True
		self.num_proofs = 0
		self.num_proofs_solved = 0
		self.in_agent_proof_mode= False
		self.in_file_proof_mode= True 
		self.commands = load_commands(self.proof_files[self.proof_file_index], progress_bar=True)
		print("Starting File :", self.proof_files[self.proof_file_index])
		
		self.proof_file_index = (self.proof_file_index + 1) % len(self.proof_files)

	def load_state_model(self) :
		with open('data/encoder_symbols.model', 'rb') as f:
			buffer = io.BytesIO(f.read())
		self.state_model = torch.load(buffer,map_location=torch.device(self.device))
		# self.state_model =  torch.load("data/encoder_symbols.model", map_location=torch.device(self.device))


	def load_language_model(self) :
		with open("data/encoder_language_symbols.pkl","rb") as f:
			self.language_model = pickle.load(f)

	# def get_state_vector(self,proof_state) :
	# 	state_text = proof_state.fg_goals[0].goal.strip()
	# 	print("State Text : ",state_text)
	# 	if self.state_type == "goal_index" :
	# 		state_sentence = get_symbols(state_text)
	# 		indexes = indexesFromSentence(self.language_model, state_sentence, ignore_missing = True)
	# 		# indexes.append(EOS_token)
	# 		return  indexes
	# 	elif self.state_type == "goal_text" :
	# 		return state_text
	# 	elif self.state_type == "goal_vector" :
	# 		return self.termvectorizer.term_to_vector(state_text)
	# 	elif self.state_type == "obligation" :
	# 		return proof_state.fg_goals[0]
	# 	elif self.state_type == "obligation_index" :
	# 		goal_state_sentence = get_symbols( proof_state.fg_goals[0].goal.strip())
	# 		goal_indexes = indexesFromSentence(self.language_model, goal_state_sentence, ignore_missing = True)
	# 		all_hyp_indexes = []
	# 		for hyp in proof_state.fg_goals[0].hypotheses :
	# 			hyp_sentence =  get_symbols( hyp.strip())
	# 			hyp_indexes = indexesFromSentence(self.language_model, hyp_sentence, ignore_missing = True)
	# 			all_hyp_indexes.append(hyp_indexes)

	# 		all_hyp_indexes.append(indexesFromSentence(self.language_model, ":", ignore_missing = True))
	# 		return [goal_indexes, all_hyp_indexes]
	# 	elif self.state_type == "obligation_vector" :
	# 		return self.obligationvectorizer.obligation_to_vector(proof_state.fg_goals[0])
	# 	else :
	# 		raise ValueError("Invalid State type", self.state_type)

	def admit_and_skip_proof(self) :
		self.in_agent_proof_mode= False
		self.in_file_proof_mode = False
		print("Admitting current proof without solving")
		self.coq.run_stmt("Admitted.", timeout= self.time_per_command)
		self.curr_proof_tactics.append("Admitted.")
		if self.wandb_log :
			wandb.log({"Num command Attempts" : self.num_commands  })
		self.prooflines_file_write("\n".join(self.curr_proof_tactics))
		# self.solve_curr_from_file()
		self.navigate_file_end_of_current_proof()
		self.in_agent_proof_mode= False
		self.in_file_proof_mode = True
		self.goto_next_proof()
		done = True
		next_state = self.get_state_vector( self.coq.proof_context)
		r = 0#-1
		self.in_agent_proof_mode= True
		self.in_file_proof_mode = False
		info = {}
		info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
		return next_state, r, done, info



	def get_available_actions_with_next_state_vectors(self) :
		# print(len( env.coq.proof_context.fg_goals),  env.coq.proof_context.fg_goals)
		# print(completed_proof(env.coq))
		relevant_lemmas = self.coq.local_lemmas[:-1]
		print(self.coq.proof_context)
		full_context_before = FullContext(relevant_lemmas, self.coq.prev_tactics,  self.coq.proof_context)
		predictions = self.predictor.predictKTactics(
			truncate_tactic_context(full_context_before.as_tcontext(),
									self.max_term_length), self.max_attempts)
		next_states = []
		list_of_pred = []
		next_state_texts = []
		print("Available actions", [_.prediction for _ in predictions])
		for prediction_idx, prediction in enumerate(predictions):
			curr_pred = prediction.prediction.strip()
			state_vec = self.check_next_state(curr_pred)
			if len(state_vec) == 0 :
				continue
			else :
				list_of_pred.append( prediction )
				next_states.append(preprocess_state(state_vec))
	
		return next_states, list_of_pred, next_state_texts





	def is_context_fresh(self, curr_proof_context) :
		# print(len(self.proof_contexts_in_path))
		for context in self.proof_contexts_in_path :
			if contextSurjective(curr_proof_context, context) :
				return False
		return True

	def is_same_context(self,context1, context2) :
		# print("Context Surjectives")
		# print(contextSurjective(context1, context2))
		# print(contextSurjective(context2, context1))
		return contextSurjective(context1, context2) and contextSurjective(context2, context1)
	
	def is_tactics_repeating(self,context, cutoff = 4) :
		tactics_used = context.prev_tactics
		if tactics_used[-cutoff:].count(tactics_used[-1]) == cutoff :
			return True
		return False
	def check_next_state(self,prediction):
		info = {}
		print("Checking next state for action -", prediction)
		tactic_class,tactic_args = split_tactic(prediction.strip().rstrip("."))
		if tactic_class.lower().strip() == "exploit" :
			if self.info_on_check :
				return [],info
			else :
				return []
		next_state = []
		context_before = self.coq.proof_context
		a= time.time()
		try:
			
			self.coq.run_stmt(prediction, timeout= self.time_per_command)
			
		except (serapi_instance.TimeoutError, serapi_instance.ParseError,
				serapi_instance.CoqExn, serapi_instance.OverflowError,
				serapi_instance.ParseError,
				RecursionError,
				serapi_instance.UnrecognizedError) as e:
			print("One of known errors", e)
		except serapi_instance.CoqAnomaly:
			print("Coq Anomaly")
			self.kill()
			quit()
		except :
			print("Some error")
			self.kill()
			quit()
		else :
			b = time.time()
			print("Time for running the above command", b-a)
			num_brackets_run = 0
			while len(unwrap(self.coq.proof_context).fg_goals) == 0 and not completed_proof(self.coq):
				if len(unwrap(self.coq.proof_context).shelved_goals) > 0:
					print("Running Unshelve.")
					self.coq.run_stmt("Unshelve.", timeout= self.time_per_command)
					num_brackets_run += 1
					continue
				print("Running }")
				self.coq.run_stmt("}", timeout= self.time_per_command)
				num_brackets_run += 1

			if len(self.coq.proof_context.fg_goals) > 1 :
				print("Context before running open brace :",self.coq.proof_context)
				print("Running {")
				self.coq.run_stmt( "{", timeout= self.time_per_command)
				print("Context after running open brace :",self.coq.proof_context)
				num_brackets_run += 1

			
			if completed_proof(self.coq) :
				for _ in range(num_brackets_run) :
					self.coq.cancel_last()
				self.coq.cancel_last()
				print("QED on this action. Cancelled - ",prediction)
				if self.info_on_check :
					info["state_text"] = "fin"
					return "fin",info
				else :
					return "fin"

			if self.coq.proof_context == None :
				print("Something is wrong. Lost context")
				quit()

			if self.is_context_fresh(self.coq.proof_context) :
				next_state_name =  self.coq.proof_context.fg_goals[0].goal
				next_state = self.get_state_vector(self.coq.proof_context)
				info["state_text"] = next_state_name.strip()
				print("Context is fresh for this actions")
			else :
				print("Context is not fresh for this action")
				next_state = []
			
			if num_brackets_run > 0 :
				print("Cancelling", num_brackets_run, "Brackets")
				for _ in range(num_brackets_run) :
					self.coq.cancel_last()

			context_mid = self.coq.proof_context
			self.coq.cancel_last()
			print("Cancelled - ",prediction)
		
		context_after = self.coq.proof_context
		
		assert self.is_same_context(context_before,context_after)
		# except :
		# 	print("History is -> {, eauto, cancel")
		# 	for obligation in context_after.all_goals :
		# 		print(obligation.goal)
			
		# 	print("History is -> {, eauto")
		# 	for obligation in context_mid.all_goals :
		# 		print(obligation.goal)
				
		# 	print("History is -> { ")
		# 	for obligation in context_before.all_goals :
		# 		print(obligation.goal)
		# 	assert self.is_same_context(context_before,context_after)

		if self.info_on_check :
			return next_state,info
		else :
			return next_state


	def step(self, action=None):
		"""
			Run one timestep of the environment's dynamics using the agent actions.
				When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
				reset this environment's state for the next episode.
				.. versionchanged:: 0.26
					The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
					to users when the environment had terminated or truncated which is critical for reinforcement learning
					bootstrapping algorithms.
				Args:
					action (ActType): an action provided by the agent to update the environment state.
				Returns:
					observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
						An example is a numpy array containing the positions and velocities of the pole in CartPole.
					reward (SupportsFloat): The reward as a result of taking the action.
					terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
						which can be positive or negative. An example is reaching the goal state or moving into the lava from
						the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
					truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
						Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
						Can be used to end the episode prematurely before a terminal state is reached.
						If true, the user needs to call :meth:`reset`.
					info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
						This might, for instance, contain: metrics that describe the agent's performance state, variables that are
						hidden from observations, or individual reward terms that are combined to produce the total reward.
						In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
						however this is deprecated in favour of returning terminated and truncated variables.
					done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
						return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
						A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
						a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
		"""
		if action == None:
			s_next,episode_r, done, info = self.admit_and_skip_proof()
			return s_next,episode_r, done, info
		done = False
		# prediction = self.get_pred(action)
		prediction = action
		info = {}
		eprint("Taking step -", action)
		a= time.time()
		try:
			self.coq.run_stmt(prediction, timeout= self.time_per_command)
		except (serapi_instance.TimeoutError, serapi_instance.ParseError,
				serapi_instance.CoqExn, serapi_instance.OverflowError,
				serapi_instance.ParseError,
				RecursionError,
				serapi_instance.UnrecognizedError) as e:
			print("One of known errors", e)
			r = 0

		except serapi_instance.CoqAnomaly:
			print("Coq Anomaly")
			self.kill()
			quit()
		except :
			print("Some error")
			self.kill()
			quit()
		else :
			b = time.time()
			self.debug_time.append(b-a)
			print("Time for running the above command", b-a)
			r = 0 #No rewards for progress
			self.curr_proof_tactics.append(prediction)
			a = time.time()
			while len(unwrap(self.coq.proof_context).fg_goals) == 0 and not completed_proof(self.coq):
				if len(unwrap(self.coq.proof_context).shelved_goals) > 0:
					print("Running Unshelve.")
					self.coq.run_stmt("Unshelve.", timeout= self.time_per_command)
					continue
				print("Running }")
				self.coq.run_stmt("}", timeout= self.time_per_command)
				self.curr_proof_tactics.append("}")
			b = time.time()
			self.debug_time.append(b-a)
			print("Time for the first while loop", b-a)

			a = time.time()
			if len(self.coq.proof_context.fg_goals) > 1 :
				print(self.coq.proof_context.fg_goals,self.coq.proof_context.bg_goals)
				print("Running {")
				self.coq.run_stmt( "{", timeout= self.time_per_command)
				self.curr_proof_tactics.append("{")
			b = time.time()
			self.debug_time.append(b-a)
			print("Time taken for opening brackets", b-a)

			a= time.time()
			if completed_proof(self.coq) :
				if self.wandb_log :
					wandb.log({"Num command Attempts" : self.num_commands  })
				b = time.time()
				self.debug_time.append(b-a)
				print("Time taken to check completed proof", b - a)
				self.coq.run_stmt( "Qed.", timeout= self.time_per_command)
				self.curr_proof_tactics.append("Qed.")
				r = 1
				print("Current proof fin with Good rewards")
				self.test_file_write("\n".join(self.curr_proof_tactics) )
				self.prooflines_file_write("\n".join(self.curr_proof_tactics))
				self.num_proofs_solved += 1
				self.in_agent_proof_mode= False
				self.in_file_proof_mode = False
				a = time.time()
				self.navigate_file_end_of_current_proof()
				b = time.time()
				self.debug_time.append(b-a)
				print("Time taken to naviagate file to the end of current proof", b -a)
				self.in_agent_proof_mode= False
				self.in_file_proof_mode = True
				a = time.time()
				self.goto_next_proof()
				b = time.time()
				self.debug_time.append(b-a)
				print("Time taken to run goto_next_proof function", b-a)
				done = True
				next_state = self.get_state_vector( self.coq.proof_context )
				info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
				return next_state, r, done, info
			b = time.time()
			self.debug_time.append(b-a)
			print("Time taken to check completed proof", b - a)

			if self.coq.proof_context == None :
				print("No context")
				quit()
			
			self.num_commands += 1
			a = time.time()
			assert self.is_context_fresh(self.coq.proof_context)
			b = time.time()
			self.debug_time.append(b-a)
			print("Time taken to check if context fresh", b - a)
			self.proof_contexts_in_path.append(self.coq.proof_context)


		if self.num_commands > self.max_proof_len :
			# r = -1 # -5
			# self.in_agent_proof_mode= False
			# self.in_file_proof_mode = True
			# print("Too many attempts, admitting and skipping current proof")
			# self.coq.run_stmt("Admitted.", timeout= self.time_per_command)
			# if self.wandb_log :
			# 	wandb.log({"Num command Attempts" : self.num_commands  })
			
			# # self.solve_curr_from_file()
			# self.navigate_file_end_of_current_proof()
			# self.goto_next_proof()
			# self.in_agent_proof_mode= True
			# self.in_file_proof_mode = False
			# done = True 
			a = time.time()
			result = self.admit_and_skip_proof()
			b = time.time()
			self.debug_time.append(b-a)
			print("Time taken to run admit and skip proof", b-a)
			return result
		# next_state = self.get_state_vector( self.coq.proof_context )
		next_state = self.coq.proof_context
		
		# if self.info_on_check :
		# return next_state,info
		# else :
		# return next_state
		info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
		return next_state, r, done, info  #next_obs, rewards, dones, infos


	def reset(self):
		self.reset_to_start_of_file()
		self.start_proof_time = 0
		self.debug_time = []
		self.goto_next_proof()
		print("Proof context after reset and next file start: ", self.coq.proof_context)
		state = self.get_state_vector( self.coq.proof_context )
		info = {}
		info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
		print("Reset done")
		return (state,info)

def child_process(pid, critical, pipe) :
	import sys
	# import io
	open("output/results/subprocess_pid%d_out.txt"%pid, 'w').close()
	open("output/errors/subprocess_pid%d_error.txt"%pid, 'w').close()
	sys.stdout = open("output/results/subprocess_pid%d_out.txt"%pid, 'a')#io.BytesIO()
	sys.stderr = open("output/errors/subprocess_pid%d_error.txt"%pid, 'a')#io.BytesIO()
	proof_file, prelude, time_per_command, state_type, info_on_check, max_proof_len = critical
	test_env = ProofEnv(proof_file, prelude, wandb = False, time_per_command = time_per_command, write_solved_proofs=False, max_proof_len=max_proof_len, state_type = state_type, info_on_check = info_on_check)
	print("child process created", pid)
	while True :
		if pipe.poll(1800) :
			func, args = pipe.recv()
		else :
			print("Terminating Child process", pid, "due to timeout")
			quit()
		print("This is inside a given child -", func, args)
		if func == 'reset' :
			result = test_env.reset()
		elif func == 'step' :
			result = test_env.step(args)
		elif func == 'check_next_state' :
			result = test_env.check_next_state(args)
			print("Results of Check next state inside a child - ",result)
		elif func == 'admit_and_skip_proof' :
			result = test_env.admit_and_skip_proof()
		elif func == 'run_to_proof' :
			result = test_env.run_to_proof(args)
		elif func == 'terminate' :
			break
		elif func == 'keepalive' :
			result = ""
		else :
			raise ValueError("Unknown function")

		pipe.send(result)
	
	return


class FastProofEnv(gym.Env):
	def __init__(self, proof_file, prelude, wandb = False, time_per_command=100, write_solved_proofs = True, state_type = "vector", 
					max_proof_len = 30, num_check_engines = 5, info_on_check = False):
		self.proof_file = proof_file
		self.prelude = prelude
		self.wandb = wandb
		self.time_per_command = time_per_command
		self.state_type = state_type
		self.num_check_engines = num_check_engines
		self.info_on_check = info_on_check
		self.max_proof_len = max_proof_len
		self.main_engine = ProofEnv(proof_file, prelude, wandb, time_per_command, write_solved_proofs = write_solved_proofs, max_proof_len=max_proof_len, state_type = state_type, info_on_check = False)
		self.language_model = self.main_engine.language_model
		self.predictor = self.main_engine.predictor
		self.create_pipes_and_children()

	@property
	def coq(self):
		return self.main_engine.coq
	@property
	def num_proofs(self) :
		return self.main_engine.num_proofs
	@property
	def num_proofs_solved(self) :
		return self.main_engine.num_proofs_solved
	@property
	def proof_time(self) :
		return self.main_engine.proof_time
	@property
	def proof_time_calculated(self) :
		return self.main_engine.proof_time_calculated
	@property
	def curr_proof_tactics(self):
		return self.main_engine.curr_proof_tactics
	def create_pipes_and_children(self) :
		self.server_end_pipes = []
		self.child_end_pipes = []
		for i in range(self.num_check_engines) :
			s,c = Pipe()
			self.server_end_pipes.append(s)
			self.child_end_pipes.append(c)
		print(self.num_check_engines)
		process_list = []
		context = multiprocessing.get_context('fork')
		for i in range(self.num_check_engines) :
			p = context.Process(target=child_process, args=(i,(self.proof_file, self.prelude, 
				self.time_per_command, self.state_type, self.info_on_check,  self.max_proof_len),self.child_end_pipes[i] ) )
			p.start()
			process_list.append(p)

		print("Exploratory Environments successfully running")		
		return
		

	def admit_and_skip_proof(self):
		print("Admitting and Skipping the current proof on all Test engines")
		for pipe in self.server_end_pipes :
			pipe.send( ["admit_and_skip_proof",None])
		for pipe in self.server_end_pipes :
			pipe.recv()
		print("Test engines sucessfully skipped proof")
		return self.main_engine.admit_and_skip_proof()

	def reset(self) :
		results = self.main_engine.reset()
		print("Reseting all Test Engines")
		for pipe in self.server_end_pipes :
			pipe.send(["reset",None])
		for pipe in self.server_end_pipes :
			print(pipe.recv())
		print("All Test Engines Reset")
		# quit()
		return results

	def step(self, action) :
		print("Stepping on all Test Engines")
		for pipe in self.server_end_pipes :
			pipe.send(["step",action])
		for pipe in self.server_end_pipes :
			pipe.recv()
		print("Stepped on all Test Engines")
		s_next,episode_r, done, info = self.main_engine.step(action)
		next_states, list_of_pred, next_state_texts = self.get_available_actions_with_next_state_vectors()
		# self.num_proofs = self.main_engine.num_proofs
		# self.num_proofs_solved = self.main_engine.num_proofs_solved
		info['reachable_states'] = next_states
		info['list_of_pred'] = list_of_pred
		info['reachable_states_text'] = next_state_texts
		return s_next,episode_r, done, info
	
	def check_next_states(self,predictions):
		print("Checking next States on all Test Engines")
		a = time.time()
		for i in range(len(predictions)) :
			self.server_end_pipes[i].send(["check_next_state",predictions[i]])
		results = []
		for pipe in self.server_end_pipes :
			recv_obj = pipe.recv()
			results.append(recv_obj)
		b = time.time()
		print("Checked next States on all Test Enignes")
		print(results)
		print("Time for check next states", b - a)
		# quit()
		if self.info_on_check :
			return list(zip(*results))
		else :
			return results
		
	def run_to_proof(self, proof_contains) :
		print("Running to proof on all Test states")
		for pipe in self.server_end_pipes :
			pipe.send(["run_to_proof",proof_contains])
		for pipe in self.server_end_pipes :
			pipe.recv()
		print("Stepped on all Test Engines")
		return self.main_engine.run_to_proof(proof_contains)
	
	def keepalive(self) :
		for pipe in self.server_end_pipes :
			pipe.send(["reset",None])
		for pipe in self.server_end_pipes :
			print(pipe.recv())
		print("Keepalive")
		# quit()
		return ""

def is_same_context(context1, context2) :
		return contextSurjective(context1, context2) and contextSurjective(context2, context1)

def is_context_fresh_utils( context_history, curr_proof_context) :
		print(len(context_history))
		for context in context_history :
			if contextSurjective(curr_proof_context, context) :
				print("False")
				return False
			else:
				print("True")
		return True

def get_available_actions_with_next_state_vectors(self) :
	relevant_lemmas = self.coq.local_lemmas[:-1]
	print(self.coq.proof_context)
	full_context_before = FullContext(relevant_lemmas, self.coq.prev_tactics,  self.coq.proof_context)
	predictions = self.predictor.predictKTactics(
		truncate_tactic_context(full_context_before.as_tcontext(),
								self.max_term_length), self.num_check_engines)
	next_states = []
	list_of_pred = []
	next_state_texts = []
	print("Available actions", [_.prediction for _ in predictions])
	all_available_pred =  [_.prediction.lstrip().rstrip() for _ in predictions]
	result = self.check_next_states(all_available_pred)
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
			list_of_pred.append(predictions[next_state_ind] )
			next_state_texts.append(curr_next_state_text)
	return next_states, list_of_pred, next_state_texts

