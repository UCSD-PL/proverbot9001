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
from coq_serapy import load_commands, kill_comments, get_hyp_type, get_indexed_vars_dict, get_stem, split_tactic, contextSurjective, summarizeContext
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
from mpire import WorkerPool
from multiprocessing import Pool
import itertools




class ProofEnv() :
	def __init__(self, proof_file, prelude, wandb = False, time_per_command=100):
		self.action_space = None
		self.observation_space = None
		self.prelude= prelude

		self.proof_file = proof_file
		self.commands = load_commands(proof_file, progress_bar=True)
		self.proof_line_num = 0
		self.wandb_log = wandb
		self.coq_running = False
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.in_agent_proof_mode= False
		self.in_file_proof_mode= True  #Likely wont be using but just for sake of testing

		self.time_per_command= time_per_command
		self.load_state_model()
		self.load_language_model()
		self.max_attempts = 30
		self.test_file = open("output/output_test_file.txt","w")
		self.curr_proof_tactics = []
		self.max_num_proofs = 15
		self.num_proofs = 0
		self.restrictions = defaultdict(lambda: [])
		self.load_list_tactic_classes()

	
	def load_list_tactic_classes(self) :
		with open("tactics.txt", "r") as f :
			whole_file = f.read()
			self.list_of_tactic_classes = whole_file.split("\n")
			for i in range(len(self.list_of_tactic_classes)) :
				self.list_of_tactic_classes[i] = self.list_of_tactic_classes[i].strip().rstrip(".")


	def goto_next_proof(self):
		print("Going to next Proof")
		assert self.in_agent_proof_mode == False
		assert self.in_file_proof_mode == True

		self.num_commands = 0
		self.proof_contexts_in_path = []
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
		
		print(" Context After finding proof : ",self.coq.proof_context)
		if self.proof_line_num >= len(self.commands) : #or self.num_proofs >= self.max_num_proofs :
			print("File Finished")
			self.test_file.write("\n ----------------------------------------------------- \n")
			self.reset_to_start_of_file()
			
			return self.goto_next_proof()

		return None #self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip())


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
		print("Proof Line Num : ", self.proof_line_num)
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

		print("Done solving from File", self.proof_line_num)

		


	def reset_to_start_of_file(self) :
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
		state = np.append(state,[self.num_commands]).astype("float") 
		return state

	def abort_and_finish_proof(self) :
		self.in_agent_proof_mode= False
		self.in_file_proof_mode = True
		print("Too many attempts, aborting training on current proof")
		self.coq.run_stmt("Abort.", timeout= self.time_per_command)
		if self.wandb_log :
			wandb.log({"Num command Attempts" : self.num_commands  })
		
		self.solve_curr_from_file()
		self.goto_next_proof()
		done = True
		next_state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal)
		r = -5
		return next_state, r, done, {}

	def is_context_fresh(self) :
		for contexts in self.proof_contexts_in_path :
			if contextSurjective(self.env.proof_context, contexts) :
				return False
		
		return True

	def is_same_context(self,context1, context2) :
		print("Context Surjectives")
		print(contextSurjective(context1, context2))
		print(contextSurjective(context2, context1))
		return contextSurjective(context1, context2) and contextSurjective(context2, context1)

	
	def check_next_state(self,prediction) :
		print("Checking next state for action -", prediction)
		next_state = []
		context_before = self.coq.proof_context
		try:
			self.coq.run_stmt(prediction, timeout= self.time_per_command)
		except (serapi_instance.TimeoutError, serapi_instance.ParseError,
				serapi_instance.CoqExn, serapi_instance.OverflowError,
				serapi_instance.ParseError,
				RecursionError,
				serapi_instance.UnrecognizedError) as e:
			print("One of known errors", e)
			self.restrictions[self.coq.proof_context.fg_goals[0].goal].append(prediction)
		except serapi_instance.CoqAnomaly:
			print("Coq Anomaly")
			self.kill()
			quit()
		except :
			print("Some error")
			self.kill()
			quit()
		else :
			num_brackets_run = 0
			while len(unwrap(self.coq.proof_context).fg_goals) == 0 and not completed_proof(self.coq):
				print("Running }")
				self.coq.run_stmt("}", timeout= self.time_per_command)
				num_brackets_run += 1

			if len(self.coq.proof_context.fg_goals) > 1 :
				print(self.coq.proof_context.fg_goals,self.coq.proof_context.bg_goals)
				print("Running {")
				self.coq.run_stmt( "{", timeout= self.time_per_command)
				num_brackets_run += 1

			
			if completed_proof(self.coq) :
				for _ in range(num_brackets_run) :
					self.coq.cancel_last()
				self.coq.cancel_last()
				print("QED on this action. Cancelled - ",prediction)
				return "fin"

			if self.coq.proof_context == None :
				print("Something is wrong. Lost context")
				quit()

			if not self.is_context_fresh() :
				self.restrictions[self.coq.proof_context.fg_goals[0].goal].append(prediction)
			else :
				next_state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal)
			
			if num_brackets_run > 0 :
				print("Cancelling", num_brackets_run, "Brackets")
				for _ in range(num_brackets_run) :
					self.coq.cancel_last()

			self.coq.cancel_last()
			print("Cancelled - ",prediction)
		
		context_after = self.coq.proof_context
		assert self.is_same_context(context_before,context_after)
		return next_state


	def step(self, action):
		done = False
		# prediction = self.get_pred(action)
		prediction = action
		self.num_commands += 1
		info = {"state_change" : True}
		print("Taking step -", action)
		print("Context Before Step - ",end=" " )
		summarizeContext(self.coq.proof_context)
		try:
			self.coq.run_stmt(prediction, timeout= self.time_per_command)
			print("Context After Step - ",end=" ")
			summarizeContext(self.coq.proof_context)

		except (serapi_instance.TimeoutError, serapi_instance.ParseError,
				serapi_instance.CoqExn, serapi_instance.OverflowError,
				serapi_instance.ParseError,
				RecursionError,
				serapi_instance.UnrecognizedError) as e:
			print("One of known errors", e)
			self.restrictions[self.coq.proof_context.fg_goals[0].goal].append(prediction)
			r = 0
			self.num_commands -= 1
			info["state_change"] = False
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
			self.curr_proof_tactics.append(prediction)

			while len(unwrap(self.coq.proof_context).fg_goals) == 0 and not completed_proof(self.coq):
				print("Running }")
				self.coq.run_stmt("}", timeout= self.time_per_command)
				self.curr_proof_tactics.append("}")

			if len(self.coq.proof_context.fg_goals) > 1 :
				print(self.coq.proof_context.fg_goals,self.coq.proof_context.bg_goals)
				print("Running {")
				self.coq.run_stmt( "{", timeout= self.time_per_command)
				self.curr_proof_tactics.append("{")
			
			if completed_proof(self.coq) :
				if self.wandb_log :
					wandb.log({"Num command Attempts" : self.num_commands  })
				self.coq.run_stmt( "Qed.", timeout= self.time_per_command)
				self.curr_proof_tactics.append("Qed.")
				r = 5
				print("Current proof fin with Good rewards")
				self.test_file.write("\n".join(self.curr_proof_tactics) )
				self.test_file.flush()
				self.num_proofs_solved += 1
				self.in_agent_proof_mode= False
				self.in_file_proof_mode = False
				self.navigate_file_end_of_current_proof()
				self.in_agent_proof_mode= False
				self.in_file_proof_mode = True
				self.goto_next_proof()
				done = True
			
			if not self.is_context_fresh() :
				r = 0
				self.num_commands -= 1
				info["state_change"] = False
				self.coq.cancel_last()
				self.restrictions[self.coq.proof_context.fg_goals[0].goal].append(prediction)

			if self.coq.proof_context == None :
				print("No context")
				quit()
				self.goto_next_proof()
				print("Went to next proof")


		if self.num_commands > self.max_attempts :
			r = -5
			self.in_agent_proof_mode= False
			self.in_file_proof_mode = True
			print("Too many attempts, aborting training on current proof")
			self.coq.run_stmt("Abort.", timeout= self.time_per_command)
			if self.wandb_log :
				wandb.log({"Num command Attempts" : self.num_commands  })
			
			self.solve_curr_from_file()
			self.goto_next_proof()
			done = True 
		
		
		next_state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip() )
		info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
		print("Context After Cancel - ",end=" ")
		summarizeContext(self.coq.proof_context)
		return next_state, r, done, info


	def reset(self):
		self.reset_to_start_of_file()
		self.goto_next_proof()
		state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip() )
		info = {}
		info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
		return (state,info)


def multiprocess_tasks(func,obj_list) :
	num_process = len(obj_list)
	with WorkerPool(n_jobs = num_process) as pool :
		results = pool.map(func , obj_list)
	return list(results)

def reset(x) :
	return x.reset()
def step(x,action) :
	return x.step(action)
def check_next_state(x, prediction) :
	return x.check_next_state(prediction)

class FastProofEnv() :
	def __init__(self, proof_file, prelude, wandb = False, time_per_command=100, num_check_engines = 5):
		self.num_check_engines = num_check_engines
		self.main_engine = ProofEnv(proof_file, prelude, wandb, time_per_command)
		self.test_engines = [ProofEnv(proof_file, prelude, wandb = False, time_per_command = time_per_command) for _ in range(num_check_engines)]
	
	def reset(self) :
		multiprocess_tasks( reset, self.test_engines)
		return self.main_engine.reset()
	
	def step(self, action) :
		multiprocess_tasks( step , zip(self.test_engines,itertools.repeat(action)) )
		return self.main_engine.step(action)
	
	def check_next_state(self,prediction) :
		return multiprocess_tasks( check_next_state , zip(self.test_engines,itertools.repeat(prediction)) )
