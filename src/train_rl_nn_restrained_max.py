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

np.random.seed(2)

class Agent_model(nn.Module) :
	def __init__(self,input_size,output_size) :
		super(Agent_model,self).__init__()
		self.lin1 = nn.Linear(input_size,1000)
		self.lin2 = nn.Linear(1000,500)
		self.lin3 = nn.Linear(500,200)
		self.lin4 = nn.Linear(200,100)
		self.lin5 = nn.Linear(100,50)
		self.lin6 = nn.Linear(100,50)
		self.lin7 = nn.Linear(100,50)
		self.lin8 = nn.Linear(100,50)
		self.linfinal = nn.Linear(50,output_size)
		self.softmax = nn.Softmax()
		self.relu = nn.LeakyReLU()
		# self.apply(self.init_weights)
		
	def forward(self,x) :
		x = self.relu(self.lin1(x))
		x = self.relu(self.lin2(x))
		x = self.relu(self.lin3(x))
		x = self.relu(self.lin4(x))
		x = self.relu(self.lin5(x))
		# x = self.relu(self.lin6(x))
		# x = self.relu(self.lin7(x))
		# x = self.relu(self.lin8(x))
		x = self.linfinal(x)
		return x


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
		
		print("After : ",self.coq.proof_context)
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
		if args.wandb_log :
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


	def step(self, action):
		done = False
		# prediction = self.get_pred(action)
		prediction = action
		self.num_commands += 1
		info = {"state_change" : True}
		
		try:
			self.coq.run_stmt(prediction, timeout= self.time_per_command)

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
				if args.wandb_log :
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
			if args.wandb_log :
				wandb.log({"Num command Attempts" : self.num_commands  })
			
			self.solve_curr_from_file()
			self.goto_next_proof()
			done = True 
		
		
		next_state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip() )
		info["state_text"] = self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
		return next_state, r, done, info


	def reset(self):
		self.reset_to_start_of_file()
		self.goto_next_proof()
		state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip() )
		info = {}
		info["state_text"] =self.coq.proof_context.fg_goals[0].goal.lstrip().rstrip()
		return state,info


class Memory :
	def __init__(self) :
		self.mem = []
		self.num_items = 0
		self.index_dict = defaultdict(lambda : None)
				
	def add(self,s_vector ,s_text, r) :
		print(self.index_dict, self.mem)
		print("Who shaoll say ....", s_text, r)
		if s_text in self.index_dict :
			i = self.index_dict[s_text]
			self.mem[i][1] = max(r, self.mem[i][1]) #Index 1 is the reward, 0 is the vector
		
		else :
			self.index_dict[s_text] = self.num_items
			self.mem.append([s_vector,r])
			self.num_items += 1
			if args.wandb_log :
				wandb.log({'Num Proof states' : self.num_items})
	
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
	state_actions = torch.tensor(state_actions,dtype=torch.float32)
	with torch.no_grad() :
		qvals = agent_model(state_actions)
	print(qvals)
	return qvals

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


def get_state_action(s, tactic_space_model, env, predictor) :
	relevant_lemmas = env.coq.local_lemmas[:-1]
	full_context_before = FullContext(relevant_lemmas, env.coq.prev_tactics,  env.coq.proof_context)
	predictions = predictor.predictKTactics(
		truncate_tactic_context(full_context_before.as_tcontext(),
								args.max_term_length), args.max_attempts)
	

	state_action = []
	list_of_pred = []
	for prediction_idx, prediction in enumerate(predictions):
		curr_pred = prediction.prediction.lstrip().rstrip()
		if curr_pred in env.restrictions[ env.coq.proof_context.fg_goals[0].goal ] :
			continue
		list_of_pred.append( prediction )
		tactic_class,tactic_args = split_tactic(curr_pred.lstrip().rstrip().rstrip("."))
		# tactic_class_vec = np.eye(len(list_of_tactic_classes), 0, list_of_tactic_classes.index(tactic_class)).flatten()
		tactic_class_vec = np.zeros(len(env.list_of_tactic_classes) + 1)
		if tactic_class in env.list_of_tactic_classes :
			tactic_class_vec[ env.list_of_tactic_classes.index(tactic_class) ] = 1
		else :
			tactic_class_vec[-1] = 1

		if tactic_args.strip() != "" and is_hyp_token(tactic_args,env.coq.proof_context.fg_goals[0]) :
			# print(current_context.fg_goals[0].hypotheses)
			# print(get_indexed_vars_dict(current_context.fg_goals[0].hypotheses))
			tactic_args = tactic_args.strip()
			index = get_indexed_vars_dict(env.coq.proof_context.fg_goals[0].hypotheses)[tactic_args]
			tactic_args_type = get_hyp_type(env.coq.proof_context.fg_goals[0].hypotheses[index])
			tactic_args_type_vec = env.get_state_vector_from_text(tactic_args_type) # tactic_space_model.get_word_vector(tactic_args)
			tactic_args_vec =  tactic_space_model.get_word_vector(tactic_args)
		else :
			# print("Nope", tactic_args)
			tactic_args_type_vec = np.zeros(shape = s.shape)
			tactic_args_vec = tactic_space_model.get_word_vector(tactic_args)

		final_state_action_vec = np.concatenate((tactic_class_vec, tactic_args_type_vec, tactic_args_vec))
		state_action.append(final_state_action_vec)

	
	return state_action, list_of_pred


def select_action(s, agent_model, tactic_space_model, env, predictor, epsilon) :

	state_action, predictions = get_state_action(s, tactic_space_model, env, predictor)
	if len(state_action) == 0 or len(predictions) == 0 :
		print("No predictions available for the current state")
		return None, None, {"qval" : -5}

	qvals = get_qvals(state_action, agent_model)
	
	print("Current Options : ", [predictions[i].prediction.lstrip().rstrip() + " : " + str(qvals[i]) for i in range(len(qvals)) ])
	action_idx = get_epsilon_greedy([i.item() for i in qvals],epsilon)

	return predictions[action_idx],state_action[action_idx],{"qval" : qvals[action_idx]}

def select_random_action(s, tactic_space_model, env, predictor) :
	state_action, predictions = get_state_action(s, tactic_space_model, env, predictor)

	if len(state_action) == 0 or len(predictions) == 0 :
		print("No predictions available for the current state")
		return None, None, None

	action_idx = np.random.choice(range(len(state_action)))
	return predictions[action_idx].prediction,state_action[action_idx]


def NN_QLearning(T_max, gamma, batch_size, args) :

	env = ProofEnv(args.proof_file.path, args.prelude)
	predictor = loadPredictorByFile(args.weightsfile)
	tactic_space_model = fasttext.train_unsupervised(args.proof_file.path, model='cbow', lr = 0.1,epoch = 1000)
	memory = Memory()

	s,info = env.reset()
	prediction, state_action = select_random_action(s, tactic_space_model, env, predictor)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	agent_model = Agent_model(state_action.shape[0],1).to(device) #DNNScorer(1,5,2) 
	optimizer =  optim.Adam(agent_model.parameters(),lr = 0.001)
	loss_object = torch.nn.MSELoss()

	# with open("data/Rf_agent_model.pkl", 'rb') as f:
	# 	# agent_model = RandomForestRegressor()
	# 	agent_model = pickle.load(f)



	
	T = 0
	episode_r = 0
	curr_epsilon = 0.5
	epsilon_increment = 0.001 #10/T_max
	perf_queue = deque(maxlen=20)

	state_rewards = []
	states_in_this_proof = []
	agent_qval = []
	live_rewards = []
	state_texts = []
	while T <= T_max :
		print(T)
		# a = get_epsilon_greedy( agent_model.get_qvals(torch.tensor(s)), curr_epsilon )
		# print(s)

		info["state_change"]= False
		while info["state_change"] == False :
			prediction,state_action,agent_info = select_action(s, agent_model, tactic_space_model, env, predictor, curr_epsilon)
			if prediction == None and state_action == None :
				print("Exiting the loop")
				s_next,episode_r, done, info = env.abort_and_finish_proof()
				break
			print("Selected action :" +  prediction.prediction  + "; Take step <press Enter>?")
			s_next,episode_r, done, info = env.step(prediction.prediction)

		print("Step taken")
		if not done :
			episode_r += prediction.certainty

		if prediction == None :
			state_rewards[-1] = episode_r
			live_rewards.append(episode_r)
		else :
			state_rewards.append(episode_r)
			state_texts.append(info["state_text"])
			states_in_this_proof.append(state_action)
			agent_qval.append(agent_info["qval"])
			live_rewards.append(episode_r)

		if args.wandb_log :
			wandb.log({"T" : T})
			wandb.log({"Num Proofs encountered" : env.num_proofs, "Num Proofs Solved" : env.num_proofs_solved})

		if done :
			total_curr_rewards = 0
			true_qvals = []
			for i in range(len(state_rewards) - 1, -1, -1) :
				total_curr_rewards = state_rewards[i] + total_curr_rewards*gamma
				memory.add(states_in_this_proof[i], state_texts[i], total_curr_rewards)
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
				
			if memory.num_items > batch_size :
				for _ in range(200) :
					mem_batch = memory.sample_random_minibatch(batch_size)
					state_actions, rewards = zip(*mem_batch)
					state_actions = torch.tensor(state_actions, dtype=torch.float32)
					rewards = torch.tensor(rewards,dtype=torch.float32)
					optimizer.zero_grad()
					y_pred = agent_model(state_actions)
					loss =  loss_object(y_pred,rewards)
					print(loss)
					loss.backward()
					optimizer.step()
					if args.wandb_log :                
						wandb.log({"Loss" : loss})


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


	total_num_steps = 10000
	gamma = 1
	batch_size = 100
	

	NN_QLearning(T_max= total_num_steps, gamma = gamma, args = args, batch_size = batch_size)


		