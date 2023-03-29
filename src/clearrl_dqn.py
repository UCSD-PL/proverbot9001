import argparse
import json
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


from search_file import loadPredictorByFile
from pathlib_revised import Path2
from pathlib import Path
import coq2vec
from coq_serapy.contexts import truncate_tactic_context, FullContext, ProofContext, Obligation
from gym import spaces

from gym_proof_env import FastProofEnv

def parse_args():
	# fmt: off
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp-name", type=str, default='dylan_exp_cleanrl',
		help="the name of this experiment")
	parser.add_argument("--seed", type=int, default=1,
		help="seed of the experiment")
	parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
		help="if toggled, `torch.backends.cudnn.deterministic=False`")
	parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
		help="if toggled, cuda will be enabled by default")
	parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
		help="if toggled, this experiment will be tracked with Weights and Biases")
	parser.add_argument("--wandb-project-name", type=str, default="Proverbot",
		help="the wandb's project name")
	parser.add_argument("--wandb-entity", type=str, default="dylanzhang",
		help="the entity (team) of wandb's project")
	parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
		help="whether to capture videos of the agent performances (check out `videos` folder)")
	parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
		help="whether to save model into the `runs/{run_name}` folder")
	parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
		help="whether to upload the saved model to huggingface")
	parser.add_argument("--hf-entity", type=str, default="",
		help="the user or org name of the model repository from the Hugging Face Hub")

	# Algorithm specific arguments
	parser.add_argument("--env-id", type=str, default="proverbot",
		help="the id of the environment")
	parser.add_argument("--total-timesteps", type=int, default=1000000,
		help="total timesteps of the experiments")
	parser.add_argument("--learning-rate", type=float, default=2.5e-4,
		help="the learning rate of the optimizer")
	parser.add_argument("--buffer-size", type=int, default=10000,
		help="the replay memory buffer size")
	parser.add_argument("--gamma", type=float, default=0.99,
		help="the discount factor gamma")
	parser.add_argument("--tau", type=float, default=1.,
		help="the target network update rate")
	parser.add_argument("--target-network-frequency", type=int, default=500,
		help="the timesteps it takes to update the target network")
	parser.add_argument("--batch-size", type=int, default=128,
		help="the batch size of sample from the reply memory")
	parser.add_argument("--start-e", type=float, default=0.05, #1,
		help="the starting epsilon for exploration")
	parser.add_argument("--end-e", type=float, default=0.05,
		help="the ending epsilon for exploration")
	parser.add_argument("--exploration-fraction", type=float, default=0.5,
		help="the fraction of `total-timesteps` it takes from start-e to go end-e")
	parser.add_argument("--learning-starts", type=int, default=1,
		help="timestep to start learning")
	parser.add_argument("--train-frequency", type=int, default=10,
		help="the frequency of training")

	#Environment specific
	parser.add_argument("--max_attempts", type=int, default=7)
	parser.add_argument("--max_proof_len", type=int, default=50)
	parser.add_argument('--prelude', default="CompCert")
	parser.add_argument('--weightsfile', default = "data/polyarg-weights.dat", type=str)
	parser.add_argument('--proof_file', default="CompCert/common/Globalenvs.v",type=str)



	args = parser.parse_args()
	# fmt: on
	return args


# def make_env(env_id, seed, idx, capture_video, run_name):
# 	def thunk():
# 		env = gym.make(env_id)
# 		env = gym.wrappers.RecordEpisodeStatistics(env)
# 		if capture_video:
# 			if idx == 0:
# 				env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
# 		env.seed(seed)
# 		env.action_space.seed(seed)
# 		env.observation_space.seed(seed)
# 		return env

# 	return thunk


# ALGO LOGIC: initialize agent here:
class Agent(nn.Module):
	def __init__(self,  coqenv) -> None:
		super(Agent, self).__init__()
		self.coqenv = coqenv
		self.device = 'cuda' if torch.cuda.is_available() else "cpu"
		self.stateVecSize = 1565

		self.network = nn.Sequential(
			nn.Linear(self.stateVecSize, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 1))
		self.softmax = nn.Softmax(dim=0)

	def forward(self, next_states) -> torch.Tensor:
		# encoded_next_states = torch.cat([self.stateEncoder(state) for state in next_states], dim=0)
		state_scores = self.network(next_states)
		return self.softmax(state_scores)
	def get_vvals_from_contexts(self, next_state_vecs) :
		with torch.no_grad():
			vvals = self.forward(next_state_vecs.to(self.device)).detach().cpu().numpy().flatten()
		if len(vvals.shape) == 1:
			vvals = vvals.reshape(1,-1)
		return vvals
	# def get_vvals_from_contexts(self, contexts) :
	# 	next_state_vecs = []
	# 	finish_idx = []
	# 	for (idx,con) in enumerate(contexts) :
	# 		if con == 'fin': 
	# 			finish_idx.append(idx)
	# 		else:
	# 			next_state_vecs.append(self.stateEncoder(con))
	# 	if len(next_state_vecs)>=1:
	# 		next_state_vecs = torch.stack(next_state_vecs,dim=0).to(self.device)
	# 		with torch.no_grad():
	# 			vvals = self.forward(next_state_vecs).detach().cpu().numpy().flatten()
	# 	else:
	# 		vvals = np.array([])
	# 	for fin_idx in sorted(finish_idx):
	# 		vvals = np.insert(vvals,fin_idx,1)
	# 	if len(vvals.shape) == 1:
	# 		vvals = vvals.reshape(1,-1)
	# 	return vvals





def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
	slope = (end_e - start_e) / duration
	return max(slope * t + start_e, end_e)



if __name__ == "__main__":
	args = parse_args()
	run_name = f"cleanrl_proverbot__{args.exp_name}__{args.seed}__{int(time.time())}"
	if args.track:
		import wandb

		wandb.init(
			project=args.wandb_project_name,
			entity=args.wandb_entity,
			sync_tensorboard=True,
			config=vars(args),
			name=run_name,
			monitor_gym=True,
			save_code=True,
		)
	writer = SummaryWriter(f"runs/{run_name}")
	writer.add_text(
		"hyperparameters",
		"|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
	)

	# TRY NOT TO MODIFY: seeding
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = args.torch_deterministic

	device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

	# env setup
	# env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
	with open("compcert_projs_splits.json",'r') as f :
		data = json.load(f)
	proof_files = data[0]["test_files"]
	env = FastProofEnv(proof_files,args.prelude, args.track, state_type = "proof_context", max_proof_len = args.max_proof_len, num_check_engines = args.max_attempts, info_on_check = True)

	# assert isinstance(env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

	q_network = Agent(env).to(device)
	optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
	target_network = Agent(env).to(device)
	target_network.load_state_dict(q_network.state_dict())

	obs,infos = env.reset()
	# obs = q_network.stateEncoder(obs)

	# print(type(env.action_space))
	rb = ReplayBuffer(
		args.buffer_size,
		spaces.Box(-1,1, shape =(1,1565), dtype=np.float32), #envs.single_observation_space,		env.single_action_space,
		spaces.Discrete(1565),
		device,
		handle_timeout_termination=False,
	)
	start_time = time.time()

	# TRY NOT TO MODIFY: start the game

	for global_step in range(args.total_timesteps):
		# ALGO LOGIC: put action logic here
		epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
		if random.random() < epsilon:
			# print("pred list",infos['list_of_pred'])
			# if len(infos['list_of_pred']) ==0 :
			# 	actions = -1
			# else:
			actions = env.action_space.sample()  #random.randrange(0,len(infos['list_of_pred']))
		else:
			# print(infos['reachable_states'])
			if env.action_space.length == 0:
				actions = -1
			else:
				q_values = q_network.get_vvals_from_contexts(env.reachable_states) #qvals == vvals for this case
				actions = np.argmax(q_values, axis=1)[0]#.cpu().numpy()
		# print(actions)
		# TRY NOT TO MODIFY: execute the game and log data.
		# print('>> List of predictions {}'.format(infos['list_of_pred']))
		# if actions == -1:
		# 	a = None
		# else:
			# print(infos['list_of_pred'])
		a = env.action_space.get_action_by_index(actions)
		next_obs, rewards, dones, infos = env.step(a)
		# next_obs = q_network.stateEncoder(next_obs)

		# TRY NOT TO MODIFY: record rewards for plotting purposes
		# for info in infos:
		if "episode" in infos.keys():
			print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
			writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
			writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
			writer.add_scalar("charts/epsilon", epsilon, global_step)
			break

		# TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
		real_next_obs = next_obs.clone()
		# for idx, d in enumerate(dones):
		# 	if d:
		# real_next_obs = infos["terminal_observation"]
		rb.add(obs, real_next_obs, np.array(actions), rewards, dones, infos)

		# TRY NOT TO MODIFY: CRUCIAL step easy to overlook
		obs = next_obs

		# ALGO LOGIC: training.
		if global_step > args.learning_starts:
			if global_step % args.train_frequency == 0:
				data = rb.sample(args.batch_size)
				with torch.no_grad():
					target_max, _ = target_network(data.next_observations).max(dim=1)
					td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
				# print("observation size",data.observations.size())
				# print("next_observation size",data.next_observations.size())
				old_val = q_network(data.observations) #.gather(1, data.actions).squeeze()
				loss = F.mse_loss(td_target, old_val)

				if global_step % 100 == 0:
					writer.add_scalar("losses/td_loss", loss, global_step)
					writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
					print("SPS:", int(global_step / (time.time() - start_time)))
					writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

				# optimize the model
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			# update target network
			if global_step % args.target_network_frequency == 0:
				for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
					target_network_param.data.copy_(
						args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
					)

	if args.save_model:
		model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
		torch.save(q_network.state_dict(), model_path)
		print(f"model saved to {model_path}")
		# from cleanrl_utils.evals.dqn_eval import evaluate

		# episodic_returns = evaluate(
		# 	model_path,
		# 	make_env,
		# 	args.env_id,
		# 	eval_episodes=10,
		# 	run_name=f"{run_name}-eval",
		# 	Model=Agent,
		# 	device=device,
		# 	epsilon=0.05,
		# )
		# for idx, episodic_return in enumerate(episodic_returns):
		# 	writer.add_scalar("eval/episodic_return", episodic_return, idx)

		# if args.upload_model:
		# 	from cleanrl_utils.huggingface import push_to_hub

		# 	repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
		# 	repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
		# 	push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

	env.close()
	writer.close()
