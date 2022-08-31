import torch
import torch.nn as nn
import torch.optim as optim
import gym
import wandb
import numpy as np
import mpire
import time
import argparse
import multiprocessing

wandb.init(project="Proverbot", entity="avarghese")


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
    
    def get_qvals(self,x) :
        with torch.no_grad() :
            qvals = self.forward(x).numpy()
        
        return qvals
    
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class MultiArgIterable() :
    def __init__(self, T_max, I_globalupdate, I_target, gamma = 1):
        self.T_max = T_max
        self.I_globalupdate = I_globalupdate
        self.I_target = I_target
        self.num_calls = 0
        self.gamma = gamma

    def __iter__(self) :
        return self
        
    def __next__(self) :
        self.num_calls += 1
        return (self.T_max, self.I_globalupdate, self.I_target, self.gamma, self.num_calls)    

class Shared_objects():
    def __init__(self, num_states, num_actions) :
        self.model = Agent_model(num_states, num_actions)
        self.T = 0

def get_epsilon_greedy(qvals,epsilon):
    coin = np.random.rand()
    if coin < epsilon :
        return np.argmax(qvals)
    else :
        return np.random.randint(low = 0, high=len(qvals))



def asynchq(global_objects, T_max, I_target, I_update, gamma, job_num) :

    env = gym.make('CartPole-v1')

    thread_model = Agent_model(4,2)
    thread_optimizer = optim.SGD(thread_model.parameters(), lr = 0.001)
    thread_optimizer.zero_grad()

    s = env.reset()
    t = 0
    episode_r = 0
    curr_epsilon = 0.2

    while global_objects.T <= T_max :
        a = get_epsilon_greedy( thread_model.get_qvals(torch.tensor(s)), curr_epsilon )
        s_next,r, done, _ = env.step(a)
        episode_r += r
        

        if done :
            y = r
        else :
            y = r + gamma * max( global_objects.model.get_qvals( torch.tensor(s_next) ) )
        
        computed_reward = thread_model.forward( torch.tensor(s) )[a]
        loss =  (y - computed_reward)**2
        wandb.log({"Loss" : loss.item()})
        wandb.log({"Qval" : computed_reward.item()})
        loss.backward()
        s = s_next
        global_objects.T += 1
        t = t + 1

        if global_objects.T% I_target == 0 :
            global_objects.model.load_state_dict( thread_model.state_dict() )

        if t% I_update == 0 or done :
            thread_optimizer.step()
            thread_optimizer.zero_grad()
            s = env.reset()
            wandb.log({"Episode Total Rewards JobId: %s"%job_num:episode_r})
            episode_r = 0
            if curr_epsilon <= 0.99 :
                curr_epsilon += 0.01
        

# class Test_class :
#     def __init__(self):
#         self.a = 1
# def test(worker_id,shared_objects,x) :
#     time.sleep(1)
#     print("Job id", x, "On CPU", worker_id, ", Before :", shared_objects.a)
#     shared_objects.a = worker_id
#     print("Job id", x, "On CPU", worker_id, ", After : ", shared_objects.a)
#     return


def run_job(worker_id,shared_objects,T_max,I_globalupdate,I_target, gamma, job_num) :
    
    print("Starting Worker number",worker_id, "with Job number", job_num)
    
    asynchq(global_objects = shared_objects, T_max= T_max, I_target = I_target, I_update = I_globalupdate, gamma = gamma, job_num = job_num)

    

if __name__ == "__main__" :
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num_workers', type=int)
    args = parser.parse_args()

    num_states = 4
    num_actions = 2
    num_workers = args.num_workers
    total_num_steps = num_workers*10000

    arguments = MultiArgIterable(T_max = total_num_steps, I_globalupdate= 500, I_target = 100)
    global_objects = Shared_objects(num_states,num_actions)

    print("number of wokers", num_workers)
    with mpire.WorkerPool(n_jobs = num_workers, pass_worker_id = True, shared_objects = global_objects, start_method="threading") as pool :
        pool.map(run_job,arguments, iterable_len = num_workers)
    
    
    # test_object = Test_class()
    # print("Given total of Num CPUs by Unity : ",mpire.cpu_count())
    # with mpire.WorkerPool(n_jobs = args.num_workers, pass_worker_id = True, shared_objects = test_object, start_method="spawn") as pool :
    #     pool.map(test,range(100))
    # print(test_object.a)
