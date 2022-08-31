import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import mpire, multiprocessing
import time, argparse
from pathlib_revised import Path2
import dataloader
import coq_serapy as serapi_instance
from search_file import completed_proof, loadPredictorByFile, truncate_tactic_context, FullContext
from train_encoder import EncoderRNN, DecoderRNN, Lang, tensorFromSentence, EOS_token
from tokenizer import get_symbols, get_words,tokenizers
import pickle
import gym
import fasttext




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
    def __init__(self, proof_file, time_per_command=100):
        self.action_space = None
        self.observation_space = None


        self.proof_file = proof_file
        self.scraped_tactics = dataloader.scraped_tactics_from_file(str(proof_file), None)
        self.scraped_tactic_index = 0

        self.coq_running = False
        # self.coq.verbose = self.args.verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_lines_pushed = 0
        self.time_per_command= time_per_command
        self.load_state_model()
        self.load_language_model()
    
    def goto_next_proof(self):
        while self.scraped_tactic_index < len(self.scraped_tactics) :
            print(self.scraped_tactic_index)
            print(self.scraped_tactics[self.scraped_tactic_index].prev_tactics[0].strip() )
            if self.scraped_tactics[self.scraped_tactic_index].tactic.strip() == "Proof." :
                tactic = self.scraped_tactics[self.scraped_tactic_index]
                self.coq.run_stmt( tactic.prev_tactics[0].lstrip().rstrip(), timeout= self.time_per_command)
                self.coq.run_stmt( "Proof.", timeout= self.time_per_command)
                self.scraped_tactic_index += 1
                print("Found Proof : ",tactic.prev_tactics[0].lstrip().rstrip())
                break
            self.scraped_tactic_index += 1
            print(self.scraped_tactic_index < len(self.scraped_tactics))
        
        if self.scraped_tactic_index >= len(self.scraped_tactics):
            print("File done")
            self.reset_to_start_of_file()
            return self.goto_next_proof()

        return self.get_state_vector_from_text(tactic.context.fg_goals[0].goal.lstrip().rstrip())

    def reset_to_start_of_file(self) :
        if self.coq_running :
            self.coq.kill()
        self.coq = serapi_instance.SerapiInstance(['sertop', '--implicit'],None,".")
        self.coq.quiet = True
        self.n_lines_pushed = 0
        self.scraped_tactic_index = 0
        self.coq_running = True


    def load_state_model(self) :
        self.state_model =  torch.load("data/encoder_symbols.model", map_location=torch.device(self.device))
    def load_language_model(self) :
        with open("data/encoder_language_symbols.pkl","rb") as f:
            self.language_model = pickle.load(f)


    def get_state_vector_from_text(self,state_text) :
        state_sentence = get_symbols(state_text)
        print(state_sentence)
        state_tensor = tensorFromSentence(self.language_model,state_sentence,self.device)
        with torch.no_grad() :
            state_model_hidden = self.state_model.initHidden(self.device)
            state_model_cell = self.state_model.initCell(self.device)
            input_length = state_tensor.size(0)
            for ei in range(input_length):
                _, state_model_hidden,state_model_cell = self.state_model(state_tensor[ei], state_model_hidden,state_model_cell)

            
            state= state_model_hidden
        state = state.cpu().detach().numpy().flatten()
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
        except :
            print("Some error")
        else :
            self.curr_context = self.coq.proof_context
            r = 0.1

            
            if completed_proof(self.coq) :
                self.coq.run_stmt( "Qed.", timeout= self.time_per_command)
                r = 10
                print("Current proof fin")
                self.goto_next_proof()
                done = True
            if self.curr_context == None :
                print("No context")
                self.goto_next_proof()
                print("Went to next proof")
           
        
        next_state = self.get_state_vector_from_text( self.coq.proof_context.fg_goals[0].goal)
        
        return next_state, r, done, {}


    def reset(self):
        self.reset_to_start_of_file()
        state = self.goto_next_proof()
        self.curr_context = self.coq.proof_context
        return state



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
    
    # def get_qvals(self,x) :
    #     with torch.no_grad() :
    #         qvals = self.forward(x).numpy()
        
    #     return qvals
    
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)



class MultiArgIterable() :
    def __init__(self, T_max, I_globalupdate, I_target, args, gamma = 1):
        self.T_max = T_max
        self.I_globalupdate = I_globalupdate
        self.I_target = I_target
        self.num_calls = 0
        self.gamma = gamma
        self.args = args

    def __iter__(self) :
        return self
        
    def __next__(self) :
        self.num_calls += 1
        return (self.T_max, self.I_globalupdate, self.I_target, self.gamma, self.args, self.num_calls)    

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



def get_qvals(s, thread_model, tactic_space_model, env, predictor):
    relevant_lemmas = env.coq.local_lemmas[:-1]
    full_context_before = FullContext(relevant_lemmas, env.coq.prev_tactics,  env.coq.proof_context)
    predictions = predictor.predictKTactics(
        truncate_tactic_context(full_context_before.as_tcontext(),
                                args.max_term_length), args.max_attempts)

    qvals = []
    for prediction_idx, prediction in enumerate(predictions):
        tactic_vec = tactic_space_model.get_word_vector(prediction.prediction)
        final_state_vec = np.concatenate((s,tactic_vec))
        final_state_tensor = torch.tensor(final_state_vec)
        with torch.no_grad() :
            curr_qval = thread_model(final_state_tensor).item()
        qvals.append(curr_qval)

        # print(prediction.prediction, prediction.certainty, curr_qval)

    return qvals

def select_action(s, thread_model, tactic_space_model, env, predictor, epsilon) :
    relevant_lemmas = env.coq.local_lemmas[:-1]
    full_context_before = FullContext(relevant_lemmas, env.coq.prev_tactics,  env.coq.proof_context)
    predictions = predictor.predictKTactics(
        truncate_tactic_context(full_context_before.as_tcontext(),
                                args.max_term_length), args.max_attempts)

    qvals = []
    for prediction_idx, prediction in enumerate(predictions):
        tactic_vec = tactic_space_model.get_word_vector(prediction.prediction)
        final_state_vec = np.concatenate((s,tactic_vec))
        final_state_tensor = torch.tensor(final_state_vec)
        curr_qval = thread_model(final_state_tensor)
        qvals.append(curr_qval)

        print(prediction.prediction, prediction.certainty, curr_qval.item())
    
    action_idx = get_epsilon_greedy([i.item() for i in qvals],epsilon)

    return predictions[action_idx],qvals[action_idx]

    


def asynchq(global_objects, T_max, I_target, I_update, gamma, args, job_num) :

    env = ProofEnv("sample-files/simple_proof.v.scrape")
    predictor = loadPredictorByFile(args.weightsfile)
    tactic_space_model = fasttext.train_unsupervised(args.scrape_file.path, model='cbow', lr = 0.1,epoch = 1000)

    s = env.reset()
    
    num_states = s.shape[0]
    num_actions = 1
    thread_space_dim = tactic_space_model.get_word_vector("auto.").shape[0]


    thread_model = Agent_model(num_states + thread_space_dim,num_actions)
    thread_optimizer = optim.SGD(thread_model.parameters(), lr = 0.001)
    thread_optimizer.zero_grad()
    

    t = 0
    episode_r = 0
    curr_epsilon = 0.2

    while global_objects.T <= T_max :
        # a = get_epsilon_greedy( thread_model.get_qvals(torch.tensor(s)), curr_epsilon )
        # print(s)
        
        prediction, computed_reward = select_action(s, thread_model, tactic_space_model, env, predictor, curr_epsilon)
        print("Selected action :" +  prediction.prediction  + "; Take step <press Enter>?")
        s_next,episode_r, done, _ = env.step(prediction.prediction)
        episode_r += prediction.certainty
        

        if done :
            y = episode_r
        else :
            y = episode_r + gamma * max( get_qvals(s, thread_model, tactic_space_model, env, predictor) )
        


        loss =  (y - computed_reward)**2
        if args.wandb_log :                
            wandb.log({"Loss" : loss.item()})
            wandb.log({"Qval" : computed_reward.item()})
            wandb.log({"True Rewards" : y})
        loss.backward()
        s = s_next
        global_objects.T += 1
        t = t + 1

        if global_objects.T% I_target == 0 :
            global_objects.model.load_state_dict( thread_model.state_dict() )

        if t% I_update == 0 or done :
            thread_optimizer.step()
            thread_optimizer.zero_grad()
            if args.wandb_log :
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


def run_job(worker_id,shared_objects,T_max,I_globalupdate,I_target, gamma, args, job_num) :
    
    print("Starting Worker number",worker_id, "with Job number", job_num)
    
    asynchq(global_objects = shared_objects, T_max= T_max, I_target = I_target, I_update = I_globalupdate, gamma = gamma, args = args, job_num = job_num)

    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrape_file", type=Path2)
    parser.add_argument("--max-tuples", default=None, type=int)
    parser.add_argument("--tokenizer",
                            choices=list(tokenizers.keys()), type=str,
                            default=list(tokenizers.keys())[0])
    parser.add_argument("--num-keywords", default=100, type=int)
    parser.add_argument("--lineend", action="store_true")
    parser.add_argument('-n','--num_workers', type=int, default=1)
    parser.add_argument('--wandb_log', action= 'store_true')
    parser.add_argument('--weightsfile', default = "data/polyarg-weights.dat", type=Path2)
    parser.add_argument("--max_term_length", type=int, default=256)
    parser.add_argument("--max_attempts", type=int, default=10)


    args = parser.parse_args()

    multiprocessing.freeze_support()

    if args.wandb_log :
        wandb.init(project="Proverbot", entity="avarghese")


    env = ProofEnv("sample-files/simple_proof.v.scrape")
    predictor = loadPredictorByFile(args.weightsfile)
    tactic_space_model = fasttext.train_unsupervised(args.scrape_file.path, model='cbow', lr = 0.1,epoch = 1000)
    s = env.reset()
    num_states = s.shape[0]
    thread_space_dim = tactic_space_model.get_word_vector("auto.").shape[0]
    del env
    del predictor
    del tactic_space_model

    num_actions = 1
    num_states = num_states  + thread_space_dim
    num_workers = args.num_workers
    total_num_steps = num_workers*10000
    

    arguments = MultiArgIterable(T_max = total_num_steps, I_globalupdate= 1, I_target = 1, args = args)
    global_objects = Shared_objects(num_states,num_actions)

    print("number of wokers", num_workers)
    with mpire.WorkerPool(n_jobs = num_workers, pass_worker_id = True, shared_objects = global_objects, start_method="threading") as pool :
        pool.map(run_job,arguments, iterable_len = num_workers)
        