from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time, argparse, random
from pathlib_revised import Path2
import coq_serapy as serapi_instance
from coq_serapy import load_commands, kill_comments, get_hyp_type, get_indexed_vars_dict, get_stem, split_tactic
from search_file import completed_proof, loadPredictorByFile, truncate_tactic_context, FullContext
from train_encoder import EncoderRNN, DecoderRNN, Lang, tensorFromSentence, EOS_token
from tokenizer import get_symbols, get_words,tokenizers
from coq_serapy.contexts import ProofContext
import pickle
import gym
import fasttext
import os, sys
import dataloader, tqdm
import time
from  sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sys import getsizeof
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

# wandb.init(project="Proverbot", entity="avarghese")


with open("tactics.txt", "r") as f :
    whole_file = f.read()
    list_of_tactic_classes = whole_file.split("\n")
    for i in range(len(list_of_tactic_classes)) :
        list_of_tactic_classes[i] = list_of_tactic_classes[i].strip().rstrip(".")

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
    

def get_state_vector_from_text(state_text, language_model, device, state_model) :
    state_sentence = get_symbols(state_text)
    state_tensor = tensorFromSentence(language_model,state_sentence,device, ignore_missing = True)
    with torch.no_grad() :
        state_model_hidden = state_model.initHidden(device)
        state_model_cell = state_model.initCell(device)
        input_length = state_tensor.size(0)
        for ei in range(input_length):
            _, state_model_hidden,state_model_cell = state_model(state_tensor[ei], state_model_hidden,state_model_cell)

		
        state= state_model_hidden
    state = state.cpu().detach().numpy().flatten()
    # state = np.append(state,[self.num_commands]).astype("float") 
    return state

		
scraped_tactics = dataloader.scraped_tactics_from_file("/home/avarghese_umass_edu/work/rl/proverbot9001/CompCert/common/Globalenvs.v.scrape", "all", 30, None)
predictor = loadPredictorByFile(Path2("data/polyarg-weights.dat"))
tactic_space_model = fasttext.train_unsupervised("/home/avarghese_umass_edu/work/rl/proverbot9001/CompCert/common/Globalenvs.v", model='cbow', lr = 0.1,epoch = 10)
tactic_space_model.save_model("fast_text_model_on_Globalenvs.bin")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_model =  torch.load("data/encoder_symbols.model", map_location=torch.device(device))
with open("data/encoder_language_symbols.pkl","rb") as f:
    language_model = pickle.load(f)

proof_start = False

state_vectors = []
correct_action_vectors = []
wrong_action_vectors = []


index = 0
num_zero_vectors = 0
for tactic in tqdm.tqdm(scraped_tactics) : 
    index += 1
    # print(index, tactic.tactic.strip())
    if proof_start :
        if tactic.tactic.lstrip().rstrip().lower() == "qed." :
            proof_start = False
        elif  len(tactic.context.fg_goals) != 0 :
            current_context = ProofContext(tactic.context.fg_goals,tactic.context.bg_goals,tactic.context.shelved_goals,tactic.context.given_up_goals)
            action = tactic.tactic.lstrip().rstrip().rstrip(".")
            state = tactic.context.fg_goals[0].goal.lstrip().rstrip()
            state_vector = get_state_vector_from_text(state, language_model, device, state_model)
            tactic_class,tactic_args = split_tactic(action)
            # tactic_class_vec = np.eye(len(list_of_tactic_classes), 0, list_of_tactic_classes.index(tactic_class)).flatten()
            tactic_class_vec = np.zeros(len(list_of_tactic_classes) + 1)
            if tactic_class in list_of_tactic_classes :
                tactic_class_vec[ list_of_tactic_classes.index(tactic_class) ] = 1
            else :
                tactic_class_vec[-1] = 1

            if tactic_args.strip() != "" and is_hyp_token(tactic_args,current_context.fg_goals[0]) :
                index = get_indexed_vars_dict(current_context.fg_goals[0].hypotheses)[tactic_args]
                tactic_args_type = get_hyp_type(current_context.fg_goals[0].hypotheses[index])
                tactic_args_type_vec = get_state_vector_from_text(tactic_args_type) # tactic_space_model.get_word_vector(tactic_args)
                tactic_args_vec =  tactic_space_model.get_word_vector(tactic_args)
            else :
                num_zero_vectors += 1
                tactic_args_type_vec = np.zeros(shape = state_vector.shape)
                tactic_args_vec = tactic_space_model.get_word_vector(tactic_args)

            final_action_vec = np.concatenate((tactic_class_vec, tactic_args_type_vec, tactic_args_vec))

            correct_action_vectors.append(final_action_vec)
            state_vectors.append(state_vector)

            curr_wrong = []
            full_context_before = FullContext(tactic.relevant_lemmas, tactic.prev_tactics,  current_context)
            predictions = predictor.predictKTactics(
                truncate_tactic_context(full_context_before.as_tcontext(),
                                        256), 10)
			
            for prediction_idx, prediction in enumerate(predictions):
                curr_pred = prediction.prediction.lstrip().rstrip()
                if curr_pred != action :
                    # print(curr_pred)
                    tactic_class,tactic_args = split_tactic(curr_pred.lstrip().rstrip().rstrip("."))
                    # tactic_class_vec = np.eye(len(list_of_tactic_classes), 0, list_of_tactic_classes.index(tactic_class)).flatten()
                    tactic_class_vec = np.zeros(len(list_of_tactic_classes) + 1)
                    if tactic_class in list_of_tactic_classes :
                        tactic_class_vec[ list_of_tactic_classes.index(tactic_class) ] = 1
                    else :
                        tactic_class_vec[-1] = 1

                    if tactic_args.strip() != "" and is_hyp_token(tactic_args,current_context.fg_goals[0]) :
                        # print(current_context.fg_goals[0].hypotheses)
                        # print(get_indexed_vars_dict(current_context.fg_goals[0].hypotheses))
                        tactic_args = tactic_args.strip()
                        index = get_indexed_vars_dict(current_context.fg_goals[0].hypotheses)[tactic_args]
                        tactic_args_type = get_hyp_type(current_context.fg_goals[0].hypotheses[index])
                        tactic_args_type_vec = get_state_vector_from_text(tactic_args_type) # tactic_space_model.get_word_vector(tactic_args)
                        tactic_args_vec =  tactic_space_model.get_word_vector(tactic_args)
                    else :
                        # print("Nope", tactic_args)
                        num_zero_vectors += 1
                        tactic_args_type_vec = np.zeros(shape = state_vector.shape)
                        tactic_args_vec = tactic_space_model.get_word_vector(tactic_args)

                    final_action_vec = np.concatenate((tactic_class_vec, tactic_args_type_vec, tactic_args_vec))
                    curr_wrong.append(final_action_vec)
            wrong_action_vectors.append(curr_wrong)
    else :
        if  tactic.tactic.lstrip().rstrip().lower() == "proof."   :
            proof_start = True
            




print(len(state_vectors),len(correct_action_vectors), len(wrong_action_vectors))
print(num_zero_vectors)
# quit()
with open("data/test_data_discrete_tactic_vector.pkl","wb") as f:
    pickle.dump((state_vectors, correct_action_vectors,wrong_action_vectors),f)

print("Dumped Vectors")





# # # Minimum : 0.0025427588
# # print("Dumped Vectors")

with open("data/test_data_discrete_tactic_vector.pkl","rb") as f:
    state_vectors,correct_action_vectors,wrong_action_vectors = pickle.load(f)
    print("Loaded Vectors")


# # min_dataset = []
# # for i in tqdm.tqdm(range(len(state_vectors))) :
# #     Dataset = []
# #     considered_state_vec  = np.concatenate((state_vectors[i],action_vectors[i]))
# #     for j in range(len(state_vectors)) :
# #         if i!= j and (action_vectors[i] != action_vectors[j]).all():
# #             comparitive_state_vec = np.concatenate((state_vectors[i],action_vectors[j]))
# #             Dataset.append( np.linalg.norm(considered_state_vec - comparitive_state_vec, ord=2) )
	
# #     min_dataset.append(min(Dataset))
# # print(min_dataset)
# # print(min(min_dataset))

X = []
Y = []
# # print(getsizeof(state_vectors)/ 1024**2, getsizeof(np.array(state_vectors))/ 1024**2, np.array(state_vectors).shape)
# # print(getsizeof(action_vectors)/ 1024**2, getsizeof(np.array(action_vectors))/ 1024**2, np.array(action_vectors).shape)
# # print(np.concatenate( ( np.array(state_vectors),np.array(action_vectors) ), axis = 1 ).nbytes / 1024**2, np.concatenate( ( np.array(state_vectors),np.array(action_vectors) ), axis = 1 ).shape)
# # print(len(state_vectors))
# # print( (np.concatenate( ( np.array(state_vectors),np.array(action_vectors) ), axis = 1 )[0].nbytes / 1024**2) * len(state_vectors)**2)
# # quit()

# # Type 1
# # for i in tqdm.tqdm(range(len(state_vectors))) :
# #     considered_state_vec  = np.concatenate((state_vectors[i],action_vectors[i]))
# #     # print(considered_state_vec.nbytes / 1024**2 )
# #     # print(considered_state_vec.nbytes / 1024**2 * len(state_vectors)**2)
# #     # quit()
# #     X.append(considered_state_vec)
# #     for j in range(len(state_vectors)) :
# #         if i!= j and (action_vectors[i] != action_vectors[j]).all():
# #             comparitive_state_vec = np.concatenate((state_vectors[i],action_vectors[j]))
# #             X.append(comparitive_state_vec)
# #             wandb.log({"Size of list" : np.array(X).nbytes/1024**2})

# # #Type 2
# # print(type(state_vectors), type(action_vectors))
# # for i in range(len(state_vectors)) :
# #     considered_state_vec  = list(state_vectors[i]) + list(action_vectors[i])
# #     X.append(considered_state_vec)
# #     for j in range(len(state_vectors)) :
# #         if i!= j and (action_vectors[i] != action_vectors[j]).all():
# #             comparitive_state_vec = list(state_vectors[i]) + list(action_vectors[j])
# #             X.append(comparitive_state_vec)
# #             wandb.log({"Size of list" : getsizeof(X)/1024**2})


# # print(X.nbytes/1024**2)

for i in tqdm.tqdm(range(len(state_vectors) )) :
    considered_state_vec  = np.concatenate((state_vectors[i],correct_action_vectors[i]))
    X.append(considered_state_vec)
    Y.append(1)
    for wrong_action_vec in wrong_action_vectors[i] :    
        comparitive_state_vec = np.concatenate((state_vectors[i],wrong_action_vec))
        X.append(comparitive_state_vec)
        Y.append(0)
        # X.append(np.random.normal(considered_state_vec,0.000025427588, size = considered_state_vec.shape ))
        # Y.append(1)
        # wandb.log({"Distance between True and False" : np.sqrt(np.sum((comparitive_state_vec - considered_state_vec)**2))} )

with open("data/rf_test_data_discrete_tactic.pkl", "wb") as f:
    pickle.dump((X,Y), f)

print("Dumped X, Y")
# class Agent_model(nn.Module) :
# 	def __init__(self,input_size,output_size) :
# 		super(Agent_model,self).__init__()
# 		self.lin1 = nn.Linear(input_size,1000)
# 		self.lin2 = nn.Linear(1000,1000)
# 		self.lin3 = nn.Linear(1000,1000)
# 		self.lin4 = nn.Linear(1000,1000)
# 		self.lin5 = nn.Linear(1000,output_size)
# 		self.relu = nn.LeakyReLU()
# 		self.apply(self.init_weights)
		
# 	def forward(self,x) :
# 		x = self.relu(self.lin1(x))
# 		x = self.relu(self.lin2(x))
# 		x = self.relu(self.lin3(x))
# 		x = self.relu(self.lin4(x))
# 		x = self.lin5(x)
# 		return x

# 	def init_weights(self,m):
# 		if isinstance(m, nn.Linear):
# 			torch.nn.init.uniform_(m.weight,-0.00001,0.00001)
# 			m.bias.data.fill_(0.001)



# with open("data/rf_test_data.pkl", "rb") as f:
# 	X,Y = pickle.load(f)

# print("dataset loaded")

X = np.array(X)
Y = np.array(Y)



pca = PCA(n_components=2)
pca_fit = pca.fit(X,Y)
check_x = pca.transform(X)

Y_ones = Y == 1
Y_zeros = Y == 0

print("Sum Y ones", sum(Y_ones))
print("Sum Y zeros", sum(Y_zeros))

plt.scatter(check_x[Y_ones][:,0],check_x[Y_ones][:,1], s=20, c='yellow')
plt.scatter(check_x[Y_zeros][:,0],check_x[Y_zeros][:,1], s=	10, c='blue')

plt.savefig("output/state_vectors.png")


print(np.sum(Y == 1), np.sum(Y == 0))
# print(np.array(X).shape)
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25)
# clf = RandomForestClassifier(max_depth = 10,n_estimators=5)
clf = RandomForestRegressor(max_depth = 10, n_estimators = 5)
print("Starting Training")
# # clf = svm.LinearSVC(dual=False,verbose=1,max_iter=100)
print(np.sum(y_train == 1), np.sum(y_train == 0))
clf.fit(X_train, y_train)
# clf.fit(X, Y)


# device = "cuda"
# agent_model = Agent_model(X_train.shape[1],1).to(device)
# optimizer = optim.SGD(agent_model.parameters(), lr = 0.00000001) #Try Adam
# optimizer.zero_grad()
# criterion = nn.CrossEntropyLoss()
# for i in range(500) :
# 	# print(type(X_train), X_train.shape,X_train.shape[0])
# 	batch_indices = np.random.choice( np.array(range(X_train.shape[0])) ,size=100,replace=False)
# 	X_train_batch = torch.tensor( X_train[batch_indices, :],dtype=torch.float32).to(device)
# 	y_train_batch = torch.tensor(y_train[batch_indices]).to(device)

# 	optimizer.zero_grad()
# 	# loss = criterion(agent_model(X_train), y_train)
# 	qvals = agent_model(X_train_batch)
# 	loss = torch.sum((qvals - y_train_batch)**2)
# 	loss.backward()
# 	optimizer.step()
# 	if i%1 == 0 :
# 		# print(torch.sum(qvals))
# 		print(loss)
	

# with open("data/rf_test_model.pkl", "wb") as f :
#     pickle.dump(clf,f)

# # print("Dumped Model")

# # with open("data/rf_test_model.pkl", "rb") as f :
# #     clf = pickle.load(f)

y_test_pred = (clf.predict(X_test))
y_test_pred = y_test_pred > 0.5
# X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test_pred = agent_model(X_test).cpu().detach().numpy()

# print( "Main File Test accuracy : ", np.sum((y_test == y_test_pred))/ len(y_test))
print("Starting Test")
print(np.sum(y_test == 1), np.sum(y_test == 0))
print(classification_report(y_test,y_test_pred))
# # print(sum(y_test_pred), len(y_test_pred))
# # print(sum(y_test), len(y_test))
# # print(np.sum(  (y_test == 1) == y_test_pred ))

# # y_ones = y_test == 1

# # y_test_ones = y_test[y_ones]
# # X_test_ones = X_test[y_ones]

# # y_ones_pred = clf.predict(X_test_ones)
# # print( np.sum(y_test_ones ==y_ones_pred)/ len(y_test_ones))


# # ----------------Different File test Phase ------------------------------------

print("Starting test Phase")


scraped_tactics = dataloader.scraped_tactics_from_file("/home/avarghese_umass_edu/work/rl/proverbot9001/CompCert/lib/Parmov.v.scrape", "all", 30, None)
predictor = loadPredictorByFile(Path2("data/polyarg-weights.dat"))
tactic_space_model = fasttext.load_model("fast_text_model_on_Globalenvs.bin")

# Deleted the contents so that I dont accidentally retrain -----vvvvv

# Deleted above block ----------^^^^^^^^^^^^^^^^^

proof_start = False

state_vectors = []
correct_action_vectors = []
wrong_action_vectors = []


index = 0
num_zero_vectors = 0
for tactic in tqdm.tqdm(scraped_tactics) : 
    index += 1
    # print(index, tactic.tactic.strip())
    if proof_start :
        if tactic.tactic.lstrip().rstrip().lower() == "qed." :
            proof_start = False
        elif  len(tactic.context.fg_goals) != 0 :
            current_context = ProofContext(tactic.context.fg_goals,tactic.context.bg_goals,tactic.context.shelved_goals,tactic.context.given_up_goals)
            action = tactic.tactic.lstrip().rstrip().rstrip(".")
            state = tactic.context.fg_goals[0].goal.lstrip().rstrip()
            state_vector = get_state_vector_from_text(state, language_model, device, state_model)
            tactic_class,tactic_args = split_tactic(action)
            # tactic_class_vec = np.eye(len(list_of_tactic_classes), 0, list_of_tactic_classes.index(tactic_class)).flatten()
            tactic_class_vec = np.zeros(len(list_of_tactic_classes) + 1)
            if tactic_class in list_of_tactic_classes :
                tactic_class_vec[ list_of_tactic_classes.index(tactic_class) ] = 1
            else :
                tactic_class_vec[-1] = 1

            if tactic_args.strip() != "" and is_hyp_token(tactic_args,current_context.fg_goals[0]) :
                index = get_indexed_vars_dict(current_context.fg_goals[0].hypotheses)[tactic_args]
                tactic_args_type = get_hyp_type(current_context.fg_goals[0].hypotheses[index])
                tactic_args_type_vec = get_state_vector_from_text(tactic_args_type) # tactic_space_model.get_word_vector(tactic_args)
                tactic_args_vec =  tactic_space_model.get_word_vector(tactic_args)
            else :
                num_zero_vectors += 1
                tactic_args_type_vec = np.zeros(shape = state_vector.shape)
                tactic_args_vec = tactic_space_model.get_word_vector(tactic_args)

            final_action_vec = np.concatenate((tactic_class_vec, tactic_args_type_vec, tactic_args_vec))

            correct_action_vectors.append(final_action_vec)
            state_vectors.append(state_vector)

            curr_wrong = []
            full_context_before = FullContext(tactic.relevant_lemmas, tactic.prev_tactics,  current_context)
            predictions = predictor.predictKTactics(
                truncate_tactic_context(full_context_before.as_tcontext(),
                                        256), 10)
			
            for prediction_idx, prediction in enumerate(predictions):
                curr_pred = prediction.prediction.lstrip().rstrip()
                if curr_pred != action :
                    # print(curr_pred)
                    tactic_class,tactic_args = split_tactic(curr_pred.lstrip().rstrip().rstrip("."))
                    # tactic_class_vec = np.eye(len(list_of_tactic_classes), 0, list_of_tactic_classes.index(tactic_class)).flatten()
                    tactic_class_vec = np.zeros(len(list_of_tactic_classes) + 1)
                    if tactic_class in list_of_tactic_classes :
                        tactic_class_vec[ list_of_tactic_classes.index(tactic_class) ] = 1
                    else :
                        tactic_class_vec[-1] = 1

                    if tactic_args.strip() != "" and is_hyp_token(tactic_args,current_context.fg_goals[0]) :
                        # print(current_context.fg_goals[0].hypotheses)
                        # print(get_indexed_vars_dict(current_context.fg_goals[0].hypotheses))
                        tactic_args = tactic_args.strip()
                        index = get_indexed_vars_dict(current_context.fg_goals[0].hypotheses)[tactic_args]
                        tactic_args_type = get_hyp_type(current_context.fg_goals[0].hypotheses[index])
                        tactic_args_type_vec = get_state_vector_from_text(tactic_args_type) # tactic_space_model.get_word_vector(tactic_args)
                        tactic_args_vec =  tactic_space_model.get_word_vector(tactic_args)
                    else :
                        # print("Nope", tactic_args)
                        num_zero_vectors += 1
                        tactic_args_type_vec = np.zeros(shape = state_vector.shape)
                        tactic_args_vec = tactic_space_model.get_word_vector(tactic_args)

                    final_action_vec = np.concatenate((tactic_class_vec, tactic_args_type_vec, tactic_args_vec))
                    curr_wrong.append(final_action_vec)
            wrong_action_vectors.append(curr_wrong)
    else :
        if  tactic.tactic.lstrip().rstrip().lower() == "proof."   :
            proof_start = True
            




with open("data/deleteme.pkl","wb") as f:
    pickle.dump((state_vectors, correct_action_vectors,wrong_action_vectors),f)


with open("data/deleteme.pkl", "rb") as f:
	(state_vectors, correct_action_vectors,wrong_action_vectors) = pickle.load(f)


# with open("data/rf_test_model.pkl", "rb") as f :
#     clf = pickle.load(f)

correct = 0
wrong = 0
total = 0
y_test= []
y_test_pred= []
correct_ones = 0
correct_zeros = 0

for i in tqdm.tqdm(range(len(state_vectors) )) :
	considered_state_vec  = np.concatenate((state_vectors[i],correct_action_vectors[i]))
	y = clf.predict([considered_state_vec])[0]
	y_test.append(1)
	y_test_pred.append(y)
	if  y == 0 :
		wrong += 1
	else :
		correct += 1
		correct_ones += 1

	total += 1
	for wrong_action_vec in wrong_action_vectors[i] :    
		comparitive_state_vec = np.concatenate((state_vectors[i],wrong_action_vec))
		y = clf.predict([comparitive_state_vec])[0]
		total+= 1
		y_test.append(0)
		y_test_pred.append(y)
		if  y == 0 :
			correct += 1
			correct_zeros += 1
		else :
			wrong += 1
		# X.append(np.random.normal(considered_state_vec,0.000025427588, size = considered_state_vec.shape ))
		# Y.append(1)
		# wandb.log({"Size of list" : np.array(X).nbytes/1024**2})

print(correct, wrong)
print(correct_ones, correct_zeros)
print(correct + wrong, len(state_vectors), total)
print(correct/ (correct+ wrong))

y_test_pred = np.array(y_test_pred)
y_test_pred = y_test_pred > 0.5
print(classification_report(y_test,y_test_pred))