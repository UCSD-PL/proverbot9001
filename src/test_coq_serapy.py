from collections import deque
import traceback
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
from coqproofenv import ProofEnv
import tqdm




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


proof_file = args.proof_file.path
commands = load_commands(proof_file, progress_bar=True)
coq = serapi_instance.SerapiInstance(['sertop', '--implicit'],None, prelude = args.prelude)
coq.verbose = 0
coq.quiet = True


# ----------------------------------------- Test 1 -------------------------------------------
def is_same_context(context1, context2) :
	print("Context Surjectives", end = " ")
	print(contextSurjective(context2, context1), end = " | ")
	print(contextSurjective(context1, context2))
	return contextSurjective(context1, context2) and contextSurjective(context2, context1)

def clear_coq_proof_context(coq) :
		while coq.proof_context != None :
			coq.cancel_last()

def is_context_fresh(proof_contexts_in_path, curr_proof_context) :
		for context in proof_contexts_in_path :
			print(contextSurjective(curr_proof_context, context), end = " ")
			if contextSurjective(curr_proof_context, context) :
				print("End")
				return False
		print("End")
		return True


# test_include = "Lemma store_init_data_list_aligned:\n  forall b il m p m',\n  store_init_data_list ge m b p il = Some m' ->\n  init_data_list_aligned p il"
# new_commands = ['Proof.', 'induction b.', '{', 'simpl.', 'try congruence.', 'intros.', 'exploit store_init_data_list_unchanged.', '{', 'eexact H.', '}',
			# '{', 'simpl.', 'eauto.', 'intros.', 'simpl.', 'red.', 'eauto.', 'eauto.', 'eauto.', 'eauto.', 'intros.', 'inv H.','simpl in H3.']
# test_include = "Remark store_zeros_unchanged:\n  forall"
# new_commands = ['Proof.', 'induction m.', 'intros.', 'exploit store_zeros_nextblock.', '{', 'eauto.']
# test_include = "Theorem find_invert_symbol:\n  forall ge id b,\n  find_symbol ge id = Some b -> invert_symbol ge b = Some id."
# new_commands = [ 'Proof.', 'clear.', 'induction ge.', 'destruct genv_public0.', '{', 'intuition.', 'exploit genv_vars_inj0.', '{','eauto.']
# test_include = """Lemma outside_interval_diff:
#     forall l l', lt l' (diff_low_bound l) \/ lt (diff_high_bound l) l' -> Loc.diff l l'.
# """
# new_commands_before = ['Proof.', "induction l'.", '{', 'intros.', 'red.', 'destruct r.', '{', 'destruct H.', '{', 'destruct l.', '{', 'red.', 'intros.','Abort.']
# new_commands = [ 'Proof.', 'intros.', "destruct l as [mr | sl ofs ty]; destruct l' as [mr' | sl' ofs' ty']; simpl in *; auto.", '-', "assert (IndexedMreg.index mr <> IndexedMreg.index mr').", '{', 'destruct H.', 'apply not_eq_sym.', 'apply Plt_ne; auto.', 'apply Plt_ne; auto.', '}', 'congruence.', '-', 'assert (RANGE: forall ty, 1 <= typesize ty <= 2).', '{', 'intros; unfold typesize.','destruct ty0; omega.','}']
# test_include = "Lemma classify_binarith_arithmetic_conversion:"
# new_commands = ['Proof.', 'destruct t1; destruct t2; try reflexivity.', '-',' destruct it; destruct it0; reflexivity.']

# test_include = "Remark store_zeros_unchanged:"
# new_commands = ['Proof.', 'simpl.', 'intros.','exploit store_init_data_perm.', '{'] + ['exploit store_init_data_perm.', '{']*27

# ["Let HF': helper_functions_declared tprog hf.", 'Proof.', 'eapply helper_functions_preserved.', 'apply get_helpers_correct.']
test_include = 'Theorem find_funct_inv:\n  forall ge v f,\n  find_funct ge v = Some f -> exists b, v = Vptr b Ptrofs.zero.'
new_commands = ['Proof.', 'induction v.', '{', 'simpl.', 'try congruence.', '}', '{', 'simpl _.', 'intros.', 'inv H.', '}', '{', 'intros.', 'econstructor.', 'try discriminate.', '}', '{', 'unfold find_funct.', 'zify.', 'inv H.', '}', '{', 'intros.', 'simpl in H.', 'destruct f.', '{', 'discriminate.', '}', '{', 'destruct s.', '{', 'discriminate.', '}', 'discriminate.', '}', '{', 'destruct pl.', '{', 'discriminate.', '}', '{', 'discriminate.', '}', 'destruct ge.', 'destruct genv_next0.', '{', 'discriminate.', '}', '{', 'destruct genv_next0.', '{', 'discriminate.', '}', '{', 'discriminate.', '}', 'discriminate.', '}', 'discriminate.', '}', 'discriminate.', '}', 'simpl _.', 'destruct Ptrofs.eq_dec.', '{', 'intros.', 'subst.', 'econstructor.', 'eauto.', '}', 'try discriminate.', "}"]

print(test_include in commands)
start_time = time.time()
for command in commands :
	# print(test_include, command, test_include in command)
	if not test_include in command :
		coq.run_stmt(command, timeout= 100)
	else :
		print("Starting that particular test")
		coq.verbose = 0
		print("Running -", command)
		coq.run_stmt(command, timeout= 100)
		# quit()
		
		contexts = []
		# for new_command in new_commands_before :
		#     # print(coq.proof_context)
		#     coq.run_stmt(new_command.lstrip().rstrip(), timeout= 100)
		#     contexts.append("Running - " + new_command.lstrip().rstrip())
		#     contexts.append(coq.proof_context)
		
		
		print(coq.proof_context)
		# contexts.append(coq.proof_context)
		
		clear_coq_proof_context(coq)
		print(coq.proof_context)
		# contexts.append(coq.proof_context)
		coq.run_stmt(command, timeout= 100)
		
		coq.run_stmt("Proof.",timeout=100)
		contexts.append(coq.proof_context)
		try :
			for new_command_ind in range(1,len( new_commands)) :
				new_command = new_commands[new_command_ind]
				# print(coq.proof_context)

				print("Running - " + new_command.lstrip().rstrip())
				print(completed_proof(coq))
				contexts.append("Running - " + new_command.lstrip().rstrip())
				coq.run_stmt(new_command.lstrip().rstrip(), timeout= 100)
				# print("Is context Fresh after the above command? -  ",is_context_fresh(contexts, coq.proof_context))
				contexts.append(coq.proof_context)
		except Exception:
			print(traceback.format_exc())
			contexts.append(traceback.format_exc())
			
		
		# print(coq.proof_context)
		# coq.cancel_last()
		# contexts.append(coq.proof_context)
		# print(is_same_context(contexts[-1], contexts[-3]))
		# print(is_same_context(contexts[-1], contexts[-5]))
		# print(is_same_context(contexts[-1], contexts[-7]))
		# print(is_same_context(contexts[-3], contexts[-5]))
		# print(is_same_context(contexts[-3], contexts[-7]))
		# print(is_same_context(contexts[-5], contexts[-7]))

		# print("\n Everything should be false below this")
		# for i in contexts :
		# 	print( is_same_context(coq.proof_context, i), contextSurjective(coq.proof_context, i) )
		# print("History is ->",  " ".join(new_commands))
		# for obligation in contexts[-1].all_goals :
		#     print(obligation.goal)
		
		# print("History is -> {, eauto")
		# for obligation in contexts[-2].all_goals :
		#     print(obligation.goal)
		
		# print("History is -> { ")
		# for obligation in contexts[-3].all_goals :
		#     print(obligation.goal)
		
		with open("output/output_test_context_file.txt", "w") as f :
			for context in contexts :
				f.write(str(context) + "\n")
		
		end_time = time.time()
		print(end_time - start_time)
		quit()
	

# ---------------------------------------------- Test 2 ----------------------------------------------------------------------

# a = time.time()
# for command in commands :
#     coq.run_stmt(command, timeout= 100)
# b = time.time()

# print(b - a)


#----------------------------------------------- Test 3 ---------------------------------------------------------------------
# test_include = "Lemma store_init_data_neutral:"
# new_commands = ['Proof.', 'intros.', 'destruct H.', ' eapply store_zeros_neutral.', '{'] + [' eapply store_zeros_neutral.', '{']*27
# time_dict = defaultdict(lambda : [])
# pred_gen_time = []
# context_check_time = []
# predictor = loadPredictorByFile(args.weightsfile)
# for command in commands :
# 	# print(test_include, command, test_include in command)
# 	if not test_include in command :
# 		coq.run_stmt(command, timeout= 100)
# 	else :
# 		print("Starting test")
# 		coq.run_stmt(command)
# 		contexts = []
# 		coq.verbose = 0
# 		time_start = time.time()
# 		for new_command_ind in range(len(new_commands) - 1) :
# 			new_command = new_commands[new_command_ind]
# 			if new_commands[new_command_ind + 1] == '{' :
# 				print("Running ", new_command)
# 				a= time.time()
# 				coq.run_stmt(new_command)
# 				b = time.time()
# 				time_dict['{'].append(b - a)
# 				continue

# 			print("Running ", new_command)
# 			a = time.time()
# 			coq.run_stmt(new_command)
# 			b = time.time()
# 			time_dict[new_command].append(b - a)
			
# 			a = time.time()
# 			relevant_lemmas = coq.local_lemmas[:-1]
# 			full_context_before = FullContext(relevant_lemmas, coq.prev_tactics,  coq.proof_context)
# 			predictions = predictor.predictKTactics(
# 				truncate_tactic_context(full_context_before.as_tcontext(),
# 										args.max_term_length), args.max_attempts)
# 			b = time.time()
# 			pred_gen_time.append(b-a)
# 			for prediction_idx, prediction in enumerate(predictions):
# 				curr_pred = prediction.prediction.strip()
# 				a = time.time()
# 				try :
# 					print("trying",curr_pred)
					
# 					coq.run_stmt(curr_pred, timeout=100)
# 					coq.cancel_last()
					
					
# 				except (serapi_instance.TimeoutError, serapi_instance.ParseError,
# 					serapi_instance.CoqExn, serapi_instance.OverflowError,
# 					serapi_instance.ParseError,
# 					RecursionError,
# 					serapi_instance.UnrecognizedError) as e:
# 					print("One of known errors",)
# 				b = time.time()
# 				time_dict[curr_pred].append(b - a)
			
# 			a = time.time()
# 			print("Is context Fresh after the above command",new_command, "?", is_context_fresh(contexts, coq.proof_context))
# 			b = time.time()
# 			context_check_time.append(b-a)
# 			contexts.append(coq.proof_context)
		
# 		break

# time_stop = time.time()
# total_sum = 0
# for i in time_dict :
# 	total_sum +=  np.sum(time_dict[i])
# 	print("%s : %.4f, %.4f, %.4f, %d, %.4f"%(i, np.mean(time_dict[i]), np.std(time_dict[i]), np.max(time_dict[i]), len(time_dict[i]), np.sum(time_dict[i])))

# print("Pred Gen time", np.mean(pred_gen_time), np.std(pred_gen_time),np.max(pred_gen_time),len(pred_gen_time),np.sum(pred_gen_time))
# print("Time taken to run the test : ",time_stop - time_start)
# print("Time Taken for running all the tactics :", total_sum)
# print("Time Taken to check all contexts :", np.sum(context_check_time))
# quit()









# -------------------------------------------- Storage -----------------------------------------------------------------------
hypotheses=['H0 : forall (i : Z) (_ : and (Z.le p i) (Z.lt i (Z.add p n))), not (P b i)',
"H : eq\n  (store_zeros\n     {|\n     Mem.mem_contents := mem_contents;\n     Mem.mem_access := mem_access;\n     Mem.nextblock := nextblock;\n     Mem.access_max := access_max;\n     Mem.nextblock_noaccess := nextblock_noaccess;\n     Mem.contents_default := contents_default |} b p n) \n  (Some m')", 
"m' : Mem\\.mem", 
'p,n : Z', 
'b : block', 
'contents_default : forall b : positive, eq (fst (PMap.get b mem_contents)) Undef', 
'nextblock_noaccess : forall (b : positive) (ofs : Z) (k : perm_kind)\n  (_ : not (Plt b nextblock)), eq (PMap.get b mem_access ofs k) None', 
"access_max : forall (b : positive) (ofs : Z),\nMem.perm_order'' (PMap.get b mem_access ofs Max)\n  (PMap.get b mem_access ofs Cur)", 
'nextblock : block', 
'mem_access : PMap.t (forall (_ : Z) (_ : perm_kind), option permission)', 
'mem_contents : PMap.t (ZMap.t memval)', 
'P : forall (_ : block) (_ : Z), Prop', 
'ge : True', 
'V : Type', 
'F : Type'], 
goal='eq (store_zeros ?Goal ?Goal0 ?Goal1 ?Goal2) (Some ?Goal3)'

hypotheses=['H0 : forall (i : Z) (_ : and (Z.le p i) (Z.lt i (Z.add p n))), not (P b i)', 
"H : eq\n  (store_zeros\n     {|\n     Mem.mem_contents := mem_contents;\n     Mem.mem_access := mem_access;\n     Mem.nextblock := nextblock;\n     Mem.access_max := access_max;\n     Mem.nextblock_noaccess := nextblock_noaccess;\n     Mem.contents_default := contents_default |} b p n) \n  (Some m')", 
"m' : Mem\\.mem", 
'p,n : Z', 
'b : block', 
'contents_default : forall b : positive, eq (fst (PMap.get b mem_contents)) Undef', 
'nextblock_noaccess : forall (b : positive) (ofs : Z) (k : perm_kind)\n  (_ : not (Plt b nextblock)), eq (PMap.get b mem_access ofs k) None', 
"access_max : forall (b : positive) (ofs : Z),\nMem.perm_order'' (PMap.get b mem_access ofs Max)\n  (PMap.get b mem_access ofs Cur)", 
'nextblock : block', 
'mem_access : PMap.t (forall (_ : Z) (_ : perm_kind), option permission)', 
'mem_contents : PMap.t (ZMap.t memval)', 
'P : forall (_ : block) (_ : Z), Prop', 
'ge : True', 
'V : Type', 
'F : Type'], 
goal="""forall _ : eq (Mem.nextblock ?Goal3) (Mem.nextblock ?Goal),\nMem.unchanged_on P\n  
{|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  
Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"""

















hypotheses=['H0 : forall (i : Z) (_ : and (Z.le p i) (Z.lt i (Z.add p n))), not (P b i)', 
"H : eq\n  (store_zeros\n     {|\n     Mem.mem_contents := mem_contents;\n     Mem.mem_access := mem_access;\n     Mem.nextblock := nextblock;\n     Mem.access_max := access_max;\n     Mem.nextblock_noaccess := nextblock_noaccess;\n     Mem.contents_default := contents_default |} b p n) \n  (Some m')", 
"m' : Mem\\.mem", 
'p,n : Z', 
'b : block', 
'contents_default : forall b : positive, eq (fst (PMap.get b mem_contents)) Undef', 
'nextblock_noaccess : forall (b : positive) (ofs : Z) (k : perm_kind)\n  (_ : not (Plt b nextblock)), eq (PMap.get b mem_access ofs k) None', 
"access_max : forall (b : positive) (ofs : Z),\nMem.perm_order'' (PMap.get b mem_access ofs Max)\n  (PMap.get b mem_access ofs Cur)", 
'nextblock : block', 'mem_access : PMap.t (forall (_ : Z) (_ : perm_kind), option permission)', 
'mem_contents : PMap.t (ZMap.t memval)', 
'P : forall (_ : block) (_ : Z), Prop', 
'ge : True', 
'V : Type', 
'F : Type'], 
goal="""forall\n  _ : eq (Mem.nextblock m')\n        
(Mem.nextblock\n           {|\n           Mem.mem_contents := mem_contents;\n          
 Mem.mem_access := mem_access;\n           Mem.nextblock := nextblock;\n           
 Mem.access_max := access_max;\n           Mem.nextblock_noaccess := nextblock_noaccess;\n           
 Mem.contents_default := contents_default |}),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  
 Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  
 Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"""

hypotheses=['H0 : forall (i : Z) (_ : and (Z.le p i) (Z.lt i (Z.add p n))), not (P b i)', 
"H : eq\n  (store_zeros\n     {|\n     Mem.mem_contents := mem_contents;\n     Mem.mem_access := mem_access;\n     Mem.nextblock := nextblock;\n     Mem.access_max := access_max;\n     Mem.nextblock_noaccess := nextblock_noaccess;\n     Mem.contents_default := contents_default |} b p n) \n  (Some m')", 
"m' : Mem\\.mem", 
'p,n : Z', 
'b : block', 
'contents_default : forall b : positive, eq (fst (PMap.get b mem_contents)) Undef', 
'nextblock_noaccess : forall (b : positive) (ofs : Z) (k : perm_kind)\n  (_ : not (Plt b nextblock)), eq (PMap.get b mem_access ofs k) None', 
"access_max : forall (b : positive) (ofs : Z),\nMem.perm_order'' (PMap.get b mem_access ofs Max)\n  (PMap.get b mem_access ofs Cur)", 
'nextblock : block', 'mem_access : PMap.t (forall (_ : Z) (_ : perm_kind), option permission)', 
'mem_contents : PMap.t (ZMap.t memval)', 
'P : forall (_ : block) (_ : Z), Prop', 
'ge : True',
'V : Type',
'F : Type'], 
goal='eq (store_zeros ?Goal ?Goal0 ?Goal1 ?Goal2) (Some ?Goal3)'




goal='eq (store_zeros ?Goal ?Goal0 ?Goal1 ?Goal2) (Some ?Goal3)'
goal="forall _ : eq (Mem.nextblock ?Goal3) (Mem.nextblock ?Goal),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"

goal="forall\n  _ : eq (Mem.nextblock m')\n        (Mem.nextblock\n           {|\n           Mem.mem_contents := mem_contents;\n           Mem.mem_access := mem_access;\n           Mem.nextblock := nextblock;\n           Mem.access_max := access_max;\n           Mem.nextblock_noaccess := nextblock_noaccess;\n           Mem.contents_default := contents_default |}),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"
goal="forall _ : eq (Mem.nextblock ?Goal3) (Mem.nextblock ?Goal),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"

goal='Mem.mem'
goal="forall _ : eq (Mem.nextblock ?Goal3) (Mem.nextblock ?Goal),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"

goal='block'
goal="forall _ : eq (Mem.nextblock ?Goal3) (Mem.nextblock ?Goal),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"

goal='Z'
goal="forall _ : eq (Mem.nextblock ?Goal3) (Mem.nextblock ?Goal),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"

goal='Z'
goal="forall _ : eq (Mem.nextblock ?Goal3) (Mem.nextblock ?Goal),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"

goal='Mem.mem'
goal="forall _ : eq (Mem.nextblock ?Goal3) (Mem.nextblock ?Goal),\nMem.unchanged_on P\n  {|\n  Mem.mem_contents := mem_contents;\n  Mem.mem_access := mem_access;\n  Mem.nextblock := nextblock;\n  Mem.access_max := access_max;\n  Mem.nextblock_noaccess := nextblock_noaccess;\n  Mem.contents_default := contents_default |} m'"
