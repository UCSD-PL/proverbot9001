import torch

dir_name = "/work/pi_brun_umass_edu/efirst_model_output"
save_file_name = "gpt2_ps_nobraces_b4" # "gpt2-medium_eos_b4"
checkpoint_num =  290000 # 29900
checkpoint = "checkpoint-" + str(checkpoint_num) 
top_k = 0
top_p = 0.8
temperature = 0.5
num_beams = 5
max_length = 32 # 512
no_repeat_ngram_size = 2
repetition_penalty = 1.0


decoding_type = ["beam_search"] # other options: sampling, greedy, beam_search
prompt_type = ["ps"] # ["proof"] # other options: ps, rel_lemmas, rel_lemma_names, ps_proof, proof

with open(save_file_name + ".dat", 'wb') as f:
    torch.save(("transformer",(dir_name + "/" + save_file_name + "/" + checkpoint,
                                prompt_type, decoding_type, temperature, top_k, top_p, 
                                num_beams, max_length, no_repeat_ngram_size, repetition_penalty)),f)

