import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

from data import (ListDataset, RawDataset,
                  EOS_token)
from util import (eprint, maybe_cuda, LongTensor, FloatTensor,
                  ByteTensor, print_time, unwrap)
import util
import math
from coq_serapy.contexts import TacticContext
from models.components import (WordFeaturesEncoder, Embedding,
                               DNNClassifier, EncoderDNN, EncoderRNN,
                               add_nn_args)
from models.tactic_predictor import (TacticPredictor, TrainablePredictor,
                                     NeuralPredictorState, Prediction,
                                     optimize_checkpoints, add_tokenizer_args)
import dataloader
from dataloader import (features_polyarg_tensors,
                        features_polyarg_tensors_with_meta,
                        sample_fpa,
                        sample_fpa_batch,
                        decode_fpa_result,
                        encode_fpa_stem,
                        encode_fpa_arg,
                        decode_fpa_stem,
                        # decode_fpa_arg,
                        # features_vocab_sizes,
                        get_num_tokens,
                        get_num_indices,
                        get_word_feature_vocab_sizes,
                        get_vec_features_size,
                        DataloaderArgs,
                        get_fpa_words)

import coq_serapy as serapi_instance

import argparse
import sys
from argparse import Namespace
from typing import (List, Tuple, NamedTuple, Optional, Sequence, Dict,
                    cast, Union, Set, Type, Any, Iterable)

from enum import Enum, auto

def only_lemma_name(hyp: str):
    var_term = hyp.split(":")[0].strip()
    return var_term

class TransformerPredictor(TacticPredictor):
    def __init__(self):
        self.training_args: Optional[argparse.Namespace] = None
        self.unparsed_args : List[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.tokenizer = None  # for example, "/work/pi_brun_umass_edu/efirst_model_output/gpt2-large_eos_b2/checkpoint-26700"
        self._model = None
        self.max_input_length = 1024
    def getOptions(self) -> List[Tuple[str, str]]:
        return [("predictor", "transformer")]
    def load_saved_state(self,model_name_or_path : str, 
                            prompt_type, decoding_type, 
                            temperature, top_k, top_p, 
                            num_beams : int, max_length : int, 
                            no_repeat_ngram_size: int,
                            repetition_penalty) -> None:
        self.tokenizer = self.tokenizer_class.from_pretrained(model_name_or_path)
        model = self.model_class.from_pretrained(model_name_or_path)
        self._model = model
        self._model.to(self.device)
        self.prompt_type = prompt_type
        self.decoding_type = decoding_type
        self.num_beams = num_beams
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self.temperature= temperature
        self.top_k = top_k
        self.top_p = top_p
    def build_prompt(self, context: TacticContext) -> str:
        lemma_statement = context.prev_tactics[0]
        prompt = ""
        if "proof" in self.prompt_type or "ps_proof" in self.prompt_type:
            prompt = "<<THEOREM>>" + lemma_statement.strip() + "\n" + "<<PROOF>>" # don't predict "Proof." (?)
            if "rel_lemmas" in self.prompt_type:
                lemmas = "\n".join(context.relevant_lemmas[-3:])
                prompt = "<<LEMMAS>>" + lemmas + "\n" + prompt
            if "rel_lemmas_name" in self.prompt_type:
                names = "\n".join([only_lemma_names(l) for l in context.relevant_lemmas[-3:]])
                prompt = "<<LEMMAS>>" + names + "\n" + prompt
        if "ps" in self.prompt_type:
            lemmas = "\n".join(context.relevant_lemmas[-3:])
            hyps = "\n".join(context.hypotheses)
            prevs = "\n".join([l.strip() for l in context.prev_tactics])
            prompt = "<<LEMMAS>>" + lemmas + "\n" + "<<HYPS>>" + hyps + "\n" + "<<GOAL>>" + context.goal + "\n" + "<<PREV>>" + prevs + "\n" + "<<STEP>>"
        return prompt

    def predictKTactics(self, context: TacticContext, k: int) -> List[Prediction]:
        prompt_text = self.build_prompt(context)
        print("PROMPT")
        print(prompt_text)
        start_idx = 0
        tok_prompt = self.tokenizer.tokenize(prompt_text)
        l = len(tok_prompt)
        if l > (self.max_input_length-self.max_length):
            start_idx = l - self.max_input_length + self.max_length
        encoded_prompt = self.tokenizer.encode(tok_prompt[start_idx:], add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = None

        if "greedy" in self.decoding_type:
            output_sequences = self._model.generate(
                input_ids=input_ids,
                max_length=(self.max_length + len(encoded_prompt[0]))
            )

        if "beam_search" in self.decoding_type:
            output_sequences = self._model.generate(
                input_ids=input_ids,
                max_length=(self.max_length + len(encoded_prompt[0])),
                num_beams=k,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                num_return_sequences=k,
                early_stopping=True
            )

        if "sampling" in self.decoding_type:
            output_sequences = self._model.generate(
                input_ids=input_ids,
                max_length=(self.max_length + len(encoded_prompt[0])),
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                num_return_sequences=k # number of samples
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            idx = text.find("<|endoftext|>") # don't include end of text token 
            if idx != -1:
                text = text[: idx]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                text[len(self.tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            print(total_sequence)
        
        return [Prediction(s, 1.0) for s in generated_sequences]

        '''
        assert self.training_args
        assert self._model

        with torch.no_grad():
            all_predictions = self.getAllPredictionIdxs(context)

        predictions = self.decodeNonDuplicatePredictions(
            context, all_predictions, k)

        return predictions
        '''

    def getOptions(self) -> List[Tuple[str, str]]: pass

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int, correct : str) -> \
        Tuple[List[Prediction], float]: pass
    def predictKTacticsWithLoss_batch(self,
                                      in_data : List[TacticContext],
                                      k : int, correct : List[str]) -> \
                                      Tuple[List[List[Prediction]], float]: pass
