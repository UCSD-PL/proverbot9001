#!/usr/bin/env python3.7

import random
import multiprocessing
import functools
import sys
import collections
from argparse import Namespace

from util import *
from data import Sentence, SOS_token, EOS_token, TOKEN_START, get_text_data, RawDataset
from data import getNGramTokenbagVector
from models.tactic_predictor import TacticPredictor, Prediction, TacticContext
from models.components import SimpleEmbedding
from tokenizer import Tokenizer, tokenizers, make_keyword_tokenizer_relevance
from models.args import start_std_args, optimizers
from models.components import DNNClassifier, EncoderDNN, DecoderGRU
import serapi_instance

from typing import List, Dict, Tuple, NamedTuple, Union, Callable, \
    Optional, Iterator, Counter
from typing import cast

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import Variable

class TacticStructure(NamedTuple):
    stem_idx : int
    hyp_idxs : List[int]

NGram = List[int]
EncodedTerm = NGram
StructDataset = List[Tuple[List[EncodedTerm], EncodedTerm, TacticStructure]]

class HypArgPredictor(TacticPredictor):
    def load_saved_state(self, filename : str) -> None:
        checkpoint = torch.load(filename)

        self.options = checkpoint['options']
        self.max_args = checkpoint['max-args']
        self.hidden_size = checkpoint['hidden-size']

        self.tokenizer = checkpoint['tokenizer']
        self.embedding = checkpoint['embedding']

        self.term_encoder = functools.partial(getNGramTokenbagVector, 1,
                                              self.tokenizer.numTokens())
        self.arg_decoder = checkpoint['arg-decoder']
        self.stem_decoder = checkpoint['stem-decoder']
        self.initial_encoder = checkpoint['initial-encoder']
        pass
    def __init__(self, options : Dict[str, Any]) -> None:
        self.load_saved_state(cast(str, options["filename"]))
        pass
    def predictOneTactic(self, in_data : TacticContext) \
        -> Prediction:

        # Size: (1, self.features_size)
        general_features : torch.FloatTensor = self.encode_general_context(in_data)
        # Size: (1, num_hypotheses, self.features_size)
        hypothesis_features : torch.FloatTensor = \
            self.encode_hypotheses(in_data.hypotheses)
        # Size: (1, num_hypotheses)
        stem_distribution : torch.FloatTensor = self.predict_stem(general_features)
        # Size(stem): (1, 1)
        # Size(probability): (1, 1)
        stem, probability = stem_distribution.topk(1)[0] # type: int, float
        # Size: (1, self.hidden_size)
        hidden_state : torch.FloatTensor = self.initial_hidden(general_features, stem)
        # Size: (1, 1)
        decoder_input : torch.LongTensor = self.initInput()
        arg_idxs : List[int] = []
        for idx in range(self.max_args):
            # Size(arg_distribution): (1, num_hypotheses)
            # Size(hidden_state): (1, self.hidden_size)
            arg_distribution, hidden_state = self.decode_arg(decoder_input, hidden_state,
                                                             hypothesis_features)
            # Size(decoder_input): (1, 1)
            # Size(next_probability): (1, 1)
            decoder_input, next_probability = [lst[0] for lst in arg_distribution.topk(1)]
            if decoder_input.item() == 0:
                break
            probability *= next_probability
            arg_idxs.append(decoder_input.item())

        result_struct = TacticStructure(stem_idx=stem, hyp_idxs=arg_idxs)

        return Prediction(decode_tactic_structure(self.embedding, result_struct,
                                                  in_data.hypotheses),
                          probability)

    def predictKTactics(self, in_data : TacticContext, k : int) \
        -> List[Prediction]:
        tactic, probability = self.predictOneTactic(in_data)
        return [Prediction(tactic, probability)] * k

    def predictKTacticsWithLoss(self, in_data : TacticContext, k : int,
                                correct : str) -> Tuple[List[Prediction], float]:
        return self.predictKTactics(in_data, k), 1.0
    def encode_general_context(self, in_data : TacticContext) \
        -> torch.FloatTensor:
        return FloatTensor(self.term_encoder(self.tokenizer.toTokenList(in_data.goal)))
    def encode_hypotheses(self, hyps : List[str]) -> torch.FloatTensor:
        return FloatTensor([self.term_encoder(self.tokenizer.toTokenList(hyp))
                            for hyp in hyps])
    def predict_stem(self, general_features : torch.FloatTensor) -> torch.FloatTensor:
        return self.stem_decoder(general_features)

    def initial_hidden(self, general_features : torch.FloatTensor, stem_idx : int):
        return self.initial_encoder(general_features, LongTensor([stem_idx]))
    def initInput(self) -> torch.LongTensor:
        return Variable(LongTensor([[SOS_token]]))
    def decode_arg(self, decoder_input : torch.LongTensor,
                   hidden_state : torch.FloatTensor,
                   hypothesis_features : torch.FloatTensor) -> \
                   Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Size: (1, self.features_size)
        last_hyp_features = hypothesis_features.add(-TOKEN_START)\
                                               .index_select(1, decoder_input.view(-1))
        # Size: (1, num_hypotheses, self.features_size)
        expanded_last_features = last_hyp_features.view(1, 1, -1)\
                                                  .expand_as(hypothesis_features)
        # Size: (1, num_hypotheses, self.features_size * 2)
        hyps_vector = torch.cat((expanded_last_features, hypothesis_features), dim=2)

        num_hyps = hypothesis_features.size()[1]
        # Size: (1, num_hypotheses, self.features_size)
        expanded_hidden = hidden_state.view(1, 1, -1)\
                                      .expand(1, num_hyps, -1)
        return self.arg_decoder(hyps_vector, expanded_hidden)
        pass

    def getOptions(self) -> List[Tuple[str, str]]:
        return self.options

def predictTacticTeach(initial_encoder : EncoderDNN,
                       stem_decoder : DNNClassifier,
                       arg_decoder : DecoderGRU,
                       batch_size : int,
                       max_args : int,
                       hypothesis_features_batches : torch.FloatTensor,
                       general_features_batch : torch.FloatTensor,
                       correct_tactstruct_batch : torch.LongTensor) -> \
                       Tuple[torch.FloatTensor, List[torch.FloatTensor]]:
    # Size: (batch_size, num_hypotheses)
    stem_distribution : torch.FloatTensor = stem_decoder(general_features_batch)
    # Size: (batch_size)
    decoder_input : torch.LongTensor = LongTensor([SOS_token] * batch_size)
    # Size: (batch_size, hidden_size)
    hidden_state : torch.FloatTensor = \
        initial_encoder(general_features_batch)

    arg_idx_distributions : List[torch.FloatTensor] = []
    print("correct_tacstruct_batch: {}".format(correct_tactstruct_batch))
    for idx in range(max_args):
        # Size(arg_distribution): (batch_size, num_hypotheses)
        # Size(hidden_state): (batch_size, hidden_size)
        arg_distribution, hidden_state = decode_arg(arg_decoder,
                                                    decoder_input,
                                                    hidden_state,
                                                    hypothesis_features_batches)
        # Size(decoder_input): (batch_size, 1)
        decoder_input = LongTensor(correct_tactstruct_batch[:,idx])

        arg_idx_distributions.append(arg_distribution)

    return stem_distribution, arg_idx_distributions

def decode_arg(arg_decoder : DecoderGRU,
               decoder_input : torch.LongTensor,
               hidden_state : torch.FloatTensor,
               hypothesis_features : torch.FloatTensor) -> \
               Tuple[torch.FloatTensor, torch.FloatTensor]:
    print(decoder_input.size())
    batch_size, max_hyps, encoded_term_size = hypothesis_features.size()
    _, hidden_size = hidden_state.size()
    # Size: (batch_size, encoded_term_size)

    # Select the features of the hypotheses indexed by
    # `decoder_input`. If `decoder_input is SOS_token, then we want to
    # use the special "no hypothesis" features, a vector of all
    # zeroes.  To do this, append the special features to the
    # beginning of the hypothesis array, and then move the
    # `decoder_input` indices down by one, since they start at "2" for
    # the first hypothesis, and we want that to refer to the
    # hypothesis at index 1, and SOS_token is `1`, and we want that to
    # refer to the zeros at index 0.
    last_hyp_features = torch.cat((torch.zeros(batch_size, 1, encoded_term_size,
                                               device='cuda'),
                                   hypothesis_features),
                                  dim=1)\
                             .index_select(1, decoder_input
                                           .add(-1))
    print(last_hyp_features.size())
    print(last_hyp_features)
    print(hypothesis_features.size())
    # Size: (batch_size, max_hyps, encoded_term_size)
    expanded_last_features = last_hyp_features.view(batch_size, 1, encoded_term_size)\
                                              .expand_as(hypothesis_features)
    print(expanded_last_features.size())
    # Size: (batch_size, max_hyps, encoded_term_size * 2)
    hyps_vector = torch.cat((expanded_last_features, hypothesis_features), dim=2)

    # Size: (batch_size, max_hyps, hidden_size)
    expanded_hidden = hidden_state.view(batch_size, 1, hidden_size)\
                                  .expand(batch_size, max_hyps, hidden_size)
    print(expanded_hidden.size())
    return cast(Tuple[torch.FloatTensor, torch.FloatTensor],
                arg_decoder(hyps_vector.view(1, batch_size * max_hyps, encoded_term_size * 2),
                            expanded_hidden.view(1, batch_size * max_hyps, hidden_size)))

def decode_tactic_structure(stem_embedding : SimpleEmbedding,
                            struct : TacticStructure, hyps : List[str]) -> str:
    stem_idx, arg_hyp_idxs = struct
    return " ".join([stem_embedding.decode_token(stem_idx)] +
                    [serapi_instance.get_first_var_in_hyp(hyps[hyp_idx-TOKEN_START])
                     for hyp_idx in arg_hyp_idxs])

def flatten_tactic_structure(tac : TacticStructure) -> List[int]:
    stem, args = tac
    return [stem] + args

def _encode(t : Tokenizer, e : 'TermEncoder', s : str) -> EncodedTerm:
    return e(t.toTokenList(s))
def _encode_hyps(t : Tokenizer, e : 'TermEncoder', num_hyps : int,
                 encoded_length : int, ls : List[str]) -> List[EncodedTerm]:
    encoded = [_encode(t, e, s) for s in ls[:num_hyps]]
    return encoded + ([[0] * encoded_length] * (num_hyps - len(encoded)))
def get_arg_idx(hyps : List[str], arg : str):
    hyp_vars = [[name.strip() for name
                 in serapi_instance.get_var_term_in_hyp(hyp).split(",")]
                for hyp in hyps]
    for hyp_idx, hyp_var_list in enumerate(hyp_vars):
        if arg in hyp_var_list:
            return hyp_idx
    return None

def encode_tactic_structure(stem_embedding : SimpleEmbedding,
                            max_args : int,
                            hyps_and_tactic : Tuple[List[str], str]) \
    -> TacticStructure:
    hyps, tactic = hyps_and_tactic
    tactic_stem, args_str = serapi_instance.split_tactic(tactic)
    arg_strs = args_str.split()[:max_args]
    stem_idx = stem_embedding.encode_token(tactic_stem)
    arg_idxs = [get_arg_idx(hyps, arg.strip()) for arg in args_str.split()]
    if len(arg_idxs) < max_args:
        arg_idxs += [EOS_token] * (max_args - len(arg_idxs))
    # If any arguments aren't hypotheses, ignore the arguments
    if not all(arg_idxs):
        arg_idxs = [EOS_token] * max_args

    return TacticStructure(stem_idx=stem_idx, hyp_idxs=arg_idxs)
def encode_hyparg_data(data : RawDataset,
                       tokenizer_type : Callable[[List[str], int], Tokenizer],
                       num_keywords : int,
                       num_reserved_tokens : int,
                       max_args : int,
                       max_hyps : int,
                       encoded_length : int,
                       entropy_data_size : int,
                       num_threads : Optional[int] = None) -> \
                       Tuple[StructDataset, Tokenizer, SimpleEmbedding]:
    stem_embedding = SimpleEmbedding()
    data_list = list(data)
    if len(data_list) <= entropy_data_size:
        subset = data_list
    else:
        subset = random.sample(data_list, entropy_data_size)
    tokenizer = make_keyword_tokenizer_relevance(
        [(context, stem_embedding.encode_token(serapi_instance.get_stem(tactic)))
         for prev_tactics, hyps, context, tactic in subset],
        tokenizer_type, num_keywords, num_reserved_tokens)
    termEncoder = functools.partial(getNGramTokenbagVector, 1, tokenizer.numTokens())
    with multiprocessing.Pool(num_threads) as pool:
        hyps, contexts, tactics = zip(*data_list)
        encoded_contexts = pool.imap_unordered(functools.partial(
            _encode, tokenizer, termEncoder), contexts)
        encoded_hyps = pool.imap_unordered(functools.partial(
            _encode_hyps, tokenizer, termEncoder, max_hyps, encoded_length), contexts)
        encoded_tactics = pool.imap_unordered(
            functools.partial(encode_tactic_structure, stem_embedding, max_args),
            zip(hyps, tactics))
        result = list(zip(encoded_hyps, encoded_contexts, encoded_tactics))
    tokenizer.freezeTokenList()
    return result, tokenizer, stem_embedding


ModuleState = Dict[Any, Any]
class Checkpoint(NamedTuple):
    initial_encoder : ModuleState
    stem_decoder : ModuleState
    arg_decoder : ModuleState
    loss : float


def main(args_list : List[str]):
    arg_parser = start_std_args("A proverbot9001 model template that can predict "
                                "tactics with hypothesis arguments")
    arg_parser.add_argument("--max-hyps", dest="max_hyps", default=10, type=int)
    arg_parser.add_argument("--max-args", dest="max_args", default=2, type=int)
    arg_parser.add_argument("--entropy-data-size", dest="entropy_data_size",
                            default=1000, type=int)
    args = arg_parser.parse_args(args_list)

    dataset = get_text_data(args.scrape_file, args.context_filter, args,
                            max_tuples=args.max_tuples, verbose=True)
    curtime = time.time()

    print("Encoding data...", end="")
    sys.stdout.flush()
    encoded_term_size = args.num_keywords + TOKEN_START + 1
    samples, tokenizer, embedding = encode_hyparg_data(dataset, tokenizers[args.tokenizer],
                                                       args.num_keywords, TOKEN_START,
                                                       args.max_args, args.max_hyps,
                                                       encoded_term_size,
                                                       args.entropy_data_size)
    print(" {:.2f}s".format(time.time() - curtime))

    checkpoints : List[Checkpoint] = train(samples, args,
                                           embedding.num_tokens(),
                                           encoded_term_size)
    for initial_encoder, stem_decoder, arg_decoder, loss in checkpoints:
        state = {'max-args': args.max_args,
                 'max-hyps': args.max_hyps,
                 'hidden-size': args.hidden_size,
                 'tokenizer': tokenizer,
                 'embedding': embedding,
                 'stem-decoder': stem_decoder,
                 'arg-decoder': arg_decoder,
                 'initial-encoder': initial_encoder,
                 'options': [
                     ("dataset size", str(len(samples))),
                     ("context filter", args.context_filter),
                     ("training loss", loss),
                     ("# stems", embedding.num_tokens()),
                     ("# tokens", args.num_keywords),
                     ("hidden size", args.hidden_size),
                     ("max # tactic args", args.max_args),
                     ("max # of hypotheses", args.max_hyps),
                     ("tokenizer entropy sample size", args.entropy_data_size),
                 ]}
        with open(args.save_file, 'wb') as f:
            torch.save(state, f)

TermEncoder = Callable[[Sentence], EncodedTerm]

def train(dataset : StructDataset, args : Namespace,
          num_stems : int, encoded_term_size : int):
    curtime = time.time()
    print("Building data loader...", end="")
    sys.stdout.flush()
    hyp_lists, goals, tactics = zip(*dataset)
    for hyp_list in hyp_lists:
        assert len(hyp_list) == len(hyp_lists[0])
        assert len(hyp_list) == args.max_hyps
        for hyp in hyp_list:
            assert len(hyp) == len(hyp_list[0]), \
                "len(hyp): {}, len(hyp_list[0]): {}".format(len(hyp), len(hyp_list[0]))
    dataloader = DataLoader(
        TensorDataset(torch.FloatTensor(hyp_lists),
                      torch.FloatTensor(goals),
                      torch.LongTensor([flatten_tactic_structure(tactic)
                                        for tactic in tactics])),
        batch_size=args.batch_size, num_workers=0,
        shuffle=True, pin_memory=True, drop_last=True)
    print(" {:.2f}s".format(time.time() - curtime))

    curtime = time.time()
    print("Initializing modules...", end="")
    sys.stdout.flush()
    initial_encoder = maybe_cuda(EncoderDNN(encoded_term_size, args.hidden_size,
                                            args.hidden_size, args.num_encoder_layers,
                                            args.batch_size))
    stem_decoder = maybe_cuda(DNNClassifier(encoded_term_size, args.hidden_size,
                                            num_stems, args.num_decoder_layers))
    arg_decoder = maybe_cuda(DecoderGRU(encoded_term_size * 2, args.hidden_size,
                                        args.num_decoder_layers, args.batch_size))
    optimizer = optimizers[args.optimizer](
        itertools.chain(initial_encoder.parameters(),
                        stem_decoder.parameters(),
                        arg_decoder.parameters()),
        lr=args.learning_rate)
    criterion = maybe_cuda(nn.NLLLoss())
    adjuster = scheduler.StepLR(optimizer, args.epoch_step, gamma=args.gamma)
    print(" {:.2f}s".format(time.time() - curtime))

    start = time.time()
    num_items = len(dataset)
    total_loss = 0

    print("Training...")
    for epoch in range(args.num_epochs):
        print("Epoch {}".format(epoch))
        adjuster.step()
        for batch_num, (hyps_batch, goal_batch, tacstruct_batch) \
            in enumerate(cast(Iterable[Tuple[torch.FloatTensor,
                                             torch.FloatTensor,
                                             torch.LongTensor]],
                              dataloader)):

            optimizer.zero_grad()

            predicted_stem_distribution_batch, predicted_arg_distributions_batches = \
                predictTacticTeach(initial_encoder, stem_decoder, arg_decoder,
                                   args.batch_size, args.max_args,
                                   maybe_cuda(hyps_batch),
                                   maybe_cuda(goal_batch),
                                   maybe_cuda(tacstruct_batch))

            loss = maybe_cuda(Variable(LongTensor(0)))

            loss += criterion(predicted_stem_distribution_batch,
                              maybe_cuda(tacstruct_batch[:,0]))
            for idx in args.max_args:
                loss += criterion(predicted_arg_distributions_batches[idx],
                                  tacstruct_batch[:,idx+1])
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * args.batch_size

            if (batch_num + 1) % args.print_every == 0:

                items_processed = (batch_num + 1) * args.batch_size + epoch * len(dataset)
                progress = items_processed / num_items
                print("{} ({:7} {:5.2f}%) {:.4f}".
                      format(timeSince(start, progress),
                             items_processed, progress * 100,
                             total_loss / items_processed))

        yield Checkpoint(initial_encoder.state_dict(),
                         stem_decoder.state_dict(),
                         arg_decoder.state_dict(),
                         total_loss / ((batch_num + 1) * args.batch_size))
