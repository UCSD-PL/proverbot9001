from typing import (List, TypeVar, Dict, Optional, Union,
                    overload, cast, Set, NamedTuple, Iterable,
                    Any)
import re
import sys
import contextlib
import itertools
import time
import random
from pathlib import Path

import torch
from torch import optim
from torch import nn
from torch.nn.modules import loss
from torch.utils import data
import torch.nn.functional as F
import torch.optim.lr_scheduler as scheduler
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.jit

from tqdm import tqdm
import numpy as np

from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch

class Obligation(NamedTuple):
    hypotheses: List[str]
    goal: str

class DummyFile:
    def write(self, x):
        pass

    def flush(self):
        pass

@contextlib.contextmanager
def silent():
    save_stderr = sys.stderr
    save_stdout = sys.stdout
    sys.stderr = DummyFile()
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stderr = save_stderr
        sys.stdout = save_stdout

with silent():
    use_cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
PAD_token = 2
EOS_token = 1
SOS_token = 0

class CoqContextVectorizer:
    term_encoder: 'CoqTermRNNVectorizer'
    max_num_hypotheses: int

    def __init__(self, term_encoder: 'CoqTermRNNVectorizer',
                 max_num_hypotheses: int) -> None:
        self.term_encoder = term_encoder
        self.max_num_hypotheses = max_num_hypotheses
    def obligation_to_vector(self, ob: Obligation) -> torch.FloatTensor:
        selected_hyps = ob.hypotheses[:self.max_num_hypotheses]
        selected_hyps += [":"] * (self.max_num_hypotheses - len(selected_hyps))
        vectors = self.term_encoder.terms_to_vectors(
            [get_hyp_type(hyp) for hyp in selected_hyps] + [ob.goal])
        return vectors
    def obligations_to_vectors(self, obs: List[Obligation]) -> torch.FloatTensor:
        selected_hyp_lists = [
            ob.hypotheses[:self.max_num_hypotheses] +
            [":"] * max(0, self.max_num_hypotheses - len(ob.hypotheses))
            for ob in obs]
        all_term_lists = [[get_hyp_type(hyp)
                           for hyp in selected_hyps] + [ob.goal]
                          for selected_hyps, ob in zip(selected_hyp_lists, obs)]
        all_terms = [term for term_list in all_term_lists for term in term_list]
        return self.term_encoder.terms_to_vectors(all_terms)\
                                .view(len(obs), 1 + self.max_num_hypotheses,
                                      self.term_encoder.hidden_size)


class CoqTermRNNVectorizer:
    symbol_mapping: Optional[Dict[str, int]]
    token_vocab: Optional[List[str]]
    model: Optional['EncoderRNN']
    _decoder: Optional['DecoderRNN']
    max_term_length: Optional[int]
    epochs_trained: int
    num_layers: int
    hidden_size: int
    device: Any
    def __init__(self) -> None:
        self.symbol_mapping = None
        self.token_vocab = None
        self.model = None
        self._decoder = None
        self.max_term_length = None
        self.epochs_trained = 0
        pass
    def load_state(self, state: Any, force_no_cuda : bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   and not force_no_cuda else "cpu")
        self.symbol_mapping, self.token_vocab, \
          model_dict, decoder_dict, \
          self.max_term_length, self.hidden_size, \
          self.num_layers, self.epochs_trained = state
        assert self.token_vocab
        self.model = torch.jit.script(
            EncoderRNN(len(self.token_vocab)+3, self.hidden_size,
                       self.num_layers).to(self.device))
        assert self.model
        self.model.load_state_dict(model_dict)
        self._decoder = torch.jit.script(
            DecoderRNN(self.hidden_size, len(self.token_vocab)+3,
                       self.num_layers).to(self.device))
        assert self._decoder
        self._decoder.load_state_dict(decoder_dict)

    def load_weights(self, model_path: Union[Path, str],
                     force_no_cuda : bool = False) -> None:
        if isinstance(model_path, str):
            model_path = Path(model_path)
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available() and not force_no_cuda
                                   else "cpu")
        self.load_state(torch.load(model_path, map_location=self.device))

    def get_state(self) -> Any:
        return (self.symbol_mapping, self.token_vocab,
                self.model.state_dict(), self._decoder.state_dict(),
                self.max_term_length, self.model.hidden_size, self.model.num_layers,
                self.epochs_trained)

    def save_weights(self, model_path: Union[Path, str]):
        if isinstance(model_path, str):
            model_path = Path(model_path)
        with model_path.open('wb') as f:
            torch.save(self.get_state(), f)

    def train(self, terms: List[str],
              hidden_size: int, learning_rate: float, n_epochs: int,
              batch_size: int, print_every: int, gamma: float,
              force_max_length: Optional[int] = None, epoch_step: int = 1,
              num_layers: int = 1, momentum: float = 0, teacher_forcing_ratio: float = 0.0,
              allow_non_cuda: bool = False, verbosity: int = 0) -> Iterable[float]:
        assert use_cuda or allow_non_cuda, "Cannot train on non-cuda device unless passed allow_non_cuda"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token_set: Set[str] = set()
        max_length_so_far = 0
        for term in tqdm(terms, desc="Getting symbols", disable=verbosity < 1):
            for symbol in get_symbols(term):
                token_set.add(symbol)
            max_length_so_far = max(len(get_symbols(term)), max_length_so_far)

        self.token_vocab = list(token_set)
        self.symbol_mapping = {}
        for idx, symbol in enumerate(self.token_vocab, start=3):
            self.symbol_mapping[symbol] = idx
        if force_max_length:
            self.max_term_length = min(force_max_length, max_length_so_far)
        else:
            self.max_term_length = max_length_so_far

        term_tensor = torch.LongTensor([self.term_to_seq(term)
            for term in tqdm(terms, desc="Tokenizing and normalizing", disable=verbosity < 1)])
        term_lengths = torch.LongTensor([min(self.term_seq_length(term)+1, self.max_term_length)
            for term in tqdm(terms, desc="Counting lengths", disable=verbosity < 1)])
        yield from self.train_with_tensors(term_tensor, term_lengths, hidden_size, learning_rate, n_epochs,
                                           batch_size, print_every, gamma, force_max_length, epoch_step,
                                           num_layers, momentum, teacher_forcing_ratio, allow_non_cuda, verbosity)
    def train_with_tensors(self, term_tensor: torch.LongTensor, term_lengths: torch.LongTensor,
                           hidden_size: int, learning_rate: float, n_epochs: int,
                           batch_size: int, print_every: int, gamma: float,
                           force_max_length: Optional[int] = None, epoch_step: int = 1,
                           num_layers: int = 1, momentum: float = 0, teacher_forcing_ratio: float = 0.0,
                           allow_non_cuda: bool = False, verbosity: int = 0) -> Iterable[float]:
        assert use_cuda or allow_non_cuda, "Cannot train on non-cuda device unless passed allow_non_cuda"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_term_length = term_tensor.size(1)

        encoder = torch.jit.script(maybe_cuda(EncoderRNN(len(self.token_vocab)+3, hidden_size, num_layers).to(self.device)))
        self.model = encoder
        decoder = torch.jit.script(maybe_cuda(DecoderRNN(hidden_size, len(self.token_vocab)+3, num_layers).to(self.device)))

        self._decoder = decoder
        dataset_size = term_tensor.size(0)
        split_ratio = 0.05
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int((dataset_size * split_ratio) / batch_size) * batch_size
        train_indices, val_indices = indices[split:], indices[:split]
        valid_batch_size = max(batch_size // 2, 1)
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        num_batches = int((dataset_size - split) / batch_size)
        num_batches_valid = int(split / valid_batch_size)

        train_dataset_size = num_batches * batch_size

        valid_data_batches = data.DataLoader(data.TensorDataset(term_tensor, term_lengths),
                                             batch_size=valid_batch_size, num_workers=0,
                                             sampler=valid_sampler, pin_memory=True, drop_last=True)
        data_batches = data.DataLoader(data.TensorDataset(term_tensor, term_lengths),
                                       batch_size=batch_size, num_workers=0,
                                       sampler=train_sampler, pin_memory=True, drop_last=True)

        optimizer = optim.SGD(itertools.chain(encoder.parameters(), decoder.parameters()),
                              lr=learning_rate, momentum=momentum)
        adjuster = scheduler.StepLR(optimizer, epoch_step,
                                    gamma=gamma)

        criterion = nn.NLLLoss(ignore_index=PAD_token)
        training_start=time.time()
        writer = SummaryWriter()
        if verbosity >= 1:
            print("Training")
        for pre_epoch in range(self.epochs_trained):
            adjuster.step()
        for epoch in range(self.epochs_trained, n_epochs):
            if verbosity >= 1:
                print("Epoch {} (learning rate {:.6f})".format(epoch, optimizer.param_groups[0]['lr']))
            epoch_loss = 0.
            epoch_tf_ratio = teacher_forcing_ratio * (1 - (epoch / (n_epochs - 1)))
            for batch_num, (term_batch, lengths_batch) in enumerate(data_batches, start=1):
                if epoch == n_epochs and batch_num == 1:
                    encoder = jit_trace_encoder(len(self.token_vocab) + 3,
                                                hidden_size, num_layers,
                                                term_batch, encoder)
                    self._model = encoder
                optimizer.zero_grad()
                lengths_sorted, sorted_idx = lengths_batch.sort(descending=True)
                padded_term_batch = pack_padded_sequence(term_batch[sorted_idx], lengths_sorted, batch_first=True)
                loss, accuracy = autoencoderBatchIter(encoder, decoder, maybe_cuda(padded_term_batch),
                                                      maybe_cuda(term_batch[sorted_idx]), maybe_cuda(lengths_sorted),
                                                      criterion, epoch_tf_ratio)
                writer.add_scalar("Batch loss/train", loss, epoch * num_batches + batch_num)
                writer.add_scalar("Batch accuracy/train", accuracy, epoch * num_batches + batch_num)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if batch_num % print_every == 0:
                    items_processed = batch_num * batch_size + \
                      epoch * train_dataset_size
                    progress = items_processed / \
                      (train_dataset_size * n_epochs)
                    if verbosity >= 1:
                        print("{} ({:7} {:5.2f}%) {:.4f}"
                              .format(timeSince(training_start, progress),
                                      items_processed, progress * 100,
                                      epoch_loss / batch_num))
            with torch.no_grad():
                valid_accuracy = maybe_cuda(torch.FloatTensor([0.]))
                valid_loss = maybe_cuda(torch.FloatTensor([0.]))
                for idx, (valid_data_batch,valid_lengths_batch) in enumerate(valid_data_batches):
                    lengths_sorted, sorted_idx = valid_lengths_batch.sort(descending=True)
                    valid_padded_batch = pack_padded_sequence(valid_data_batch[sorted_idx], lengths_sorted, batch_first=True)
                    batch_loss, batch_accuracy = autoencoderBatchIter(encoder, decoder, maybe_cuda(valid_padded_batch),
                                                                      maybe_cuda(valid_data_batch[sorted_idx]), lengths_sorted,
                                                                      criterion, 0., verbosity=verbosity if idx == len(valid_data_batches)-1 else 0, model=self)
                    valid_loss = cast(torch.FloatTensor, valid_loss + batch_loss)
                    valid_accuracy = cast(torch.FloatTensor, valid_accuracy + batch_accuracy)
            writer.add_scalar("Loss/valid", valid_loss / num_batches_valid,
                              epoch * num_batches + batch_num)
            writer.add_scalar("Accuracy/valid", valid_accuracy / num_batches_valid,
                              epoch * num_batches + batch_num)
            if verbosity >= 1:
                print(f"Validation loss: {valid_loss.item() / num_batches_valid:.4f}; "
                      f"Validation accuracy: {valid_accuracy.item() * 100 / num_batches_valid:.2f}%")

            adjuster.step()
            self.model = encoder
            self._decoder = decoder
            yield valid_loss.item() / num_batches_valid
            pass
        pass
    def term_to_seq(self, term_text: str) -> List[int]:
        return normalize_sentence_length([self.symbol_mapping[symb]
                                          for symb in get_symbols(term_text)
                                          if symb in self.symbol_mapping][:self.max_term_length-1] + [EOS_token],
                                         self.max_term_length,
                                         PAD_token)
    def term_seq_length(self, term_text: str) -> int:
        assert self.symbol_mapping
        return len([True for symb in get_symbols(term_text) if symb in self.symbol_mapping])
    def seq_to_symbol_list(self, seq: List[int]) -> List[str]:
        assert self.token_vocab
        output_symbols = []
        for item in seq:
            if item == EOS_token:
                break
            assert item >= 3
            output_symbols.append(self.token_vocab[item - 3])
        return output_symbols
    def output_seq_to_term(self, seq: List[int]) -> str:
        return " ".join(self.seq_to_symbol_list(seq)[::-1])
    def input_seq_to_term(self, seq: List[int]) -> str:
        return " ".join(self.seq_to_symbol_list(seq))
    def term_to_vector(self, term_text: str) -> torch.FloatTensor:
        seq = self.term_to_seq(term_text)
        return self.seq_to_vector(seq)
    def terms_to_vectors(self, term_texts: List[str]) -> torch.FloatTensor:
        seqs = [self.term_to_seq(term_text) for term_text in term_texts]
        return self.seqs_to_vectors(seqs)

    def seqs_to_vectors(self, term_seqs: List[List[int]]) -> torch.FloatTensor:
        assert self.symbol_mapping, "No loaded weights!"
        assert self.model, "No loaded weights!"
        input_lengths = [len([t for t in term_seq if t != PAD_token])
                         for term_seq in term_seqs]
        terms_tensor = pack_padded_sequence(torch.LongTensor(term_seqs).to(self.device),
                                            torch.LongTensor(input_lengths),
                                            batch_first=True, enforce_sorted=False)
        with torch.no_grad():
            hidden = self.model.initHidden(len(term_seqs), self.device)
            cell = self.model.initCell(len(term_seqs), self.device)
            _, hidden, cell = self.model(terms_tensor, hidden, cell)
        return hidden.cpu().squeeze(0)
    def seq_to_vector(self, term_seq: List[int]) -> torch.FloatTensor:
        return self.seqs_to_vectors([term_seq])[0]
    def vector_to_term(self, term_vec: torch.FloatTensor) -> str:
        return self.output_seq_to_term(self.vector_to_seq(term_vec))
    def vector_to_seq(self, term_vec: torch.FloatTensor) -> List[int]:
        assert self.symbol_mapping, "No loaded weights!"
        assert self.model, "No loaded weights!"
        assert self._decoder
        assert self.max_term_length
        assert self.token_vocab
        assert term_vec.size() == torch.Size([self.model.num_layers, self.model.hidden_size]), f"Wrong dimensions for input {term_vec.size()}"
        self._decoder.to(self.device)
        with torch.no_grad():
            decoder_hidden = term_vec.unsqueeze(1).to(self.device)
            decoder_input = torch.tensor([[SOS_token]], device=self.device)
            decoder_cell = self._decoder.initCell(1, self.device)
            output_seq = []
            for di in range(self.max_term_length):
                decoder_output, decoder_hidden, decoder_cell = self._decoder(decoder_input, decoder_hidden, decoder_cell)
                topv, topi = decoder_output.topk(1)
                next_char = topi.view(1).detach()
                output_seq.append(next_char.item())
                decoder_input = next_char
        return output_seq


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.num_layers = num_layers

    def forward(self, input: PackedSequence, hidden: torch.FloatTensor,
                cell: torch.FloatTensor):
        batch_size = input.batch_sizes[0]
        max_input_length = len(input.batch_sizes)
        embedded = PackedSequence(F.relu(self.embedding(input.data)), input.batch_sizes,
                                  input.sorted_indices, input.unsorted_indices)
        output, (hidden, cell) = self.lstm(embedded, (hidden,cell))
        return output, hidden, cell

    @torch.jit.export
    def initHidden(self,batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    @torch.jit.export
    def initCell(self,batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, num_layers: int = 1) -> None:
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.num_layers = num_layers
        self.output_size = output_size

    def forward(self, input, hidden, cell):
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        output, (hidden, cell) = self.lstm(F.relu(embedded), (hidden, cell))
        token_dist = self.softmax(self.out(output))
        return token_dist, hidden, cell

    @torch.jit.export
    def initHidden(self,batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    @torch.jit.export
    def initCell(self,batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

def autoencoderBatchIter(encoder: EncoderRNN, decoder: DecoderRNN, data: torch.LongTensor, output: torch.LongTensor, lengths: torch.LongTensor,
                         criterion: loss._Loss, teacher_forcing_ratio: float, verbosity:int = 0, model: Optional[CoqTermRNNVectorizer] = None) -> torch.FloatTensor:
    batch_size = data.batch_sizes[0]
    input_length = len(data.batch_sizes)
    target_length = input_length
    device = torch.device("cuda" if use_cuda else "cpu")

    loss: torch.FloatTensor = maybe_cuda(torch.tensor([0.]))
    accuracy_sum = 0.
    hidden = encoder.initHidden(batch_size, device)
    cell = encoder.initCell(batch_size, device)
    _, hidden, cell = encoder(data, hidden, cell)
    decoder_hidden = hidden.clone()
    decoder_input = torch.tensor([[SOS_token]]*batch_size, device=device)
    decoder_cell = decoder.initCell(batch_size, device)
    decoder_results = []
    target_input = torch.tensor([[SOS_token]]*batch_size, device=device)
    for di in range(target_length):
        # target = output[:,target_length-(di+1)]
        if random.random() < teacher_forcing_ratio:
             decoder_output, decoder_hidden, decoder_cell = decoder(target_input, decoder_hidden, decoder_cell)
        else:
             decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input, decoder_hidden, decoder_cell)
        target = maybe_cuda(torch.LongTensor([output[i, lengths[i]-(di+2)] if di < lengths[i]-1 else EOS_token if di == lengths[i]-1 else PAD_token for i in range(batch_size)]))
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.view(batch_size).detach()
        target_input = target

        item_loss = criterion(decoder_output.view(batch_size, decoder.output_size), target)
        loss = cast(torch.FloatTensor, loss + item_loss)
        decoder_results.append(decoder_input)
        accuracy_sum += torch.sum((decoder_input == target) * (target != PAD_token).int().float()).item()
    accuracy_denominator = torch.sum((output != PAD_token).int().float()).item()
    if verbosity > 1:
        for i in range(batch_size):
            encoded_state = hidden[:,i].tolist()
            decoded_result = [decoder_results[j][i].item() for j in range(target_length)]
            print(f"{model.input_seq_to_term(output[i])} -> {output[i].tolist()} -> {encoded_state} -> {decoded_result} -> {model.output_seq_to_term(decoded_result)}")
    elif verbosity > 0:
        for i in range(min(batch_size, 4)):
            # target = maybe_cuda(torch.LongTensor([output[i, lengths[i]-(j+2)] if j < lengths[i]-1
            #                                       else EOS_token if j == lengths[i]-1 else PAD_token
            #                                       for j in range(target_length)]))
            # print(f"Target is {model.output_seq_to_term(target)} -> {target.tolist()}")
            decoded_result = [decoder_results[j][i].item() for j in range(target_length)]
            print(f"{model.input_seq_to_term(output[i])} [======>>\n{model.output_seq_to_term(decoded_result)}")
            # sample_correct = torch.sum((maybe_cuda(torch.tensor(decoded_result)) == target) * (target != PAD_token).int().float()).item()
            # sample_denominator = torch.sum((target != PAD_token).int().float()).item()
            # print(f"Number of matching tokens: {torch.sum(maybe_cuda(torch.tensor(decoded_result)) == target)}")
            # print(f"Training accuracy of sample: {sample_correct * 100 / sample_denominator:.2f}% ({sample_correct} / {sample_denominator})")
            # assert output[i, 0] not in [model.symbol_mapping[c] for c in [".", ")"]], f"Input term {output[i]} doesn't make any sense!"
        #print(f"Accuracy: {accuracy_sum} / {accuracy_denominator}")

    return loss / target_length, accuracy_sum / accuracy_denominator
def jit_trace_encoder(vocab_size: int, hidden_size: int, num_layers: int,
                      term_batch: torch.LongTensor,
                      scriptedEncoder: torch.jit.ScriptModule) \
                      -> torch.jit.ScriptModule:
    batch_size = data.batch_sizes[0]
    input_length = len(data.batch_sizes)
    target_length = input_length
    device = torch.device("cuda" if use_cuda else "cpu")
    hidden = scriptedEncoder.initHidden(batch_size, device)
    cell = scriptedEncoder.initCell(batch_size, device)
    return torch.jit.trace_module(EncoderRNN(vocab_size, hidden_size,
                                             num_layers).to(device),
                                  {"initHidden": (batch_size, device),
                                   "initCell": (batch_size, device),
                                   "forward": (term_batch, hidden, cell)})
    _, hidden, cell = encoder(data, hidden, cell)
    pass
def tune_termrnn_hyperparameters(terms: List[str], n_epochs: int,
                                 batch_size: int, print_every: int,
                                 force_max_length: Optional[int] = None, epoch_step: int = 1,
                                 allow_non_cuda: bool = False,
                                 search_space: Optional[Dict[str, Any]] = None) -> None:
    def objective(config: Dict[str, Union[float, int]], terms: List[str]) -> float:
        vectorizer = CoqTermRNNVectorizer()
        for epoch, valid_loss in enumerate(vectorizer.train(terms,
                                                            hidden_size=config['hidden_size'],
                                                            learning_rate=config['learning_rate'],
                                                            num_layers=config['num_layers'],
                                                            momentum=config['momentum'],
                                                            teacher_forcing_ratio=config['teacher_forcing_ratio'],
                                                            n_epochs=n_epochs,
                                                            batch_size=64, print_every=16,
                                                            gamma=config['gamma'], epoch_step=1,
                                                            force_max_length=30)):
            session.report({"valid_loss": valid_loss})
    if not search_space:
        search_space={'hidden_size': tune.lograndint(64, 4096), "learning_rate": tune.loguniform(1e-4, 10), "teacher_forcing_ratio": tune.uniform(0.0, 1.0),
                      'momentum': tune.uniform(0.1, 0.9), 'num_layers': tune.randint(1, 3), 'gamma': tune.uniform(0.1, 1.0)}
    algo=OptunaSearch()
    tuner = tune.Tuner(tune.with_resources(
                         tune.with_parameters(objective, terms=terms),
                         {"cpu": 1, "gpu": 1}),
                       tune_config=tune.TuneConfig(
                         metric="valid_loss",
                         mode="min",
                         search_alg=algo,
                         num_samples=64),
                       run_config=air.RunConfig(
                         stop={"training_iteration": 5}),
                       param_space=search_space)
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)

symbols_regexp = (r',|(?::>)|(?::(?!=))|(?::=)|\)|\(|;|@\{|~|\+{1,2}|\*{1,2}|&&|\|\||'
                  r'(?<!\\)/(?!\\)|/\\|\\/|(?<![<*+-/|&])=(?!>)|%|(?<!<)-(?!>)|'
                  r'<-|->|<=|>=|<>|\^|\[|\]|(?<!\|)\}|\{(?!\|)|\.(?=$|\s+)')
def get_symbols(string: str) -> List[str]:
    return [word for word in re.sub(
        r'(' + symbols_regexp + ')',
        r' \1 ', string).split()
            if word.strip() != '']

T1 = TypeVar('T1', bound=nn.Module)
T2 = TypeVar('T2', bound=torch.Tensor)
@overload
def maybe_cuda(component: T1) -> T1:
    ...

@overload
def maybe_cuda(component: T2) -> T2:
    ...

if use_cuda:
    def maybe_cuda(component):
        return component.to(device=torch.device("cuda"))
else:
    def maybe_cuda(component):
        return component

def normalize_sentence_length(sentence: List[int], target_length: int, fill_value: int) -> List[int]:
    if len(sentence) > target_length:
        return sentence[:target_length]
    elif len(sentence) < target_length:
        return sentence + [fill_value] * (target_length - len(sentence))
    else:
        return sentence

def timeSince(since : float, percent : float) -> str:
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "{} (- {})".format(asMinutes(s), asMinutes(rs))

def asMinutes(s : float) -> str:
    m = int(s / 60)
    s -= m * 60
    return "{:3}m {:5.2f}s".format(m, s)

hypcolon_regex = re.compile(":(?!=)")

def get_hyp_type(hyp: str) -> str:
    splits = hypcolon_regex.split(hyp, maxsplit=1)
    if len(splits) == 1:
        return ""
    else:
        return splits[1].strip()
