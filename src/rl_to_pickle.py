import argparse
import torch
import coq2vec
import pickle
import math
from v_model import VModel
from coq_serapy import Obligation, SexpObligation
from typing import Any

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file")
  parser.add_argument("output_file")
  args = parser.parse_args()

  with open(args.input_file, "rb") as f1:
      _, _, _, (network_state, encoder_state, _, training_args), \
          _, _, _ = torch.load(f1)

  with open(args.output_file, 'wb') as f2:
    pickle.dump(LearnedEstimator(network_state, encoder_state, training_args),
                f2)

class LearnedEstimator:
  network: VModel
  encoder: coq2vec.CoqContextVectorizer
  def __init__(self, network_state, encoder_state,
               args: argparse.Namespace) -> None:
    self.init(network_state, encoder_state, args)
  def init(self, network_state, encoder_state,
           args: argparse.Namespace) -> None:
    self.args = args
    term_encoder = coq2vec.CoqTermRNNVectorizer()
    term_encoder.load_state(encoder_state)
    num_hyps = 5
    self.encoder = coq2vec.CoqContextVectorizer(
      term_encoder, num_hyps)
    term_encoder.hidden_size
    self.encoding_size = term_encoder.hidden_size * (num_hyps + 1)
    self.network = VModel(self.encoding_size, args.tactic_vocab_size,
                          args.tactic_embedding_size, args.hidden_size,
                          args.num_layers)
    self.network.load_state_dict(network_state)

  def predict_obl(self, lemma_name: str,
                  obl: Obligation, sexp_obl: SexpObligation,
                  previous_tactic_encoded: int) \
    -> float:
    encoded = self.encoder.obligations_to_vectors([obl])\
      .view(1, self.encoding_size)
    scores = self.network(encoded,
                          torch.LongTensor([
                            previous_tactic_encoded])).view(1)
    return math.log(scores[0].item())

  def __getstate__(self) -> Any:
    return (self.network.state_dict(),
            self.encoder.term_encoder.get_state(),
            self.args)
    pass
  def __setstate__(self, state: Any) -> None:
    self.init(*state)
    pass

if __name__ == "__main__":
  main()
