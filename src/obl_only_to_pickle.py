import argparse
import torch
import coq2vec
import pickle
import math
from lemma_models import SVCThreshold
from coq_serapy import Obligation, SexpObligation
from typing import Any
import torch

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file")
  parser.add_argument("output_file")
  parser.add_argument("-v", "--verbose", dest="verbosity",
                      action='count', default=0)
  args = parser.parse_args()

  with print_time("Loading original model", guard=args.verbosity >= 1):
    with open(args.input_file, "rb") as f1:
      model = pickle.load(f1)

  with print_time("Building new model", guard=args.verbosity >= 1):
    new_model = OblOnlyLearnedEstimator(model)

  with print_time("Writing new model", guard=args.verbosity >=1 ):
    with open(args.output_file, 'wb') as f2:
      pickle.dump(new_model,
                  f2)

class OblOnlyLearnedEstimator:
  inner: SVCThreshold
  def __init__(self, model: SVCThreshold) -> None:
    self.inner = model
  def predict_obl(self, lemma_name: str,
                  obl: Obligation, sexp_obl: SexpObligation,
                  previous_tactic_encoded: int) \
    -> float:
    with torch.no_grad():
        return self.inner.predict_obl(obl)[0]

  def __getstate__(self) -> Any:
    return self.inner
  def __setstate__(self, state: Any) -> None:
    self.inner = state

if __name__ == "__main__":
  main()
