
from abc import ABC, abstractmethod, abstractproperty
from tabnanny import verbose
from typing import Dict, Iterable, Iterator, List, Tuple, TypeVar, TypedDict, Union, Any, Optional, Callable
from dataclasses import dataclass

import coq2vec

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression # for linear regression of type vs difficulty
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection # for k-fold splitting, see source at https://datascience.stanford.edu/news/splitting-data-randomly-can-ruin-your-model
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

import sexpdata as sexp

from itertools import chain, tee

from numba import njit

# import random as rand

import json

Symbol = sexp.Symbol

Sexpr = Union[sexp.Symbol, int, str, List['Sexpr']]

class ProofCtx(TypedDict):
  hypos: List[Sexpr]
  type: Sexpr
  goal_str: str
  hyp_strs: List[str]

@dataclass
class Lemma:
  name: str
  ctx: ProofCtx

  @property
  def type(self):
    return self.ctx["type"]
  
  @property
  def hypos(self):
    for x in self.ctx["hypos"]:
      yield x

class ProofStep(TypedDict):
  tacs: List[str]
  ctx: ProofCtx

def name_transform(x):
  if type(x) == StandardScaler: return 'standardscalar'
  elif type(x) == SVR: return 'svr'
  elif type(x) == KNeighborsClassifier: return 'knn'
  elif type(x) == PCA: return 'pca'
  elif type(x) == SVC: return 'svc'
  elif type(x) == LinearSVC: return 'linearsvc'
  elif type(x) == NuSVC: return 'nusvc'
  elif type(x) == Nystroem: return 'nystroem'
  elif type(x) == SGDClassifier: return 'sgdclassifier'
  elif type(x) == GridSearchCV: return 'gridsearchcv'
  else:
    print("unhandled transform", x)
    assert False

def make_pipeline(*args):
  return Pipeline(steps=[(name_transform(x), x) for x in args])

def step_len(st: ProofStep):
  return len(st["tacs"])

def pretty_step_tac(ps: ProofStep):
  return ";".join(ps["tacs"])

def steps_from_json(x: Dict[str, Any]) -> Optional[ProofStep]:
  if "tacs" in x and "ctx" in x:
    # print('tacs:', [str(tac) for tac in x["tacs"]])
    ctx = x["ctx"]
    
    return {  
        "tacs" : [str(tac) for tac in x["tacs"]]
      , "ctx" : { 
          "hypos": [eval(x) for x in ctx["hypos"]]
        , "type": eval(ctx["type"]) 
        , "goal_str" : ctx["goal_str"]
        , "hyp_strs" : ctx["hypo_strs"]
      } 
    }
  else:
    return None

@dataclass
class ProofState(Iterable):
  steps: List[ProofStep]
  children: List['ProofState']

  def __iter__(self) -> Iterator['ProofState']:
    for child in self.children:
      for inner_child in child:
        yield inner_child
    yield self

  def proof_len(self):
    return sum([step_len(step) for x in self for step in x.steps])

  def pretty_proof(self, depth : int = 0) -> str:
    line_prefix = "  "*depth
    lines = []
    for step in self.steps:
      lines.append(line_prefix + pretty_step_tac(step))
    output = "\n".join(lines) + "\n"

    for child in self.children:
      output += line_prefix + "{\n"
      output += child.pretty_proof(depth=depth+1)
      output += line_prefix + "}\n"
    
    return output

  @staticmethod
  def from_json(x: Dict[str, Any]) -> Optional['ProofState']:
    if "steps" in x and "children" in x:
      try:
        steps = []
        for step in x["steps"]:
          next_step = steps_from_json(step)
          if next_step:
            steps.append(next_step)
          else:
            return None
        
        children = []
        for child_json in x["children"]:
          child = ProofState.from_json(child_json)
          if child:
            children.append(child)
          else:
            return None
          
        return ProofState(steps, children)
      except:
        # print(f"couldn't convert {x} to a proof state")
        return None
    else:
      return None

  def steps_to_lemmas(self, parent_prefix = "state") -> Iterator[Tuple[Lemma, int]]:
    for x in self.branch_prefix_to_lemmas(parent_prefix):
      yield x
    
    for i, child in enumerate(self.children):
      for x in child.steps_to_lemmas(f"{parent_prefix}_{i}"):
        yield x



  def branch_prefix_to_lemmas(self, parent_prefix = "state") -> Iterator[Tuple[Lemma, int]]:
    diff = 0
    curr_len = self.proof_len()
    for idx, step in enumerate(self.steps):
      curr_diff = step_len(step)
      diff += curr_diff
      yield Lemma(f"{parent_prefix}_{idx}", step["ctx"]), curr_len - diff + curr_diff

A = TypeVar('A')
B = TypeVar('B')
def split_diffs(a_diffs: List[Tuple[A, float]], thresholds: List[float]):
  print(f'splitting {len(a_diffs)} into {thresholds}')

  for i, thresh in enumerate(thresholds):
    if i == 0:
      lo = float('-inf')
    else:
      lo = thresholds[i-1]
    yield [(a, diff) for a, diff in a_diffs if lo < diff and diff <= thresh ]



def iter_fst(xs: Iterable[Tuple[A, B]]) -> Iterable[A]:
  xs, _ = tee(xs)
  for x, _ in xs:
    yield x

def iter_snd(xs: Iterable[Tuple[A, B]]) -> Iterable[B]:
  xs, _ = tee(xs)
  for _, x in xs:
    yield x


class ILearner(ABC):

  _use_hypos : bool

  def __init__(self, *args, **kwargs) -> None:
    if "use_hypos" in kwargs:
      self._use_hypos = kwargs["use_hypos"]
    else:
      self._use_hypos = False

  def learn(self, lemmas: Iterable[Tuple[Lemma, int]]) -> None:
    self.init(iter_fst(lemmas), iter_snd(lemmas))
    xs = self.transform_xs(iter_fst(lemmas))
    ys = self.transform_ys(iter_snd(lemmas))
    self.train_transformed(xs, ys)

  def predict(self, lemma: Lemma) -> float:
    xs = self.transform_xs([lemma])
    return self.predict_transformed(xs)[0]

  # clients that need to do initialization should override this
  def init(self, lemmas: Iterable[Lemma], diffs: Iterable[int]) -> None:
    return None

  @abstractmethod
  def transform_xs(self, lemmas: Iterable[Lemma]) -> np.ndarray:
    raise NotImplemented

  def transform_ys(self, diffs: Iterable[int]) -> np.ndarray:
    return np.fromiter(diffs, np.single)

  @abstractmethod
  def train_transformed(self, xs: np.ndarray, ys: np.ndarray) -> None:
    raise NotImplemented

  @abstractmethod
  def predict_transformed(self, xs: np.ndarray) -> np.ndarray:
    raise NotImplemented

  @property
  @abstractmethod
  def name(self) -> str:
    raise NotImplemented

class UnhandledExpr(Exception):

  def __init__(self, e: Sexpr, info: Optional[str] = None) -> None:
    super().__init__(e, info)

class NaiveMeanLearner(ILearner):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._mean: float = 0

  def transform_xs(self, lemmas: Iterable[Lemma]):
    return np.array([np.array([1.0]) for _ in lemmas])

  def transform_ys(self, diffs: Iterable[int]):
    return np.fromiter(diffs, np.single)

  def train_transformed(self, xs: np.ndarray, ys: np.ndarray):
    self._mean = np.mean(ys)

  def predict_transformed(self, xs: np.ndarray):
    return np.fromiter((self._mean for _ in xs), np.single, count=len(xs))

  @property
  def name(self):
    return "naive mean"

class NaiveMedianLearner(ILearner):

  _median: np.single

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._median: np.single = np.single(0.0)

  def transform_xs(self, lemmas: Iterable[Lemma]):
    return np.array([np.array([1.0]) for _ in lemmas])

  def transform_ys(self, diffs: Iterable[int]):
    return np.fromiter(diffs, np.single)

  def train_transformed(self, xs: np.ndarray, ys: np.ndarray):
    self._median = np.median(ys)

  def predict_transformed(self, xs: np.ndarray):
    return np.fromiter((self._median for _ in xs), np.single, count=len(xs))

  @property
  def name(self):
    return "naive median"

class LinRegressionLearner(ILearner):

  _model : LinearRegression

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._model = LinearRegression(n_jobs=-1)

  def lem_to_x(self, lemma : Lemma) -> np.ndarray:
    types = (list(lemma.hypos) if self._use_hypos else []) + [lemma.type]
    sizes = np.fromiter((float(nested_size(x)) for x in types), np.single, count=len(types))
    return np.array([np.sum(sizes)])

  def transform_xs(self, lemmas: Iterable[Lemma]):
    xs = [self.lem_to_x(lem) for lem in lemmas]
    return np.asarray(xs)

  def train_transformed(self, xs: np.ndarray, ys: np.ndarray):
    self._model.fit(xs, ys)

  def predict_transformed(self, xs: np.ndarray):
    return self._model.predict(xs)

  @property
  def name(self):
    return "LR size"


class ISVRLearner(ILearner):
  
  _should_CV: bool
  _model : BaseEstimator

  _svr_params : dict[Any, Any]

  @property 
  def svr_params(self):
    return self._svr_params

  def __init__(self, svr_params = None, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if svr_params:
      self._svr_params = svr_params
      self._model = make_pipeline(
          StandardScaler()
        , PCA(n_components=0.95)
        , SVR(**svr_params)
      )
      self._should_CV = False
    else:
      self._should_CV = True
      self._svr_params = {}
      ranges = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
      epsilon_ranges = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
      self._model = make_pipeline(
          StandardScaler()
        , PCA(n_components=0.95)
        , GridSearchCV(
            estimator=SVR()
          , param_grid=[
                {'C': ranges, 'epsilon': epsilon_ranges, 'gamma': ranges, 'kernel': ['rbf']},
            ]
          , n_jobs=-1
          , verbose=3
        )
      )

  def train_transformed(self, xs: np.ndarray, ys: np.ndarray):
    self._model.fit(xs, ys)
    if self._should_CV:
      self._svr_params = self._model[-1].best_params_
      print("best svr params:")
      print(self._svr_params)

  def predict_transformed(self, xs: np.ndarray):
    return self._model.predict(xs)


class SVRLength(ISVRLearner):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
  
  def transform_xs(self, lems: Iterable[Lemma]):
    return np.array([
      np.array(lemma_nested_size(lemma, use_hypos=self._use_hypos)) for lemma in lems
    ])
    
  @property
  def name(self):
    return "SVR size"

class SVRIdent(ISVRLearner):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def transform_xs(self, lems: Iterable[Lemma]):
    return np.array([
      np.array(lemma_size(lemma, use_hypos=self._use_hypos)) for lemma in lems
    ])

  @property
  def name(self):
    return "SVR ident"

class SVRIdentLength(ISVRLearner):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
  
  def transform_xs(self, lems: Iterable[Lemma]):
    return np.array([
      np.array(lemma_nested_size(lemma, use_hypos=self._use_hypos) + lemma_size(lemma, use_hypos=self._use_hypos)) for lemma in lems
    ])
    
  @property
  def name(self):
    return "SVR ident + size"

# >>> import coq2vec
# >>> vectorizer = coq2vec.CoqRNNVectorizer()
# >>> vectorizer.load_weights("coq2vec/encoder_model.dat")
# >>> vectorizer.term_to_vector("forall x: nat, x = x")

class SVRNNGoal(ISVRLearner):

  _vectorizer: coq2vec.CoqTermRNNVectorizer
  _normalizer: StandardScaler

  def __init__(self, encoder_weights_path : str = "encoder_model.dat", *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._vectorizer = coq2vec.CoqTermRNNVectorizer()
    self._normalizer = StandardScaler()

    self._vectorizer.load_weights(encoder_weights_path)

  def init(self, lemmas: Iterable[Lemma], diffs: Iterable[int]) -> None:
    super().init(lemmas, diffs)
    xs = np.array([
      self._vectorizer.term_to_vector(lemma.ctx["goal"]) for lemma in lemmas
    ])
    self._normalizer.fit(xs)
  
  def transform_xs(self, lems: Iterable[Lemma]):
    xs = np.array([
      self._vectorizer.term_to_vector(lemma.ctx["goal"]) for lemma in lems
    ])


    return self._normalizer.transform(xs, copy=False)
    
  @property
  def name(self):
    return "SVR NN goal"

class SVRC2V(ISVRLearner):

  _vectorizer: coq2vec.CoqContextVectorizer

  def __init__(self, encoder_weights_path : str = "encoder_model.dat", *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    goal_vectorizer = coq2vec.CoqTermRNNVectorizer()
    goal_vectorizer.load_weights(encoder_weights_path)

    self._vectorizer = coq2vec.CoqContextVectorizer(goal_vectorizer, 4)

  def transform_xs(self, lemmas: Iterable[Lemma]) -> np.ndarray:
    return super().transform_xs(lemmas)

  @property
  def name(self):
    return "SVR C2V Contexts"

  def __getstate__(self):
    state = self.__dict__.copy()
    
    state["_vectorizer"] = self._vectorizer.term_encoder.get_state()

    return state

  def __setstate__(self, state):
    vectorizer = coq2vec.CoqTermRNNVectorizer()
    vectorizer.load_state(state["_vectorizer"])
    state["_vectorizer"] = coq2vec.CoqContextVectorizer(vectorizer, 4)
    self.__dict__.update(state)

  def predict_obl(self, obl: coq2vec.Obligation) -> np.ndarray:
    x : np.ndarray = self._vectorizer.obligation_to_vector(obl).numpy().flatten()

    return self.predict_transformed(np.array([x]))

class SVRCounts(ISVRLearner):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def transform_xs(self, lemmas: Iterable[Lemma]) -> np.ndarray:
    return super().transform_xs(lemmas)

  @property
  def name(self):
    return "SVR Counts"


class SVRNNGoalIdentLength(ISVRLearner):

  _vectorizer: coq2vec.CoqTermRNNVectorizer
  _normalizer: StandardScaler

  def __init__(self, encoder_weights_path: str ="encoder_model.dat", *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    self._vectorizer = coq2vec.CoqTermRNNVectorizer()
    self._normalizer = StandardScaler()
    self._vectorizer.load_weights(encoder_weights_path)

  def init(self, lemmas: Iterable[Lemma], diffs: Iterable[int]) -> None:
    super().init(lemmas, diffs)
    xs = np.array([
      self._vectorizer.term_to_vector(lemma.ctx["goal"]) for lemma in lemmas
    ])
    self._normalizer.fit(xs)
  
  def transform_xs(self, lems: Iterable[Lemma]):

    return np.array([
     np.concatenate(  (   
            self._normalizer.transform(np.array([self._vectorizer.term_to_vector(lemma.ctx["goal"])]), copy=False)[0]
          , np.array(lemma_nested_size(lemma, use_hypos=self._use_hypos) + lemma_size(lemma, use_hypos=self._use_hypos))
        )
        , axis=None
      )
      for lemma in lems
    ])
    
  @property
  def name(self):
    return "SVR NN IL goal"


def ranges(N: int):
  return [float(k) / float(N) for k, _ in enumerate(range(1,N), start=1)] + [1.0]

def find_quantile(buckets: list[float], x: float):
  assert len(buckets) > 0, "empty quantiles?"
  for i, buck in enumerate(buckets, start=0):
    if x <= buck: return i
  return i

def make_ident_vector(lem: Lemma, idxs: dict[str, int], use_hyps : bool) -> list[float]:
  if use_hyps:
    idents = lemma_hyp_idents(lem)
  else:
    idents = lemma_ty_idents(lem)
  
  ret = [0.0] * len(idxs)
  for ident, value in idents.items():
    ret[idxs[ident]] = float(value)
  return ret





# for now, assumes that each of the individual learners use the same x data as the ensemble learner
# so basically, only use individual learners that don't really make use of x data (or fit on the same data as the ensemble learner)
class IEnsembleLearner(ILearner):

  @property
  @abstractmethod
  def model(self) -> BaseEstimator:
    pass

  _locals : List[ILearner]
  _buckets : int
  _ty_ident_idx : dict[str, int]
  _hyp_ident_idx : dict[str, int]
  _thresholds : list[float]
  
  _init : bool
  

  def __init__(self, local_builder : Callable[[int], ILearner], buckets: int, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    
    self._locals = []
    self._ty_ident_idx = {}
    self._hyp_ident_idx = {}
    self._buckets = buckets
    self._locals = [local_builder(i) for i in range(buckets)]

    self._thresholds = []

    self._init = False

  def init_idents(self, lemmas: Iterable[Lemma]):
    all_ty_idents : set[str] = set()
    all_hyp_idents : set[str] = set()
    # print("gathering idents...")
    for lem in lemmas:
      all_ty_idents |= lemma_ty_idents(lem)
      if self._use_hypos:
        all_hyp_idents |= lemma_hyp_idents(lem)

    self._ty_ident_idx = {ident : i for i, ident in enumerate(all_ty_idents)}
    self._hyp_ident_idx = {ident : i for i, ident in enumerate(all_hyp_idents)}

  def init_buckets(self, diffs: Iterable[int]):
    slices = ranges(self._buckets)
    self._thresholds = np.quantile(list(map(float, diffs)), slices)
    

  def init(self, lemmas: Iterable[Lemma], diffs: Iterable[int]):
    lems, diffs = list(lemmas), list(diffs)
    self.init_idents(lems)
    self.init_buckets(diffs)

    for learner in self._locals:
      learner.init(lems, diffs)

    self._init = True


  def train_transformed(self, xs: np.ndarray, ys: np.ndarray):
    bucket_ys = np.fromiter((find_quantile(self._thresholds, y) for y in ys), ys.dtype, count=len(ys))
    self.fit_classifier(xs, bucket_ys)

    print("bounds:")
    print(self._thresholds)
    threshed_vals = split_diffs(list(zip(xs, ys)), self._thresholds)

    for buck, xys in enumerate(threshed_vals):
      
      # tups = ((x, y) for x, y in zip(xs, ys) if find_quantile(self._thresholds, y) == buck)
      local_xs = np.array(list(iter_fst(xys)), xs.dtype)
      local_ys = np.fromiter(iter_snd(xys), ys.dtype)
      self._locals[buck].train_transformed(local_xs, local_ys)


  def predict_transformed(self, xs: np.ndarray) -> np.ndarray:

    def pred_local(x): 
      bf = self.model.predict([x])
      return self._locals[int(bf)].predict_transformed([x])[0]

    predictions = [pred_local(x) for x in xs]
    # print("predictions:")
    # print(predictions)
    return np.array(predictions)

    # return np.fromiter(
    #     (pred_local(x) for x in xs)
    #   , np.single
    #   , count=len(xs)
    # )


  # assumes lemma_ys has already been transformed into buckets
  def fit_classifier(self, lemma_xs: np.ndarray, lemma_ys: np.ndarray):
    self.model.fit(lemma_xs, lemma_ys)

  def pred_bucket(self, lemma, val : Optional[int] = None):
    buck_float = self.model.predict(self.convert_lemmas([lemma]))
    # print(f'predicted to {int(buck_float)}')
    buck = int(buck_float) # could also try combining the two adjacent quantiles
    
    if val:
      return (buck, find_quantile(self._thresholds, float(val)))
    else:
      return (buck, None)

  def diff_to_buck(self, diff: int) -> int: 
    return find_quantile(self._thresholds, float(diff))

  def accuracy(self, lemmas : List[Tuple[Lemma, int]]):
    return np.mean([1.0 if self.pred_bucket(l, val)[0] == self.pred_bucket(l, val)[1] else 0.0 for l, val in lemmas])

  def np_accuracy(self, lem_xs: np.ndarray, lem_ys: np.ndarray):
    y_pred = self.model.predict(lem_xs)
    return accuracy_score(lem_ys, y_pred)


  @property
  def name(self):
    inner = ",".join([x.name for x in self._locals])
    return f"Ensemble {str(self.model)} + [{inner}]"

  def convert_lemmas(self, lemmas: Iterable[Lemma]):
    lemmas = list(lemmas)
    fst = lemmas[0]
    assert self._init, "uninitialized lemmas! (need to call init_lemmas)"

    first_x_hyp = make_ident_vector(fst, self._hyp_ident_idx, True) if self._use_hypos else []
    x_dim = len(first_x_hyp + make_ident_vector(fst, self._ty_ident_idx, False))

    dater = np.fromiter(chain.from_iterable((make_ident_vector(lem, self._hyp_ident_idx, True) if self._use_hypos else []) + make_ident_vector(lem, self._ty_ident_idx, False) for lem in lemmas), float, count=x_dim*len(lemmas))

    dater.shape = len(lemmas), x_dim
    return dater
  
  def transform_xs(self, lemmas: Iterable[Lemma]) -> np.ndarray:
    return self.convert_lemmas(lemmas)


class KNNIdents(IEnsembleLearner):

  _model : BaseEstimator

  @property
  def model(self): 
    return self._model

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._model = make_pipeline(
        PCA(n_components=0.95)
      , KNeighborsClassifier(weights='distance', n_jobs=-1)
    )


class SVMCV(IEnsembleLearner):

  _model : BaseEstimator
  _should_cv : bool

  @property
  def model(self): 
    return self._model

  def __init__(self, svc_params = None, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    ranges = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    if svc_params:
      self._model = make_pipeline(
          PCA(n_components=0.95)
        , SVC(**svc_params)
      )

      self._should_cv = False
    else:
      self._model = make_pipeline(
          PCA(n_components=0.95)
        , GridSearchCV(
            estimator=SVC()
          , param_grid=[
                {'C': ranges, 'kernel': ['linear']}
              , {'C': ranges, 'gamma': ranges, 'kernel': ['rbf']},
            ]
          , verbose=1
          , n_jobs=-1
        )
      )

      self._should_cv = True


  def fit_classifier(self, lemma_xs: np.ndarray, lemma_ys: np.ndarray):
    ret = super().fit_classifier(lemma_xs, lemma_ys)

    if self._should_cv:
      print("cross validation params:")
      print(self._model[-1].best_params_)
    return ret


def nested_size(obj: Sexpr) -> int:
  if isinstance(obj, sexp.Symbol) or isinstance(obj, str) or isinstance(obj, int):
    return 1
  elif isinstance(obj, List):
    return sum([nested_size(x) for x in obj]) + 1
  else:
    print("weird type?", obj, type(obj))
    raise Exception()

def lemma_size(lemma: Lemma, use_hypos = False):
  pref = []
  if use_hypos:
    pref = [ident_size(x) for x in lemma.hypos]
  return [sum(pref), ident_size(lemma.type)]

def lemma_nested_size(lemma: Lemma, use_hypos = False):
  pref = []
  if use_hypos:
    pref = [float(nested_size(x)) for x in lemma.hypos]
  return [sum(pref), float(nested_size(lemma.type))]

def ident_size(obj: Sexpr) -> int:
  inner = strip_toplevel(obj)
  if not inner:
    return 0

  idents = gather_idents(inner)

  return len(idents)

def strip_toplevel(e: Sexpr) -> Optional[Sexpr]:
  match e:
    case ['CoqConstr', x]:
      return x
    case _ :
      return None

def union_idents(l: Dict[str, int], r: Dict[str, int]) -> Dict[str, int]:
  ret = dict(l)
  for k, v in r.items():
    if k in ret:
      ret[k] += v
    else:
      ret[k] = v
  
  return ret

def gather_idents(e: Sexpr) -> Dict[str, int]:
  match e:
    case [name, *args]:
      if name == Symbol("Ind"):
        return gather_idents(args[0][0][0])
      elif name == Symbol("Prod"):
        match e:
          case [_, nme_binding, typ, bod]:
            inter = union_idents(gather_idents(nme_binding[0]), gather_idents(typ))
            return union_idents(inter, gather_idents(bod))
          case _ : raise UnhandledExpr(e, "Prod")
      elif name == Symbol("Const"):
        match e:
          case [_, [inner, _]]: return gather_idents(inner)
          case _ : raise UnhandledExpr(e, "Const")
      elif name == Symbol("App"):
        f = args[0]
        es = args[1]
        res = gather_idents(f)
        for inner in es:
          res = union_idents(res, gather_idents(inner))
        return res
      elif name == Symbol('binder_name'):
        # [Symbol('binder_name'), [Symbol('Name'), [Symbol('Id'), Symbol('notin')]]]
        if args == [Symbol('Anonymous')]:
            return dict()
        inner = conv_id(args[0][1])
        if inner:
          return {inner : 1}
        else:
          return dict()
      elif name == Symbol("LetIn"):
        # let arg0 : arg1 = arg2 in arg3
        return union_idents(gather_idents(args[1]), union_idents(gather_idents(args[2]), gather_idents(args[3])))
      elif name == Symbol("Lambda"):
        # \ arg0 : arg1 . arg2
        return union_idents(gather_idents(args[0][0]), union_idents(gather_idents(args[1]), gather_idents(args[2])))
      elif name == Symbol("Fix"):
        ret = dict()
        # fixpoints are complicated, see https://coq.github.io/doc/master/api/coq-core/Constr/index.html for the structure
        _, inner = args[0]
        binders, types, bodies = inner
        for nme, _ in binders:
          ret = union_idents(ret, gather_idents(nme))
        for typ in types:
          ret = union_idents(ret, gather_idents(typ))
        for body in bodies:
          ret = union_idents(ret, gather_idents(body))
        return ret
      elif name == Symbol("Rel") or name == Symbol("Var") or name == Symbol("Sort") or name == Symbol("MPbound"):
        return dict()
      elif name == Symbol("Construct"):
        return gather_idents(args[0][0][0][0])

      elif name == Symbol("Case"):

        base = union_idents(gather_idents(args[0][0][1][0]), union_idents(gather_idents(args[1]), gather_idents(args[2])))
        for inner in args[3]:
          base = union_idents(base, gather_idents(inner))
        return base
      elif name == Symbol("MutInd"):
        pref = join_module(args[0])
        if pref:
          return {f"{pref}.{conv_id(args[1])}" : 1}
        else:
          return dict()

      elif name == Symbol('Constant'):
        pref = join_module(args[0])
        if pref:
          return {f"{pref}.{conv_id(args[1])}" : 1}
        else:
          return dict()

      elif name == Symbol('Evar'):
        return dict()
      elif name == Symbol('Cast'):
        return union_idents(gather_idents(args[0]), gather_idents(args[2]))
      elif name == Symbol('Instance'):
          # TODO
        return dict()
      elif args == [] and type(name) is list and len(name) > 0:
        gather_idents(name[0])
      else:
        print("unrecognized symbol", name)
        print("with args", args)
        raise UnhandledExpr(e, "sexpr head")
        # out = set()
        # for arg in args:
        #   out |= gather_idents(arg)
        # return out
    case int(_) | str(_): return dict()
    case _: raise UnhandledExpr(e, "Top level")

def join_module(e: Sexpr) -> Optional[str]:
  match e:
    case [name, [_, path]] if name == Symbol("MPfile"):
      paths = [conv_id(x) for x in path]
      return '.'.join(paths)
    case [name, inner, outer] if name == Symbol("MPdot"):
      return f"{join_module(inner)}.{conv_id(outer)}"
    case [name, *args] if name == Symbol("MPbound"):
      return None
    case _:
      raise UnhandledExpr(e, "joint module")
  return None

def conv_id(e: Sexpr) -> Optional[str]:
  match e:
    case [l, r] if l == Symbol('Id'):
      if isinstance(r, Symbol):
        return r.value()
      else:
        return None
    case _ : return None
  return None

def lemma_ty_idents(l: Lemma) -> dict[str, int]:
  x = strip_toplevel(l.type)
  if x: return gather_idents(x)
  else: 
    print(f"couldn't handle type idents for {l.name}")
    print(l.type)
    print(type(l.type))
    assert False

def lemma_hyp_idents(l: Lemma) -> dict[str, int]:
  ret = dict()
  for hyp in l.hypos:
    x = strip_toplevel(hyp)
    if x: ret = gather_idents(x)
    else: 
      print(f"couldn't handle hyp idents for {l.name}")
      print(hyp)
      print(type(hyp))
      assert False
  return ret

class SVCCategories:
  
  _should_CV: bool
  _model : BaseEstimator

  _svr_params : dict[Any, Any]

  _upper_quantile : np.single
  _lower_quantile : np.single

  @property 
  def svr_params(self):
    return self._svr_params

  def __init__(self, lower, upper, svr_params = None, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._lower_quantile = lower
    self._upper_quantile = upper
    if svr_params:
      self._svr_params = svr_params
      self._model = make_pipeline(
          StandardScaler()
        , PCA(n_components=0.95)
        , SVC(**svr_params, class_weight='balanced')
      )
      self._should_CV = False
    else:
      self._should_CV = True
      self._svr_params = {}
      ranges = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
      self._model = make_pipeline(
          StandardScaler()
        , PCA(n_components=0.95)
        , GridSearchCV(
            estimator=SVR()
          , param_grid=[
                {'C': ranges, 'gamma': ranges + ['scale'], 'kernel': ['rbf']},
            ]
          , n_jobs=-1
          , verbose=3
        )
      )

  def train(self, xs: np.ndarray, ys: np.ndarray):
    self._upper_quantile = np.quantile(ys, self._upper_quantile)
    self._lower_quantile = np.quantile(ys, self._lower_quantile)

    ys_categories = self.categorize(ys)

    self._model.fit(xs, ys_categories)

  def predict(self, xs: np.ndarray) -> np.ndarray:
    if self._should_CV:
      print("cross validation params:")
      print(self._model[-1].best_params_)
    return self._model.predict(xs)

  def categorize(self, ys: np.ndarray):
    return np.fromiter(
      (categorize_value(y, self._upper_quantile, self._lower_quantile) for y in ys),
      np.single,
      count=len(ys)
    )


class SVCCatBinary(SVCCategories):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def train(self, xs: np.ndarray, ys: np.ndarray):
    self._upper_quantile = np.single(np.inf)
    self._lower_quantile = np.single(3.0)

    ys_categories = self.categorize(ys)

    self._model.fit(xs, ys_categories)

@njit
def categorize_value(v: np.single, upper: np.single, lower: np.single):
  return \
    1.0 if v <= lower else \
    2.0 if v <= upper else \
    3.0

class SVCCatEnsemble:
  
  _model_lower : SVRC2V
  _model_middle : SVRC2V
  _model_upper : SVRC2V

  _model_cat : SVCCategories


  def __init__(self, svc_cat, lower_params, middle_params, upper_params, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._model_cat = svc_cat
    self._model_lower = SVRC2V(svr_params = lower_params)
    self._model_middle = SVRC2V(svr_params = middle_params)
    self._model_upper = SVRC2V(svr_params = upper_params)

  def train(self, xs: np.ndarray, ys: np.ndarray):

    ys_categories = self._model_cat.categorize(ys)

    xs_low = np.fromiter((x_ for i, x in enumerate(xs) for x_ in x.flatten() if ys_categories[i] == 1.0), np.single)
    xs_low = xs_low.reshape((-1, xs.shape[-1]))
    ys_low = np.fromiter((x for i, x in enumerate(ys) if ys_categories[i] == 1.0), np.single)

    self._model_lower.train_transformed(xs_low, ys_low)
    xs_low = []
    ys_low = []

    xs_middle = np.fromiter((x_ for i, x in enumerate(xs) for x_ in x.flatten() if ys_categories[i] == 2.0), np.single)
    xs_middle = xs_middle.reshape((-1, xs.shape[-1]))
    ys_middle = np.fromiter((x for i, x in enumerate(ys) if ys_categories[i] == 2.0), np.single)

    self._model_middle.train_transformed(xs_middle, ys_middle)
    xs_middle = []
    ys_middle = []

    xs_upper = np.fromiter((x_ for i, x in enumerate(xs) for x_ in x.flatten() if ys_categories[i] == 3.0), np.single)
    xs_upper = xs_upper.reshape((-1, xs.shape[-1]))
    ys_upper = np.fromiter((x for i, x in enumerate(ys) if ys_categories[i] == 3.0), np.single)
    self._model_upper.train_transformed(xs_upper, ys_upper)

  # assumes that the input has been transformed by a c2v pass already
  def predict(self, xs: np.ndarray):
    return np.fromiter((
      self._model_lower.predict_transformed(np.array([x]))[0] if self._model_cat.predict(np.array([x]))[0] == 1.0 else
      self._model_middle.predict_transformed(np.array([x]))[0] if self._model_cat.predict(np.array([x]))[0] == 2.0 else 
      self._model_upper.predict_transformed(np.array([x]))[0]
      for x in xs
    ), np.single, count=len(xs))

  def predict_obl(self, obl: coq2vec.Obligation) -> np.ndarray:
    xs = np.array([self._model_lower._vectorizer.obligation_to_vector(obl).numpy().flatten()])
    return self.predict(xs)
