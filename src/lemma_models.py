
from abc import ABC, abstractmethod, abstractproperty
from tabnanny import verbose
from typing import Dict, Iterable, Iterator, List, Tuple, TypeVar, TypedDict, Union, Any, Optional, Callable
from dataclasses import dataclass

import coq2vec

from coq2vec import (
  Obligation
)

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

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

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

def lem2obl(lem: Lemma) -> Obligation:
  return Obligation(hypotheses=lem.ctx["hyp_strs"], goal=lem.ctx["goal_str"])

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
          "hypos": [eval(x, None, {"Symbol" : Symbol}) for x in ctx["hypos"]]
        , "type": eval(ctx["type"], None, {"Symbol" : Symbol})
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

class ITransformer(ABC):
  @abstractmethod
  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) -> np.ndarray:
    raise NotImplemented

  def transform_ys(self, diffs: Iterable[int]) -> np.ndarray:
    return np.fromiter(diffs, np.single)

  @abstractmethod
  def predict(self, obl: Obligation, lem: Lemma):
    raise NotImplemented

class ILearner(ITransformer):

  _use_hypos : bool

  def __init__(self, *args, **kwargs) -> None:
    if "use_hypos" in kwargs:
      self._use_hypos = kwargs["use_hypos"]
    else:
      self._use_hypos = False

  def learn(self, lemmas: Iterable[Tuple[Tuple[Obligation, Lemma], int]]) -> None:
    self.init(iter_fst(lemmas), iter_snd(lemmas))
    xs = self.transform_xs(iter_fst(lemmas))
    ys = self.transform_ys(iter_snd(lemmas))
    self.train_transformed(xs, ys)

  def predict(self, obl: Obligation, lem: Lemma) -> float:
    xs = self.transform_xs([(obl, lem)])
    return self.predict_transformed(xs)[0]

  def init(self, lemmas: Iterable[Tuple[Obligation, Lemma]], diffs: Iterable[int]) -> None:
    return None

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

  _mean: np.float64

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._mean = np.float64(0.0)

  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) :
    return np.array([np.array([1.0]) for _ in lemmas])

  def transform_ys(self, diffs: Iterable[int]):
    return np.fromiter(diffs, np.single)

  def train_transformed(self, xs: np.ndarray, ys: np.ndarray):
    self._mean = np.mean(ys, dtype=np.float64)

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

  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) :
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

  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) :
    xs = [self.lem_to_x(lem) for _, lem in lemmas]
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
      ranges = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] # TODO: ma
      epsilon_ranges = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
      self._model = make_pipeline(
          StandardScaler()
        , PCA(n_components=0.95)
        , GridSearchCV(
            estimator=SVR()
          , param_grid=[
                {'C': ranges, 'epsilon': epsilon_ranges, 'gamma': ranges + ['scale'], 'kernel': ['rbf']},
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

class ISVRFactory(ABC):
  @abstractmethod
  def make_svr(self, *args, **kwargs) -> ISVRLearner: pass


class SVRLength(ISVRLearner):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
  
  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) :
    return np.array([
      np.array(lemma_nested_size(lemma, use_hypos=self._use_hypos)) for _, lemma in lemmas
    ])
    
  @property
  def name(self):
    return "SVR size"
  
def build_names(lem: Lemma) -> dict[str, int]:
  return union_idents(lemma_hyp_idents(lem), lemma_ty_idents(lem))

class SVRLFactory(ISVRFactory):
  def make_svr(self, *args, **kwargs) -> ISVRLearner:
    return SVRLength(*args, **kwargs)

class SVRIdent(ISVRLearner):

  _names: dict[str, int] # map from idents (strings) to number of times the ident occur

  # be careful with names and make sure that it is only built from training data
  def __init__(self, names: dict[str, int], *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._names = names

  def init(self, lemmas: Iterable[Tuple[Obligation, Lemma]], diffs: Iterable[int]) -> None:
    for _, lem in lemmas:
      self._names = union_idents(self._names, union_idents(lemma_hyp_idents(lem), lemma_ty_idents(lem)))

  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) :
    return np.array([
      make_ident_vector(lem, self._names, True) + make_ident_vector(lem, self._names, False) for _, lem in lemmas
    ])

  @property
  def name(self):
    return "SVR ident"
  
  def predict(self, obl: Obligation, lem: Lemma) -> np.ndarray:
    x = self.transform_xs([(obl, lem)])
    return self._model.predict(x)[0]
  

class SVRIdentFactory(ISVRFactory):
  def make_svr(self, names: dict[str, int], *args, **kwargs) -> ISVRLearner:
    return SVRIdent(names, *args, **kwargs)

class SVRC2V(ISVRLearner):

  _vectorizer: coq2vec.CoqContextVectorizer

  def __init__(self, encoder_weights_path : str = "encoder_model.dat", *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    goal_vectorizer = coq2vec.CoqTermRNNVectorizer()
    goal_vectorizer.load_weights(encoder_weights_path)

    self._vectorizer = coq2vec.CoqContextVectorizer(goal_vectorizer, 4)

  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) -> np.ndarray:
    obls = [obl for obl, _ in lemmas]
    return np.array([x.flatten() for x in self._vectorizer.obligations_to_vectors(obls).numpy()])

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

  def predict_obl(self, obl: Obligation) -> np.ndarray:
    x : np.ndarray = self._vectorizer.obligation_to_vector(obl).numpy().flatten()
    
    return self.predict_transformed(np.array([x]))
  
class SVRC2VFactory(ISVRFactory):
  def make_svr(self, encoder_weights_path : str = "encoder_model.dat", *args, **kwargs) -> ISVRLearner:
    return SVRC2V(encoder_weights_path=encoder_weights_path, *args, **kwargs)

class SVRCounts(ISVRLearner):

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) -> np.ndarray:
    return super().transform_xs(lemmas)

  @property
  def name(self):
    return "SVR Counts"
  
class SVRCountsFactory(ISVRFactory):
  def make_svr(self, *args, **kwargs) -> ISVRLearner:
    return SVRCounts(*args, **kwargs)

def make_ident_vector(lem: Lemma, idxs: dict[str, int], use_hyps : bool) -> list[float]:
  if use_hyps:
    idents = lemma_hyp_idents(lem)
  else:
    idents = lemma_ty_idents(lem)
  
  ret = [0.0] * (len(idxs)+1)
  for ident, value in idents.items():
    if ident in idxs:
      ret[idxs[ident] + 1] = float(value)
    else:
      ret[0] += 1.0
  return ret

# class KNNIdents(IEnsembleLearner):

#   _model : BaseEstimator

#   @property
#   def model(self): 
#     return self._model

#   def __init__(self, *args, **kwargs) -> None:
#     super().__init__(*args, **kwargs)
#     self._model = make_pipeline(
#         PCA(n_components=0.95)
#       , KNeighborsClassifier(weights='distance', n_jobs=-1)
#     )


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
      raise UnhandledExpr(e, "join module")
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

class SVCCategories(ITransformer):
  
  _should_CV: bool
  _model : BaseEstimator

  _svr_params : dict[Any, Any]

  _upper_quantile : np.single
  _lower_quantile : np.single

  @property 
  def svc_params(self):
    return self._svc_params

  def __init__(self, lower, upper, svc_params = None, *args, **kwargs) -> None:
    self._lower_quantile = lower
    self._upper_quantile = upper
    if svc_params:
      self._svc_params = svc_params
      self._model = make_pipeline(
          StandardScaler()
        , PCA(n_components=0.95)
        , SVC(**svc_params, class_weight='balanced')
      )
      self._should_CV = False
    else:
      self._should_CV = True
      self._svc_params = {}
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

    if self._should_CV:
      print("cross validation params:")
      print(self._model[-1].best_params_)

  def predict_transformed(self, xs: np.ndarray) -> np.ndarray:
    return self._model.predict(xs)

  def categorize(self, ys: np.ndarray):
    return np.fromiter(
      (categorize_value(y, self._upper_quantile, self._lower_quantile) for y in ys),
      np.single,
      count=len(ys)
    )


class SVCCatBinary(SVCCategories):

  def __init__(self, *args, **kwargs) -> None:
    kwargs = {k: v for k, v in kwargs.items()}
    kwargs["lower"] = 0.0
    kwargs["upper"] = 0.0
    super().__init__(*args, **kwargs) # lower, upper filled in in training

  def train(self, xs: np.ndarray, ys: np.ndarray):
    self._upper_quantile = np.single(np.inf)
    self._lower_quantile = np.single(3.0)

    ys_categories = self.categorize(ys)

    self._model.fit(xs, ys_categories)

class C2VCategories(SVCCategories):

  _vectorizer: coq2vec.CoqContextVectorizer

  def __init__(self, encoder_weights_path : str = "encoder_model.dat", *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    goal_vectorizer = coq2vec.CoqTermRNNVectorizer()
    goal_vectorizer.load_weights(encoder_weights_path)

    self._vectorizer = coq2vec.CoqContextVectorizer(goal_vectorizer, 4)

  def __getstate__(self):
    state = self.__dict__.copy()
    
    state["_vectorizer"] = self._vectorizer.term_encoder.get_state()

    return state

  def __setstate__(self, state):
    vectorizer = coq2vec.CoqTermRNNVectorizer()
    vectorizer.load_state(state["_vectorizer"])
    state["_vectorizer"] = coq2vec.CoqContextVectorizer(vectorizer, 4)
    self.__dict__.update(state)

  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) -> np.ndarray:
    obls = [obl for obl, _ in lemmas]
    return np.array([x.flatten() for x in self._vectorizer.obligations_to_vectors(obls).numpy()])
  
  def predict(self, obl: Obligation, lem: Lemma):
    xs = self.transform_xs([(obl, lem)])
    return super().predict_transformed(xs)


class SVRCategories(SVCCategories):

  _learner : ISVRLearner # use this model just for the transform_xs method

  def __init__(self, learner_factory: ISVRFactory, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._learner = learner_factory.make_svr(*args, **kwargs)

  def transform_xs(self, lemmas: Iterable[Tuple[Obligation, Lemma]]) -> np.ndarray:
    return self._learner.transform_xs(lemmas=lemmas)
  
  def predict(self, obl: Obligation, lem: Lemma):
    xs = self.transform_xs([(obl, lem)])
    return super().predict_transformed(xs)
  

@njit
def categorize_value(v: np.single, upper: np.single, lower: np.single):
  return \
    1.0 if v <= lower else \
    2.0 if v <= upper else \
    3.0

class SVRCatEnsemble:
  
  _model_lower : ISVRLearner
  _model_middle : ISVRLearner
  _model_upper : ISVRLearner

  _model_cat : SVCCategories

  def __init__(self, svc_cat, inter_model_builder, lower_params=None, middle_params=None, upper_params=None, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._model_cat = svc_cat
    if "use_hypos" in kwargs:
      self._use_hypos = kwargs["use_hypos"]
    else:
      self._use_hypos = False
      
    self._model_lower = inter_model_builder(svr_params = lower_params)
    self._model_middle = inter_model_builder(svr_params = middle_params)
    self._model_upper = inter_model_builder(svr_params = upper_params)

  def train(self, xs: np.ndarray, ys: np.ndarray):

    # logging.info(f"total: {len(xs)}")

    ys_categories = self._model_cat.categorize(ys)

    xs_low = np.fromiter((x_ for i, x in enumerate(xs) for x_ in x.flatten() if ys_categories[i] == 1.0), np.single)
    xs_low = xs_low.reshape((-1, xs.shape[-1]))
    # logging.info(f"total lower: {len(xs_low)}")
    ys_low = np.fromiter((x for i, x in enumerate(ys) if ys_categories[i] == 1.0), np.single)

    self._model_lower.train_transformed(xs_low, ys_low)
    # logging.info("trained lower")
    xs_low = []
    ys_low = []

    xs_middle = np.fromiter((x_ for i, x in enumerate(xs) for x_ in x.flatten() if ys_categories[i] == 2.0), np.single)
    xs_middle = xs_middle.reshape((-1, xs.shape[-1]))
    # logging.info(f"total middle: {len(xs_middle)}")
    ys_middle = np.fromiter((x for i, x in enumerate(ys) if ys_categories[i] == 2.0), np.single)

    self._model_middle.train_transformed(xs_middle, ys_middle)
    # logging.info("trained middle")
    xs_middle = []
    ys_middle = []

    xs_upper = np.fromiter((x_ for i, x in enumerate(xs) for x_ in x.flatten() if ys_categories[i] == 3.0), np.single)
    xs_upper = xs_upper.reshape((-1, xs.shape[-1]))
    # logging.info(f"total upper: {len(xs_upper)}")
    ys_upper = np.fromiter((x for i, x in enumerate(ys) if ys_categories[i] == 3.0), np.single)
    self._model_upper.train_transformed(xs_upper, ys_upper)
    # logging.info("trained upper")

  # assumes that the input has been transformed by a c2v pass already
  def predict(self, xs: np.ndarray, xs_cat: np.ndarray):
    return np.fromiter((
      self._model_lower.predict_transformed(np.array([x]))[0] if self._model_cat.predict_transformed(np.array([xs_cat[i]]))[0] == 1.0 else
      self._model_middle.predict_transformed(np.array([x]))[0] if self._model_cat.predict_transformed(np.array([xs_cat[i]]))[0] == 2.0 else 
      self._model_upper.predict_transformed(np.array([x]))[0]
      for i, x in enumerate(xs)
    ), np.single, count=len(xs))

  def predict_obl(self, obl: Obligation, lem: Lemma) -> np.ndarray:
    xs = self._model_lower.transform_xs([(obl, lem)])
    # xs = np.array([self._model_lower._vectorizer.obligation_to_vector(obl).numpy().flatten()])
    xs_cat = self._model_cat.transform_xs([(obl, lem)])
    return self.predict(xs, xs_cat)

class SVCCV:

  _should_CV: bool
  _model : BaseEstimator

  _svr_params : dict[Any, Any]
  _vectorizer: coq2vec.CoqContextVectorizer

  @property 
  def svc_params(self):
    return self._svc_params

  def __init__(self, encoder_weights_path : str = "encoder_model.dat", svc_params = None, *args, **kwargs) -> None:

    goal_vectorizer = coq2vec.CoqTermRNNVectorizer()
    goal_vectorizer.load_weights(encoder_weights_path)

    self._vectorizer = coq2vec.CoqContextVectorizer(goal_vectorizer, 4)


    if svc_params:
      self._svc_params = svc_params
      self._model = make_pipeline(
          StandardScaler()
        , PCA(n_components=0.95)
        , SVC(**svc_params, class_weight='balanced')
      )
      self._should_CV = False
    else:
      self._should_CV = True
      self._svc_params = {}
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
    self._model.fit(xs, ys)

    if self._should_CV:
      print("cross validation params after training:")
      print(self._model[-1].best_params_)

  def predict_transformed(self, xs: np.ndarray) -> np.ndarray:
    return self._model.predict(xs)

  def __getstate__(self):
    state = self.__dict__.copy()

    state["_vectorizer"] = self._vectorizer.term_encoder.get_state()

    return state

  def __setstate__(self, state):
    vectorizer = coq2vec.CoqTermRNNVectorizer()
    vectorizer.load_state(state["_vectorizer"])
    state["_vectorizer"] = coq2vec.CoqContextVectorizer(vectorizer, 4)
    self.__dict__.update(state)

  def predict_obl(self, obl: Obligation) -> np.ndarray:

    xs = np.array([self._vectorizer.obligation_to_vector(obl).numpy().flatten()])
    return self.predict_transformed(xs)

class SVCThreshold:
  _inner_cat: SVCCV
  _inner_model: ILearner
  _max: float

  def __init__(self, _inner_cat: SVCCV, _max: float, _inner_model: ILearner, *args, **kwargs) -> None:
    self._max = _max
    self._inner_cat = _inner_cat
    self._inner_model = _inner_model

  def predict_transformed(self, xs: np.ndarray) -> np.ndarray:
    return np.fromiter((self._inner_model.predict_transformed(np.array([x]))[0][0] if not(self._inner_cat.predict_transformed(np.array([x]))[0][0]) else self._max for x in xs), dtype=np.double, count=len(xs)).reshape((1, -1))

  def predict_obl(self, obl: Obligation) -> np.ndarray:

    xs = np.array([self._inner_cat._vectorizer.obligation_to_vector(obl).numpy().flatten()])
    return self.predict_transformed(xs)
