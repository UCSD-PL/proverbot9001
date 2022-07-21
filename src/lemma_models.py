
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple, Union, Any, Optional, Callable
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression # for linear regression of type vs difficulty
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection # for k-fold splitting, see source at https://datascience.stanford.edu/news/splitting-data-randomly-can-ruin-your-model
from sklearn.model_selection import train_test_split
import numpy as np

import sexpdata as sexp

import random as rand

Symbol = sexp.Symbol

Sexpr = Union[sexp.Symbol, int, str, List['Sexpr']]

@dataclass
class Lemma:
  name: str
  type: Sexpr

class ILearner(ABC):
  @abstractmethod
  def learn(self, lemmas: List[Tuple[Lemma, int]]) -> None:
    raise NotImplemented

  @abstractmethod
  def predict(self, lemma: Lemma) -> float:
    raise NotImplemented

  @property
  @abstractmethod
  def name(self) -> str:
    raise NotImplemented

class UnhandledExpr(Exception):

  def __init__(self, e: Sexpr) -> None:
    super().__init__(e)

class NaiveMeanLearner(ILearner):

  def __init__(self) -> None:
    super().__init__()
    self._mean: float = 0

  def learn(self, lemmas):
    self._mean = float(np.mean([x for _, x in lemmas]))

  def predict(self, lemma): return self._mean

  @property
  def name(self):
    return "naive mean"

class LinRegressionLearner(ILearner):

  _model : LinearRegression

  def __init__(self) -> None:
    super().__init__()
    self._model = LinearRegression()

  def learn(self, lemmas):
    lems, ys = zip(*lemmas)
    xs = [[float(nested_size(l.type))] for l in lems]

    self._model.fit(xs, ys)

  def predict(self, lemma):
    return self._model.predict([[float(nested_size(lemma.type))]])

  @property
  def name(self):
    return "LR size"


class SVRLength(ILearner):


  def __init__(self) -> None:
    super().__init__()
    self._model : Any = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

  def learn(self, lemmas):
    lems, ys = zip(*lemmas)
    xs = [[float(nested_size(l.type))] for l in lems]

    self._model.fit(xs, ys)

  def predict(self, lemma):
    return self._model.predict([[float(nested_size(lemma.type))]])

  @property
  def name(self):
    return "SVR size"

class SVRIdent(ILearner):

  def __init__(self) -> None:
    super().__init__()
    self._model : Any = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

  def learn(self, lemmas):
    lems, ys = zip(*lemmas)
    xs = [[float(ident_size(l.type))] for l in lems]

    self._model.fit(xs, ys)

  def predict(self, lemma):
    return self._model.predict([[float(ident_size(lemma.type))]])

  @property
  def name(self):
    return "SVR idents"

class SVRIdentLength(ILearner):

  def __init__(self) -> None:
    super().__init__()
    self._model : Any = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

  def learn(self, lemmas):
    lems, ys = zip(*lemmas)
    xs = [[float(ident_size(l.type)), float(nested_size(l.type))] for l in lems]

    self._model.fit(xs, ys)

  def predict(self, lemma):
    return self._model.predict([[float(ident_size(lemma.type)), float(nested_size(lemma.type))]])

  @property
  def name(self):
    return "SVR idents + size"

def ranges(N: int):
  return [float(k) / float(N) for k, _ in enumerate(range(N))]

def find_quantile(buckets: list[float], x: float):
  for i, buck in enumerate(buckets):
    if x <= buck: return i-1
  return i-1

def make_ident_vector(lem: Lemma, idxs: dict[str, int]) -> list[float]:
  idents = {x for x in lemma_idents(lem)}
  return [1.0 if ident in idents else 0.0 for ident in idxs ]

class KNNIdents(ILearner):

  _model : Any
  _locals : List[ILearner]
  _buckets : int
  _ident_idx : dict[str, int]

  def __init__(self, local_learner : Callable[[int], ILearner], buckets: int, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._model = make_pipeline(StandardScaler(), KNeighborsClassifier(weights='distance'))
    self._locals = []
    self._buckets = 0
    self._ident_idx = {}
    self._buckets = buckets
    self._locals = [local_learner(i) for i in range(buckets)]

  def learn(self, lemmas):
    # gather up all of the idents
    lems, difficulties = zip(*lemmas)
    all_idents : set[str] = set()
    print("gathering idents...")
    for lem in lems:
      all_idents |= lemma_idents(lem)

    ident_idx : dict[str, int] = {ident : i for i, ident in enumerate(all_idents)}

    self._ident_idx = ident_idx

    # train idents -> quantile
    # print("gathering ident data...")
    slices = ranges(self._buckets)
    thresholds = np.quantile(difficulties, slices)
    # print('computed thresholds:', thresholds)
    ident_xs : list[list[float]] = [ make_ident_vector(lem, ident_idx) for lem in lems]
    ident_ys : list[int] = [ find_quantile(thresholds, diff) for diff in difficulties ]


    print('fitting model for ident -> bucket')
    self._model.fit(ident_xs, ident_ys)

    print('score:', self._model.score(ident_xs, ident_ys))


    # next, train learners for each of the buckets


    for i, thresh in enumerate(thresholds):
      if i == len(thresholds) - 1:
        lo = thresh
        hi = max(difficulties)
      else:
        lo = thresh
        hi = thresholds[i+1]

      data = [(lem, d) for lem, d in lemmas if lo <= d and d <= hi]

      learner = self._locals[i]

      # print(f"learning using {learner.name}")
      learner.learn(data)
    print("done training individual learners")

    # just a sanity check, pick a random element and doublecheck the computed stuff
    # test_x, test_y = rand.choice(lemmas)

    # thresh = find_quantile(thresholds, test_y)
    # print("value and computed quantile:", test_y, thresh)
    # pred_bucket = int(self._model.predict([make_ident_vector(test_x, self._ident_idx)]))
    # print("predicted bucket:", pred_bucket)
    # for buck in range(len(thresholds)):
    #   print(f"output of learner at bucket {buck}:", self._locals[buck].predict(test_x))


  def predict(self, lemma):
    # get the bucket
    buck_float = self._model.predict([make_ident_vector(lemma, self._ident_idx)])
    # print(f'predicted to {int(buck_float)}')
    buck = int(buck_float) # could also try combining the two adjacent quantiles

    return self._locals[buck].predict(lemma)

  @property
  def name(self):
    inner = ",".join([x.name for x in self._locals])
    return f"KNN idents + [{inner}]"

def nested_size(obj: Sexpr) -> int:
  if isinstance(obj, sexp.Symbol) or isinstance(obj, str) or isinstance(obj, int):
    return 1
  elif isinstance(obj, List):
    return sum([nested_size(x) for x in obj]) + 1
  else:
    print("weird type?", obj, type(obj))
    raise Exception()

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

def gather_idents(e: Sexpr) -> set[str]:
  match e:
    case [name, *args]:
      if name == Symbol("Ind"):
        return gather_idents(args[0][0][0])
      elif name == Symbol("Prod"):
        match e:
          case [_, nme_binding, typ, bod]:
            return gather_idents(nme_binding[0]) | gather_idents(typ) | gather_idents(bod)
          case _ : raise UnhandledExpr(e)
      elif name == Symbol("Const"):
        match e:
          case [_, [inner, _]]: return gather_idents(inner)
          case _ : raise UnhandledExpr(e)
      elif name == Symbol("App"):
        f = args[0]
        es = args[1]
        res = gather_idents(f)
        for inner in es:
          res |= gather_idents(inner)
        return res
      elif name == Symbol('binder_name'): 
        # [Symbol('binder_name'), [Symbol('Name'), [Symbol('Id'), Symbol('notin')]]]
        if args == [Symbol('Anonymous')]:
            return set()
        inner = conv_id(args[0][1])
        if inner:
          return {inner}
        else:
          return set()
      elif name == Symbol("LetIn"):
        # let arg0 : arg1 = arg2 in arg3
        return gather_idents(args[1]) | gather_idents(args[2]) | gather_idents(args[3])
      elif name == Symbol("Lambda"):
        # \ arg0 : arg1 . arg2
        return gather_idents(args[0][0]) | gather_idents(args[1]) | gather_idents(args[2])
      elif name == Symbol("Fix"):
        ret = set()
        # fixpoints are complicated, see https://coq.github.io/doc/master/api/coq-core/Constr/index.html for the structure
        _, inner = args[0]
        binders, types, bodies = inner
        for nme, _ in binders:
          ret |= gather_idents(nme)
        for type in types:
          ret |= gather_idents(type)
        for body in bodies:
          ret |= gather_idents(body)
        return ret
      elif name == Symbol("Rel") or name == Symbol("Var") or name == Symbol("Sort") or name == Symbol("MPbound"):
        return set()
      elif name == Symbol("Construct"):
        return gather_idents(args[0][0][0][0])

      elif name == Symbol("Case"):

        base = gather_idents(args[0][0][1][0]) | gather_idents(args[1]) | gather_idents(args[2])
        for inner in args[3]:
          base |= gather_idents(inner)
        return base
      elif name == Symbol("MutInd"):
        pref = join_module(args[0])
        if pref:
          return {f"{pref}.{conv_id(args[1])}"}
        else:
          return set()

      elif name == Symbol('Constant'):
        pref = join_module(args[0])
        if pref:
          return {f"{pref}.{conv_id(args[1])}"}
        else:
          return set()

      elif name == Symbol('Evar'):
          return set()
      else:
        print("unrecognized symbol", name)
        raise UnhandledExpr(e)
        # out = set()
        # for arg in args:
        #   out |= gather_idents(arg)
        # return out
    case int(_) | str(_): return set()
    case _: raise UnhandledExpr(e)

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
      raise UnhandledExpr(e)
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

def lemma_idents(l: Lemma) -> set[str]:
  x = strip_toplevel(l.type)
  if x: return gather_idents(x)
  else: return set()
