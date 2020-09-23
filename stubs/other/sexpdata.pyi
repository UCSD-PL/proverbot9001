
from typing import Union, List, Any, Optional

Sexp = Union['SExpBase', List[Any]]


class SExpBase:
    ...


class Symbol(SExpBase):
    def __init__(self, name: str) -> None: ...
    ...


class ExpectClosingBracket(Exception):
    ...


def loads(s: str, nil: Optional[str] = None) -> Sexp: ...


def dumps(s: Sexp) -> str: ...
