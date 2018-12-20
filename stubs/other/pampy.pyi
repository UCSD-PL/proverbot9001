
from typing import Any

# Unfortunately the true type signature here requires being able to
# specify the type "form" of variable arguments: that the first
# argument has one time, and then every *pair* of arguments after has
# another.
def match(*args : Any) -> Any:
    ...

_ = ...

TAIL = ...
