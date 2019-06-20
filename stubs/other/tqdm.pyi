
import sys
from typing import Dict, Optional, TextIO, Iterable
class tqdm:
  def __init__(self, iterable=None, desc=None, total=None, leave=True,
               file=None, ncols=None, mininterval=0.1,
               maxinterval=10.0, miniters=None, ascii=None, disable=False,
               unit='it', unit_scale=False, dynamic_ncols=False,
               smoothing=0.3, bar_format=None, initial=0, position=None,
               postfix=None, unit_divisor=1000) -> None:
      ...
  def update(self, n:int=1) -> None:
      """
      Manually update the progress bar, useful for streams
      such as reading files.
      E.g.:
      >>> t = tqdm(total=filesize) # Initialise
      >>> for current_buffer in stream:
      ...    ...
      ...    t.update(len(current_buffer))
      >>> t.close()
      The last line is highly recommended, but possibly not necessary if
      ``t.update()`` will be called in such a way that ``filesize`` will be
      exactly reached and printed.

      Parameters
      ----------
      n  : int, optional
          Increment to add to the internal counter of iterations
          [default: 1].
      """
      ...
  def __enter__(self) -> "tqdm":
    ...
  def __exit__(self, *args) -> None:
    ...
  def close(self) -> None:
      """Cleanup and (if leave=False) close the progressbar."""
      ...
  def clear(self, nomove:bool=False) -> None:
      """Clear current bar display."""
      ...
  def refresh(self) -> None:
      """Force refresh the display of this bar."""
      ...
  def unpause(self) -> None:
      """Restart tqdm timer from last print time."""
      ...
  def reset(self, total:Optional[int]=None) -> None:
      """
      Resets to 0 iterations for repeated use.

      Consider combining with ``leave=True``.

      Parameters
      ----------
      total  : int, optional. Total to use for the new bar.
      """
      ...
  def set_description(self, desc:Optional[str]=None, refresh:bool=True) -> None:
      """
      Set/modify description of the progress bar.

      Parameters
      ----------
      desc  : str, optional
      refresh  : bool, optional
          Forces refresh [default: True].
      """
      ...

  def set_postfix(self, ordered_dict:Optional[Dict]=None, refresh:bool=True, **kwargs) \
      -> None:
      """
      Set/modify postfix (additional stats)
      with automatic formatting based on datatype.

      Parameters
      ----------
      ordered_dict  : dict or OrderedDict, optional
      refresh  : bool, optional
          Forces refresh [default: True].
      kwargs  : dict, optional
      """
      ...

  @classmethod
  def write(cls, s, file:TextIO=sys.stdout, end:str="\n") -> None:
      """Print a message via tqdm (without overlap with bars)."""
      ...

  @property
  def format_dict(self) -> Dict[str,str]:
      """Public API for read-only member access."""
      ...

  def display(self, msg:Optional[str]=None, pos:Optional[int]=None) -> None:
      """
      Use ``self.sp`` to display ``msg`` in the specified ``pos``.

      Consider overloading this function when inheriting to use e.g.:
      ``self.some_frontend(**self.format_dict)`` instead of ``self.sp``.

      Parameters
      ----------
      msg  : str, optional. What to display (default: ``repr(self)``).
      pos  : int, optional. Position to ``moveto``
        (default: ``abs(self.pos)``).
      """
      ...

def trange(*args, **kwargs) -> Iterable[int]:
    """
    A shortcut for tqdm(xrange(*args), **kwargs).
    On Python3+ range is used instead of xrange.
    """
    ...
