from typing import overload, IO
class Path():
    def __init__(self, str_path : str) -> None:
        ...
    def with_suffix(self, new_suffix : str) -> Path:
        ...
    @property
    def stem(self) -> str:
        ...
    @property
    def suffix(self) -> str:
        ...
    @property
    def parent(self) -> Path:
        pass
    def open(self, mode='r',newline='\n') -> IO[str]:
        ...
    @overload
    def __truediv__(self, other : Path) -> Path:
        ...
    @overload
    def __truediv__(self, other : str) -> Path:
        ...
    pass
class Path2(Path):
    @overload
    def __init__(self, str_path : str) -> None:
        ...
    @overload
    def __init__(self, str_path : Path2) -> None:
        ...
    @property
    def parent(self) -> Path2:
        pass
    def with_suffix(self, new_suffix : str) -> Path2:
        ...
    def exists(self) -> bool:
        ...
    def copyfile(self, dest : Path) -> None:
        ...
    @overload
    def __truediv__(self, other : Path) -> Path2:
        ...
    @overload
    def __truediv__(self, other : str) -> Path2:
        ...
    pass
