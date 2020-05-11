
from typing import List, Optional

class ScrapedTactic:
    relevant_lemmas : List[str]
    prev_tactics: List[str]
    prev_hyps: List[str]
    prev_goal: str
    tactic: str

def scraped_tactics_from_file(filename : str, num_tactics : Optional[int]) -> List[ScrapedTactic]:
    ...
