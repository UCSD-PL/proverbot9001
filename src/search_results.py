#!/usr/bin/env python3

from enum import Enum, auto
from typing import NamedTuple, Optional, List, Union
from coq_serapy import ProofContext

class ReportStats(NamedTuple):
    filename: str
    num_proofs: int
    num_proofs_failed: int
    num_proofs_completed: int


class SearchStatus(str, Enum):
    SUCCESS = 'SUCCESS'
    INCOMPLETE = 'INCOMPLETE'
    SKIPPED = 'SKIPPED'
    FAILURE = 'FAILURE'
    CRASHED = 'CRASHED'

class TacticInteraction(NamedTuple):
    tactic: str
    context_before: ProofContext

    @classmethod
    def from_dict(cls, data):
        tactic = data['tactic']
        context_before = ProofContext.from_dict(data['context_before'])
        return cls(tactic, context_before)

    def to_dict(self):
        return {"tactic": self.tactic,
                "context_before": self.context_before.to_dict()}


class SearchResult(NamedTuple):
    status: SearchStatus
    context_lemmas: List[str]
    commands: Optional[List[TacticInteraction]]
    steps_taken: int

    @classmethod
    def from_dict(cls, data):
        status = SearchStatus(data['status'])
        if data['commands'] is None:
            commands = None
        else:
            commands = list(map(TacticInteraction.from_dict,
                                data['commands']))
        return cls(status, data['context_lemmas'], commands, data['steps_taken'])

    def to_dict(self):
        return {'status': self.status.name,
                'context_lemmas': self.context_lemmas,
                'commands': list(map(TacticInteraction.to_dict,
                                     self.commands)),
                'steps_taken': self.steps_taken}

class VernacBlock(NamedTuple):
    commands: List[str]

class ProofBlock(NamedTuple):
    lemma_statement: str
    unique_lemma_statement: str
    module: Optional[str]
    status: SearchStatus
    predicted_tactics: List[TacticInteraction]
    original_tactics: List[TacticInteraction]

DocumentBlock = Union[VernacBlock, ProofBlock]

class ArgsMismatchException(Exception):
    pass


class SourceChangedException(Exception):
    pass

class KilledException(Exception):
    pass
