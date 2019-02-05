
from models.tactic_predictor import TacticContext

from typing import List
import re

def numEvarsInGoal(context : TacticContext) -> List[float]:
    return [float(len(re.findall("\s\?\w", context.goal)))]
def topLevelTokenInGoal(context : TacticContext) -> List[float]:
    pass
def numUnboundIdentifiersInGoal(context : TacticContext) -> List[float]:
    pass
def numEqualitiesInHyps(context : TacticContext) -> List[float]:
    return [float(sum([re.match("\w+ : eq ", hypothesis) != None
                       for hypothesis in context.hypotheses]))]

feature_functions = [numEvarsInGoal, numEqualitiesInHyps]
