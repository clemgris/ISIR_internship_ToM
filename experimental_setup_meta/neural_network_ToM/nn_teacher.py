
import numpy as np

from learner import compute_policy, bayesian_update, projection
from environment import ButtonsToy
from teacher import Teacher, cost

# ToM teacher: NN model of the learner
def ToMNetTeacher(Teacher):
    pass