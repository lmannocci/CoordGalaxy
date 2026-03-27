from utils.Checkpoint.Checkpoint import *
from IntegrityConstraintManager.IntegrityConstraintManager import *

import os
import pandas as pd
file_name = os.path.splitext(os.path.basename(__file__))[0]


class CoAction:
    def __init__(self, co_action, similarity_function):
        """
            CoAction constructor.
            :param co_action: [str] The type of co-action.
            Admissible values for parameter:
            - co-retweet
            :param similarity_function: [str] The function of similarity to compute the weight of edge.
            Admissible values for parameter:
            - overlapping
        """
        self.lm = LogManager("main")

        self.co_action = co_action
        self.similarity_function = similarity_function

        self.icm = IntegrityConstraintManager(file_name)
        self.icm.check_co_action_availability(co_action, similarity_function)
        self.ch = Checkpoint()

    def get_co_action(self):
        """
            Get co-action type.
            :return: [str] Co-action type, e.g., 'co-retweet'.
        """
        return self.co_action

    def get_similarity_function(self):
        """
            Get co-action similarity function.
            :return: [str] Co-action similarity_function, e.g., 'overlapping'.
        """
        return self.similarity_function