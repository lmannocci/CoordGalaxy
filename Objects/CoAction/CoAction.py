import os

from IntegrityConstraintManager.IntegrityConstraintManager import IntegrityConstraintManager
from utils.Checkpoint.Checkpoint import Checkpoint
from utils.LogManager.LogManager import LogManager
from utils.common_variables import normalize_co_action_id


file_name = os.path.splitext(os.path.basename(__file__))[0]


class CoAction:
    def __init__(self, co_action: str, similarity_function: str) -> None:
        """
            Create a co-action configuration.
            :param co_action: [str] Co-action id or alias, for example co-hashtag, hashtag, or co-commentText.
            :param similarity_function: [str] Similarity function used to compute edge weights, for example
                tfidf_cosine_similarity, average_cosine_similarity, overlapping, or overlapping_coefficient.
            :return: None.
        """
        self.lm = LogManager("main")
        self.co_action = normalize_co_action_id(co_action)
        self.similarity_function = similarity_function

        self.icm = IntegrityConstraintManager(file_name)
        self.icm.check_co_action_availability(self.co_action, similarity_function)
        self.ch = Checkpoint()

    def get_co_action(self) -> str:
        """
            Return the canonical co-action id.
            :return: [str] Canonical co-action id, for example co-hashtag.
        """
        return self.co_action

    def get_similarity_function(self) -> str:
        """
            Return the configured similarity function.
            :return: [str] Similarity function name.
        """
        return self.similarity_function
