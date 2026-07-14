from DirectoryManager import DirectoryManager
from IntegrityConstraintManager.IntegrityConstraintManager import *
from utils.Checkpoint.Checkpoint import *
from utils.ConversionManager.ConversionManager import *
import os
from CommunityComparisonManager.analysis.analysis_io import OverlapAnalysisIOMixin
from CommunityComparisonManager.analysis.coordination import CoordinationAnalysisMixin
from CommunityComparisonManager.analysis.metric_preparation import OverlapMetricPreparationMixin
from CommunityComparisonManager.analysis.validation import ValidationAnalysisMixin
from CommunityComparisonManager.computation.nmi import SingleLayerNMIAnalysisMixin
from CommunityComparisonManager.computation.overlap_computation import OverlapComputationMixin
from CommunityComparisonManager.computation.overlap_metrics import OverlapMetricCalculator
from CommunityComparisonManager.plotting import (
    NodeMetricsPlotter,
    OverlappingFluxPlotter,
    OverlappingHeatmapPlotter,
    SingleLayerMetricsPlotter,
)
from CommunityComparisonManager.utils import CommunityDataPreparer, MatrixFilter


absolute_path = os.path.dirname(__file__)
file_name = os.path.splitext(os.path.basename(__file__))[0]
directory_level_name = "CommunityComparisonManager"
package_path = os.path.dirname(absolute_path)
results = os.path.join(package_path, f"..{os.sep}results{os.sep}")


class CommunityComparisonWorker(
    OverlapAnalysisIOMixin,
    OverlapComputationMixin,
    OverlapMetricPreparationMixin,
    SingleLayerNMIAnalysisMixin,
    CoordinationAnalysisMixin,
    ValidationAnalysisMixin,
    OverlappingHeatmapPlotter,
    OverlappingFluxPlotter,
    SingleLayerMetricsPlotter,
    NodeMetricsPlotter,
):
    def __init__(
        self,
        dataset_name: str,
        user_fraction: float | None,
        type_filter: str,
        tw,
        list_ca: list,
        dict_ca_filter: dict,
        file_prefix: str,
        chm_x=None,
        chm_y=None,
        community_size_th: int | None = None,
    ) -> None:
        """
        Create the internal worker used by CommunityComparisonManager.

        The worker owns the shared state needed by overlap computation, NMI, metric preparation,
        coordination/validation summaries, and comparison plots. Main scripts should use
        CommunityComparisonManager as the public facade; this class is kept as the internal worker
        aggregator that forwards work to focused mixins.

        :param dataset_name: [str] Dataset identifier used to resolve result directories.
        :param user_fraction: [float | None] User-selection fraction used in the result path.
        :param type_filter: [str] User-selection strategy used in the result path.
        :param tw: Time-window object used to resolve the network/community result path.
        :param list_ca: [list] Co-action objects included in the compared multiplex configuration.
        :param dict_ca_filter: [dict] Filter configuration keyed by co-action id.
        :param file_prefix: [str] Prefix used for saved comparison artifacts.
        :param chm_x: Optional first CharacterizationManager instance for pairwise comparisons.
        :param chm_y: Optional second CharacterizationManager instance for pairwise comparisons.
        :param community_size_th: [int | None] Optional minimum community size used by filtered comparisons.
        :return: None.
        """
        self.dataset_name = dataset_name
        self.user_fraction = user_fraction
        self.type_filter = type_filter
        self.tw = tw
        self.list_ca = list_ca
        self.dict_ca_filter = dict_ca_filter

        self.list_ca = list_ca
        self.available_list_ca = list(available_co_action.keys())
        self.dict_ca_filter = dict_ca_filter
        self.icm = IntegrityConstraintManager(directory_level_name)
        # check if for each co_action in the list, it is passed the corresponding threshold in the dictionary of the threshold.
        self.icm.check_co_action(list_ca, dict_ca_filter)

        self.file_prefix = file_prefix
        self.community_size_th = community_size_th

        self.lm = LogManager('main')
        self.ch = Checkpoint()
        self.cm = ConversionManager()
        self.overlap_metric_calculator = OverlapMetricCalculator()
        self.community_data_preparer = CommunityDataPreparer()
        self.matrix_filter = MatrixFilter()

        # General information, common to all chm_x, chm_y. useful to extract the info for the file name
        self.dm = DirectoryManager(directory_level_name, dataset_name, results=results, user_fraction=user_fraction,
                                   type_filter=type_filter,
                                   tw=tw, list_ca=list_ca, dict_ca_filter=dict_ca_filter)
       

        if chm_x is not None:
            self.chm_x = chm_x
            self.dm_x = self.chm_x.get_directory_manager()
            self.ch_x = self.chm_x.get_checkpoint()
            self.type_algorithm_x = self.chm_x.get_type_algorithm()
            self.list_ca_x = self.chm_x.get_list_ca()
            self.dict_ca_filter_x = self.chm_x.get_dict_ca_filter()
            self.cda_x = self.chm_x.get_cda()
            self.type_algorithm_x = self.dm_x.get_type_algorithm()

        if chm_y is not None:
            self.chm_y = chm_y
            self.dm_y = self.chm_y.get_directory_manager()
            self.ch_y = self.chm_y.get_checkpoint()
            self.type_algorithm_y = self.chm_y.get_type_algorithm()
            self.list_ca_y = self.chm_y.get_list_ca()
            self.dict_ca_filter_y = self.chm_y.get_dict_ca_filter()
            self.cda_y = self.chm_y.get_cda()
            self.type_algorithm_y = self.dm_y.get_type_algorithm()
