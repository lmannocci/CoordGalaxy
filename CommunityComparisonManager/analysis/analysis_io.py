"""Shared file readers for overlap-derived analysis modules."""

from __future__ import annotations

import pandas as pd

from utils.common_variables import dtype


class OverlapAnalysisIOMixin:
    """Provide common readers used by coordination and validation analyses."""

    def _read_overlap_flux_df(self, metric: str, mid_th: float) -> pd.DataFrame:
        """
        Read the community-level flux dataframe.

        :param metric: [str] Overlap metric encoded in the filename.
        :param mid_th: [float] Flux threshold encoded in the filename.
        :return: [pd.DataFrame] Flux dataframe.
        """
        return self.ch.read_dataframe(
            f"{self.dm.path_overlapping_flux_df}{self.file_prefix}_{metric}_"
            f"th_size_{str(self.community_size_th)}_mid_th_{str(mid_th)}_flux_df.csv",
            dtype=dtype,
        )
