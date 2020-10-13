import numpy as np
import pandas as pd
from .base import BaseModel
from .utils import run_check


class BoutAnalyser(BaseModel):
    """
    Analyses Freeze/Active bouts

    Common usage:

    >>> analyser = BoutAnalyser()
    >>> analyser.fit(freeze_array)
    >>> results = analyser.get_results()

    """

    @staticmethod
    def _bout_idx(arr, val):
        """
        Find the indexes of all vouts of a value in an array.

        Returned as a (n, 2) numpy array where n is the number of bouts.

        Args:
            arr (array-like): array in which to search for bouts
            value (float): value for which bouts will be searched
        """
        is_val = np.equal(arr, val).view(np.uint8)
        is_val = np.concatenate((np.array([0]), is_val, np.array([0])))
        absdiff = np.abs(np.diff(is_val))
        return np.where(absdiff == 1)[0].reshape(-1, 2)

    @staticmethod
    def transition_count(arr):
        """
        Returns the number of transitions between freezing and active states
        """
        return

    def fit(self, arr, freeze_val=1, active_val=0, sampling_rate=1):
        self.arr = arr
        self.freeze_val = freeze_val
        self.active_val = active_val
        self.sampling_period = 1 / sampling_rate

        self.freeze_bout_idx_ = self._bout_idx(arr, val=self.freeze_val)
        self.freeze_bout_lengths_ = np.apply_along_axis(
            lambda x: x[1] - x[0], 1, self.freeze_bout_idx_
        )
        self.mean_freeze_boutlength_ = np.mean(self.freeze_bout_lengths_)
        self.active_bout_idx_ = self._bout_idx(arr, val=self.active_val)
        self.active_bout_lengths_ = (
            np.apply_along_axis(lambda x: x[1] - x[0], 1, self.active_bout_idx_)
            * self.sampling_period
        )
        self.mean_active_boutlength_ = (
            np.mean(self.active_bout_lengths_) * self.sampling_period
        )

        self.transition_count_ = np.sum(np.diff(arr) != 0)
        self.time_freezing_ = np.sum(self.arr == self.freeze_val) * self.sampling_period
        self.time_active_ = np.sum(self.arr == self.active_val) * self.sampling_period
        self.freeze_bout_count_ = len(self.freeze_bout_lengths_)
        self.active_bout_count_ = len(self.active_bout_lengths_)

        self.calculated_metrics_ = {
            "mean_freeze_bout_length": self.mean_freeze_boutlength_,
            "mean_active_bout_length": self.mean_active_boutlength_,
            "time_freezing": self.time_freezing_,
            "time_active": self.time_active_,
            "transition_count": self.transition_count_,
            "freeze_bout_count": self.freeze_bout_count_,
            "active_bout_count": self.active_bout_count_,
        }
        self._fitted = True
        return self

    @run_check
    def get_results(self):
        return pd.Series(self.calculated_metrics_)
