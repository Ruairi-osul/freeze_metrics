import numpy as np
import pandas as pd
from scipy.stats import uniform
from .base import BaseModel
from .utils import run_check


class MarkovFreezing(BaseModel):
    """
    Model freezing behaviour as a two state markov model

    Common usage:

    >>> model = MarkovFreezing()
    >>> model.fit(freezing_data)
    >>> results = model.get_results()
    >>> simulated_freezing = model.predict(size=10000)

    """

    @staticmethod
    def _transitions(arr):
        """
        Return an array of length(arr) with 1s at transition points
        and 0s otherwise
        """
        return np.concatenate([np.diff(arr), np.array([0])])

    def _get_state(self, roll, state):
        if state == self.freeze_val:
            return self.freeze_val if roll < self.p_f_given_f_ else self.active_val
        elif state == self.active_val:
            return self.freeze_val if roll < self.p_f_given_a_ else self.active_val
        else:
            raise ValueError("Unknow State Value: {}".format(state))

    def fit(self, arr, freeze_val=1, active_val=0):
        """
        Fit the model to an array of freezing behaviour data
        """
        self.arr = arr
        self.arr_prime_ = self._transitions(arr)
        self.freeze_val = freeze_val
        self.active_val = active_val

        self.p_f_ = np.mean(self.arr == 1)
        self.p_a_ = np.mean(self.arr == 0)

        self.p_f_given_f_ = np.mean(self.arr_prime_[self.arr == 1] == 0)
        self.p_a_given_f_ = np.mean(self.arr_prime_[self.arr == 1] != 0)

        self.p_a_given_a_ = np.mean(self.arr_prime_[self.arr == 0] == 0)
        self.p_f_given_a_ = np.mean(self.arr_prime_[self.arr == 0] != 0)
        self._fitted = True
        return self

    @run_check
    def predict(self, size, initial_state=None):

        """
        Use the model to simulate freezing behaviour.

        Model must first be fit.

        Args:
            size: The number of samples to simulate
            initial_state: The initial freezing state of the animal. Defaults
                           to random state according to state probabilities.
                           Value must be in the set of {freeze_val, active_val}
                           specified at instantiation.
        Returns:
            A numpy array of freeze state predictions 
        """
        if not self._fitted:
            raise ValueError("Model not fit")
        rolls = uniform(0, 1).rvs(size)
        states = np.empty(size)
        if initial_state is None:
            states[0] = self.freeze_val if rolls[0] < self.p_f_ else self.active_val
        else:
            assert initial_state in (
                self.freeze_val,
                self.active_val,
            ), "Invalid Initial State"
        for i in range(1, size):
            states[i] = self._get_state(rolls[i], states[i - 1])
        return states

    @run_check
    def get_results(self):
        """
        Return the calculated model parameters as a labeled pandas.Series object
        """
        if not self._fitted:
            raise ValueError("Model not fit")
        return pd.Series(
            {
                "p(f)": self.p_f_,
                "p(a)": self.p_a_,
                "p(f|f)": self.p_f_given_f_,
                "p(a|f)": self.p_a_given_f_,
                "p(a|a)": self.p_a_given_a_,
                "p(f|a)": self.p_f_given_a_,
            }
        )


if __name__ == "__main__":

    f = np.random.choice([0, 1], size=50)
    model = MarkovFreezing().fit(f)
    print(f[:10])
    print(model.get_results())
