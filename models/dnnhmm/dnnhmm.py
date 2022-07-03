from typing import Optional
import keras
import numpy as np


# TODO: add class documentation
class DNNHMM(object):
    def __init__(self, transitions: np.ndarray, emission_model: keras.Model,
                 state_frequencies: np.ndarray, priors: np.ndarray = None):
        # TODO: add checks
        self.__n_states = priors.shape[0]
        self.__transitions = transitions
        self.__priors = priors if priors is not None else np.full(self.__n_states, 1 / self.__n_states)
        self.__emission_model = emission_model
        self.__state_frequencies = state_frequencies

    @property
    def state_frequencies(self) -> np.ndarray:
        return self.__state_frequencies

    @state_frequencies.setter
    def state_frequencies(self, state_frequencies: np.ndarray):
        # TODO: add checks about state frequencies (e.g. sum exactly to 1)
        self.__state_frequencies = state_frequencies

    def _compute_emission_matrix(self, y, state_range) -> np.ndarray:

        # Range of indexes of the emission model output to take into account
        if not state_range[0] < state_range[1] <= self.__emission_model.output_shape[-1]:
            raise ValueError(
                f"The emission model output range must be between 0 and {self.__emission_model.output_shape[-1]}"
            )

        if state_range[1] - state_range[0] != self.__n_states:
            raise ValueError(f"The emission model output range must be exactly {self.__n_states}-elements long")

        # Get posterior probabilities for each observation of the sequence
        posteriors_sequence = self.__emission_model(y).numpy()[:, state_range[0]:state_range[1]]

        # Observation prior is allotted to be a constant value since all observations are assumed to be independent
        n_obs = len(y)
        observation_prior = 1 / n_obs
        observation_index = 0
        observations_likelihood = np.zeros(shape=(self.__n_states, n_obs))

        # For each observation
        for posterior in posteriors_sequence:

            # Convert the posterior into likelihood
            likelihood = (posterior * observation_prior) / self.__state_frequencies

            '''
            # Convert likelihood to log-likelihood, guarding against log(0) if some posterior equals to 0
            log_likelihood = np.zeros(len(posterior))
            log_likelihood = np.log(
                likelihood,
                out=log_likelihood,
                where=[False if x == 0 else True for x in likelihood]
            )
            '''

            # Add the obtained likelihood to the result matrix
            observations_likelihood[:, observation_index] = likelihood
            observation_index += 1

        return observations_likelihood

    def viterbi(self, y: np.ndarray, state_range: Optional[tuple[int, int]] = None) -> (np.ndarray, np.float64):
        emission_matrix = self._compute_emission_matrix(y, state_range)
        most_likely_path, t1, t2 = DNNHMM._viterbi(
            n_obs=len(y),
            a=self.__transitions,
            b=emission_matrix,
            pi=self.__priors
        )
        return most_likely_path, np.max(t1[:, len(y) - 1])

    @staticmethod
    def _viterbi(n_obs: int, a: np.ndarray, b: np.ndarray, pi: Optional[np.ndarray] = None):
        """
        Return the MAP estimate of state trajectory of Hidden Markov Model.

        Parameters
        ----------
        n_obs : number of observations
        a : array (n_states, n_states)
            State transition matrix. See HiddenMarkovModel.state_transition  for
            details.
        b : array (n_states, n_obs)
            Emission matrix containing the likelihood of the i-th observation given the state (works both for continuous
            and discrete distributions since it must give an emission probability for each OBSERVED state and not).
        pi: optional, (n_states,)
            Initial state probabilities: pi[i] is the probability most_likely_path[0] == i. If
            None, uniform initial distribution is assumed (pi[:] == 1/n_states).

        Returns
        -------
        most_likely_path : array (n_obs,)
            Maximum a posteriori probability estimate of hidden state trajectory,
            conditioned on observation sequence y under the model parameters a, b,pi.
        t1: array (n_states, n_obs)
            the probability of the most likely path so far
        t2: array (n_states, n_obs)
            the x_j-1 of the most likely path so far
        """
        # Cardinality of the state space
        n_states = a.shape[0]
        # Initialize the priors with default (uniform dist) if not given by caller
        pi = pi if pi is not None else np.full(n_states, 1 / n_states)
        t1 = np.empty((n_states, n_obs), 'd')
        t2 = np.empty((n_states, n_obs), 'b')

        # Initialize the tracking tables from first observation
        t1[:, 0] = pi * b[:, 0]
        t2[:, 0] = 0

        # Iterate through the observations updating the tracking tables
        for i in range(1, n_obs):
            t1[:, i] = np.max(t1[:, i - 1] * a.T * b[np.newaxis, :, i].T, 1)
            t2[:, i] = np.argmax(t1[:, i - 1] * a.T, 1)

        # Build the output, optimal model trajectory
        most_likely_path = np.empty(n_obs, 'b')
        most_likely_path[-1] = np.argmax(t1[:, n_obs - 1])
        for i in reversed(range(1, n_obs)):
            most_likely_path[i - 1] = t2[most_likely_path[i], i]

        return most_likely_path, t1, t2

