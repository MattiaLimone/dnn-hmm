from typing import Optional
import keras
import numpy as np


# TODO: finish off class and method documentation
class DNNHMM(object):
    """
    This class represents a DNN-HMM model e.g. an HMM whose observation emission probabilities P(x | s) (where x is the
     observation and s is the state) are defined by a neural network model.
    """

    def __init__(self, transitions: np.ndarray, emission_model: keras.Model, state_frequencies: np.ndarray,
                 priors: Optional[np.ndarray] = None):
        """
        Constructor. Instantiates a DNN-HMM model starting from transition probabilities, prior probabilities and state
        frequencies.
        :param transitions: transition probability matrix from each state to each state; shape: (n_states, n_states).
        :param emission_model: built and trained emission model to approximate state posterior distribution P(s | x),
            where s is the state and x is the observation.
        :param state_frequencies: approximation of the state distribution P(s), for s = 1, 2, ..., n_states; shape
            (n_states, ).
        :param priors: prior probability distribution of each state P(s_0 = s), for s = 1, 2, ..., n_states; shape
            (n_states, ).
        :raise ValueError if any given shape is incorrect, if any of the given transition matrix row doesn't sum up to
            1, if state_frequencies elements don't sum up to 1 or priors doesn't sum up to 1.
        """
        self.__n_states = transitions.shape[0]

        # Validate transition matrix
        self._validate_transition_matrix(transitions)

        # Validate priors
        if priors is not None:
            self._validate_priors(priors)

        # Validate state frequencies
        self._validate_state_frequencies(state_frequencies)

        # Validate emission model
        self._validate_emission_model(emission_model)

        # Setup instance variables
        self.__transitions = transitions
        self.__priors = priors if priors is not None else np.full(self.__n_states, 1 / self.__n_states)
        self.__emission_model = emission_model
        self.__state_frequencies = state_frequencies

    @property
    def n_states(self) -> int:
        return self.__n_states

    @property
    def transitions(self) -> np.ndarray:
        return self.__transitions

    @transitions.setter
    def transitions(self, transitions: np.ndarray):
        # Validate transition matrix
        self._validate_transition_matrix(transitions)
        self.__n_states = transitions.shape[0]
        self.__transitions = transitions

    @property
    def priors(self) -> np.ndarray:
        return self.__priors

    @priors.setter
    def priors(self, priors: np.ndarray):
        # Validate priors
        self._validate_priors(priors)
        self.__priors = priors

    @property
    def state_frequencies(self) -> np.ndarray:
        return self.__state_frequencies

    @state_frequencies.setter
    def state_frequencies(self, state_frequencies: np.ndarray):
        # Validate state frequencies
        self._validate_state_frequencies(state_frequencies)
        self.__state_frequencies = state_frequencies

    def _validate_transition_matrix(self, transitions: np.ndarray):

        if self.__n_states != transitions.shape[1] or transitions.ndim != 2:
            raise ValueError("Transition matrix must have shape (n_states, n_states)")

        # Check if each transition matrix sum up to 1 (since they are probabilities of transition out of a state)
        for i in range(0, transitions.shape[0]):
            if sum(transitions[i, :]) != 1:
                raise ValueError("Each transition matrix row must sum up to 1")

    def _validate_state_frequencies(self, state_frequencies: np.ndarray):
        if state_frequencies.shape != (self.__n_states, ):
            raise ValueError("State frequencies array must have shape (n_states, )")

        # Check if frequencies sum up to 1 (since they are estimates of the probabilities of each state)
        if np.sum(state_frequencies) != 1:
            raise ValueError("State frequencies array must sum up to 1")

    def _validate_priors(self, priors: np.ndarray):
        if priors.shape != (self.__n_states, ):
            raise ValueError("State prior probabilities array must have shape (n_states, )")

        # Check if priors sum up to 1 (since they are estimates of the probabilities of each state)
        if np.sum(priors) != 1:
            raise ValueError("State prior probabilities array must sum up to 1")

    def _validate_emission_model(self, emission_model: keras.Model):
        # TODO: implement emission_model checks
        pass

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

        # Observation prior is allotted to be a constant value since all observations are assumed to be independent, and
        # thus it can be ignored completely
        n_obs = len(y)
        # observation_prior = 1 / n_obs  # This should be ignored
        observation_index = 0
        observations_likelihood = np.zeros(shape=(self.__n_states, n_obs))

        # For each observation
        for posterior in posteriors_sequence:

            # Convert the posterior into likelihood (observation prior can be ignored since it's constant)
            # likelihood = (posterior * observation_prior) / self.__state_frequencies
            likelihood = posterior / self.__state_frequencies

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
        Computes the Viterbi estimate of state trajectory of HMM (e.g. most likely hidden state sequence, given an
        observation sequence.

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
