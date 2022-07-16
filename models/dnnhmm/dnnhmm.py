from typing import Optional, final
import keras
import numpy as np


class DNNHMM(object):
    """
    This class represents a DNN-HMM model e.g. an HMM whose observation emission probabilities P(x | s) (where x is the
     observation and s is the state) are defined by a neural network model.
    """

    __MODES_VITERBI: final = {"log", "mult"}

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
        :raise ValueError: if any given shape is incorrect, if any of the given transition matrix rows doesn't sum up to
            1, if state_frequencies elements or priors if given emission model is invalid (not compiled and built
            or has a number of output neurons less than n_states, e.g. emission model.output_shape[-1] < n_states).
            don't sum up to 1.
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
        """
        Retrieves the number of states of the DNN-HMM model.

        :return: the number of states of the DNN-HMM model.
        """
        return self.__n_states

    @property
    def transitions(self) -> np.ndarray:
        """
        Retrieves transition matrix of the DNN-HMM model.

        :return: the transition matrix of the DNN-HMM model, with shape (n_states, n_states).
        """
        return self.__transitions

    @transitions.setter
    def transitions(self, transitions: np.ndarray):
        """
        Sets transition matrix of the DNN-HMM model.

        :param transitions: novel (n_states, n_states)-shaped transition matrix of the DNN-HMM model.
        :raises ValueError: if shape is incorrect or any of the given transition matrix rows doesn't sum up to 1.
        """
        # Validate transition matrix
        self._validate_transition_matrix(transitions)
        self.__n_states = transitions.shape[0]
        self.__transitions = transitions

    @property
    def priors(self) -> np.ndarray:
        """
        Retrieves prior distribution of the DNN-HMM states (e.g P(s_0 = s), for each state s).

        :return: an array with shape (n_states, ) containing P(s_0 = s), for each HMM state s.
        """
        return self.__priors

    @priors.setter
    def priors(self, priors: np.ndarray):
        """
        Sets prior distribution of the DNN-HMM states (e.g P(s_0 = s), for each state s).

        :param priors: novel prior distribution array of shape (n_states, ).
        :raises ValueError: if shape is incorrect or any of the given probabilities don't sum up to 1.
        """
        # Validate priors
        self._validate_priors(priors)
        self.__priors = priors

    @property
    def state_frequencies(self) -> np.ndarray:
        """
        Retrieves frequency distribution of the DNN-HMM states (e.g P(s_0 = s), for each state s).

        :return: an array with shape (n_states, ) containing P(s_0 = s), for each HMM state s.
        """
        return self.__state_frequencies

    @state_frequencies.setter
    def state_frequencies(self, state_frequencies: np.ndarray):
        """
        Sets frequency distribution of the DNN-HMM states (e.g P(s), for each state s).

        :param state_frequencies: novel frequency distribution array of shape (n_states, ).
        :raises ValueError: if shape is incorrect or any of the given frequencies don't sum up to 1.
        """
        # Validate state frequencies
        self._validate_state_frequencies(state_frequencies)
        self.__state_frequencies = state_frequencies

    def _validate_transition_matrix(self, transitions: np.ndarray):
        """
        Validates transition matrix of the DNN-HMM model.
        :param transitions: novel (n_states, n_states)-shaped transition matrix of the DNN-HMM model.
        :raises ValueError: if shape is incorrect or any of the given transition matrix rows doesn't sum up to 1.
        """

        if self.__n_states != transitions.shape[1] or transitions.ndim != 2:
            raise ValueError("Transition matrix must have shape (n_states, n_states)")

        # Check if each transition matrix sum up to 1 (since they are probabilities of transition out of a state)
        '''
        for i in range(0, transitions.shape[0]):
            if sum(transitions[i, :]) != 1:
                raise ValueError("Each transition matrix row must sum up to 1")
        '''

    def _validate_state_frequencies(self, state_frequencies: np.ndarray):
        """
        Validates frequency distribution of the DNN-HMM states (e.g P(s), for each state s).

        :param state_frequencies: novel frequency distribution array of shape (n_states, ).
        :raises ValueError: if shape is incorrect or any of the given frequencies don't sum up to 1.
        """
        if state_frequencies.shape != (self.__n_states, ):
            raise ValueError("State frequencies array must have shape (n_states, )")

        # Check if frequencies sum up to 1 (since they are estimates of the probabilities of each state)
        # if np.sum(state_frequencies) != 1:
        #    raise ValueError("State frequencies array must sum up to 1")

    def _validate_priors(self, priors: np.ndarray):
        """
        Validates prior distribution of the DNN-HMM states (e.g P(s_0 = s), for each state s).

        :param priors: novel prior distribution array of shape (n_states, ).
        :raises ValueError: if shape is incorrect or any of the given probabilities don't sum up to 1.
        """
        if priors.shape != (self.__n_states, ):
            raise ValueError("State prior probabilities array must have shape (n_states, )")

        # Check if priors sum up to 1 (since they are estimates of the probabilities of each state)
        # if np.sum(priors) != 1:
        #    raise ValueError("State prior probabilities array must sum up to 1")

    def _validate_emission_model(self, emission_model: keras.Model):
        """
        Validates emission model for the DNN-HMM model.

        :param emission_model: the emission model to validate.
        :raises ValueError: if emission model is invalid (not compiled and built or has a number of output neurons less
            than n_states, e.g. emission model.output_shape[-1] < n_states).
        """
        # TODO: implement more emission_model checks
        if not emission_model.built:
            raise ValueError("Emission model must be built prior being used for DNN-HMM state predictions.")
        if emission_model.output_shape[-1] < self.n_states:
            raise ValueError("Emission model mus")
        pass

    def _compute_emission_matrix(self, y: np.ndarray, state_range: tuple[int, int]) -> np.ndarray:
        """
        Computes emission matrix for given observations, taking into account states in the given range (e.g. output
        neurons of the emission models, corresponding to HMM states).

        :param y: a numpy array of shape (n_obs, n_features) representing observation sequence to compute the emission
            matrix for.
        :param state_range: a 2-element tuple representing the range of the states to take into account (e.g. output
            neurons of the emission models, corresponding to HMM states).
        :return: a (n_states, n_obs)-shaped emission matrix, containing the emission probabilities for each
            observation in y.
        :raises ValueError: if the given state range is invalid (state_range[1] -
            state_range[0] != n_states or state_range[0] < state_range[1] <= emission_model.output_shape[-1]).
        """

        # Range of indexes of the emission model output to take into account
        if not state_range[0] < state_range[1] <= self.__emission_model.output_shape[-1]:
            raise ValueError(
                f"The emission model output range must be between 0 and {self.__emission_model.output_shape[-1]}"
            )

        if state_range[1] - state_range[0] != self.__n_states:
            raise ValueError(f"The emission model output range must be exactly {self.__n_states}-elements long")

        # Get posterior probabilities for each observation of the sequence
        posteriors_sequence = self.__emission_model(
            np.expand_dims(y, axis=0)
        ).numpy()[0, :, state_range[0]:state_range[1]]

        # Observation prior is allotted to be a constant value since all observations are assumed to be independent, and
        # thus it can be ignored completely
        n_obs = y.shape[0]
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

    def viterbi(self, y: np.ndarray, state_range: Optional[tuple[int, int]] = None, mode: str = 'log') -> \
            (np.ndarray, np.float64):
        """
        Computes the Viterbi estimate of state trajectory of HMM (e.g. most likely hidden state sequence, given an
        observation sequence and its probability.

        :param y: a numpy array of shape (n_obs, n_features) representing observation sequence to compute the viterbi
            algorithm for.
        :param state_range: a 2-element tuple representing the range of the states to take into account (e.g. output
            neurons of the emission models, corresponding to HMM states).
        :param mode: either 'log' or 'mult', indicates wherever or not to make probability calculations in the log
            domain (which is the default, and strongly recommended).
        :return: the most likely state sequence given the observations, and the corresponding posterior probability.
        :raises ValueError: if mode is neither 'mult' or 'log', if y is not 2-dimensional or the given state range is
            invalid (state_range[1] - state_range[0] != n_states or state_range[0] < state_range[1] <=
            emission_model.output_shape[-1]).
        """

        if y.ndim != 2:
            raise ValueError("Observation array must be 1-dimensional")

        if state_range is None:
            state_range = (0, self.__emission_model.output_shape[-1])

        if mode not in DNNHMM.__MODES_VITERBI:
            raise ValueError(f"Mode must be either: {DNNHMM.__MODES_VITERBI}")

        # Compute emission matrix
        emission_matrix = self._compute_emission_matrix(y, state_range)

        # Compute the most likely state sequence
        most_likely_path, t1, t2 = DNNHMM._viterbi(
            n_obs=y.shape[0],
            a=self.__transitions,
            b=emission_matrix,
            pi=self.__priors,
            mode=mode
        )

        # Compute most likely state sequence probability
        most_likely_path_prob = np.max(t1[:, y.shape[0] - 1])

        return most_likely_path, most_likely_path_prob

    @staticmethod
    def _viterbi(n_obs: int, a: np.ndarray, b: np.ndarray, pi: Optional[np.ndarray] = None, mode: str = 'log'):
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
        mode: either 'log' or 'mult', indicates wherever or not to make probability calculations in the log
            domain (which is the default, and strongly recommended).

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
        # TODO: modify this to work in log-domain to avoid too much zeros (caused by multiple prob multiplications)

        # Cardinality of the state space
        n_states = a.shape[0]

        # Initialize the priors with default (uniform dist) if not given by caller
        pi = pi if pi is not None else np.full(n_states, 1 / n_states)
        t1 = np.empty((n_states, n_obs), 'd')
        t2 = np.empty((n_states, n_obs), 'b')

        if mode == 'mult':
            # Initialize the tracking tables from first observation
            t1[:, 0] = pi * b[:, 0]
            t2[:, 0] = 0

            # Iterate through the observations updating the tracking tables
            for t in range(1, n_obs):
                t1[:, t] = np.max(t1[:, t - 1] * a.T * b[np.newaxis, :, t].T, 1)
                t2[:, t] = np.argmax(t1[:, t - 1] * a.T, 1)

        elif mode == 'log':
            # Convert a, b, pi to the log-domain, adding a small eps to each probability to avoid log(0)s
            eps = np.finfo(0.).tiny
            a_log = np.log(a + eps)
            b_log = np.log(b + eps)
            pi_log = np.log(pi + eps)

            # Initialize the tracking tables from first observation
            t1[:, 0] = pi_log + b_log[:, 0]
            t2[:, 0] = 0

            # Iterate through the observations and states, updating the tracking tables
            for t in range(1, n_obs):
                for i in range(n_states):
                    temp_sum = a_log[:, i] + t1[:, t-1]
                    t1[i, t] = np.max(temp_sum) + b_log[i, t]
                    t2[i, t-1] = np.argmax(temp_sum)

            '''
            # Iterate through the observations updating the tracking tables
            for i in range(1, n_obs):
                t1[:, i] = np.max(t1[:, i - 1] * a.T * b[np.newaxis, :, i].T, 1)
                t2[:, i] = np.argmax(t1[:, i - 1] * a.T, 1)
            '''

        # Build the output, optimal model trajectory, backtracking from the last state
        most_likely_path = np.empty(n_obs, 'b')
        most_likely_path[-1] = np.argmax(t1[:, n_obs - 1])
        for t in reversed(range(1, n_obs)):
            most_likely_path[t - 1] = t2[most_likely_path[t], t]

        return most_likely_path, t1, t2
