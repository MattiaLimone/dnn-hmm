from typing import Optional, Union
import numpy as np
from hmmlearn import base
from keras.models import Model
from scipy.stats import rv_continuous


class _DNNHMMRandomVariable(rv_continuous):
    """
    Random variable that models the posterior distribution of a deep neural network output classes, given an instance x:
    P(q | x), for each output class q.
    """

    def __init__(self, model: Model, name: str = "DNNHMMRandomVariable", longname: str = "DNNHMMRandomVariable",
                 seed: Optional[int] = None):
        super().__init__(name=name, longname=longname, seed=seed)
        # TODO: add checks about compilation and fitting of the model and consistent outputs
        self.__model = model

    def _pdf(self, x, *args):
        # TODO: convert into likelihood
        return self.__model(x).numpy()

    @property
    def model(self) -> Model:
        return self.__model

    @model.setter
    def model(self, model: Model):
        # TODO: add checks about compilation and fitting of the model and consistent outputs
        self.__model = model

    def rvs(self, size=None, random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
            scale=1, **kwargs):
        """
        Random variates of given type.

        Parameters
        ----------
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (by default, it's equal to the given NN output vector size).
        random_state : {None, int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            If `seed` is `None` the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is None.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        if size is None:
            size = self.__model.output_shape[-1]
        return super().rvs(size=size, random_state=random_state, scale=scale, **kwargs)


class DNNHMM(base._BaseHMM):
    def __init__(self, timesteps: int, observation_prior: float, state_frequencies: np.ndarray,
                 emission_model: Optional[Model] = None, emission_model_output_range: Optional[tuple[int, int]] = None,
                 n_components: int = 1, startprob_prior=1.0, transmat_prior=1.0, algorithm: str = "viterbi",
                 random_state: Optional[int] = None, n_iter: int = 10, tol: float = 1e-2, verbose: bool = False,
                 params: str = "st", init_params: str = "st"):
        base._BaseHMM.__init__(
            self,
            n_components,
            startprob_prior=startprob_prior,
            transmat_prior=transmat_prior,
            algorithm=algorithm,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            params=params,
            verbose=verbose,
            init_params=init_params
        )

        # TODO: add checks about state frequencies (e.g. sum exactly to 1)
        self.__state_frequencies = state_frequencies

        # Init prior state probabilities from given state frequencies array
        self.startprob_ = self.__state_frequencies

        # Init transition matrix to 1/n_components since the model is ergodic
        self.transmat_ = np.ones((self.n_components, self.n_components)) / self.n_components

        # TODO: add check for the prior to be between 0 and 1
        self.__observation_prior = observation_prior

        # TODO: add checks about timesteps
        self.__timesteps = timesteps

        # TODO: add checks about compilation and fitting of the model and consistent outputs
        self.__emission_model = emission_model
        if emission_model is not None:
            super(base._BaseHMM, self).n_features_in_ = emission_model.input_shape[-1]
        else:
            super(base._BaseHMM, self).n_features_in_ = None

        # Range of indexes of the emission model output to take into account
        if self.n_features_in_ is not None and \
                not emission_model_output_range[0] < emission_model_output_range[1] <= emission_model.output_shape[-1]:
            raise ValueError(f"The emission model output range must be between 0 and {emission_model.output_shape[-1]}")

        if self.n_features_in_ is not None and \
                emission_model_output_range[1] - emission_model_output_range[0] != n_components:
            raise ValueError(f"The emission model output range must be exactly {n_components}-elements long")

        self.__emission_model_output_range = emission_model_output_range

    @property
    def emission_model(self) -> Model:
        return self.__emission_model

    @emission_model.setter
    def emission_model(self, emission_model: Model):
        # TODO: add checks about compilation and fitting of the model and consistent outputs
        self.__emission_model = emission_model
        super(base._BaseHMM, self).n_features_in_ = emission_model.input_shape[-1]

    @property
    def n_features_in_(self) -> int:
        return super(base._BaseHMM, self).n_features_in_

    @n_features_in_.setter
    def n_features_in_(self, n_features_in_: int):
        raise ValueError("n_features_in_ cannot be set directly, the emission_model must be first changed.")

    @property
    def timesteps(self) -> int:
        return self.__timesteps

    @property
    def observation_prior(self) -> float:
        return self.__observation_prior

    @observation_prior.setter
    def observation_prior(self, observation_prior: float):
        # TODO: add check for the prior to be between 0 and 1
        self.__observation_prior = observation_prior

    @property
    def state_frequencies(self) -> np.ndarray:
        return self.__state_frequencies

    @state_frequencies.setter
    def state_frequencies(self, state_frequencies: np.ndarray):
        # TODO: add checks about state frequencies (e.g. sum exactly to 1)
        self.__state_frequencies = state_frequencies

    def _check_n_features(self, X: np.ndarray, reset: bool = False):
        """
        Checks if the given array is suitable for fitting.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features) or (n_sequences, n_samples, n_features)
            The input samples.
        reset : bool
            This has to be False, and will rise ValueError otherwise.
        """
        if reset:
            raise ValueError("reset must be False in DNN-HMM")

        if X.shape[-1] != self.n_features_in_:
            raise ValueError("X.shape[-1] must be equal to n_features_in_")

    def _init(self, X, lengths=None):
        self._check_n_features(X)
        super()._init(X, lengths=lengths)

    def _check(self):
        super()._check()

        # TODO: add checks about compilation and fitting of the model

    def _get_n_fit_scalars_per_param(self):
        nc = self.n_components
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
        }

    def _compute_log_likelihood(self, X):
        if X.ndim > 2:
            raise ValueError("X must be a 2-dimensional array")

        n_observations = X.shape[0]
        splitted = [X[i: i + self.timesteps] for i in range(0, len(X), self.timesteps)]  # split the sequence array

        # Define a result matrix containing the log-likelihood of each observation given each state
        observations_log_likelihood = np.zeros(shape=(n_observations, self.n_components))

        # Observation index in the given data
        observation_index = 0

        # For each sequence in X
        for sequence in splitted:

            # Get posterior probabilities for each observation of the sequence
            posteriors_sequence = self.__emission_model(
                sequence
            ).numpy()[:, self.__emission_model_output_range[0]:self.__emission_model_output_range[1]]

            # For each observation
            for posterior in posteriors_sequence:
                log_likelihood = np.zeros(len(posterior))

                # Convert the posterior into likelihood
                likelihood = (posterior * self.__observation_prior) / self.state_frequencies

                # Convert likelihood to log-likelihood, guarding against log(0) if some posterior equals to 0
                log_likelihood = np.log(
                    likelihood,
                    out=log_likelihood,
                    where=[False if x == 0 else True for x in likelihood]
                )

                # Add the obtained log-likelihood to the result matrix
                observations_log_likelihood[observation_index, :] = log_likelihood
                observation_index += 1

        return observations_log_likelihood

    def _generate_sample_from_state(self, state, random_state=None):
        dnn_rv = _DNNHMMRandomVariable(
            self.__emission_model,
            name="DNNHMMRandomVariable",
            longname="DNNHMMRandomVariable",
            seed=random_state
        )
        return dnn_rv.rvs(size=self.__emission_model.output_shape[-1], random_state=random_state)
