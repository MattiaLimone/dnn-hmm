import random
from typing import final

from audiomentations import Gain, GainTransition, TimeStretch

from preprocessing.constants import RANDOM_SEED


# Randomness-related constants
random.seed(RANDOM_SEED)

# Transformations-related constants
AUGMENTATION_RATIO: final = 2
TRANSFORMATIONS: final = frozenset([
    Gain(),
    GainTransition(),
    TimeStretch()
])
