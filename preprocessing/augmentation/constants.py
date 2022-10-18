import random
from typing import final
from audiomentations import Gain, TimeStretch, TanhDistortion, AddGaussianNoise
from preprocessing.constants import RANDOM_SEED


# Randomness-related constants
random.seed(RANDOM_SEED)

# Transformations-related constants
AUGMENTATION_RATIO: final = 2
TRANSFORMATIONS: final = frozenset([
    Gain(p=0.9),
    # GainTransition(),
    TimeStretch(p=0.9),
    AddGaussianNoise(p=0.3),
    TanhDistortion(p=0.2)
])
