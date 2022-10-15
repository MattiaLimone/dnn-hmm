import random
import tensorflow as tf
from audiomentations import Compose, Gain, GainTransition, TimeStretch
from preprocessing.constants import RANDOM_SEED

random.seed(RANDOM_SEED)


def build_augmentation_pipeline():
    """
    It creates a pipeline of audio augmentations that will be applied to the audio files
    :return: A list of augmentations
    """
    augmentations_pipeline = Compose(
        [
            Gain(),
            GainTransition(),
            TimeStretch()
        ]
    )
    return augmentations_pipeline


def apply_pipeline_func(waveform, sr):
    """
    It takes in a sound file and returns a sound file that has been augmented by the pipeline

    :param waveform: the audio time series
    :param sr: sampling rate of the original signal
    :return: The shifted audio file
    """
    pipeline = build_augmentation_pipeline()
    shifted = pipeline(waveform, sr)
    return shifted


@tf.function
def tf_apply_pipeline(waveform, sr, audio_path, speaker):
    """
    It takes in a tensor of audio data and a tensor of the sampling rate, and returns a tensor of the augmented audio data
    and the sampling rate

    :param waveform: the audio data
    :param sr: sampling rate
    :param audio_path: The path to the audio file
    :param speaker: the speaker id
    :return: The augmented feature, sampling rate, audio path, and speaker
    """

    augmented_feature_shape = tf.shape(waveform)

    augmented_feature = tf.numpy_function(
        apply_pipeline_func, inp=[waveform, sr], Tout=tf.float32, name="apply_pipeline"
    )

    augmented_feature = tf.reshape(augmented_feature, augmented_feature_shape)

    return augmented_feature, sr, audio_path, speaker


def augment_audio_dataset(dataset: tf.data.Dataset):
    """
    It takes a dataset of audio files and applies the augmentation pipeline to each one

    :param dataset: The dataset to augment
    :type dataset: tf.data.Dataset
    :return: A dataset with the audio files augmented.
    """
    dataset = dataset.map(tf_apply_pipeline)
    return dataset


#augumented_dataset = augment_audio_dataset(ds)
#augumented_dataset = augumented_dataset.map(lambda y, sr: (tf.expand_dims(y, axis=0), sr))