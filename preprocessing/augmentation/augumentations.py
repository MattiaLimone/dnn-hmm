import tensorflow as tf
from audiomentations import Compose
from preprocessing.augmentation.constants import AUGMENTATION_RATIO, TRANSFORMATIONS
from preprocessing.constants import AUTOTUNE


def build_augmentation_pipeline():
    """
    It creates a pipeline of audio augmentations that will be applied to the audio files.
    :return: an augmentation pipeline.
    """
    augmentations_pipeline = Compose(list(TRANSFORMATIONS))
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

    # Save the input waveform shape
    augmented_feature_shape = tf.shape(waveform)

    # Apply pipeline
    augmented_feature = tf.numpy_function(
        apply_pipeline_func, inp=[waveform, sr], Tout=tf.float32, name="apply_pipeline"
    )

    # Recover previous waveform shape, lost in tf.numpy_function()
    augmented_feature = tf.reshape(augmented_feature, augmented_feature_shape)

    return augmented_feature, sr, audio_path, speaker


def augment_waveform_dataset(dataset: tf.data.Dataset):

    # Define identity dummy function to copy the dataset
    def __identity(waveform, sr, audio_path, speaker):
        return waveform, sr, audio_path, speaker

    # Copy the dataset
    copy = dataset.map(__identity, num_parallel_calls=AUTOTUNE)

    # Replicate each element in dataset AUGMENTATION_RATIO times
    copy = copy.repeat(AUGMENTATION_RATIO)

    # Apply augmentation transformations
    copy = copy.map(tf_apply_pipeline, num_parallel_calls=AUTOTUNE)

    # Merge the augmented data to the original data
    dataset = dataset.concatenate(copy)

    return dataset
