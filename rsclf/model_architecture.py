# SPDX-FileCopyrightText: Copyright © Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: S. Pavankumar Dubagunta <pavankumar.dubagunta@idiap.ch>
# SPDX-FileContributor: Mathew Magimai Doss <mathew@idiap.ch>
# SPDX-FileContributor: Olivier Canévet <olivier.canevet@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

"""Define the model architecture for keras."""

import keras


def model_architecture(arch, inputFeatDim=4000, outputFeatDim=1):
    """Define the speech classification model."""
    if arch == "subseg":
        m = keras.models.Sequential(
            [
                keras.layers.Input((inputFeatDim,)),
                keras.layers.Reshape((inputFeatDim, 1)),
                keras.layers.Conv1D(filters=128, kernel_size=30, strides=10),
                keras.layers.Activation("relu"),
                keras.layers.MaxPooling1D(2),
                keras.layers.Conv1D(filters=256, kernel_size=10, strides=5),
                keras.layers.Activation("relu"),
                keras.layers.MaxPooling1D(3),
                keras.layers.Conv1D(filters=512, kernel_size=4, strides=2),
                keras.layers.Activation("relu"),
                keras.layers.Conv1D(filters=512, kernel_size=3, strides=1),
                keras.layers.Activation("relu"),
                # keras.layers.Flatten(),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(10),
                keras.layers.Activation("relu"),
            ],
        )

    elif arch == "seg":
        m = keras.models.Sequential(
            [
                keras.layers.Input((inputFeatDim,)),
                keras.layers.Reshape((inputFeatDim, 1)),
                keras.layers.Conv1D(filters=128, kernel_size=300, strides=100),
                keras.layers.Activation("relu"),
                keras.layers.MaxPooling1D(2),
                keras.layers.Conv1D(filters=256, kernel_size=5, strides=2),
                keras.layers.Activation("relu"),
                keras.layers.Conv1D(filters=512, kernel_size=4, strides=2),
                keras.layers.Activation("relu"),
                keras.layers.Conv1D(filters=512, kernel_size=3, strides=1),
                keras.layers.Activation("relu"),
                # keras.layers.Flatten(),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(10),
                keras.layers.Activation("relu"),
            ],
        )

    else:
        raise TypeError(f"Unknown architecture: {arch}")

    # Add the final layer
    m.add(keras.layers.Dense(outputFeatDim))
    if outputFeatDim == 1:
        m.add(keras.layers.Activation("sigmoid"))
    else:
        m.add(keras.layers.Activation("softmax"))

    return m
