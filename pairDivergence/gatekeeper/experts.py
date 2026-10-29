from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from .normalization import featureColumns, horizonNames


regimeNames = [
    "Lower",
    "Middle",
    "Upper"
]

expertHuberDeltas = {
    "F1": {
        "Lower": 0.15,
        "Middle": 0.125,
        "Upper": 1.30
    },
    "F3": {
        "Lower": 0.15,
        "Middle": 0.12,
        "Upper": 0.15
    },
    "F5": {
        "Lower": 0.15,
        "Middle": 0.11,
        "Upper": 0.15
    }
}


def buildExpert(modelName):
    """
    Build one horizon-regime regression expert.
    """
    inputs = Input(shape=(len(featureColumns),))

    x = Dense(128)(inputs)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = Dropout(0.1)(x)

    x = Dense(64)(x)
    x = LeakyReLU(negative_slope=0.01)(x)

    outputs = Dense(1)(x)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name=modelName
    )


def compileExpert(model, horizon, regime):
    """
    Compile one expert using its horizon-regime Huber loss.
    """
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(delta=expertHuberDeltas[horizon][regime])
    )


def fitHorizonThresholds(horizonTargets):
    """
    Fit Lower and Upper regime thresholds from training targets.
    """
    horizonThresholds = {}

    for horizon in horizonNames:
        horizonThresholds[horizon] = {
            "Lower": np.quantile(horizonTargets[horizon], 0.10),
            "Upper": np.quantile(horizonTargets[horizon], 0.90)
        }

    return horizonThresholds


def getHorizonMasks(horizonTargets, horizonThresholds):
    """
    Assign targets to Lower, Middle and Upper regimes.
    """
    horizonMasks = {}

    for horizon in horizonNames:
        target = horizonTargets[horizon]
        lowerThreshold = horizonThresholds[horizon]["Lower"]
        upperThreshold = horizonThresholds[horizon]["Upper"]

        horizonMasks[horizon] = {
            "Lower": target < lowerThreshold,
            "Middle": (
                (target >= lowerThreshold) &
                (target <= upperThreshold)
            ),
            "Upper": target > upperThreshold
        }

    return horizonMasks


def buildExpertCallbacks(checkpointPath):
    """
    Build the callbacks used while tuning expert epoch counts.
    """
    return [
        TerminateOnNaN(),

        ModelCheckpoint(
            checkpointPath,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode="min",
            verbose=0
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            mode="min",
            verbose=0
        )
    ]


def tuneExperts(
    XTrain,
    trainHorizonTargets,
    XValidation,
    validationHorizonTargets
):
    """
    Tune all nine experts and determine their best epoch counts.
    """
    horizonThresholds = fitHorizonThresholds(trainHorizonTargets)

    trainHorizonMasks = getHorizonMasks(
        trainHorizonTargets,
        horizonThresholds
    )

    validationHorizonMasks = getHorizonMasks(
        validationHorizonTargets,
        horizonThresholds
    )

    experts = {}
    histories = {}
    bestEpochs = {}

    with TemporaryDirectory() as checkpointDirectory:
        checkpointDirectory = Path(checkpointDirectory)

        for horizon in horizonNames:
            experts[horizon] = {}
            histories[horizon] = {}
            bestEpochs[horizon] = {}

            for regime in regimeNames:
                model = buildExpert(f"expert{horizon}{regime}")

                compileExpert(
                    model,
                    horizon,
                    regime
                )

                trainMask = trainHorizonMasks[horizon][regime]
                validationMask = validationHorizonMasks[horizon][regime]

                checkpointPath = checkpointDirectory/f"expert{horizon}{regime}.keras"

                history = model.fit(
                    XTrain[trainMask],
                    trainHorizonTargets[horizon][trainMask],
                    validation_data=(
                        XValidation[validationMask],
                        validationHorizonTargets[horizon][validationMask]
                    ),
                    epochs=100,
                    batch_size=256,
                    callbacks=buildExpertCallbacks(checkpointPath),
                    shuffle=True,
                    verbose=0
                )

                bestEpochs[horizon][regime] = np.argmin(history.history["val_loss"]) + 1

                experts[horizon][regime] = model
                histories[horizon][regime] = history

    return {
        "experts": experts,
        "histories": histories,
        "bestEpochs": bestEpochs,
        "horizonThresholds": horizonThresholds,
        "trainHorizonMasks": trainHorizonMasks,
        "validationHorizonMasks": validationHorizonMasks
    }


def trainExperts(
    XTrain,
    trainHorizonTargets,
    horizonMasks,
    bestEpochs,
    modelPrefix="expert"
):
    """
    Train all nine experts for fixed epoch counts.
    """
    experts = {}

    for horizon in horizonNames:
        experts[horizon] = {}

        for regime in regimeNames:
            model = buildExpert(f"{modelPrefix}{horizon}{regime}")

            compileExpert(
                model,
                horizon,
                regime
            )

            trainMask = horizonMasks[horizon][regime]

            model.fit(
                XTrain[trainMask],
                trainHorizonTargets[horizon][trainMask],
                epochs=bestEpochs[horizon][regime],
                batch_size=256,
                shuffle=True,
                verbose=0
            )

            experts[horizon][regime] = model

    return experts


def predictExperts(experts, XData):
    """
    Predict every horizon-regime expert for the supplied features.
    """
    predictions = {}

    for horizon in horizonNames:
        predictions[horizon] = {}

        for regime in regimeNames:
            predictions[horizon][regime] = experts[horizon][regime].predict(
                XData,
                verbose=0
            ).flatten()

    return predictions