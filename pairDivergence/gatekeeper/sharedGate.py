import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam

from .experts import regimeNames
from .normalization import featureColumns, horizonNames


gateInputDimension = len(featureColumns) + len(horizonNames) * len(regimeNames)
batchSize = 1024
mcInferenceBatchSize = 131072


class LearningRateRecorder(Callback):
    """Record the learning rate used at the start of every training epoch."""

    def __init__(self):
        super().__init__()
        self.learningRates = []

    def on_epoch_begin(self, epoch, logs=None):
        learningRate = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        self.learningRates.append(float(learningRate))


class FixedLearningRateSchedule(Callback):
    """Replay a fixed per-epoch learning-rate schedule."""

    def __init__(self, learningRates):
        super().__init__()
        self.learningRates = [float(value) for value in learningRates]

    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.learning_rate.assign(self.learningRates[epoch])


class TrainingProgress(Callback):
    """Print concise in-epoch progress for long Gatekeeper fits."""

    def __init__(self, label):
        super().__init__()
        self.label = str(label)
        self.epochStart = None
        self.totalSteps = None
        self.reportEvery = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epochStart = perf_counter()
        self.totalSteps = int(self.params.get("steps") or 0)
        self.reportEvery = max(1, self.totalSteps // 10) if self.totalSteps else 1

        print(
            f"{self.label} epoch {epoch + 1}/{self.params.get('epochs', '?')} "
            f"started ({self.totalSteps:,} batches)",
            flush=True
        )

    def on_train_batch_end(self, batch, logs=None):
        completed = batch + 1

        if self.totalSteps and (completed % self.reportEvery == 0 or completed == self.totalSteps):
            elapsedMinutes = (perf_counter() - self.epochStart) / 60
            percent = 100 * completed / self.totalSteps
            loss = None if logs is None else logs.get("loss")
            lossText = "" if loss is None else f", loss={loss:.6f}"

            print(
                f"  {completed:,}/{self.totalSteps:,} batches "
                f"({percent:.1f}%, {elapsedMinutes:.1f} min{lossText})",
                flush=True
            )

    def on_epoch_end(self, epoch, logs=None):
        elapsedMinutes = (perf_counter() - self.epochStart) / 60
        loss = None if logs is None else logs.get("loss")
        validationLoss = None if logs is None else logs.get("val_loss")

        details = []

        if loss is not None:
            details.append(f"loss={loss:.6f}")

        if validationLoss is not None:
            details.append(f"val_loss={validationLoss:.6f}")

        detailText = "" if not details else ", " + ", ".join(details)

        print(
            f"{self.label} epoch {epoch + 1} finished in "
            f"{elapsedMinutes:.1f} min{detailText}",
            flush=True
        )


class EpochCheckpoint(Callback):
    """Persist a resumable final shared-gate checkpoint after every epoch."""

    def __init__(self, checkpointPath, schedule):
        super().__init__()
        self.checkpointPath = Path(checkpointPath)
        self.progressPath = self.checkpointPath.with_suffix(".progress.json")
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs=None):
        self.checkpointPath.parent.mkdir(parents=True, exist_ok=True)

        temporaryModelPath = self.checkpointPath.with_name(
            self.checkpointPath.stem + ".tmp" + self.checkpointPath.suffix
        )

        self.model.save(temporaryModelPath, overwrite=True)
        os.replace(temporaryModelPath, self.checkpointPath)

        progress = {
            "completedEpochs": int(epoch + 1),
            "batchSize": int(batchSize),
            "epochs": int(self.schedule["epochs"]),
            "learningRates": [float(value) for value in self.schedule["learningRates"]]
        }

        temporaryProgressPath = self.progressPath.with_name(self.progressPath.name + ".tmp")

        with open(temporaryProgressPath, "w") as file:
            json.dump(progress, file, indent=4)

        os.replace(temporaryProgressPath, self.progressPath)

        print(f"  checkpoint saved after epoch {epoch + 1}", flush=True)


def buildGateLabels(horizonTargets, horizonThresholds):
    """Build Lower, Middle and Upper class labels for the shared gate."""
    gateLabels = {}

    for horizon in horizonNames:
        target = horizonTargets[horizon]
        lowerThreshold = horizonThresholds[horizon]["Lower"]
        upperThreshold = horizonThresholds[horizon]["Upper"]

        labels = np.ones(len(target), dtype=int)
        labels[target < lowerThreshold] = 0
        labels[target > upperThreshold] = 2
        gateLabels[horizon] = labels

    return gateLabels


def fitGateClassWeights(gateTrainLabels):
    """Fit inverse-frequency class weights from gate training labels."""
    classWeights = {}
    sampleWeights = {}

    for horizon in horizonNames:
        counts = np.bincount(gateTrainLabels[horizon], minlength=3)
        nEvents = len(gateTrainLabels[horizon])
        weights = nEvents / (3 * counts)

        classWeights[horizon] = weights
        sampleWeights[horizon] = weights[gateTrainLabels[horizon]]

    return classWeights, sampleWeights


def getGateSampleWeights(gateLabels, classWeights):
    """Apply fitted gate class weights to another label set."""
    return {
        horizon: classWeights[horizon][gateLabels[horizon]]
        for horizon in horizonNames
    }


def buildSharedGate(modelName):
    """Build the shared three-horizon regime gate."""
    inputs = Input(shape=(gateInputDimension,))

    x = Dense(128)(inputs)
    x = LeakyReLU(negative_slope=0.01)(x)
    x = Dropout(0.1)(x)

    x = Dense(64)(x)
    x = LeakyReLU(negative_slope=0.01)(x)

    outputF1 = Dense(3, activation="softmax", name="F1")(x)
    outputF3 = Dense(3, activation="softmax", name="F3")(x)
    outputF5 = Dense(3, activation="softmax", name="F5")(x)

    return Model(inputs=inputs, outputs=[outputF1, outputF3, outputF5], name=modelName)


def compileSharedGate(model):
    """Compile the shared gate."""
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=[
            "sparse_categorical_crossentropy",
            "sparse_categorical_crossentropy",
            "sparse_categorical_crossentropy"
        ],
        metrics=[["accuracy"], ["accuracy"], ["accuracy"]]
    )


def buildSharedGateCallbacks(checkpointPath):
    """Build callbacks used while tuning the shared gate."""
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


def _fitTunedSharedGate(
    model,
    XTrainGate,
    gateTrainLabels,
    XValidationGate,
    validationGateLabels,
    trainSampleWeights,
    validationSampleWeights,
    checkpointPath,
    showProgress
):
    learningRateRecorder = LearningRateRecorder()

    callbacks = [
        *buildSharedGateCallbacks(checkpointPath),
        learningRateRecorder
    ]

    if showProgress:
        callbacks.append(TrainingProgress("Shared-gate tuning"))

    history = model.fit(
        XTrainGate,
        [
            gateTrainLabels["F1"],
            gateTrainLabels["F3"],
            gateTrainLabels["F5"]
        ],
        validation_data=(
            XValidationGate,
            [
                validationGateLabels["F1"],
                validationGateLabels["F3"],
                validationGateLabels["F5"]
            ],
            [
                validationSampleWeights["F1"],
                validationSampleWeights["F3"],
                validationSampleWeights["F5"]
            ]
        ),
        sample_weight=[
            trainSampleWeights["F1"],
            trainSampleWeights["F3"],
            trainSampleWeights["F5"]
        ],
        epochs=100,
        batch_size=batchSize,
        callbacks=callbacks,
        shuffle=True,
        verbose=0
    )

    bestEpoch = int(np.argmin(history.history["val_loss"]) + 1)

    return history, bestEpoch, learningRateRecorder.learningRates


def tuneSharedGate(
    XTrainGate,
    gateTrainLabels,
    XValidationGate,
    validationGateLabels,
    checkpointPath=None,
    showProgress=False
):
    """Tune the shared gate using chronological gate training and validation data."""
    classWeights, trainSampleWeights = fitGateClassWeights(gateTrainLabels)
    validationSampleWeights = getGateSampleWeights(validationGateLabels, classWeights)

    model = buildSharedGate("sharedGate")
    compileSharedGate(model)

    if checkpointPath is None:
        with TemporaryDirectory() as checkpointDirectory:
            temporaryCheckpointPath = Path(checkpointDirectory)/"sharedGate.keras"
            history, bestEpoch, learningRates = _fitTunedSharedGate(
                model,
                XTrainGate,
                gateTrainLabels,
                XValidationGate,
                validationGateLabels,
                trainSampleWeights,
                validationSampleWeights,
                temporaryCheckpointPath,
                showProgress
            )
    else:
        checkpointPath = Path(checkpointPath)
        checkpointPath.parent.mkdir(parents=True, exist_ok=True)
        history, bestEpoch, learningRates = _fitTunedSharedGate(
            model,
            XTrainGate,
            gateTrainLabels,
            XValidationGate,
            validationGateLabels,
            trainSampleWeights,
            validationSampleWeights,
            checkpointPath,
            showProgress
        )

    return {
        "model": model,
        "history": history,
        "bestEpoch": bestEpoch,
        "learningRates": learningRates,
        "trainingRows": len(XTrainGate),
        "classWeights": classWeights,
        "trainSampleWeights": trainSampleWeights,
        "validationSampleWeights": validationSampleWeights
    }


def buildFinalLearningRateSchedule(
    finalTrainingRows,
    tuningRows,
    bestEpoch,
    tuningLearningRates
):
    """Convert the tuned gate trajectory to an equivalent optimizer-step schedule."""
    bestEpoch = int(bestEpoch)
    tuningRows = int(tuningRows)
    finalTrainingRows = int(finalTrainingRows)

    if bestEpoch < 1:
        raise ValueError("bestEpoch must be at least 1.")

    if tuningRows < 1 or finalTrainingRows < 1:
        raise ValueError("Gate training row counts must be positive.")

    if len(tuningLearningRates) < bestEpoch:
        raise ValueError("tuningLearningRates does not contain every epoch through bestEpoch.")

    tuningLearningRates = [float(value) for value in tuningLearningRates[:bestEpoch]]

    tuningBatches = int(np.ceil(tuningRows / batchSize))
    finalBatches = int(np.ceil(finalTrainingRows / batchSize))
    targetOptimizerSteps = bestEpoch * tuningBatches

    finalEpochs = max(1, int(np.floor(targetOptimizerSteps / finalBatches + 0.5)))
    finalLearningRates = []

    for finalEpoch in range(finalEpochs):
        optimizerStepsBeforeEpoch = finalEpoch * finalBatches
        tuningEpochIndex = min(optimizerStepsBeforeEpoch // tuningBatches, bestEpoch - 1)
        finalLearningRates.append(tuningLearningRates[tuningEpochIndex])

    return {
        "learningRates": finalLearningRates,
        "epochs": finalEpochs,
        "targetOptimizerSteps": targetOptimizerSteps,
        "actualOptimizerSteps": finalEpochs * finalBatches,
        "tuningBatchesPerEpoch": tuningBatches,
        "finalBatchesPerEpoch": finalBatches,
        "batchSize": batchSize
    }


def _loadFinalCheckpoint(checkpointPath, schedule):
    checkpointPath = Path(checkpointPath)
    progressPath = checkpointPath.with_suffix(".progress.json")

    if not checkpointPath.exists() or not progressPath.exists():
        return None, 0

    with open(progressPath, "r") as file:
        progress = json.load(file)

    expectedLearningRates = [float(value) for value in schedule["learningRates"]]

    if int(progress.get("batchSize", -1)) != batchSize:
        raise RuntimeError("Existing final-gate checkpoint uses a different batch size.")

    if int(progress.get("epochs", -1)) != int(schedule["epochs"]):
        raise RuntimeError("Existing final-gate checkpoint uses a different epoch schedule.")

    if [float(value) for value in progress.get("learningRates", [])] != expectedLearningRates:
        raise RuntimeError("Existing final-gate checkpoint uses a different learning-rate schedule.")

    completedEpochs = int(progress["completedEpochs"])
    model = tf.keras.models.load_model(checkpointPath)

    return model, completedEpochs


def trainFinalSharedGate(
    XTrainGate,
    gateTrainLabels,
    bestEpoch,
    tuningRows,
    tuningLearningRates,
    randomSeed=42,
    checkpointPath=None,
    showProgress=False
):
    """Train the final shared gate using the tuned optimizer-step trajectory."""
    classWeights, sampleWeights = fitGateClassWeights(gateTrainLabels)

    schedule = buildFinalLearningRateSchedule(
        len(XTrainGate),
        tuningRows,
        bestEpoch,
        tuningLearningRates
    )

    model = None
    completedEpochs = 0

    if checkpointPath is not None:
        model, completedEpochs = _loadFinalCheckpoint(checkpointPath, schedule)

    if model is None:
        model = buildSharedGate("sharedGate")
        compileSharedGate(model)
        np.random.seed(randomSeed)
        tf.random.set_seed(randomSeed)
    else:
        print(
            f"Resuming final shared gate from epoch {completedEpochs}/{schedule['epochs']}.",
            flush=True
        )

    if completedEpochs < schedule["epochs"]:
        callbacks = [
            TerminateOnNaN(),
            FixedLearningRateSchedule(schedule["learningRates"])
        ]

        if showProgress:
            callbacks.append(TrainingProgress("Final shared gate"))

        if checkpointPath is not None:
            callbacks.append(EpochCheckpoint(checkpointPath, schedule))

        model.fit(
            XTrainGate,
            [
                gateTrainLabels["F1"],
                gateTrainLabels["F3"],
                gateTrainLabels["F5"]
            ],
            sample_weight=[
                sampleWeights["F1"],
                sampleWeights["F3"],
                sampleWeights["F5"]
            ],
            initial_epoch=completedEpochs,
            epochs=schedule["epochs"],
            batch_size=batchSize,
            shuffle=True,
            callbacks=callbacks,
            verbose=0
        )

    return {
        "model": model,
        "classWeights": classWeights,
        "sampleWeights": sampleWeights,
        "schedule": schedule
    }


def predictSharedGate(model, XData):
    """Predict deterministic shared-gate regime classes."""
    probabilities = model.predict(XData, batch_size=4096, verbose=0)

    return {
        "probabilities": {
            "F1": probabilities[0],
            "F3": probabilities[1],
            "F5": probabilities[2]
        },
        "predictions": {
            "F1": np.argmax(probabilities[0], axis=1),
            "F3": np.argmax(probabilities[1], axis=1),
            "F5": np.argmax(probabilities[2], axis=1)
        }
    }


def predictSharedGateMC(
    model,
    XData,
    mcRuns=100,
    inferenceBatchSize=mcInferenceBatchSize,
    showProgress=False
):
    """Predict gate regimes and MC-dropout uncertainty without storing every MC pass."""
    nRows = len(XData)

    if nRows == 0:
        raise ValueError("XData must contain at least one row.")

    if mcRuns < 1:
        raise ValueError("mcRuns must be at least 1.")

    means = {
        horizon: np.zeros((nRows, 3), dtype=np.float64)
        for horizon in horizonNames
    }

    m2 = {
        horizon: np.zeros((nRows, 3), dtype=np.float64)
        for horizon in horizonNames
    }

    reportEvery = max(1, mcRuns // 10)

    for runIndex in range(mcRuns):
        count = runIndex + 1

        for batchStart in range(0, nRows, inferenceBatchSize):
            batchEnd = min(batchStart + inferenceBatchSize, nRows)
            predictions = model(XData[batchStart:batchEnd], training=True)

            for horizonIndex, horizon in enumerate(horizonNames):
                values = predictions[horizonIndex].numpy().astype(np.float64, copy=False)
                currentMean = means[horizon][batchStart:batchEnd]
                currentM2 = m2[horizon][batchStart:batchEnd]

                delta = values - currentMean
                currentMean += delta / count
                delta2 = values - currentMean
                currentM2 += delta * delta2

        if showProgress and (count % reportEvery == 0 or count == mcRuns):
            print(f"MC dropout: {count}/{mcRuns} runs ({100 * count / mcRuns:.0f}%)", flush=True)

    meanProbabilities = {}
    gatePredictions = {}
    uncertainty = {}

    for horizon in horizonNames:
        variance = m2[horizon] / mcRuns
        meanProbabilities[horizon] = means[horizon].astype(np.float32)
        gatePredictions[horizon] = np.argmax(means[horizon], axis=1)
        uncertainty[horizon] = np.sqrt(variance).mean(axis=1).astype(np.float32)

    return {
        "meanProbabilities": meanProbabilities,
        "predictions": gatePredictions,
        "uncertainty": uncertainty
    }
