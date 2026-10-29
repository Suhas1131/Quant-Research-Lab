import gc
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .experts import fitHorizonThresholds, getHorizonMasks, predictExperts, regimeNames, trainExperts, tuneExperts
from .normalization import featureColumns, fitNormalizationParameters, fitTargetScales, getTargetScaleVector, horizonNames, normalizeFeatureData, normalizeTargetData
from .sharedGate import buildGateLabels, predictSharedGate, trainFinalSharedGate, tuneSharedGate


defaultTargetPrecision = 0.93
defaultConfidenceLevels = sorted(set([0.025] + [value / 100 for value in range(1, 101)]))
mcInferenceBatchSize = 131072

horizonSteps = {
    "F1": 1,
    "F3": 3,
    "F5": 5
}

expertPredictionColumns = [
    f"{horizon}{regime}"
    for horizon in horizonNames
    for regime in regimeNames
]


def _getJsonSafeValue(value):
    if isinstance(value, dict):
        return {
            str(key): _getJsonSafeValue(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _getJsonSafeValue(item) for item in value
        ]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        if np.isposinf(value):
            return "Infinity"

        if np.isneginf(value):
            return "-Infinity"

    return value


def _restoreJsonNumber(value):
    if value == "Infinity":
        return np.inf

    if value == "-Infinity":
        return -np.inf

    return value


class Gatekeeper:
    """Pair-divergence convergence gate."""

    def __init__(
        self,
        targetPrecision=defaultTargetPrecision,
        confidenceLevels=None,
        mcRuns=100,
        randomSeed=42,
        oofStartDate="2021-01-01",
        recipe=None
    ):
        self.targetPrecision = float(targetPrecision)

        if self.targetPrecision <= 0 or self.targetPrecision > 1:
            raise ValueError(
                "targetPrecision must be greater than 0 and less than or equal to 1."
            )

        self.confidenceLevels = (
            list(defaultConfidenceLevels)
            if confidenceLevels is None
            else sorted(set(float(level) for level in confidenceLevels))
        )

        if any(
            level <= 0 or level > 1
            for level in self.confidenceLevels
        ):
            raise ValueError(
                "confidenceLevels must be greater than 0 and less than or equal to 1."
            )

        if 1.00 not in self.confidenceLevels:
            self.confidenceLevels.append(1.00)
            self.confidenceLevels.sort()

        self.deterministicConfidenceLevel = None
        self.mcConfidenceLevel = None

        self.mcRuns = int(mcRuns)

        if self.mcRuns < 1:
            raise ValueError("mcRuns must be at least 1.")

        self.randomSeed = int(randomSeed)
        self.oofStartDate = pd.Timestamp(oofStartDate)

        self.recipe = None
        self.modelDirectory = None
        self.normalizationParameters = None
        self.targetScales = None
        self.targetScaleVector = None
        self.horizonThresholds = None
        self.experts = None
        self.sharedGate = None
        self.oofExpertPredictions = None
        self.oofGateLabels = None

        self.tuningNormalizationParameters = None
        self.tuningTargetScales = None
        self.tuningTargetScaleVector = None
        self.tuningHorizonThresholds = None
        self.tuningExperts = None
        self.tuningSharedGate = None

        self.fitted = False

        if recipe is not None:
            self.loadRecipe(recipe)

    def getRecipe(self):
        """Return a copy of the frozen tuning recipe."""
        if self.recipe is None:
            raise RuntimeError(
                "Gatekeeper has not been tuned and no recipe has been loaded."
            )

        return deepcopy(self.recipe)

    def loadRecipe(
        self,
        recipe
    ):
        """Load a previously tuned Gatekeeper recipe."""
        requiredKeys = {
            "expertBestEpochs",
            "recipeAvailableDate",
            "mcRuns",
            "randomSeed",
            "oofStartDate"
        }

        missingKeys = requiredKeys - set(recipe)

        if missingKeys:
            raise ValueError(
                f"Recipe is missing required keys: {sorted(missingKeys)}"
            )

        loadedRecipe = deepcopy(recipe)

        if int(loadedRecipe["mcRuns"]) != self.mcRuns:
            raise ValueError(
                "Recipe mcRuns does not match this Gatekeeper."
            )

        if int(loadedRecipe["randomSeed"]) != self.randomSeed:
            raise ValueError(
                "Recipe randomSeed does not match this Gatekeeper."
            )

        if pd.Timestamp(loadedRecipe["oofStartDate"]) != self.oofStartDate:
            raise ValueError(
                "Recipe oofStartDate does not match this Gatekeeper."
            )

        for key in [
            "deterministicConfidenceCutoff",
            "mcConfidenceCutoff"
        ]:
            if key in loadedRecipe:
                loadedRecipe[key] = float(
                    _restoreJsonNumber(
                        loadedRecipe[key]
                    )
                )

        if loadedRecipe.get("confidenceMethod") == "deterministicMargin+fullPipelineMCSNR":
            if "targetPrecision" in loadedRecipe:
                self._setTargetPrecision(
                    loadedRecipe["targetPrecision"]
                )

            if "confidenceLevels" in loadedRecipe:
                self.confidenceLevels = self._validateConfidenceLevels(
                    loadedRecipe["confidenceLevels"]
                )

            if "deterministicConfidenceLevel" in loadedRecipe:
                self.deterministicConfidenceLevel = float(
                    loadedRecipe["deterministicConfidenceLevel"]
                )

            if "mcConfidenceLevel" in loadedRecipe:
                self.mcConfidenceLevel = float(
                    loadedRecipe["mcConfidenceLevel"]
                )

        self.recipe = loadedRecipe

    def save(
        self,
        modelDirectory=None
    ):
        """Save the fitted Gatekeeper models and prediction metadata."""
        if not self.fitted:
            raise RuntimeError(
                "Gatekeeper must be fitted before it can be saved."
            )

        if modelDirectory is None:
            modelDirectory = (
                self.modelDirectory
                if self.modelDirectory is not None
                else Path(__file__).resolve().parent/"models"
            )
        else:
            modelDirectory = Path(modelDirectory)

        self.modelDirectory = modelDirectory

        modelDirectory.mkdir(
            parents=True,
            exist_ok=True
        )

        for horizon in horizonNames:
            for regime in regimeNames:
                modelPath = modelDirectory/f"expert_{horizon}_{regime}.keras"

                self.experts[horizon][regime].save(
                    modelPath,
                    overwrite=True
                )

        self.sharedGate.save(
            modelDirectory/"sharedGate.keras",
            overwrite=True
        )

        self.saveMetadata(modelDirectory)

        return modelDirectory

    def saveMetadata(
        self,
        modelDirectory=None
    ):
        """Save Gatekeeper metadata without rewriting model weights."""
        if not self.fitted:
            raise RuntimeError(
                "Gatekeeper must be fitted before metadata can be saved."
            )

        if modelDirectory is None:
            modelDirectory = (
                self.modelDirectory
                if self.modelDirectory is not None
                else Path(__file__).resolve().parent/"models"
            )
        else:
            modelDirectory = Path(modelDirectory)

        self.modelDirectory = modelDirectory

        modelDirectory.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata = {
            "formatVersion": 3,
            "targetPrecision": self.targetPrecision,
            "confidenceLevels": self.confidenceLevels,
            "deterministicConfidenceLevel": self.deterministicConfidenceLevel,
            "mcConfidenceLevel": self.mcConfidenceLevel,
            "mcRuns": self.mcRuns,
            "randomSeed": self.randomSeed,
            "oofStartDate": self.oofStartDate,
            "recipe": self.recipe,
            "normalizationParameters": self.normalizationParameters,
            "targetScales": self.targetScales,
            "horizonThresholds": self.horizonThresholds
        }

        with open(modelDirectory/"metadata.json", "w") as file:
            json.dump(
                _getJsonSafeValue(metadata),
                file,
                indent=4
            )

        return modelDirectory/"metadata.json"

    def saveTuningModels(
        self,
        modelDirectory=None
    ):
        """Save the exact pre-final models used during chronological tuning."""
        if (
            self.tuningExperts is None or
            self.tuningSharedGate is None or
            self.tuningNormalizationParameters is None or
            self.tuningTargetScales is None or
            self.tuningHorizonThresholds is None
        ):
            raise RuntimeError(
                "Gatekeeper tuning models are not available. Run tune() before saveTuningModels()."
            )

        if self.recipe is None:
            raise RuntimeError(
                "Gatekeeper recipe is not available. Run tune() before saveTuningModels()."
            )

        if modelDirectory is None:
            modelDirectory = Path(__file__).resolve().parent/"models"/"tuning"
        else:
            modelDirectory = Path(modelDirectory)

        modelDirectory.mkdir(
            parents=True,
            exist_ok=True
        )

        for horizon in horizonNames:
            for regime in regimeNames:
                modelPath = modelDirectory/f"expert_{horizon}_{regime}.keras"

                self.tuningExperts[horizon][regime].save(
                    modelPath,
                    overwrite=True
                )

        self.tuningSharedGate.save(
            modelDirectory/"sharedGate.keras",
            overwrite=True
        )

        metadata = {
            "formatVersion": 3,
            "trainEnd": self.recipe["tuneTrainEnd"],
            "validationEnd": self.recipe["tuneValidationEnd"],
            "randomSeed": self.randomSeed,
            "targetPrecision": self.targetPrecision,
            "confidenceLevels": self.confidenceLevels,
            "deterministicConfidenceLevel": self.deterministicConfidenceLevel,
            "mcConfidenceLevel": self.mcConfidenceLevel,
            "normalizationParameters": self.tuningNormalizationParameters,
            "targetScales": self.tuningTargetScales,
            "horizonThresholds": self.tuningHorizonThresholds,
            "expertBestEpochs": self.recipe["expertBestEpochs"],
            "sharedGateBestEpoch": self.recipe["sharedGateBestEpoch"],
            "sharedGateTuningRows": self.recipe["sharedGateTuningRows"],
            "sharedGateLearningRates": self.recipe["sharedGateLearningRates"]
        }

        with open(modelDirectory/"metadata.json", "w") as file:
            json.dump(
                _getJsonSafeValue(metadata),
                file,
                indent=4
            )

        return modelDirectory

    @staticmethod
    def loadTuningModels(
        modelDirectory=None
    ):
        """Load the exact pre-final models and preprocessing used during tuning."""
        if modelDirectory is None:
            modelDirectory = Path(__file__).resolve().parent/"models"/"tuning"
        else:
            modelDirectory = Path(modelDirectory)

        metadataPath = modelDirectory/"metadata.json"

        if not metadataPath.exists():
            raise FileNotFoundError(
                f"Gatekeeper tuning metadata not found: {metadataPath}"
            )

        with open(metadataPath, "r") as file:
            metadata = json.load(file)

        if metadata.get("formatVersion") not in [1, 2, 3]:
            raise ValueError("Unsupported Gatekeeper tuning-model format.")

        experts = {}

        for horizon in horizonNames:
            experts[horizon] = {}

            for regime in regimeNames:
                modelPath = modelDirectory/f"expert_{horizon}_{regime}.keras"

                if not modelPath.exists():
                    raise FileNotFoundError(
                        f"Gatekeeper tuning expert not found: {modelPath}"
                    )

                experts[horizon][regime] = tf.keras.models.load_model(
                    modelPath,
                    compile=False
                )

        sharedGatePath = modelDirectory/"sharedGate.keras"

        if not sharedGatePath.exists():
            raise FileNotFoundError(
                f"Gatekeeper tuning shared gate not found: {sharedGatePath}"
            )

        sharedGate = tf.keras.models.load_model(
            sharedGatePath,
            compile=False
        )

        return {
            "metadata": metadata,
            "normalizationParameters": metadata["normalizationParameters"],
            "targetScales": metadata["targetScales"],
            "targetScaleVector": getTargetScaleVector(metadata["targetScales"]),
            "horizonThresholds": metadata["horizonThresholds"],
            "experts": experts,
            "sharedGate": sharedGate
        }

    def clearTuningModels(self):
        """Release tuning-stage model objects after they have been saved."""
        self.tuningExperts = None
        self.tuningSharedGate = None

        gc.collect()
        tf.keras.backend.clear_session()

    @classmethod
    def load(
        cls,
        modelDirectory=None
    ):
        """Load a previously fitted Gatekeeper."""
        if modelDirectory is None:
            modelDirectory = Path(__file__).resolve().parent/"models"
        else:
            modelDirectory = Path(modelDirectory)

        metadataPath = modelDirectory/"metadata.json"

        if not metadataPath.exists():
            raise FileNotFoundError(
                f"Gatekeeper metadata not found: {metadataPath}"
            )

        with open(metadataPath, "r") as file:
            metadata = json.load(file)

        formatVersion = metadata.get("formatVersion")

        if formatVersion not in [1, 2, 3]:
            raise ValueError("Unsupported Gatekeeper model format.")

        if formatVersion == 3:
            confidenceLevels = metadata.get(
                "confidenceLevels",
                defaultConfidenceLevels
            )

            targetPrecision = metadata.get(
                "targetPrecision",
                defaultTargetPrecision
            )
        else:
            confidenceLevels = defaultConfidenceLevels
            targetPrecision = defaultTargetPrecision

        gatekeeper = cls(
            targetPrecision=targetPrecision,
            confidenceLevels=confidenceLevels,
            mcRuns=metadata["mcRuns"],
            randomSeed=metadata["randomSeed"],
            oofStartDate=metadata["oofStartDate"],
            recipe=metadata["recipe"]
        )

        if (
            gatekeeper.deterministicConfidenceLevel is None and
            metadata.get("deterministicConfidenceLevel") is not None
        ):
            gatekeeper.deterministicConfidenceLevel = float(
                metadata["deterministicConfidenceLevel"]
            )

        if (
            gatekeeper.mcConfidenceLevel is None and
            metadata.get("mcConfidenceLevel") is not None
        ):
            gatekeeper.mcConfidenceLevel = float(
                metadata["mcConfidenceLevel"]
            )

        gatekeeper.normalizationParameters = metadata["normalizationParameters"]
        gatekeeper.targetScales = metadata["targetScales"]
        gatekeeper.targetScaleVector = getTargetScaleVector(gatekeeper.targetScales)
        gatekeeper.horizonThresholds = metadata["horizonThresholds"]
        gatekeeper.experts = {}

        for horizon in horizonNames:
            gatekeeper.experts[horizon] = {}

            for regime in regimeNames:
                modelPath = modelDirectory/f"expert_{horizon}_{regime}.keras"

                if not modelPath.exists():
                    raise FileNotFoundError(
                        f"Gatekeeper expert not found: {modelPath}"
                    )

                gatekeeper.experts[horizon][regime] = tf.keras.models.load_model(
                    modelPath,
                    compile=False
                )

        sharedGatePath = modelDirectory/"sharedGate.keras"

        if not sharedGatePath.exists():
            raise FileNotFoundError(
                f"Shared gate not found: {sharedGatePath}"
            )

        gatekeeper.sharedGate = tf.keras.models.load_model(
            sharedGatePath,
            compile=False
        )

        gatekeeper.modelDirectory = modelDirectory
        gatekeeper.fitted = True

        return gatekeeper

    def tune(
        self,
        data,
        trainEnd,
        validationEnd,
        targetPrecision=None
    ):
        """Tune expert epochs, the shared gate and confidence thresholds chronologically."""
        trainEnd = pd.Timestamp(trainEnd)
        validationEnd = pd.Timestamp(validationEnd)

        if validationEnd <= trainEnd:
            raise ValueError(
                "validationEnd must be later than trainEnd."
            )

        self._validateDataColumns(
            data,
            requireTargets=True
        )

        if targetPrecision is not None:
            self._setTargetPrecision(
                targetPrecision
            )

        dateValues = pd.to_datetime(data["date"])
        targetDateValues = pd.to_datetime(data["targetDate_F5"])

        trainMask = (
            (dateValues < trainEnd) &
            (targetDateValues < trainEnd)
        )

        validationMask = (
            (dateValues >= trainEnd) &
            (dateValues < validationEnd) &
            (targetDateValues < validationEnd)
        )

        trainData = data.loc[trainMask].copy()
        validationData = data.loc[validationMask].copy()

        if len(trainData) == 0:
            raise ValueError(
                "No legal training rows are available for tuning."
            )

        if len(validationData) == 0:
            raise ValueError(
                "No legal validation rows are available for tuning."
            )

        self._setRandomSeed()

        normalizationParameters = fitNormalizationParameters(trainData)

        normalizedTrainData = normalizeFeatureData(
            trainData,
            normalizationParameters
        )

        normalizedValidationData = normalizeFeatureData(
            validationData,
            normalizationParameters
        )

        targetScales = fitTargetScales(trainData)
        targetScaleVector = getTargetScaleVector(targetScales)

        trainHorizonTargets = self._getHorizonTargets(
            trainData,
            targetScales
        )

        validationHorizonTargets = self._getHorizonTargets(
            validationData,
            targetScales
        )

        tuningResults = tuneExperts(
            normalizedTrainData.to_numpy(),
            trainHorizonTargets,
            normalizedValidationData.to_numpy(),
            validationHorizonTargets
        )

        expertBestEpochs = {
            horizon: {
                regime: int(
                    tuningResults["bestEpochs"][horizon][regime]
                )
                for regime in regimeNames
            }
            for horizon in horizonNames
        }

        oofExpertPredictions, gateTrainLabels = self._buildOofTrainingData(
            trainData,
            targetScales,
            expertBestEpochs
        )

        XTrainGate = np.column_stack([
            normalizedTrainData.loc[oofExpertPredictions.index].to_numpy(),
            oofExpertPredictions[expertPredictionColumns].to_numpy()
        ])

        validationExpertPredictions = predictExperts(
            tuningResults["experts"],
            normalizedValidationData.to_numpy()
        )

        XValidationGate = np.column_stack([
            normalizedValidationData.to_numpy(),
            self._stackExpertPredictions(validationExpertPredictions)
        ])

        validationGateLabels = buildGateLabels(
            validationHorizonTargets,
            tuningResults["horizonThresholds"]
        )

        sharedGateTuningResults = tuneSharedGate(
            XTrainGate,
            gateTrainLabels,
            XValidationGate,
            validationGateLabels
        )

        confidenceResults = self._selectConfidenceMix(
            tuningResults["experts"],
            sharedGateTuningResults["model"],
            normalizedValidationData.to_numpy(dtype=np.float32),
            validationData,
            targetScaleVector
        )

        self.tuningNormalizationParameters = deepcopy(normalizationParameters)
        self.tuningTargetScales = deepcopy(targetScales)
        self.tuningTargetScaleVector = targetScaleVector.copy()
        self.tuningHorizonThresholds = deepcopy(tuningResults["horizonThresholds"])
        self.tuningExperts = tuningResults["experts"]
        self.tuningSharedGate = sharedGateTuningResults["model"]

        self.recipe = {
            "expertBestEpochs": expertBestEpochs,
            "sharedGateBestEpoch": int(
                sharedGateTuningResults["bestEpoch"]
            ),
            "sharedGateTuningRows": int(
                sharedGateTuningResults["trainingRows"]
            ),
            "sharedGateLearningRates": [
                float(value)
                for value in sharedGateTuningResults["learningRates"]
            ],
            "confidenceMethod": "deterministicMargin+fullPipelineMCSNR",
            "targetPrecision": self.targetPrecision,
            "confidenceLevels": list(self.confidenceLevels),
            "deterministicConfidenceLevel": self.deterministicConfidenceLevel,
            "mcConfidenceLevel": self.mcConfidenceLevel,
            "deterministicConfidenceCutoff": confidenceResults["deterministicConfidenceCutoff"],
            "mcConfidenceCutoff": confidenceResults["mcConfidenceCutoff"],
            "confidenceValidationPrecision": confidenceResults["precision"],
            "confidenceValidationTradeRate": confidenceResults["tradeRate"],
            "confidenceValidationSignals": confidenceResults["signals"],
            "confidenceCalibrationRows": int(len(validationData)),
            "recipeAvailableDate": validationEnd.isoformat(),
            "tuneTrainEnd": trainEnd.isoformat(),
            "tuneValidationEnd": validationEnd.isoformat(),
            "mcRuns": self.mcRuns,
            "randomSeed": self.randomSeed,
            "oofStartDate": self.oofStartDate.isoformat()
        }

        del confidenceResults
        del validationExpertPredictions
        del tuningResults
        del sharedGateTuningResults
        gc.collect()

        return self.getRecipe()

    def calibrateConfidence(
        self,
        calibrationData,
        tuningModelDirectory=None,
        targetPrecision=None,
        showProgress=False
    ):
        """Select the highest-coverage confidence mix that meets the validation precision target."""
        if not self.fitted:
            raise RuntimeError(
                "Gatekeeper must be loaded or fitted before confidence calibration."
            )

        requiredColumns = {
            "date",
            "targetDate_F5",
            *featureColumns,
            "divZProjectionResidualRel_F5"
        }

        missingColumns = requiredColumns - set(calibrationData.columns)

        if missingColumns:
            raise ValueError(
                f"Missing confidence-calibration columns: {sorted(missingColumns)}"
            )

        if targetPrecision is not None:
            self._setTargetPrecision(
                targetPrecision
            )

        if tuningModelDirectory is None and self.modelDirectory is not None:
            tuningModelDirectory = self.modelDirectory/"tuning"

        tuningModels = self.loadTuningModels(
            tuningModelDirectory
        )

        tuningMetadata = tuningModels["metadata"]

        if int(tuningMetadata["randomSeed"]) != self.randomSeed:
            raise ValueError(
                "Tuning models use a different random seed from the loaded Gatekeeper."
            )

        trainEnd = pd.Timestamp(
            tuningMetadata["trainEnd"]
        )

        validationEnd = pd.Timestamp(
            tuningMetadata["validationEnd"]
        )

        dateValues = pd.to_datetime(
            calibrationData["date"]
        )

        targetDateValues = pd.to_datetime(
            calibrationData["targetDate_F5"]
        )

        calibrationMask = (
            (dateValues >= trainEnd) &
            (dateValues < validationEnd) &
            (targetDateValues < validationEnd)
        )

        legalCalibrationData = calibrationData.loc[
            calibrationMask
        ].copy()

        if len(legalCalibrationData) == 0:
            raise ValueError(
                "No legal rows are available for confidence calibration."
            )

        normalizedCalibrationData = normalizeFeatureData(
            legalCalibrationData,
            tuningModels["normalizationParameters"]
        ).to_numpy(
            dtype=np.float32
        )

        confidenceResults = self._selectConfidenceMix(
            tuningModels["experts"],
            tuningModels["sharedGate"],
            normalizedCalibrationData,
            legalCalibrationData,
            tuningModels["targetScaleVector"],
            showProgress=showProgress
        )

        if self.recipe is None:
            self.recipe = {}

        self.recipe.pop("confidenceCutoffs", None)

        self.recipe.update({
            "confidenceMethod": "deterministicMargin+fullPipelineMCSNR",
            "targetPrecision": self.targetPrecision,
            "confidenceLevels": list(self.confidenceLevels),
            "deterministicConfidenceLevel": self.deterministicConfidenceLevel,
            "mcConfidenceLevel": self.mcConfidenceLevel,
            "deterministicConfidenceCutoff": confidenceResults["deterministicConfidenceCutoff"],
            "mcConfidenceCutoff": confidenceResults["mcConfidenceCutoff"],
            "confidenceValidationPrecision": confidenceResults["precision"],
            "confidenceValidationTradeRate": confidenceResults["tradeRate"],
            "confidenceValidationSignals": confidenceResults["signals"],
            "confidenceCalibrationRows": int(len(legalCalibrationData)),
            "confidenceCalibrationStart": trainEnd.isoformat(),
            "confidenceCalibrationEnd": validationEnd.isoformat(),
            "recipeAvailableDate": validationEnd.isoformat()
        })

        result = {
            "targetPrecision": self.targetPrecision,
            "deterministicConfidenceLevel": self.deterministicConfidenceLevel,
            "mcConfidenceLevel": self.mcConfidenceLevel,
            "deterministicConfidenceCutoff": confidenceResults["deterministicConfidenceCutoff"],
            "mcConfidenceCutoff": confidenceResults["mcConfidenceCutoff"],
            "validationPrecision": confidenceResults["precision"],
            "validationTradeRate": confidenceResults["tradeRate"],
            "validationSignals": confidenceResults["signals"],
            "calibrationRows": int(len(legalCalibrationData))
        }

        del confidenceResults
        del tuningModels
        gc.collect()

        return result

    def fit(
        self,
        trainingData
    ):
        """Fit final experts and shared gate using a previously tuned recipe."""
        if self.recipe is None:
            raise RuntimeError(
                "Gatekeeper requires a tuned or loaded recipe before fit()."
            )

        requiredSharedGateKeys = {
            "sharedGateBestEpoch",
            "sharedGateTuningRows",
            "sharedGateLearningRates"
        }

        missingSharedGateKeys = requiredSharedGateKeys - set(self.recipe)

        if missingSharedGateKeys:
            raise RuntimeError(
                "Gatekeeper recipe is missing corrected shared-gate training metadata: "
                f"{sorted(missingSharedGateKeys)}"
            )

        self._validateDataColumns(
            trainingData,
            requireTargets=True
        )

        if len(trainingData) == 0:
            raise ValueError("trainingData must contain at least one row.")

        self._setRandomSeed()

        self.fitted = False
        self.experts = None
        self.sharedGate = None
        gc.collect()

        self.normalizationParameters = fitNormalizationParameters(trainingData)

        normalizedTrainingData = normalizeFeatureData(
            trainingData,
            self.normalizationParameters
        )

        self.targetScales = fitTargetScales(trainingData)
        self.targetScaleVector = getTargetScaleVector(self.targetScales)

        trainHorizonTargets = self._getHorizonTargets(
            trainingData,
            self.targetScales
        )

        self.horizonThresholds = fitHorizonThresholds(trainHorizonTargets)

        horizonMasks = getHorizonMasks(
            trainHorizonTargets,
            self.horizonThresholds
        )

        self.experts = trainExperts(
            normalizedTrainingData.to_numpy(),
            trainHorizonTargets,
            horizonMasks,
            self.recipe["expertBestEpochs"],
            modelPrefix="finalExpert"
        )

        self.oofExpertPredictions, self.oofGateLabels = self._buildOofTrainingData(
            trainingData,
            self.targetScales,
            self.recipe["expertBestEpochs"]
        )

        XFinalGateTrain = np.column_stack([
            normalizedTrainingData.loc[self.oofExpertPredictions.index].to_numpy(),
            self.oofExpertPredictions[expertPredictionColumns].to_numpy()
        ])

        sharedGateResults = trainFinalSharedGate(
            XFinalGateTrain,
            self.oofGateLabels,
            bestEpoch=self.recipe["sharedGateBestEpoch"],
            tuningRows=self.recipe["sharedGateTuningRows"],
            tuningLearningRates=self.recipe["sharedGateLearningRates"],
            randomSeed=self.randomSeed
        )

        self.sharedGate = sharedGateResults["model"]

        sharedGateSchedule = sharedGateResults["schedule"]

        self.recipe["sharedGateFinalEpochs"] = int(
            sharedGateSchedule["epochs"]
        )

        self.recipe["sharedGateFinalLearningRates"] = [
            float(value)
            for value in sharedGateSchedule["learningRates"]
        ]

        self.recipe["sharedGateTargetOptimizerSteps"] = int(
            sharedGateSchedule["targetOptimizerSteps"]
        )

        self.recipe["sharedGateFinalOptimizerSteps"] = int(
            sharedGateSchedule["actualOptimizerSteps"]
        )

        self.fitted = True

        return self

    def predict(
        self,
        currentData,
        showProgress=False
    ):
        """Predict Gatekeeper trade/no-trade decisions for current pair events."""
        if not self.fitted:
            raise RuntimeError(
                "Gatekeeper must be fitted before predict()."
            )

        self._validateDataColumns(
            currentData,
            requireTargets=False
        )

        self._validateConfidenceRecipe()

        if len(currentData) == 0:
            return pd.DataFrame(index=currentData.index)

        recipeAvailableDate = pd.Timestamp(
            self.recipe["recipeAvailableDate"]
        )

        if (
            pd.to_datetime(currentData["date"]) <
            recipeAvailableDate
        ).any():
            raise ValueError(
                "Gatekeeper recipe uses information that was not available for one or more prediction dates."
            )

        normalizedCurrentData = normalizeFeatureData(
            currentData,
            self.normalizationParameters
        ).to_numpy(
            dtype=np.float32
        )

        deterministicResults = self._predictDeterministicPipeline(
            self.experts,
            self.sharedGate,
            normalizedCurrentData,
            currentData,
            self.targetScaleVector
        )

        self._setRandomSeed()

        mcResults = self._predictFullPipelineMC(
            self.experts,
            self.sharedGate,
            normalizedCurrentData,
            currentData,
            self.targetScaleVector,
            showProgress=showProgress
        )

        deterministicConfidencePass = (
            deterministicResults["predictedConvergence"] &
            (
                deterministicResults["marginUncertainty"] <=
                self.recipe["deterministicConfidenceCutoff"]
            )
        )

        mcConfidencePass = (
            mcResults["predictedConvergence"] &
            (
                mcResults["snrUncertainty"] <=
                self.recipe["mcConfidenceCutoff"]
            )
        )

        predictedConvergence = (
            deterministicResults["predictedConvergence"] &
            mcResults["predictedConvergence"]
        )

        trade = (
            deterministicConfidencePass &
            mcConfidencePass
        )

        output = pd.DataFrame(
            index=currentData.index
        )

        for column in [
            "date",
            "symbolA",
            "symbolB"
        ]:
            if column in currentData.columns:
                output[column] = currentData[column]

        output["trade"] = trade
        output["predictedConvergence"] = predictedConvergence
        output["deterministicPredictedConvergence"] = deterministicResults["predictedConvergence"]
        output["mcPredictedConvergence"] = mcResults["predictedConvergence"]
        output["deterministicConfidencePass"] = deterministicConfidencePass
        output["mcConfidencePass"] = mcConfidencePass
        output["deterministicMargin"] = deterministicResults["margin"]
        output["deterministicMarginUncertainty"] = deterministicResults["marginUncertainty"]
        output["mcSnrUncertainty"] = mcResults["snrUncertainty"]
        output["mcF5PredictedChange"] = mcResults["meanChange"]
        output["mcF5StdChange"] = mcResults["stdChange"]

        for horizon in horizonNames:
            output[f"{horizon}GateClass"] = deterministicResults["gatePredictions"][horizon]
            output[f"{horizon}ResidualPrediction"] = deterministicResults["routedPredictions"][horizon]
            output[f"{horizon}PredictedChange"] = deterministicResults["predictedChanges"][horizon]
            output[f"{horizon}PredictedConvergence"] = deterministicResults["predictedChanges"][horizon] < 0

        return output

    def _predictDeterministicPipeline(
        self,
        experts,
        sharedGate,
        normalizedData,
        data,
        targetScaleVector
    ):
        expertPredictions = predictExperts(
            experts,
            normalizedData
        )

        XGate = np.column_stack([
            normalizedData,
            self._stackExpertPredictions(expertPredictions)
        ]).astype(
            np.float32,
            copy=False
        )

        gateResults = predictSharedGate(
            sharedGate,
            XGate
        )

        routedPredictions = self._routeExpertPredictions(
            expertPredictions,
            gateResults["predictions"],
            targetScaleVector
        )

        predictedChanges = self._getPredictedChanges(
            data,
            routedPredictions
        )

        f5Probabilities = gateResults["probabilities"]["F5"]

        sortedProbabilities = np.sort(
            f5Probabilities,
            axis=1
        )

        margin = (
            sortedProbabilities[:,2] -
            sortedProbabilities[:,1]
        ).astype(
            np.float32,
            copy=False
        )

        marginUncertainty = (
            1.0 -
            margin
        ).astype(
            np.float32,
            copy=False
        )

        return {
            "gateProbabilities": gateResults["probabilities"],
            "gatePredictions": gateResults["predictions"],
            "routedPredictions": routedPredictions,
            "predictedChanges": predictedChanges,
            "predictedConvergence": predictedChanges["F5"] < 0,
            "margin": margin,
            "marginUncertainty": marginUncertainty
        }

    def _predictFullPipelineMC(
        self,
        experts,
        sharedGate,
        normalizedData,
        data,
        targetScaleVector,
        showProgress=False
    ):
        nRows = len(normalizedData)

        if nRows == 0:
            raise ValueError(
                "normalizedData must contain at least one row."
            )

        meanChange = np.zeros(
            nRows,
            dtype=np.float64
        )

        m2Change = np.zeros(
            nRows,
            dtype=np.float64
        )

        f5Index = horizonNames.index("F5")
        f5TargetScale = targetScaleVector[f5Index]

        divSlope5Rel = data["divSlope5Rel"].to_numpy(
            dtype=np.float32,
            copy=False
        )

        reportEvery = max(
            1,
            self.mcRuns // 10
        )

        for runIndex in range(self.mcRuns):
            count = runIndex + 1

            for batchStart in range(
                0,
                nRows,
                mcInferenceBatchSize
            ):
                batchEnd = min(
                    batchStart + mcInferenceBatchSize,
                    nRows
                )

                XBatch = normalizedData[
                    batchStart:
                    batchEnd
                ]

                stochasticExpertPredictions = []
                stochasticF5Predictions = []

                for horizon in horizonNames:
                    for regime in regimeNames:
                        prediction = (
                            experts[horizon][regime](
                                XBatch,
                                training=True
                            )
                            .numpy()
                            .reshape(-1)
                        )

                        stochasticExpertPredictions.append(
                            prediction
                        )

                        if horizon == "F5":
                            stochasticF5Predictions.append(
                                prediction
                            )

                expertMatrix = np.column_stack(
                    stochasticExpertPredictions
                ).astype(
                    np.float32,
                    copy=False
                )

                gateInput = np.column_stack([
                    XBatch,
                    expertMatrix
                ]).astype(
                    np.float32,
                    copy=False
                )

                probabilities = sharedGate(
                    gateInput,
                    training=True
                )[f5Index].numpy()

                gateClasses = np.argmax(
                    probabilities,
                    axis=1
                )

                f5ExpertMatrix = np.column_stack(
                    stochasticF5Predictions
                )

                batchLength = batchEnd - batchStart

                routedResidual = (
                    f5ExpertMatrix[
                        np.arange(batchLength),
                        gateClasses
                    ] *
                    f5TargetScale
                )

                predictedChange = (
                    5 *
                    divSlope5Rel[
                        batchStart:
                        batchEnd
                    ] +
                    routedResidual
                )

                currentMean = meanChange[
                    batchStart:
                    batchEnd
                ]

                currentM2 = m2Change[
                    batchStart:
                    batchEnd
                ]

                delta = (
                    predictedChange -
                    currentMean
                )

                currentMean += (
                    delta /
                    count
                )

                delta2 = (
                    predictedChange -
                    currentMean
                )

                currentM2 += (
                    delta *
                    delta2
                )

            if (
                showProgress and
                (
                    count % reportEvery == 0 or
                    count == self.mcRuns
                )
            ):
                print(
                    f"Full-pipeline MC: "
                    f"{count}/{self.mcRuns} "
                    f"({100 * count / self.mcRuns:.0f}%)",
                    flush=True
                )

        meanChange = meanChange.astype(
            np.float32
        )

        stdChange = np.sqrt(
            m2Change /
            self.mcRuns
        ).astype(
            np.float32
        )

        snrUncertainty = (
            stdChange /
            (
                np.abs(meanChange) +
                1e-8
            )
        ).astype(
            np.float32
        )

        return {
            "meanChange": meanChange,
            "stdChange": stdChange,
            "snrUncertainty": snrUncertainty,
            "predictedConvergence": meanChange < 0
        }

    def _selectConfidenceMix(
        self,
        experts,
        sharedGate,
        normalizedData,
        data,
        targetScaleVector,
        showProgress=False
    ):
        deterministicResults = self._predictDeterministicPipeline(
            experts,
            sharedGate,
            normalizedData,
            data,
            targetScaleVector
        )

        self._setRandomSeed()

        mcResults = self._predictFullPipelineMC(
            experts,
            sharedGate,
            normalizedData,
            data,
            targetScaleVector,
            showProgress=showProgress
        )

        actualChange = (
            horizonSteps["F5"] *
            data["divSlope5Rel"].to_numpy(
                dtype=np.float32,
                copy=False
            ) +
            data["divZProjectionResidualRel_F5"].to_numpy(
                dtype=np.float32,
                copy=False
            )
        )

        actualConvergence = actualChange < 0

        deterministicCutoffs = self._fitConfidenceCutoffs(
            deterministicResults["predictedConvergence"],
            deterministicResults["marginUncertainty"]
        )

        mcCutoffs = self._fitConfidenceCutoffs(
            mcResults["predictedConvergence"],
            mcResults["snrUncertainty"]
        )

        deterministicRequired = self._getRequiredConfidenceIndices(
            deterministicResults["predictedConvergence"],
            deterministicResults["marginUncertainty"],
            deterministicCutoffs
        )

        mcRequired = self._getRequiredConfidenceIndices(
            mcResults["predictedConvergence"],
            mcResults["snrUncertainty"],
            mcCutoffs
        )

        eligibleMask = (
            deterministicResults["predictedConvergence"] &
            mcResults["predictedConvergence"]
        )

        if not eligibleMask.any():
            raise ValueError(
                "No validation events are predicted to converge by both confidence pipelines."
            )

        countGrid, correctGrid = self._buildConfidenceGrid(
            deterministicRequired,
            mcRequired,
            actualConvergence,
            eligibleMask
        )

        precisionGrid = np.divide(
            correctGrid,
            countGrid,
            out=np.full(
                countGrid.shape,
                np.nan,
                dtype=float
            ),
            where=countGrid > 0
        )

        valid = (
            (countGrid > 0) &
            (precisionGrid >= self.targetPrecision)
        )

        if not valid.any():
            maximumPrecision = float(
                np.nanmax(precisionGrid)
            )

            raise ValueError(
                f"No confidence-level combination achieved targetPrecision={self.targetPrecision:.4f}. "
                f"Maximum validation precision was {maximumPrecision:.4f}."
            )

        candidates = np.argwhere(
            valid
        )

        candidateCounts = countGrid[
            valid
        ]

        maximumSignals = candidateCounts.max()

        bestCandidates = candidates[
            candidateCounts == maximumSignals
        ]

        if len(bestCandidates) > 1:
            candidatePrecisions = np.array([
                precisionGrid[row, column]
                for row, column in bestCandidates
            ])

            bestPosition = np.argmax(
                candidatePrecisions
            )

            deterministicIndex, mcIndex = bestCandidates[
                bestPosition
            ]
        else:
            deterministicIndex, mcIndex = bestCandidates[0]

        self.deterministicConfidenceLevel = float(
            self.confidenceLevels[deterministicIndex]
        )

        self.mcConfidenceLevel = float(
            self.confidenceLevels[mcIndex]
        )

        result = {
            "deterministicConfidenceLevel": self.deterministicConfidenceLevel,
            "mcConfidenceLevel": self.mcConfidenceLevel,
            "deterministicConfidenceCutoff": float(
                deterministicCutoffs[deterministicIndex]
            ),
            "mcConfidenceCutoff": float(
                mcCutoffs[mcIndex]
            ),
            "signals": int(
                countGrid[
                    deterministicIndex,
                    mcIndex
                ]
            ),
            "tradeRate": float(
                countGrid[
                    deterministicIndex,
                    mcIndex
                ] /
                len(data)
            ),
            "precision": float(
                precisionGrid[
                    deterministicIndex,
                    mcIndex
                ]
            )
        }

        del deterministicResults
        del mcResults

        return result

    def _fitConfidenceCutoffs(
        self,
        predictedConvergence,
        uncertainty
    ):
        convergenceUncertainty = np.sort(
            uncertainty[
                predictedConvergence
            ]
        )

        if len(convergenceUncertainty) == 0:
            raise ValueError(
                "No predicted F5 convergence events are available to fit confidence cutoffs."
            )

        cutoffs = []

        for fraction in self.confidenceLevels:
            if fraction == 1.00:
                cutoffs.append(
                    np.inf
                )
                continue

            nSelected = int(
                np.ceil(
                    len(convergenceUncertainty) *
                    fraction
                )
            )

            cutoffs.append(
                float(
                    convergenceUncertainty[
                        nSelected - 1
                    ]
                )
            )

        return np.asarray(
            cutoffs,
            dtype=float
        )

    def _getRequiredConfidenceIndices(
        self,
        predictedConvergence,
        uncertainty,
        cutoffs
    ):
        requiredIndices = np.full(
            len(uncertainty),
            -1,
            dtype=np.int16
        )

        indices = np.where(
            predictedConvergence
        )[0]

        requiredIndices[
            indices
        ] = np.searchsorted(
            cutoffs,
            uncertainty[
                indices
            ],
            side="left"
        ).astype(
            np.int16
        )

        return requiredIndices

    def _buildConfidenceGrid(
        self,
        deterministicRequired,
        mcRequired,
        actualConvergence,
        eligibleMask
    ):
        nLevels = len(
            self.confidenceLevels
        )

        countGrid = np.zeros(
            (
                nLevels,
                nLevels
            ),
            dtype=np.int64
        )

        correctGrid = np.zeros(
            (
                nLevels,
                nLevels
            ),
            dtype=np.int64
        )

        deterministicIndices = deterministicRequired[
            eligibleMask
        ]

        mcIndices = mcRequired[
            eligibleMask
        ]

        correct = actualConvergence[
            eligibleMask
        ].astype(
            np.int64
        )

        np.add.at(
            countGrid,
            (
                deterministicIndices,
                mcIndices
            ),
            1
        )

        np.add.at(
            correctGrid,
            (
                deterministicIndices,
                mcIndices
            ),
            correct
        )

        countGrid = countGrid.cumsum(
            axis=0
        ).cumsum(
            axis=1
        )

        correctGrid = correctGrid.cumsum(
            axis=0
        ).cumsum(
            axis=1
        )

        return countGrid, correctGrid

    def _setTargetPrecision(
        self,
        targetPrecision
    ):
        targetPrecision = float(
            targetPrecision
        )

        if targetPrecision <= 0 or targetPrecision > 1:
            raise ValueError(
                "targetPrecision must be greater than 0 and less than or equal to 1."
            )

        self.targetPrecision = targetPrecision

    def _validateConfidenceLevels(
        self,
        confidenceLevels
    ):
        confidenceLevels = sorted(
            set(
                float(level)
                for level in confidenceLevels
            )
        )

        if any(
            level <= 0 or level > 1
            for level in confidenceLevels
        ):
            raise ValueError(
                "confidenceLevels must be greater than 0 and less than or equal to 1."
            )

        if 1.00 not in confidenceLevels:
            confidenceLevels.append(1.00)
            confidenceLevels.sort()

        return confidenceLevels

    def _validateConfidenceRecipe(self):
        if self.recipe is None:
            raise RuntimeError(
                "Gatekeeper confidence recipe is not available."
            )

        requiredKeys = {
            "targetPrecision",
            "deterministicConfidenceLevel",
            "mcConfidenceLevel",
            "deterministicConfidenceCutoff",
            "mcConfidenceCutoff",
            "recipeAvailableDate"
        }

        missingKeys = requiredKeys - set(self.recipe)

        if missingKeys:
            raise RuntimeError(
                "Gatekeeper confidence cutoffs have not been calibrated. "
                "Run calibrateConfidence() using the saved tuning models before prediction."
            )

    def _buildOofTrainingData(
        self,
        data,
        globalTargetScales,
        expertBestEpochs
    ):
        """Build chronological OOF expert predictions and gate labels."""
        globalTargetScaleVector = getTargetScaleVector(globalTargetScales)
        dateValues = pd.to_datetime(data["date"])

        predictionFrames = []
        labelFrames = []

        for (
            trainEnd,
            predictionStart,
            predictionEnd
        ) in self._getOofPeriods(data):
            periodTrainData = self._getLegalPeriodTrainingData(
                data,
                trainEnd
            )

            periodPredictionMask = (
                (dateValues >= predictionStart) &
                (dateValues < predictionEnd)
            )

            periodPredictionData = data.loc[
                periodPredictionMask
            ].copy()

            if (
                len(periodTrainData) == 0 or
                len(periodPredictionData) == 0
            ):
                continue

            periodNormalizationParameters = fitNormalizationParameters(
                periodTrainData
            )

            periodXTrain = normalizeFeatureData(
                periodTrainData,
                periodNormalizationParameters
            ).to_numpy()

            periodXPrediction = normalizeFeatureData(
                periodPredictionData,
                periodNormalizationParameters
            ).to_numpy()

            periodTargetScales = fitTargetScales(
                periodTrainData
            )

            periodTargetScaleVector = getTargetScaleVector(
                periodTargetScales
            )

            periodTrainTargets = self._getHorizonTargets(
                periodTrainData,
                periodTargetScales
            )

            periodPredictionTargets = self._getHorizonTargets(
                periodPredictionData,
                periodTargetScales
            )

            periodThresholds = fitHorizonThresholds(
                periodTrainTargets
            )

            periodMasks = getHorizonMasks(
                periodTrainTargets,
                periodThresholds
            )

            periodExperts = trainExperts(
                periodXTrain,
                periodTrainTargets,
                periodMasks,
                expertBestEpochs,
                modelPrefix=f"oof{predictionStart.year}"
            )

            periodPredictions = predictExperts(
                periodExperts,
                periodXPrediction
            )

            predictionFrame = pd.DataFrame(
                index=periodPredictionData.index
            )

            for horizonIndex, horizon in enumerate(horizonNames):
                for regime in regimeNames:
                    predictionsRaw = (
                        periodPredictions[horizon][regime] *
                        periodTargetScaleVector[horizonIndex]
                    )

                    predictionFrame[f"{horizon}{regime}"] = (
                        predictionsRaw /
                        globalTargetScaleVector[horizonIndex]
                    )

            periodLabels = buildGateLabels(
                periodPredictionTargets,
                periodThresholds
            )

            labelFrame = pd.DataFrame({
                horizon: periodLabels[horizon]
                for horizon in horizonNames
            }, index=periodPredictionData.index)

            predictionFrames.append(
                predictionFrame
            )

            labelFrames.append(
                labelFrame
            )

            del periodExperts
            del periodPredictions
            gc.collect()

        if not predictionFrames:
            raise ValueError(
                "No valid OOF periods could be constructed."
            )

        oofExpertPredictions = pd.concat(
            predictionFrames,
            axis=0
        )

        gateLabelFrame = pd.concat(
            labelFrames,
            axis=0
        )

        if oofExpertPredictions.isna().any().any():
            raise ValueError(
                "OOF expert predictions contain missing values."
            )

        if (
            oofExpertPredictions.index.has_duplicates or
            gateLabelFrame.index.has_duplicates
        ):
            raise ValueError(
                "OOF training data contains duplicate indices."
            )

        if not oofExpertPredictions.index.equals(gateLabelFrame.index):
            raise ValueError(
                "OOF expert predictions and gate labels are misaligned."
            )

        gateLabels = {
            horizon: gateLabelFrame[horizon].astype(int).to_numpy()
            for horizon in horizonNames
        }

        return (
            oofExpertPredictions[expertPredictionColumns],
            gateLabels
        )

    def _getOofPeriods(
        self,
        data
    ):
        """Return yearly OOF periods represented by the supplied data."""
        maximumDate = pd.to_datetime(data["date"]).max()

        periods = []

        for year in range(
            self.oofStartDate.year,
            maximumDate.year + 1
        ):
            predictionStart = pd.Timestamp(
                year=year,
                month=1,
                day=1
            )

            predictionEnd = pd.Timestamp(
                year=year + 1,
                month=1,
                day=1
            )

            if predictionStart < self.oofStartDate:
                continue

            periods.append((
                predictionStart,
                predictionStart,
                predictionEnd
            ))

        return periods

    def _getLegalPeriodTrainingData(
        self,
        data,
        trainEnd
    ):
        """Return OOF rows whose event and F5 target were available before trainEnd."""
        dateValues = pd.to_datetime(data["date"])
        targetDateValues = pd.to_datetime(data["targetDate_F5"])

        mask = (
            (dateValues < trainEnd) &
            (targetDateValues < trainEnd)
        )

        return data.loc[mask].copy()

    def _getHorizonTargets(
        self,
        data,
        targetScales
    ):
        normalizedTargets = normalizeTargetData(
            data,
            targetScales
        ).to_numpy()

        return {
            horizon: normalizedTargets[:,horizonIndex]
            for horizonIndex, horizon in enumerate(horizonNames)
        }

    def _stackExpertPredictions(
        self,
        expertPredictions
    ):
        columns = [
            expertPredictions[horizon][regime]
            for horizon in horizonNames
            for regime in regimeNames
        ]

        return np.column_stack(columns)

    def _routeExpertPredictions(
        self,
        expertPredictions,
        gatePredictions,
        targetScaleVector
    ):
        routedPredictions = {}

        for horizonIndex, horizon in enumerate(horizonNames):
            expertMatrix = np.column_stack([
                expertPredictions[horizon]["Lower"],
                expertPredictions[horizon]["Middle"],
                expertPredictions[horizon]["Upper"]
            ])

            routedNormalized = expertMatrix[
                np.arange(len(expertMatrix)),
                gatePredictions[horizon]
            ]

            routedPredictions[horizon] = (
                routedNormalized *
                targetScaleVector[horizonIndex]
            )

        return routedPredictions

    def _getPredictedChanges(
        self,
        data,
        routedPredictions
    ):
        divSlope5Rel = data["divSlope5Rel"].to_numpy()

        return {
            horizon: (
                horizonSteps[horizon] *
                divSlope5Rel +
                routedPredictions[horizon]
            )
            for horizon in horizonNames
        }

    def _setRandomSeed(self):
        np.random.seed(
            self.randomSeed
        )

        tf.random.set_seed(
            self.randomSeed
        )

    def _validateDataColumns(
        self,
        data,
        requireTargets
    ):
        requiredColumns = {
            "date",
            *featureColumns
        }

        if requireTargets:
            requiredColumns.update({
                "targetDate_F5",
                "divZProjectionResidualRel_F1",
                "divZProjectionResidualRel_F3",
                "divZProjectionResidualRel_F5"
            })

        missingColumns = requiredColumns - set(data.columns)

        if missingColumns:
            raise ValueError(
                f"Missing Gatekeeper columns: {sorted(missingColumns)}"
            )