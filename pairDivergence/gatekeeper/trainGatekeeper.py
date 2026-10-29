from pathlib import Path
from time import perf_counter

import pandas as pd

from pairDivergence.gatekeeper.gatekeeper import Gatekeeper
from pairDivergence.gatekeeper.normalization import featureColumns, targetColumns


repositoryRoot = Path(__file__).resolve().parents[2]
gatekeeperDirectory = Path(__file__).resolve().parent

featurePath = repositoryRoot/"data"/"pairDivergence"/"pairDivergenceFeatures.parquet"
modelDirectory = gatekeeperDirectory/"models"
tuningModelDirectory = modelDirectory/"tuning"

tuneTrainEnd = pd.Timestamp("2024-01-01")
tuneValidationEnd = pd.Timestamp("2025-01-01")
finalTrainEnd = pd.Timestamp("2025-01-01")

targetPrecision = 0.93


print("Loading Gatekeeper training data...")

loadStart = perf_counter()

data = pd.read_parquet(
    featurePath,
    columns=[
        "date",
        "targetDate_F5",
        *featureColumns,
        *targetColumns
    ],
    filters=[
        (
            "date",
            "<",
            finalTrainEnd
        ),
        (
            "targetDate_F5",
            "<",
            finalTrainEnd
        )
    ]
)

data["date"] = pd.to_datetime(data["date"])

data["targetDate_F5"] = pd.to_datetime(data["targetDate_F5"])

print(f"Rows loaded: {len(data):,}")
print(f"Load time: {perf_counter() - loadStart:.2f} seconds")

print()
print("Tuning Gatekeeper...")

gatekeeper = Gatekeeper(targetPrecision=targetPrecision)

tuneStart = perf_counter()

recipe = gatekeeper.tune(
    data,
    trainEnd=tuneTrainEnd,
    validationEnd=tuneValidationEnd,
    targetPrecision=targetPrecision
)

print()
print(f"Tuning time: {(perf_counter() - tuneStart) / 3600:.2f} hours")

print(f"Selected deterministic confidence level: {recipe['deterministicConfidenceLevel']:.1%}")

print(f"Selected MC confidence level: {recipe['mcConfidenceLevel']:.1%}")

print(f"Validation precision: {recipe['confidenceValidationPrecision']:.4%}")

print(f"Validation trade rate: {recipe['confidenceValidationTradeRate']:.4%}")

print()
print("Saving exact tuning models...")

gatekeeper.saveTuningModels(tuningModelDirectory)

gatekeeper.clearTuningModels()

print()
print("Training final Gatekeeper...")

finalTrainingData = data.loc[
    (data["date"] < finalTrainEnd) & (data["targetDate_F5"] < finalTrainEnd)
].copy()

fitStart = perf_counter()

gatekeeper.fit(finalTrainingData)

print(f"Final training time: {(perf_counter() - fitStart) / 3600:.2f} hours")

print()
print("Saving final Gatekeeper...")

gatekeeper.save(modelDirectory)

print()
print(f"Models saved to: {modelDirectory}")