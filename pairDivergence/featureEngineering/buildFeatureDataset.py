"""
Build and save the pair-divergence trigger-event feature dataset.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from featureEngineering import (
    buildPairEvents,
    finalEventColumns,
    futureColumns,
    modelFeatureColumns,
    prepMarketFeatures,
    targetDateColumns
)


repositoryRoot = Path(__file__).resolve().parents[2]

marketDataPath = repositoryRoot/"data"/"pairDivergence"/"pairDivergenceData.parquet"
pairDataPath = repositoryRoot/"data"/"pairDivergence"/"pairDivergencePairs.parquet"
outputPath = repositoryRoot/"data"/"pairDivergence"/"pairDivergenceFeatures.parquet"

maxWorkers = 6
batchSize = 1000

workerReturns = None
workerVolumeRatios = None


def initializeFeatureWorker(returns, volumeRatios):
    """
    Store the market feature matrices in each worker process.

    Inputs:
        - returns (dataFrame): Date-indexed log returns
        - volumeRatios (dataFrame): Date-indexed relative volume

    Returns:
        - None
    """
    global workerReturns
    global workerVolumeRatios

    workerReturns = returns
    workerVolumeRatios = volumeRatios


def processPairBatch(pairBatch):
    """
    Construct trigger events for one batch of stock pairs.

    Inputs:
        - pairBatch (array): Stock pairs processed by one worker

    Returns:
        - events (dataFrame): Trigger events produced by the batch
    """
    eventFrames = []

    for symbolA, symbolB in pairBatch:
        events = buildPairEvents(
            workerReturns,
            workerVolumeRatios,
            symbolA,
            symbolB
        )

        if not events.empty:
            eventFrames.append(events)

    if not eventFrames:
        return pd.DataFrame(columns=finalEventColumns)

    return pd.concat(eventFrames, ignore_index=True)


def buildPairBatches(pairUniverse):
    """
    Divide the saved pair universe into processing batches.

    Inputs:
        - pairUniverse (dataFrame): Eligible stock-pair universe

    Returns:
        - pairBatches (list): Pair batches for parallel processing
    """
    pairValues = pairUniverse[["symbolA", "symbolB"]].to_numpy()

    pairBatches = []

    for startIndex in range(0, len(pairValues), batchSize):
        pairBatches.append(pairValues[startIndex:startIndex + batchSize])

    return pairBatches


def validatePairUniverse(pairUniverse):
    """
    Validate the saved candidate pair universe.

    Inputs:
        - pairUniverse (dataFrame): Eligible stock-pair universe

    Returns:
        - None
    """
    duplicatePairs = pairUniverse.duplicated(["symbolA", "symbolB"]).sum()

    sameSymbolPairs = (pairUniverse["symbolA"] == pairUniverse["symbolB"]).sum()

    incorrectlyOrderedPairs = (pairUniverse["symbolA"] >= pairUniverse["symbolB"]).sum()

    if duplicatePairs != 0:
        raise ValueError(f"Duplicate pairs found: {duplicatePairs}")

    if sameSymbolPairs != 0:
        raise ValueError(f"Same-symbol pairs found: {sameSymbolPairs}")

    if incorrectlyOrderedPairs != 0:
        raise ValueError(f"Incorrectly ordered pairs found: {incorrectlyOrderedPairs}")


def validateFeatureData(featureData):
    """
    Validate the completed trigger-event feature dataset.

    Inputs:
        - featureData (dataFrame): Completed trigger-event dataset

    Returns:
        - None
    """
    duplicateEvents = featureData.duplicated(["date", "symbolA", "symbolB"]).sum()

    missingValues = featureData.isna().sum().sum()

    numericData = featureData.select_dtypes(include=np.number)
    infiniteValues = (~np.isfinite(numericData.to_numpy())).sum()

    leakedColumns = sorted(set(modelFeatureColumns).intersection(futureColumns))

    invalidF1Dates = (featureData["targetDate_F1"] <= featureData["date"]).sum()
    invalidF3Dates = (featureData["targetDate_F3"] <= featureData["targetDate_F1"]).sum()
    invalidF5Dates = (featureData["targetDate_F5"] <= featureData["targetDate_F3"]).sum()

    if duplicateEvents != 0:
        raise ValueError(f"Duplicate trigger events found: {duplicateEvents}")

    if missingValues != 0:
        raise ValueError(f"Missing values found: {missingValues}")

    if infiniteValues != 0:
        raise ValueError(f"Infinite values found: {infiniteValues}")

    if leakedColumns:
        raise ValueError(f"Future columns found in model features: {leakedColumns}")

    if invalidF1Dates != 0:
        raise ValueError(f"Invalid F1 target dates found: {invalidF1Dates}")

    if invalidF3Dates != 0:
        raise ValueError(f"Invalid F3 target dates found: {invalidF3Dates}")

    if invalidF5Dates != 0:
        raise ValueError(f"Invalid F5 target dates found: {invalidF5Dates}")

    print(f"Duplicate events: {duplicateEvents:,}")
    print(f"Missing values: {missingValues:,}")
    print(f"Infinite values: {infiniteValues:,}")
    print(f"Feature/future overlap: {leakedColumns}")
    print(f"Invalid target dates: {invalidF1Dates + invalidF3Dates + invalidF5Dates:,}")


def main():
    print("Loading cleaned market data")
    marketData = pd.read_parquet(marketDataPath)

    print("Loading eligible pair universe")
    pairUniverse = pd.read_parquet(pairDataPath)

    # Remove Comment for test run
    # pairUniverse = pairUniverse.head(10000).copy()

    validatePairUniverse(pairUniverse)

    print("Preparing market features")
    returns, volumeRatios = prepMarketFeatures(marketData)

    pairSymbols = pd.Index(pd.unique(pairUniverse[[
        "symbolA",
        "symbolB"
    ]].values.ravel())).sort_values()

    missingSymbols = pairSymbols.difference(returns.columns)

    if len(missingSymbols) != 0:
        raise ValueError(f"Pair symbols missing from market data: {missingSymbols.tolist()}")

    returns = returns.loc[:, pairSymbols]
    volumeRatios = volumeRatios.loc[:, pairSymbols]

    pairBatches = buildPairBatches(pairUniverse)

    eventFrames = []
    eventCount = 0

    print()
    print(f"Market rows: {len(marketData):,}")
    print(f"Pairs: {len(pairUniverse):,}")
    print(f"Pair symbols: {len(pairSymbols):,}")
    print(f"Batches: {len(pairBatches):,}")
    print(f"Workers: {maxWorkers}")
    print()

    with ProcessPoolExecutor(
        max_workers=maxWorkers,
        initializer=initializeFeatureWorker,
        initargs=(returns, volumeRatios)
    ) as executor:
        batchResults = executor.map(processPairBatch, pairBatches)

        for batchNumber, events in enumerate(batchResults, start=1):
            if not events.empty:
                eventFrames.append(events)
                eventCount += len(events)

            if batchNumber % 10 == 0 or batchNumber == len(pairBatches):
                processedPairs = min(batchNumber * batchSize, len(pairUniverse))

                print(f"Processed {processedPairs:,} / {len(pairUniverse):,} pairs")
                print(f"Trigger events: {eventCount:,}")
                print()

    if not eventFrames:
        raise ValueError("No complete trigger events were produced")

    featureData = pd.concat(eventFrames, ignore_index=True)

    featureData = featureData.sort_values([
        "date",
        "symbolA",
        "symbolB"
    ]).reset_index(drop=True)

    print("Validating feature dataset")
    validateFeatureData(featureData)

    outputPath.parent.mkdir(parents=True, exist_ok=True)

    featureData.to_parquet(outputPath, index=False)

    print()
    print(f"Trigger events: {len(featureData):,}")
    print(f"Pairs with triggers: {featureData[['symbolA', 'symbolB']].drop_duplicates().shape[0]:,}")
    print(f"First date: {featureData['date'].min().date()}")
    print(f"Last date: {featureData['date'].max().date()}")
    print(f"Model features: {len(modelFeatureColumns)}")
    print(f"Future targets: {len(futureColumns)}")
    print(f"Target dates: {len(targetDateColumns)}")
    print(f"Output columns: {len(featureData.columns)}")
    print(f"Output: {outputPath}")


if __name__ == "__main__":
    main()