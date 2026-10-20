"""
Functions for constructing pair-divergence features, trigger events, and prediction targets.
"""

import numpy as np
import pandas as pd


shortCorrelationWindow = 42
mediumCorrelationWindow = 126
longCorrelationWindow = 252

shortCorrelationThreshold = 0.75
mediumCorrelationThreshold = 0.50

regressionWindow = 126
divergenceWindow = 5
divergenceBaselineWindow = 63
correlationVariabilityWindow = 211
volumeWindow = 63
slopeWindow = 5

futureHorizons = [1, 3, 5]


identifierColumns = [
    "date",
    "symbolA",
    "symbolB"
]

baseModelFeatureColumns = [
    "A_Return1",
    "B_Return1",
    "corr42",
    "corr126",
    "corr252",
    "hedgeRatio",
    "div",
    "divMean",
    "divStd",
    "divZ",
    "corrVar",
    "A_Vol21",
    "A_Vol252",
    "B_Vol21",
    "B_Vol252",
    "divMean_PQ",
    "divStd_PQ",
    "divMeanDiff",
    "divStdDiff",
    "A_VolumeRatio63",
    "B_VolumeRatio63"
]

modelFeatureColumns = baseModelFeatureColumns + [
    "divSlope5Rel"
]

slopeColumns = [
    "divSlope5",
    "divSlope5Rel"
]

targetDateColumns = [
    "targetDate_F1",
    "targetDate_F3",
    "targetDate_F5"
]

futureColumns = [
    "divZProjectionResidualRel_F1",
    "divZProjectionResidualRel_F3",
    "divZProjectionResidualRel_F5",
    "A_Return_F1",
    "A_Return_F3",
    "A_Return_F5",
    "B_Return_F1",
    "B_Return_F3",
    "B_Return_F5",
    "A_Direction_F1",
    "A_Direction_F3",
    "A_Direction_F5",
    "B_Direction_F1",
    "B_Direction_F3",
    "B_Direction_F5"
]

baseEventColumns = identifierColumns + baseModelFeatureColumns

finalEventColumns = (
    identifierColumns +
    baseModelFeatureColumns +
    slopeColumns +
    targetDateColumns +
    futureColumns
)


def rollingVolatility(returns, window, periodsPerYear=252):
    """
    Calculate annualized rolling volatility.

    Inputs:
        - returns (series): Ordered return observations
        - window (int): Number of observations in each rolling window
        - periodsPerYear (int): Number of return observations per year

    Returns:
        - volatility (series): Annualized rolling volatility
    """
    return returns.rolling(window).std(ddof=1) * np.sqrt(periodsPerYear)


def prepMarketFeatures(data):
    """
    Calculate one-observation log returns and relative volume.

    Returns are calculated only when the previous observation for a stock occurred on the immediately preceding market
    trading date. Relative volume compares current volume with the preceding 63 observations for that stock.

    Inputs:
        - data (dataFrame): Cleaned market data

    Returns:
        - returns (dataFrame): Date-indexed one-observation log returns
        - volumeRatios (dataFrame): Date-indexed relative-volume values
    """
    marketData = data[
        [
            "date",
            "symbol",
            "adjClose",
            "volume"
        ]
    ].copy()

    marketData = marketData.sort_values(["symbol", "date"]).reset_index(drop=True)

    tradingDates = pd.Index(sorted(marketData["date"].unique()))

    tradingDayPosition = pd.Series(
        np.arange(len(tradingDates)),
        index=tradingDates
    )

    marketData["tradingDay"] = marketData["date"].map(tradingDayPosition)

    marketData["previousTradingDay"] = (
        marketData
        .groupby("symbol")["tradingDay"]
        .shift(1)
    )

    marketData["previousAdjClose"] = (
        marketData
        .groupby("symbol")["adjClose"]
        .shift(1)
    )

    consecutiveObservation = marketData["tradingDay"] - marketData["previousTradingDay"] == 1

    marketData["return1"] = np.where(
        consecutiveObservation,
        np.log(marketData["adjClose"] / marketData["previousAdjClose"]),
        np.nan
    )

    def calculatePreviousVolumeMean(volume):
        return volume.rolling(volumeWindow).mean().shift(1)

    previousVolumeMean = (
        marketData
        .groupby("symbol")["volume"]
        .transform(calculatePreviousVolumeMean)
    )

    marketData["volumeRatio63"] = marketData["volume"] / previousVolumeMean.replace(0, np.nan)

    returns = marketData.pivot(
        index="date",
        columns="symbol",
        values="return1"
    ).sort_index()

    volumeRatios = marketData.pivot(
        index="date",
        columns="symbol",
        values="volumeRatio63"
    ).sort_index()

    returns.columns.name = None
    volumeRatios.columns.name = None

    return returns, volumeRatios


def prepPairData(returns, volumeRatios, firstSymbol, secondSymbol):
    """
    Prepare aligned return and relative-volume data for one stock pair.

    Stock A and Stock B are assigned alphabetically. Dates without a valid return for either stock are excluded before
    pair calculations begin.

    Inputs:
        - returns (dataFrame): Date-indexed log returns
        - volumeRatios (dataFrame): Date-indexed relative volume
        - firstSymbol (str): First requested symbol
        - secondSymbol (str): Second requested symbol

    Returns:
        - pairData (dataFrame): Aligned pair observations
    """
    symbolA, symbolB = sorted([firstSymbol, secondSymbol])

    pairData = pd.DataFrame({
        "A_Return1": returns[symbolA],
        "B_Return1": returns[symbolB],
        "A_VolumeRatio63": volumeRatios[symbolA],
        "B_VolumeRatio63": volumeRatios[symbolB]
    })

    pairData = pairData.dropna(subset=["A_Return1", "B_Return1"])

    pairData.insert(0, "symbolB", symbolB)
    pairData.insert(0, "symbolA", symbolA)

    return pairData


def calcPairFeatures(pairData):
    """
    Calculate rolling relationship, divergence, and trigger features.

    The rolling regression uses the previous 126 aligned observations and excludes the current observation. Divergence
    is the sum of the latest five daily residuals.

    Inputs:
        - pairData (dataFrame): Aligned pair observations

    Returns:
        - pairData (dataFrame): Pair data with calculated features
    """
    stockA = pairData["A_Return1"]
    stockB = pairData["B_Return1"]

    pairData["corr42"] = stockA.rolling(shortCorrelationWindow).corr(stockB)

    pairData["corr126"] = stockA.rolling(mediumCorrelationWindow).corr(stockB)

    pairData["corr252"] = stockA.rolling(longCorrelationWindow).corr(stockB)

    rollingCovariance = stockA.rolling(regressionWindow, closed="left").cov(stockB)

    rollingVariance = stockB.rolling(regressionWindow, closed="left").var()

    pairData["hedgeRatio"] = rollingCovariance / rollingVariance.replace(0, np.nan)

    meanA = stockA.rolling(
        regressionWindow,
        closed="left"
    ).mean()

    meanB = stockB.rolling(
        regressionWindow,
        closed="left"
    ).mean()

    pairData["rollingIntercept"] = meanA - pairData["hedgeRatio"] * meanB

    pairData["dailyResidual"] = stockA - pairData["rollingIntercept"] - pairData["hedgeRatio"] * stockB

    pairData["A_Return5"] = stockA.rolling(divergenceWindow).sum()

    pairData["B_Return5"] = stockB.rolling(divergenceWindow).sum()

    pairData["div"] = pairData["dailyResidual"].rolling(divergenceWindow).sum()

    pairData["divMean"] = pairData["div"].rolling(divergenceBaselineWindow, closed="left").mean()

    pairData["divStd"] = pairData["div"].rolling(divergenceBaselineWindow, closed="left").std()

    pairData["divZ"] = (pairData["div"] - pairData["divMean"]) / pairData["divStd"].replace(0, np.nan)

    pairData["corrVar"] = pairData["corr42"].rolling(correlationVariabilityWindow, closed="left").std()

    pairData["A_Vol21"] = rollingVolatility(stockA, window=21)

    pairData["A_Vol252"] = rollingVolatility(stockA, window=252)

    pairData["B_Vol21"] = rollingVolatility(stockB, window=21)

    pairData["B_Vol252"] = rollingVolatility(stockB, window=252)

    pairData["divMean_PQ"] = pairData["divMean"].shift(divergenceBaselineWindow)

    pairData["divStd_PQ"] = pairData["divStd"].shift(divergenceBaselineWindow)

    pairData["divMeanDiff"] = pairData["divMean"] - pairData["divMean_PQ"]

    pairData["divStdDiff"] = pairData["divStd"] - pairData["divStd_PQ"]

    pairData["eligible"] = (
        (pairData["corr42"].abs() >= shortCorrelationThreshold) &
        (pairData["corr126"].abs() >= mediumCorrelationThreshold) &
        (np.sign(pairData["corr42"]) == np.sign(pairData["corr126"])) &
        pairData["corr252"].notna()
    )

    pairData["trigger"] = (
        pairData["eligible"] &
        (pairData["divZ"].abs() > 1) &
        (pairData["divZ"].shift(1).abs() <= 1)
    )

    return pairData


def extractTriggerEvents(pairData):
    """
    Extract complete trigger-event observations from one stock pair.

    Inputs:
        - pairData (dataFrame): Pair data containing calculated features and trigger indicators

    Returns:
        - events (dataFrame): Complete trigger-event observations
    """
    events = pairData.loc[pairData["trigger"]].copy()

    events = events.dropna(subset=baseModelFeatureColumns)

    events = events.reset_index()
    events = events[baseEventColumns]

    return events


def calculateSlope(values):
    """
    Calculate the slope of a straight line fitted through equally spaced values.

    Inputs:
        - values (array): Ordered one-dimensional observations

    Returns:
        - slope (float): Linear slope per observation
    """
    timeValues = np.arange(
        len(values),
        dtype=float
    )

    slope = np.polyfit(
        timeValues,
        values,
        1
    )[0]

    return slope


def calculateSlopeFeatures(pairData, triggerDates):
    """
    Calculate the five-observation divergence slope for requested trigger events.

    Historical divergence values are standardized using the trigger-date divergence mean and standard deviation.

    Inputs:
        - pairData (dataFrame): Daily pair feature data
        - triggerDates (array): Trigger dates for the pair

    Returns:
        - slopeData (dataFrame): Trigger-date divergence slope features
    """
    eventRows = []

    for triggerDate in triggerDates:
        triggerPosition = pairData.index.get_loc(triggerDate)

        historyStart = triggerPosition - slopeWindow + 1

        if historyStart < 0:
            continue

        triggerRow = pairData.iloc[triggerPosition]

        divMean = triggerRow["divMean"]
        divStd = triggerRow["divStd"]
        triggerDivZ = triggerRow["divZ"]

        if (
            pd.isna(divMean) or
            pd.isna(divStd) or
            divStd == 0 or
            pd.isna(triggerDivZ)
        ):
            continue

        historicalDiv = (
            pairData["div"]
            .iloc[historyStart:triggerPosition + 1]
            .to_numpy()
        )

        if len(historicalDiv) != slopeWindow or np.isnan(historicalDiv).any():
            continue

        historicalDivZ = (historicalDiv - divMean) / divStd

        divSlope5 = calculateSlope(historicalDivZ)

        triggerDirection = np.sign(triggerDivZ)

        eventRows.append({
            "date": triggerDate,
            "divSlope5": divSlope5,
            "divSlope5Rel": triggerDirection * divSlope5
        })

    return pd.DataFrame(
        eventRows,
        columns=["date"] + slopeColumns
    )


def calculateFutureTargets(pairData, pairEvents):
    """
    Calculate F1, F3, and F5 targets for existing trigger events.

    Divergence targets use the trigger-date divergence baseline and recent slope. Future stock returns are cumulative
    log returns over the same aligned pair observations used for the divergence horizon.

    Inputs:
        - pairData (dataFrame): Daily feature data for one pair
        - pairEvents (dataFrame): Existing trigger events for that pair

    Returns:
        - futureData (dataFrame): Multi-horizon target dates and targets
    """
    eventRows = []
    maximumHorizon = max(futureHorizons)

    for _, pairEvent in pairEvents.iterrows():
        triggerDate = pairEvent["date"]

        triggerPosition = pairData.index.get_loc(triggerDate)

        finalFuturePosition = triggerPosition + maximumHorizon

        if finalFuturePosition >= len(pairData):
            continue

        triggerRow = pairData.iloc[triggerPosition]

        if not np.isclose(
            triggerRow["divZ"],
            pairEvent["divZ"],
            rtol=1e-10,
            atol=1e-12
        ):
            raise ValueError(
                f"divZ mismatch for {pairEvent['symbolA']}-{pairEvent['symbolB']} on {triggerDate}"
            )

        if not np.isclose(
            triggerRow["A_Return1"],
            pairEvent["A_Return1"],
            rtol=1e-10,
            atol=1e-12
        ):
            raise ValueError(
                f"A_Return1 mismatch for {pairEvent['symbolA']}-{pairEvent['symbolB']} on {triggerDate}"
            )

        if not np.isclose(
            triggerRow["B_Return1"],
            pairEvent["B_Return1"],
            rtol=1e-10,
            atol=1e-12
        ):
            raise ValueError(
                f"B_Return1 mismatch for {pairEvent['symbolA']}-{pairEvent['symbolB']} on {triggerDate}"
            )

        divMean = pairEvent["divMean"]
        divStd = pairEvent["divStd"]
        triggerDivZ = pairEvent["divZ"]
        divSlope5 = pairEvent["divSlope5"]

        triggerDirection = np.sign(triggerDivZ)

        eventRow = {"date": triggerDate}

        completeEvent = True

        for horizon in futureHorizons:
            futurePosition = triggerPosition + horizon
            futureDate = pairData.index[futurePosition]

            futureDiv = pairData["div"].iloc[futurePosition]

            if pd.isna(futureDiv):
                completeEvent = False
                break

            futureDivZ = (futureDiv - divMean) / divStd

            linearDivZ_F = triggerDivZ + divSlope5 * horizon

            projectionResidual = futureDivZ - linearDivZ_F

            projectionResidualRel = triggerDirection * projectionResidual

            futurePairData = pairData.iloc[triggerPosition + 1:futurePosition + 1]

            if len(futurePairData) != horizon:
                completeEvent = False
                break

            futureReturnA = futurePairData["A_Return1"].sum()
            futureReturnB = futurePairData["B_Return1"].sum()

            eventRow[f"targetDate_F{horizon}"] = futureDate
            eventRow[f"divZProjectionResidualRel_F{horizon}"] = projectionResidualRel
            eventRow[f"A_Return_F{horizon}"] = futureReturnA
            eventRow[f"B_Return_F{horizon}"] = futureReturnB
            eventRow[f"A_Direction_F{horizon}"] = int(futureReturnA > 0)
            eventRow[f"B_Direction_F{horizon}"] = int(futureReturnB > 0)

        if completeEvent:
            eventRows.append(eventRow)

    return pd.DataFrame(
        eventRows,
        columns=["date"] + targetDateColumns + futureColumns
    )


def buildPairEvents(returns, volumeRatios, symbolA, symbolB):
    """
    Construct the complete trigger-event dataset for one stock pair.

    Inputs:
        - returns (dataFrame): Date-indexed log returns
        - volumeRatios (dataFrame): Date-indexed relative volume
        - symbolA (str): First stock symbol
        - symbolB (str): Second stock symbol

    Returns:
        - events (dataFrame): Complete trigger events for the pair
    """
    pairData = prepPairData(
        returns,
        volumeRatios,
        symbolA,
        symbolB
    )

    if len(pairData) < longCorrelationWindow:
        return pd.DataFrame(columns=finalEventColumns)

    pairData = calcPairFeatures(pairData)

    events = extractTriggerEvents(pairData)

    if events.empty:
        return pd.DataFrame(columns=finalEventColumns)

    slopeData = calculateSlopeFeatures(
        pairData,
        events["date"].to_numpy()
    )

    if slopeData.empty:
        return pd.DataFrame(columns=finalEventColumns)

    events = events.merge(
        slopeData,
        on="date",
        how="inner",
        validate="one_to_one"
    )

    if events.empty:
        return pd.DataFrame(columns=finalEventColumns)

    futureData = calculateFutureTargets(
        pairData,
        events
    )

    if futureData.empty:
        return pd.DataFrame(columns=finalEventColumns)

    events = events.merge(
        futureData,
        on="date",
        how="inner",
        validate="one_to_one"
    )

    events = events[finalEventColumns]

    return events