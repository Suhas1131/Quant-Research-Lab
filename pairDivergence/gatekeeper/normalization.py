import numpy as np
import pandas as pd


featureColumns = [
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
    "B_VolumeRatio63",
    "divSlope5Rel"
]

targetColumns = [
    "divZProjectionResidualRel_F1",
    "divZProjectionResidualRel_F3",
    "divZProjectionResidualRel_F5"
]

horizonNames = [
    "F1",
    "F3",
    "F5"
]


def fitNormalizationParameters(data):
    """
    Fit feature normalization parameters.
    """
    parameters = {}

    returnCombined = pd.concat([
        data["A_Return1"],
        data["B_Return1"]
    ])

    parameters["returnMedian"] = returnCombined.median()

    parameters["returnScale"] = (
        returnCombined.quantile(0.99) -
        returnCombined.quantile(0.01)
    )

    parameters["hedgeRatioScale"] = data["hedgeRatio"].abs().max()

    for column in [
        "div",
        "divMean",
        "divStd"
    ]:
        parameters[f"{column}Median"] = data[column].median()

        parameters[f"{column}Scale"] = data[column].quantile(0.99) - data[column].quantile(0.01)

    parameters["divZScale"] = data["divZ"].quantile(0.99) - data["divZ"].quantile(0.01)

    vol21Combined = pd.concat([data["A_Vol21"], data["B_Vol21"]])

    parameters["vol21Scale"] = vol21Combined.quantile(0.99) - vol21Combined.quantile(0.01)

    vol252Combined = pd.concat([data["A_Vol252"], data["B_Vol252"]])

    parameters["vol252Scale"] = vol252Combined.quantile(0.99) - vol252Combined.quantile(0.01)

    for column in ["divMean_PQ", "divStd_PQ"]:
        parameters[f"{column}Scale"] = data[column].quantile(0.99) - data[column].quantile(0.01)

    volumeLogCombined = pd.concat([np.log1p(data["A_VolumeRatio63"]), np.log1p(data["B_VolumeRatio63"])])

    parameters["volumeRatioScale"] = volumeLogCombined.max()

    return parameters


def normalizeFeatureData(data, parameters):
    """
    Normalize model features using fitted parameters.
    """
    normalizedData = pd.DataFrame(index=data.index)

    normalizedData["A_Return1"] = (data["A_Return1"] - parameters["returnMedian"]) / parameters["returnScale"]

    normalizedData["B_Return1"] = (data["B_Return1"] - parameters["returnMedian"]) / parameters["returnScale"]

    for column in [
        "corr42",
        "corr126",
        "corr252"
    ]:
        normalizedData[column] = data[column]

    normalizedData["hedgeRatio"] = data["hedgeRatio"] / parameters["hedgeRatioScale"]

    for column in [
        "div",
        "divMean",
        "divStd"
    ]:
        normalizedData[column] = (data[column] - parameters[f"{column}Median"]) / parameters[f"{column}Scale"]

    normalizedData["divZ"] = data["divZ"] /parameters["divZScale"]

    normalizedData["corrVar"] = data["corrVar"]

    normalizedData["A_Vol21"] = data["A_Vol21"] / parameters["vol21Scale"]

    normalizedData["B_Vol21"] = data["B_Vol21"] / parameters["vol21Scale"]

    normalizedData["A_Vol252"] = data["A_Vol252"] / parameters["vol252Scale"]

    normalizedData["B_Vol252"] = data["B_Vol252"] / parameters["vol252Scale"]

    for column in ["divMean_PQ", "divStd_PQ"]:
        normalizedData[column] = data[column] / parameters[f"{column}Scale"]

    normalizedData["divMeanDiff"] = data["divMeanDiff"]
    normalizedData["divStdDiff"] = data["divStdDiff"]

    normalizedData["A_VolumeRatio63"] = np.log1p(data["A_VolumeRatio63"]) / parameters["volumeRatioScale"]

    normalizedData["B_VolumeRatio63"] = np.log1p(data["B_VolumeRatio63"]) / parameters["volumeRatioScale"]

    normalizedData["divSlope5Rel"] = np.arctan(data["divSlope5Rel"]) / (np.pi / 2)

    return normalizedData[featureColumns].astype(np.float32)


def fitTargetScales(data):
    """
    Fit target scales using the training data.
    """
    targetScales = {}

    for column in targetColumns:
        targetScales[column] = data[column].quantile(0.99) - data[column].quantile(0.01)

    return targetScales


def getTargetScaleVector(targetScales):
    """
    Return target scales in F1, F3, F5 order.
    """
    return np.array([targetScales[column] for column in targetColumns])


def normalizeTargetData(data, targetScales):
    """
    Normalize the F1, F3 and F5 targets.
    """
    targetScaleVector = getTargetScaleVector(targetScales)

    return data[targetColumns] / targetScaleVector