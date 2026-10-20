"""
Generic walk-forward backtesting engine.

The backtester controls chronological data access, model retraining, and prediction-time information. Portfolio
simulation and reporting are handled separately.
"""

import numpy as np
import pandas as pd

validWindowTypes = [
    "expanding",
    "rolling"
]

validRetrainingCadences = [
    "daily",
    "weekly",
    "biweekly",
    "monthly",
    "yearly"
]


class Backtester:
    """
    Run one or more models through a chronological walk-forward backtest.

    One dataset may be shared by every model, or one dataset may be supplied for each model.
    """

    def __init__(
        self,
        datasets,
        models,
        modelSettings,
        startDate=None,
        endDate=None
    ):
        self.datasets = datasets
        self.models = models
        self.modelSettings = modelSettings

        self.startDate = pd.Timestamp(startDate) if startDate is not None else None
        self.endDate = pd.Timestamp(endDate) if endDate is not None else None

        self.dataCaches = {}
        self.modelStates = []

        self.validateConfiguration()
        self.prepareModelStates()

    def validateConfiguration(self):
        """
        Validate datasets, models, and backtester settings.

        Inputs:
            - None

        Returns:
            - None
        """
        if not self.datasets:
            raise ValueError("At least one dataset is required")

        if not self.models:
            raise ValueError("At least one model is required")

        if len(self.models) != len(self.modelSettings):
            raise ValueError("Each model must have one corresponding settings dictionary")

        if len(self.datasets) not in [1, len(self.models)]:
            raise ValueError("Supply either one shared dataset or one dataset per model")

        if (
            self.startDate is not None
            and self.endDate is not None
            and self.startDate > self.endDate
        ):
            raise ValueError("startDate cannot be after endDate")

        modelNames = []

        for modelNumber, (model, settings) in enumerate(
            zip(self.models, self.modelSettings)
        ):
            if not callable(getattr(model, "fit", None)):
                raise TypeError(f"Model {modelNumber} does not define fit()")

            if not callable(getattr(model, "predict", None)):
                raise TypeError(f"Model {modelNumber} does not define predict()")

            requiredSettings = [
                "name",
                "windowType",
                "retrainingCadence",
                "predictionColumns"
            ]

            missingSettings = [
                setting
                for setting in requiredSettings
                if setting not in settings
            ]

            if missingSettings:
                raise ValueError(f"Missing settings for model {modelNumber}: {missingSettings}")

            modelName = settings["name"]

            if not isinstance(modelName, str) or not modelName:
                raise ValueError(f"Model {modelNumber} requires a non-empty name")

            modelNames.append(modelName)

            windowType = settings["windowType"]

            if windowType not in validWindowTypes:
                raise ValueError(f"Invalid windowType for {modelName}: {windowType}")

            if windowType == "rolling":
                windowSize = settings.get("windowSize")

                if not isinstance(windowSize, int) or windowSize <= 0:
                    raise ValueError(f"{modelName} requires a positive rolling windowSize")

            retrainingCadence = settings["retrainingCadence"]

            if retrainingCadence not in validRetrainingCadences:
                raise ValueError(f"Invalid retrainingCadence for {modelName}: {retrainingCadence}")

            minimumTrainingRows = settings.get("minimumTrainingRows", 1)

            if (not isinstance(minimumTrainingRows, int) or minimumTrainingRows <= 0):
                raise ValueError(f"{modelName} requires a positive minimumTrainingRows")

            predictionColumns = settings["predictionColumns"]

            if (not isinstance(predictionColumns, list) or not predictionColumns):
                raise ValueError(f"{modelName} requires a non-empty predictionColumns list")

            targetDateColumn = settings.get("targetDateColumn")

            if (targetDateColumn is not None and not isinstance(targetDateColumn, str)):
                raise ValueError(f"{modelName} targetDateColumn must be a string")

        if len(modelNames) != len(set(modelNames)):
            raise ValueError("Model names must be unique")

    def getDataset(self, modelNumber):
        """
        Return the dataset assigned to one model.

        Inputs:
            - modelNumber (int): Position of the model

        Returns:
            - data (dataFrame): Dataset assigned to the model
        """
        if len(self.datasets) == 1:
            return self.datasets[0]

        return self.datasets[modelNumber]

    def prepareDataCache(
        self,
        data,
        dateColumn,
        targetDateColumn
    ):
        """
        Prepare reusable chronological information for one dataset.

        If targetDateColumn is supplied, it represents the date on which a historical row becomes legal training
        information. If it is omitted, the event date itself determines availability.

        Inputs:
            - data (dataFrame): Model dataset
            - dateColumn (str): Event-date column
            - targetDateColumn (str or None): Training-label availability date

        Returns:
            - cache (dict): Prepared dataset and chronological indices
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Each dataset must be a pandas DataFrame")

        if dateColumn not in data.columns:
            raise ValueError(f"Date column not found: {dateColumn}")

        if (
            targetDateColumn is not None
            and targetDateColumn not in data.columns
        ):
            raise ValueError(f"Target-date column not found: {targetDateColumn}")

        preparedData = data

        dateNeedsConversion = not pd.api.types.is_datetime64_any_dtype(preparedData[dateColumn])

        targetNeedsConversion = (targetDateColumn is not None and not pd.api.types.is_datetime64_any_dtype(
            preparedData[targetDateColumn]
        ))

        if dateNeedsConversion or targetNeedsConversion:
            preparedData = preparedData.copy()

            if dateNeedsConversion:
                preparedData[dateColumn] = pd.to_datetime(preparedData[dateColumn])

            if targetNeedsConversion:
                preparedData[targetDateColumn] = pd.to_datetime(preparedData[targetDateColumn])

        if preparedData[dateColumn].isna().any():
            raise ValueError(f"Missing values found in {dateColumn}")

        if (
            targetDateColumn is not None
            and preparedData[targetDateColumn].isna().any()
        ):
            raise ValueError(f"Missing values found in {targetDateColumn}")

        if not preparedData[dateColumn].is_monotonic_increasing:
            preparedData = (
                preparedData
                .sort_values(dateColumn)
                .reset_index(drop=True)
            )

        dateValues = preparedData[dateColumn].to_numpy(dtype="datetime64[ns]")

        predictionDates = pd.DatetimeIndex(
            preparedData[dateColumn]
            .drop_duplicates()
            .to_numpy()
        )

        dateStartPositions = np.flatnonzero(
            np.concatenate([
                [True],
                dateValues[1:] != dateValues[:-1]
            ])
        )

        dateEndPositions = np.concatenate([
            dateStartPositions[1:],
            [len(preparedData)]
        ])

        dateSlices = {}

        for startPosition, endPosition in zip(
            dateStartPositions,
            dateEndPositions
        ):
            currentDate = pd.Timestamp(dateValues[startPosition])

            dateSlices[currentDate] = (
                startPosition,
                endPosition
            )

        if targetDateColumn is None:
            availabilityValues = dateValues
        else:
            availabilityValues = preparedData[
                targetDateColumn
            ].to_numpy(dtype="datetime64[ns]")

        availabilityOrder = np.argsort(
            availabilityValues,
            kind="stable"
        )

        sortedAvailabilityDates = availabilityValues[availabilityOrder]

        return {
            "data": preparedData,
            "dateColumn": dateColumn,
            "targetDateColumn": targetDateColumn,
            "dateValues": dateValues,
            "predictionDates": predictionDates,
            "dateSlices": dateSlices,
            "availabilityOrder": availabilityOrder,
            "sortedAvailabilityDates": sortedAvailabilityDates
        }

    def prepareModelStates(self):
        """
        Prepare dataset caches and runtime state for each model.

        Inputs:
            - None

        Returns:
            - None
        """
        for modelNumber, (model, settings) in enumerate(
            zip(self.models, self.modelSettings)
        ):
            data = self.getDataset(modelNumber)

            dateColumn = settings.get(
                "dateColumn",
                "date"
            )

            targetDateColumn = settings.get("targetDateColumn")

            predictionColumns = settings["predictionColumns"]

            missingPredictionColumns = [
                column
                for column in predictionColumns
                if column not in data.columns
            ]

            if missingPredictionColumns:
                raise ValueError(
                    f"Prediction columns missing for "
                    f"{settings['name']}: "
                    f"{missingPredictionColumns}"
                )

            cacheKey = (
                id(data),
                dateColumn,
                targetDateColumn
            )

            if cacheKey not in self.dataCaches:
                self.dataCaches[cacheKey] = self.prepareDataCache(
                    data,
                    dateColumn,
                    targetDateColumn
                )

            self.modelStates.append({
                "name": settings["name"],
                "model": model,
                "settings": settings,
                "cache": self.dataCaches[cacheKey],
                "isFitted": False,
                "lastRetrainDate": None
            })

    def getPredictionDates(self):
        """
        Construct the combined chronological prediction clock.

        Inputs:
            - None

        Returns:
            - predictionDates (dateTimeIndex): Dates on which at least one model has observations
        """
        modelDates = []

        for state in self.modelStates:
            predictionDates = state["cache"]["predictionDates"]

            if self.startDate is not None:
                predictionDates = predictionDates[predictionDates >= self.startDate]

            if self.endDate is not None:
                predictionDates = predictionDates[predictionDates <= self.endDate]

            modelDates.append(
                predictionDates.to_numpy(
                    dtype="datetime64[ns]"
                )
            )

        combinedDates = np.unique(np.concatenate(modelDates))

        return pd.DatetimeIndex(combinedDates)

    def shouldRetrain(
        self,
        currentDate,
        lastRetrainDate,
        cadence
    ):
        """
        Determine whether a fitted model should be retrained.

        Inputs:
            - currentDate (timestamp): Current prediction date
            - lastRetrainDate (timestamp or None): Most recent successful fit
            - cadence (str): Requested retraining cadence

        Returns:
            - shouldRetrain (bool): Whether the model should be fitted
        """
        if lastRetrainDate is None:
            return True

        if cadence == "daily":
            return True

        if cadence == "weekly":
            return (
                currentDate - lastRetrainDate
                >= pd.Timedelta(days=7)
            )

        if cadence == "biweekly":
            return (
                currentDate - lastRetrainDate
                >= pd.Timedelta(days=14)
            )

        if cadence == "monthly":
            return (
                currentDate.to_period("M")
                != lastRetrainDate.to_period("M")
            )

        if cadence == "yearly":
            return (
                currentDate.year
                != lastRetrainDate.year
            )

        raise ValueError(
            f"Unsupported retraining cadence: {cadence}"
        )

    def getRollingStartDate(
        self,
        predictionDates,
        currentDate,
        windowSize
    ):
        """
        Find the first event date included in a rolling training window.

        windowSize counts historical dates, not dataset rows.

        Inputs:
            - predictionDates (dateTimeIndex): Ordered dataset dates
            - currentDate (timestamp): Current prediction date
            - windowSize (int): Number of historical dates in the window

        Returns:
            - rollingStartDate (timestamp or None): Earliest allowed event date
        """
        historicalDates = predictionDates[predictionDates < currentDate]

        if len(historicalDates) == 0:
            return None

        if len(historicalDates) <= windowSize:
            return historicalDates[0]

        return historicalDates[-windowSize]

    def buildTrainingData(
        self,
        state,
        currentDate
    ):
        """
        Construct the legal training dataset for one model and date.

        The event itself must occur before the current prediction date. If the model defines targetDateColumn, that
        target must also have become observable before the current prediction date.

        Inputs:
            - state (dict): Runtime state for one model
            - currentDate (timestamp): Current prediction date

        Returns:
            - trainingData (dataFrame): Chronologically legal training rows
        """
        settings = state["settings"]
        cache = state["cache"]

        data = cache["data"]
        dateColumn = cache["dateColumn"]

        currentDateValue = currentDate.to_datetime64()

        availabilityPosition = np.searchsorted(
            cache["sortedAvailabilityDates"],
            currentDateValue,
            side="left"
        )

        availableIndices = cache["availabilityOrder"][:availabilityPosition]

        if len(availableIndices) == 0:
            return data.iloc[0:0]

        availableDates = cache["dateValues"][availableIndices]

        legalRows = (availableDates < currentDateValue)

        if settings["windowType"] == "rolling":
            rollingStartDate = self.getRollingStartDate(
                cache["predictionDates"],
                currentDate,
                settings["windowSize"]
            )

            if rollingStartDate is None:
                return data.iloc[0:0]

            rollingStartValue = (rollingStartDate.to_datetime64())

            legalRows &= (availableDates >= rollingStartValue)

        trainingIndices = availableIndices[legalRows]

        if len(trainingIndices) == 0:
            return data.iloc[0:0]

        trainingIndices = np.sort(trainingIndices)

        trainingData = data.iloc[trainingIndices]

        if not trainingData[dateColumn].is_monotonic_increasing:
            raise ValueError(
                f"Training data for {state['name']} "
                f"is not chronological"
            )

        return trainingData

    def getCurrentData(
        self,
        state,
        currentDate
    ):
        """
        Return prediction-safe observations for one model and date.

        Only explicitly approved prediction columns are passed to the model.

        Inputs:
            - state (dict): Runtime state for one model
            - currentDate (timestamp): Current prediction date

        Returns:
            - currentData (dataFrame or None): Current prediction observations
        """
        cache = state["cache"]

        if currentDate not in cache["dateSlices"]:
            return None

        startPosition, endPosition = cache["dateSlices"][currentDate]

        predictionColumns = state["settings"]["predictionColumns"]

        currentData = (
            cache["data"]
            .iloc[startPosition:endPosition]
            .loc[:, predictionColumns]
            .copy()
        )

        return currentData

    def run(self):
        """
        Run the chronological walk-forward backtest.

        Models are fitted only on legally observable historical rows. Predictions are generated on every available
        prediction date while the most recently fitted model remains frozen between retraining dates.

        Inputs:
            - None

        Returns:
            - results (dict): Predictions, training history, and prediction history
        """
        predictionDates = self.getPredictionDates()

        predictionFrames = []
        trainingRecords = []
        predictionRecords = []

        for currentDate in predictionDates:
            for state in self.modelStates:
                currentData = self.getCurrentData(
                    state,
                    currentDate
                )

                if (
                    currentData is None
                    or currentData.empty
                ):
                    continue

                settings = state["settings"]

                retrainModel = self.shouldRetrain(
                    currentDate,
                    state["lastRetrainDate"],
                    settings["retrainingCadence"]
                )

                if retrainModel:
                    trainingData = self.buildTrainingData(
                        state,
                        currentDate
                    )

                    minimumTrainingRows = settings.get(
                        "minimumTrainingRows",
                        1
                    )

                    if (len(trainingData) >= minimumTrainingRows):
                        state["model"].fit(trainingData)

                        state["isFitted"] = True
                        state["lastRetrainDate"] = currentDate

                        trainingRecords.append({
                            "date": currentDate,
                            "model": state["name"],
                            "trainingRows": len(trainingData),
                            "firstTrainingDate": trainingData[state["cache"]["dateColumn"]].min(),
                            "lastTrainingDate": trainingData[state["cache"]["dateColumn"]].max()
                        })

                if not state["isFitted"]:
                    predictionRecords.append({
                        "date": currentDate,
                        "model": state["name"],
                        "inputRows": len(currentData),
                        "outputRows": 0,
                        "status": "notFitted"
                    })

                    continue

                predictions = state["model"].predict(currentData)

                if predictions is None:
                    predictions = pd.DataFrame()

                if not isinstance(
                    predictions,
                    pd.DataFrame
                ):
                    raise TypeError(f"{state['name']}.predict() must return a pandas DataFrame")

                predictions = predictions.copy()

                if not predictions.empty:
                    predictions["model"] = state["name"]

                    predictions["predictionDate"] = currentDate

                    predictionFrames.append(predictions)

                predictionRecords.append({
                    "date": currentDate,
                    "model": state["name"],
                    "inputRows": len(currentData),
                    "outputRows": len(predictions),
                    "status": "predicted"
                })

        if predictionFrames:
            predictions = pd.concat(
                predictionFrames,
                ignore_index=True
            )
        else:
            predictions = pd.DataFrame()

        trainingLog = pd.DataFrame(trainingRecords)

        predictionLog = pd.DataFrame(predictionRecords)

        return {
            "predictions": predictions,
            "trainingLog": trainingLog,
            "predictionLog": predictionLog
        }