# ai/trainer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import joblib
from config import BusinessConfig

class SalesForecaster:
    def __init__(self, config: BusinessConfig):
        self.config = config
        self.model = None
        self.feature_columns = []
        self.item_mapping = None  # Maps item names to encoded values
        self.item_metadata = {}  # Maps item names to {size, cost, avg_hourly} data
        self.hourly_patterns = {}  # Maps item names to hourly sales patterns
        self.global_avg_quantity = 0  # Fallback for unknown items
        self.category_avg_quantity = {}  # Category-level fallback
        self.sales_history = None  # Store historical sales for lag features during prediction

    def _compute_lag_features(self, df: pd.DataFrame, target_column: str = "quantity") -> pd.DataFrame:
        """Compute lag features (previous day, previous week) and rolling averages per item"""

        df = df.copy()
        df = df.sort_values(["item_name", "date"])

        # Group by item to compute lags correctly
        lag_features = []

        for item in df["item_name"].unique():
            item_df = df[df["item_name"] == item].copy()
            item_df = item_df.sort_values("date")

            # Lag features - previous day's sales (lag_1) and same day last week (lag_7)
            item_df["lag_1"] = item_df[target_column].shift(1)
            item_df["lag_7"] = item_df[target_column].shift(7)

            # Rolling averages - 7 day and 30 day
            item_df["rolling_7_avg"] = item_df[target_column].shift(1).rolling(window=7, min_periods=1).mean()
            item_df["rolling_30_avg"] = item_df[target_column].shift(1).rolling(window=30, min_periods=1).mean()

            # Rolling standard deviation (captures volatility)
            item_df["rolling_7_std"] = item_df[target_column].shift(1).rolling(window=7, min_periods=1).std()

            lag_features.append(item_df)

        result = pd.concat(lag_features, ignore_index=True)

        # Fill NaN lag features with rolling averages, then global mean
        for col in ["lag_1", "lag_7", "rolling_7_avg", "rolling_30_avg", "rolling_7_std"]:
            if col in result.columns:
                # First try to fill with item's rolling average
                result[col] = result[col].fillna(result["rolling_7_avg"])
                # Then fill remaining with global average
                result[col] = result[col].fillna(result[target_column].mean())
                # Finally fill any remaining with 0
                result[col] = result[col].fillna(0)

        return result

    def _compute_prediction_lag_features(self, pred_df: pd.DataFrame, items: list) -> pd.DataFrame:
        """Compute lag features for prediction using stored sales history"""

        pred_df = pred_df.copy()

        # Initialize lag columns
        pred_df["lag_1"] = 0.0
        pred_df["lag_7"] = 0.0
        pred_df["rolling_7_avg"] = 0.0
        pred_df["rolling_30_avg"] = 0.0
        pred_df["rolling_7_std"] = 0.0

        if self.sales_history is None or len(self.sales_history) == 0:
            # No history available, use global/category averages
            for idx, row in pred_df.iterrows():
                item = row["item_name"]
                if item in self.item_metadata:
                    avg = self.item_metadata[item].get("avg_daily_qty", self.global_avg_quantity)
                else:
                    avg = self.global_avg_quantity
                pred_df.at[idx, "lag_1"] = avg
                pred_df.at[idx, "lag_7"] = avg
                pred_df.at[idx, "rolling_7_avg"] = avg
                pred_df.at[idx, "rolling_30_avg"] = avg
                pred_df.at[idx, "rolling_7_std"] = avg * 0.2  # Estimate 20% std
            return pred_df

        # Use stored history to compute lag features
        history = self.sales_history.copy()
        history["date"] = pd.to_datetime(history["date"])

        for idx, row in pred_df.iterrows():
            item = row["item_name"]
            pred_date = pd.to_datetime(row["date"])

            # Get this item's history
            item_history = history[history["item_name"] == item].sort_values("date")

            if len(item_history) == 0:
                # No history for this item, use category or global average
                avg = self._get_fallback_quantity(item)
                pred_df.at[idx, "lag_1"] = avg
                pred_df.at[idx, "lag_7"] = avg
                pred_df.at[idx, "rolling_7_avg"] = avg
                pred_df.at[idx, "rolling_30_avg"] = avg
                pred_df.at[idx, "rolling_7_std"] = avg * 0.2
                continue

            # lag_1: yesterday's sales
            yesterday = pred_date - timedelta(days=1)
            yesterday_sales = item_history[item_history["date"] == yesterday]
            if len(yesterday_sales) > 0:
                pred_df.at[idx, "lag_1"] = yesterday_sales["quantity"].values[0]
            else:
                pred_df.at[idx, "lag_1"] = item_history["quantity"].iloc[-1]  # Most recent

            # lag_7: same day last week
            last_week = pred_date - timedelta(days=7)
            last_week_sales = item_history[item_history["date"] == last_week]
            if len(last_week_sales) > 0:
                pred_df.at[idx, "lag_7"] = last_week_sales["quantity"].values[0]
            else:
                pred_df.at[idx, "lag_7"] = pred_df.at[idx, "lag_1"]

            # Rolling averages from recent history
            recent_7 = item_history[item_history["date"] >= pred_date - timedelta(days=7)]
            recent_30 = item_history[item_history["date"] >= pred_date - timedelta(days=30)]

            if len(recent_7) > 0:
                pred_df.at[idx, "rolling_7_avg"] = recent_7["quantity"].mean()
                pred_df.at[idx, "rolling_7_std"] = recent_7["quantity"].std() if len(recent_7) > 1 else recent_7["quantity"].mean() * 0.2
            else:
                pred_df.at[idx, "rolling_7_avg"] = item_history["quantity"].mean()
                pred_df.at[idx, "rolling_7_std"] = item_history["quantity"].std() if len(item_history) > 1 else item_history["quantity"].mean() * 0.2

            if len(recent_30) > 0:
                pred_df.at[idx, "rolling_30_avg"] = recent_30["quantity"].mean()
            else:
                pred_df.at[idx, "rolling_30_avg"] = item_history["quantity"].mean()

        # Fill any remaining NaN
        for col in ["lag_1", "lag_7", "rolling_7_avg", "rolling_30_avg", "rolling_7_std"]:
            pred_df[col] = pred_df[col].fillna(self.global_avg_quantity)

        return pred_df

    def _get_fallback_quantity(self, item_name: str) -> float:
        """Get fallback quantity for unknown or new items"""

        # Try item metadata first
        if item_name in self.item_metadata:
            return self.item_metadata[item_name].get("avg_daily_qty", self.global_avg_quantity)

        # Try to find category from similar items
        for cat, items in self.categories.items():
            # Check if item name partially matches any known item
            for known_item in items:
                if item_name.lower() in known_item.lower() or known_item.lower() in item_name.lower():
                    return self.category_avg_quantity.get(cat, self.global_avg_quantity)

        # Return global average
        return self.global_avg_quantity

    def prepare_features(
        self,
        df: pd.DataFrame,
        weather_data: list = None,
        holidays: list = None,
        sports_games: list = None,
        paydays: list = None,
        school_calendar: list = None,
        is_training: bool = False
    ) -> pd.DataFrame:
        """Convert raw data into features for training"""

        df = df.copy()
        # Handle various date formats (including day-first international formats)
        try:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        except Exception:
            df["date"] = pd.to_datetime(df["date"], format='mixed', dayfirst=True)
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

        # Date features
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encoding for day of week and month (helps model understand cyclical nature)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Time features (if available)
        if "time" in df.columns:
            df["time_block"] = df["time"].apply(self.config.get_time_block)
            df["time_block_encoded"] = pd.factorize(df["time_block"])[0]

        # Weather features with missing flag
        df["temp_max"] = np.nan
        df["temp_min"] = np.nan
        df["precipitation"] = np.nan
        df["weather_code"] = np.nan
        df["weather_missing"] = 1  # Flag for missing weather data

        if weather_data:
            weather_dict = {w["date"]: w for w in weather_data}
            for idx, row in df.iterrows():
                weather = weather_dict.get(row["date_str"])
                if weather:
                    temp_max = weather.get("temp_max")
                    temp_min = weather.get("temp_min")
                    precip = weather.get("precipitation")
                    code = weather.get("weather_code")

                    if temp_max is not None:
                        df.at[idx, "temp_max"] = float(temp_max)
                        df.at[idx, "weather_missing"] = 0
                    if temp_min is not None:
                        df.at[idx, "temp_min"] = float(temp_min)
                    if precip is not None:
                        df.at[idx, "precipitation"] = float(precip)
                    if code is not None:
                        df.at[idx, "weather_code"] = int(code)

        # Fill missing weather with monthly averages from the data, not hardcoded values
        for col in ["temp_max", "temp_min", "precipitation", "weather_code"]:
            if df[col].isna().any():
                # Try to fill with same month's average
                monthly_avg = df.groupby("month")[col].transform("mean")
                df[col] = df[col].fillna(monthly_avg)
                # If still missing, use overall median
                df[col] = df[col].fillna(df[col].median())
                # Final fallback
                if col == "temp_max":
                    df[col] = df[col].fillna(70.0)
                elif col == "temp_min":
                    df[col] = df[col].fillna(50.0)
                else:
                    df[col] = df[col].fillna(0.0)

        # Holiday features
        df["is_holiday"] = 0
        df["days_to_holiday"] = 30  # Days until next holiday (capped at 30)
        df["days_from_holiday"] = 30  # Days since last holiday (capped at 30)

        if holidays:
            holiday_dates = set(h["date"] for h in holidays)
            df["is_holiday"] = df["date_str"].isin(holiday_dates).astype(int)

            # Calculate days to/from nearest holiday
            holiday_dates_dt = sorted([pd.to_datetime(h["date"]) for h in holidays])
            for idx, row in df.iterrows():
                current_date = row["date"]

                # Days to next holiday
                future_holidays = [h for h in holiday_dates_dt if h >= current_date]
                if future_holidays:
                    days_to = (future_holidays[0] - current_date).days
                    df.at[idx, "days_to_holiday"] = min(days_to, 30)

                # Days from last holiday
                past_holidays = [h for h in holiday_dates_dt if h <= current_date]
                if past_holidays:
                    days_from = (current_date - past_holidays[-1]).days
                    df.at[idx, "days_from_holiday"] = min(days_from, 30)

        # Sports features
        df["has_sports"] = 0
        if sports_games:
            sports_dates = set(g["date"] for g in sports_games)
            df["has_sports"] = df["date_str"].isin(sports_dates).astype(int)

        # Payday features
        df["is_payday"] = 0
        df["days_to_payday"] = 15  # Days until next payday (capped)

        if paydays:
            payday_dates = set(p["date"] for p in paydays)
            df["is_payday"] = df["date_str"].isin(payday_dates).astype(int)

            # Days to next payday
            payday_dates_dt = sorted([pd.to_datetime(p["date"]) for p in paydays])
            for idx, row in df.iterrows():
                current_date = row["date"]
                future_paydays = [p for p in payday_dates_dt if p >= current_date]
                if future_paydays:
                    days_to = (future_paydays[0] - current_date).days
                    df.at[idx, "days_to_payday"] = min(days_to, 15)

        # School features - optimized with date range indexing
        df["is_school_break"] = 0
        if school_calendar:
            # Build break ranges for efficient lookup
            break_ranges = []
            for event in school_calendar:
                if event.get("type") == "break":
                    start = event["start_date"]
                    end = event.get("end_date", start)
                    break_ranges.append((start, end))

            # Vectorized check
            for start, end in break_ranges:
                mask = (df["date_str"] >= start) & (df["date_str"] <= end)
                df.loc[mask, "is_school_break"] = 1

        # Encode item names - use existing mapping if available (for predictions)
        if self.item_mapping is not None:
            # Use stored mapping from training (reverse lookup: name -> code)
            reverse_mapping = {v: k for k, v in self.item_mapping.items()}
            df["item_encoded"] = df["item_name"].map(reverse_mapping)

            # Handle unknown items gracefully - use category average encoding or -1
            unknown_mask = df["item_encoded"].isna()
            if unknown_mask.any():
                # For unknown items, try to find similar category
                for idx in df[unknown_mask].index:
                    item = df.at[idx, "item_name"]
                    # Find most similar known item based on category
                    fallback_code = self._get_similar_item_code(item)
                    df.at[idx, "item_encoded"] = fallback_code

            df["item_encoded"] = df["item_encoded"].fillna(-1).astype(int)
        else:
            # Create new mapping during training
            codes, uniques = pd.factorize(df["item_name"])
            df["item_encoded"] = codes
            self.item_mapping = dict(enumerate(uniques))

        return df

    def _get_similar_item_code(self, item_name: str) -> int:
        """Find a similar item's code for unknown items"""

        if not self.item_mapping or not self.categories:
            return -1

        # Try to find category match
        for cat, items in self.categories.items():
            for known_item in items:
                if item_name.lower() in known_item.lower() or known_item.lower() in item_name.lower():
                    # Found similar item, return its code
                    reverse_mapping = {v: k for k, v in self.item_mapping.items()}
                    if known_item in reverse_mapping:
                        return reverse_mapping[known_item]

        # Return most common item code as fallback
        if self.item_metadata:
            most_common = max(self.item_metadata.items(),
                            key=lambda x: x[1].get("avg_daily_qty", 0),
                            default=(None, None))
            if most_common[0]:
                reverse_mapping = {v: k for k, v in self.item_mapping.items()}
                return reverse_mapping.get(most_common[0], -1)

        return -1

    def safe_mape(self, y_true, y_pred):
        """Calculate MAPE safely, handling zeros"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0

        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        if np.isnan(mape) or np.isinf(mape):
            return 0.0

        return float(mape)

    def train(
        self,
        df: pd.DataFrame,
        weather_data: list = None,
        holidays: list = None,
        sports_games: list = None,
        paydays: list = None,
        school_calendar: list = None,
        target_column: str = "quantity",
        test_size: float = 0.2,
        raw_df: pd.DataFrame = None,
        tune_hyperparameters: bool = True,
        n_cv_splits: int = 3
    ) -> dict:
        """Train the forecasting model with improvements

        Args:
            df: Aggregated dataframe with date, item_name, quantity
            raw_df: Optional raw dataframe with all columns (size, time, unit_price) for metadata extraction
            tune_hyperparameters: Whether to perform hyperparameter tuning
            n_cv_splits: Number of cross-validation splits for time-series CV
        """

        # Reset item mapping before training (will be created fresh in prepare_features)
        self.item_mapping = None

        # Use raw_df for metadata extraction if provided, otherwise use df
        metadata_df = raw_df if raw_df is not None else df

        # Extract item metadata (size, cost) from original data before aggregation
        self._extract_item_metadata(metadata_df)

        # Extract hourly patterns if time data is available
        self._extract_hourly_patterns(metadata_df, target_column)

        # Compute global and category averages for fallback predictions
        self._compute_fallback_averages(df, target_column)

        # Store sales history for prediction lag features
        self._store_sales_history(df, target_column)

        # Prepare features
        prepared_df = self.prepare_features(
            df,
            weather_data=weather_data,
            holidays=holidays,
            sports_games=sports_games,
            paydays=paydays,
            school_calendar=school_calendar,
            is_training=True
        )

        # Add lag features and rolling averages
        prepared_df = self._compute_lag_features(prepared_df, target_column)

        # Select feature columns (now including lag features)
        self.feature_columns = [
            # Date features
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "is_weekend",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            # Weather features
            "temp_max",
            "temp_min",
            "precipitation",
            "weather_code",
            "weather_missing",
            # Event features
            "is_holiday",
            "days_to_holiday",
            "days_from_holiday",
            "has_sports",
            "is_payday",
            "days_to_payday",
            "is_school_break",
            # Item features
            "item_encoded",
            # Lag features (NEW)
            "lag_1",
            "lag_7",
            "rolling_7_avg",
            "rolling_30_avg",
            "rolling_7_std"
        ]

        # Add time block if available
        if "time_block_encoded" in prepared_df.columns:
            self.feature_columns.append("time_block_encoded")

        # Ensure all columns exist
        for col in self.feature_columns:
            if col not in prepared_df.columns:
                prepared_df[col] = 0

        X = prepared_df[self.feature_columns]
        y = prepared_df[target_column]

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        cv_scores = []

        if tune_hyperparameters and len(X) >= 100:
            # Hyperparameter tuning with RandomizedSearchCV
            param_distributions = {
                "n_estimators": [50, 100, 150, 200, 250],
                "max_depth": [3, 4, 5, 6, 7, 8],
                "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                "min_child_weight": [1, 3, 5, 7],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2, 0.3]
            }

            base_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=42
            )

            # Use fewer iterations for faster tuning
            n_iter = min(20, len(X) // 50)  # Scale with data size
            n_iter = max(5, n_iter)  # At least 5 iterations

            search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=n_iter,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                random_state=42,
                n_jobs=-1
            )

            search.fit(X, y)
            self.model = search.best_estimator_
            best_params = search.best_params_

            # Get CV scores from the search
            cv_scores = [-score for score in search.cv_results_["mean_test_score"]]
        else:
            # Use default parameters or simple training for small datasets
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="reg:squarederror",
                random_state=42
            )
            best_params = None

            # Manual cross-validation for scoring
            for train_idx, val_idx in tscv.split(X):
                X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
                y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]

                temp_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="reg:squarederror",
                    random_state=42
                )
                temp_model.fit(X_cv_train, y_cv_train)
                cv_pred = temp_model.predict(X_cv_val)
                cv_scores.append(mean_absolute_error(y_cv_val, cv_pred))

            # Final fit on all data
            self.model.fit(X, y)

        # Final train/test split for reporting (still useful for final metrics)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        results = {
            "train_mae": float(round(mean_absolute_error(y_train, train_pred), 2)),
            "test_mae": float(round(mean_absolute_error(y_test, test_pred), 2)),
            "train_mape": float(round(self.safe_mape(y_train, train_pred), 2)),
            "test_mape": float(round(self.safe_mape(y_test, test_pred), 2)),
            "cv_mae_mean": float(round(np.mean(cv_scores), 2)) if cv_scores else None,
            "cv_mae_std": float(round(np.std(cv_scores), 2)) if cv_scores else None,
            "feature_importance": self._get_feature_importance(),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "best_params": best_params,
            "improvements_applied": [
                "lag_features",
                "rolling_averages",
                "time_series_cv",
                "unknown_item_handling",
                "weather_missing_flag",
                "hyperparameter_tuning" if tune_hyperparameters else None,
                "cyclical_encoding",
                "days_to_holiday",
                "days_to_payday"
            ]
        }

        return results

    def _compute_fallback_averages(self, df: pd.DataFrame, target_column: str = "quantity"):
        """Compute global and category-level averages for unknown item fallback"""

        # Global average
        self.global_avg_quantity = float(df[target_column].mean()) if target_column in df.columns else 0

        # Category averages
        self.category_avg_quantity = {}
        if hasattr(self, 'categories') and self.categories:
            for cat, items in self.categories.items():
                cat_df = df[df["item_name"].isin(items)]
                if len(cat_df) > 0 and target_column in cat_df.columns:
                    self.category_avg_quantity[cat] = float(cat_df[target_column].mean())

    def _store_sales_history(self, df: pd.DataFrame, target_column: str = "quantity"):
        """Store recent sales history for computing lag features during prediction"""

        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])

        # Keep last 60 days of history for each item
        max_date = df_copy["date"].max()
        cutoff = max_date - timedelta(days=60)

        self.sales_history = df_copy[df_copy["date"] >= cutoff][["date", "item_name", target_column]].copy()
        self.sales_history.columns = ["date", "item_name", "quantity"]

    def _get_feature_importance(self) -> dict:
        """Get feature importance from trained model"""

        if self.model is None:
            return {}

        importance = self.model.feature_importances_
        return {
            col: round(float(imp), 4)
            for col, imp in sorted(
                zip(self.feature_columns, importance),
                key=lambda x: x[1],
                reverse=True
            )
        }

    def predict(
        self,
        future_dates: list,
        items: list,
        weather_data: list = None,
        holidays: list = None,
        sports_games: list = None,
        paydays: list = None,
        school_calendar: list = None
    ) -> pd.DataFrame:
        """Make predictions for future dates"""

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create prediction dataframe
        predictions = []

        for date in future_dates:
            for item in items:
                predictions.append({
                    "date": date,
                    "item_name": item
                })

        pred_df = pd.DataFrame(predictions)

        # Prepare features
        prepared_df = self.prepare_features(
            pred_df,
            weather_data=weather_data,
            holidays=holidays,
            sports_games=sports_games,
            paydays=paydays,
            school_calendar=school_calendar,
            is_training=False
        )

        # Add lag features for prediction
        prepared_df = self._compute_prediction_lag_features(prepared_df, items)

        # Ensure all required feature columns exist (some may be missing during prediction)
        for col in self.feature_columns:
            if col not in prepared_df.columns:
                # Add missing column with default value
                prepared_df[col] = 0

        # Make predictions
        X = prepared_df[self.feature_columns]
        prepared_df["predicted_quantity"] = self.model.predict(X)
        prepared_df["predicted_quantity"] = prepared_df["predicted_quantity"].round().astype(int)
        prepared_df["predicted_quantity"] = prepared_df["predicted_quantity"].clip(lower=0)

        return prepared_df[["date", "item_name", "predicted_quantity"]]

    def _extract_item_metadata(self, df: pd.DataFrame):
        """Extract size, cost, and category metadata for each item"""
        self.item_metadata = {}

        for item in df["item_name"].unique():
            item_df = df[df["item_name"] == item]
            metadata = {"item_name": item}

            # Extract category/type if available
            if "type" in df.columns:
                categories = item_df["type"].dropna().unique()
                if len(categories) > 0:
                    metadata["category"] = str(categories[0])  # Use first category found

            # Extract size if available
            if "size" in df.columns:
                sizes = item_df["size"].dropna().unique()
                if len(sizes) > 0:
                    metadata["sizes"] = [str(s) for s in sizes]
                    # Get most common size
                    mode_size = item_df["size"].mode()
                    metadata["primary_size"] = str(mode_size.iloc[0]) if len(mode_size) > 0 else str(sizes[0])

                    # Calculate size distribution (proportion of sales by size)
                    size_distribution = {}
                    total_qty = item_df["quantity"].sum() if "quantity" in item_df.columns else len(item_df)
                    for size in sizes:
                        size_df = item_df[item_df["size"] == size]
                        size_qty = size_df["quantity"].sum() if "quantity" in size_df.columns else len(size_df)
                        if total_qty > 0:
                            size_distribution[str(size)] = float(size_qty / total_qty)
                    if size_distribution:
                        metadata["size_distribution"] = size_distribution

                    # Extract price per size
                    if "unit_price" in df.columns:
                        size_prices = {}
                        for size in sizes:
                            size_df = item_df[item_df["size"] == size]
                            size_price = size_df["unit_price"].dropna().mean()
                            if not pd.isna(size_price):
                                size_prices[str(size)] = float(size_price)
                        if size_prices:
                            metadata["size_prices"] = size_prices

            # Extract cost/price if available
            if "unit_price" in df.columns:
                prices = item_df["unit_price"].dropna()
                if len(prices) > 0:
                    metadata["unit_price"] = float(prices.mean())
                    metadata["min_price"] = float(prices.min())
                    metadata["max_price"] = float(prices.max())

            # Calculate average daily quantity
            if "quantity" in df.columns and "date" in df.columns:
                daily_qty = item_df.groupby(pd.to_datetime(item_df["date"]).dt.date)["quantity"].sum()
                if len(daily_qty) > 0:
                    metadata["avg_daily_qty"] = float(daily_qty.mean())

            self.item_metadata[item] = metadata

        # Build category index for quick lookup
        self.categories = {}
        for item, meta in self.item_metadata.items():
            cat = meta.get("category", "Other")
            if cat not in self.categories:
                self.categories[cat] = []
            self.categories[cat].append(item)

    def _extract_hourly_patterns(self, df: pd.DataFrame, target_column: str = "quantity"):
        """Extract hourly sales patterns for each item"""
        self.hourly_patterns = {}

        if "time" not in df.columns:
            return

        df_copy = df.copy()

        # Parse time to get hour
        try:
            df_copy["hour"] = pd.to_datetime(df_copy["time"], format='mixed').dt.hour
        except Exception:
            try:
                df_copy["hour"] = pd.to_datetime(df_copy["time"]).dt.hour
            except Exception:
                return

        # Calculate hourly patterns for each item
        for item in df_copy["item_name"].unique():
            item_df = df_copy[df_copy["item_name"] == item]
            hourly_data = item_df.groupby("hour")[target_column].agg(["sum", "count", "mean"]).reset_index()

            # Calculate percentage of daily sales per hour
            total = hourly_data["sum"].sum()
            if total > 0:
                hourly_data["percentage"] = (hourly_data["sum"] / total * 100).round(1)
            else:
                hourly_data["percentage"] = 0

            self.hourly_patterns[item] = hourly_data.to_dict(orient="records")

        # Also calculate overall hourly pattern (all items combined)
        overall_hourly = df_copy.groupby("hour")[target_column].agg(["sum", "count", "mean"]).reset_index()
        total = overall_hourly["sum"].sum()
        if total > 0:
            overall_hourly["percentage"] = (overall_hourly["sum"] / total * 100).round(1)
        else:
            overall_hourly["percentage"] = 0

        self.hourly_patterns["_overall"] = overall_hourly.to_dict(orient="records")

    def save_model(self, filepath: str):
        """Save trained model to file"""

        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "item_mapping": self.item_mapping,
            "item_metadata": self.item_metadata,
            "hourly_patterns": self.hourly_patterns,
            "categories": getattr(self, 'categories', {}),
            "config": self.config,
            # New fields for improved model
            "global_avg_quantity": self.global_avg_quantity,
            "category_avg_quantity": self.category_avg_quantity,
            "sales_history": self.sales_history
        }, filepath)

    def load_model(self, filepath: str):
        """Load trained model from file"""

        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.item_mapping = data["item_mapping"]
        self.item_metadata = data.get("item_metadata", {})
        self.hourly_patterns = data.get("hourly_patterns", {})
        self.categories = data.get("categories", {})
        # New fields for improved model
        self.global_avg_quantity = data.get("global_avg_quantity", 0)
        self.category_avg_quantity = data.get("category_avg_quantity", {})
        self.sales_history = data.get("sales_history", None)
