# ai/analyzer.py
import pandas as pd
from datetime import datetime
from typing import Optional
from config import BusinessConfig

class DataAnalyzer:
    def __init__(self, config: BusinessConfig):
        self.config = config
        self.anomalies = []
    
    def add_context_columns(
        self,
        df: pd.DataFrame,
        weather_data: list = None,
        holidays: list = None,
        sports_games: list = None,
        paydays: list = None
    ) -> pd.DataFrame:
        """Add contextual columns to the dataframe for grouping"""
        
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        # Day of week
        df["day_of_week"] = df["date"].dt.day_name()
        
        # Time block (if time column exists)
        if "time" in df.columns:
            df["time_block"] = df["time"].apply(self.config.get_time_block)
        else:
            df["time_block"] = "all_day"
        
        # Convert date to string for matching
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        
        # Weather conditions
        df["weather_type"] = "normal"
        if weather_data:
            weather_dict = {w["date"]: w for w in weather_data}
            for idx, row in df.iterrows():
                weather = weather_dict.get(row["date_str"])
                if weather:
                    if weather.get("precipitation", 0) > 10:
                        df.at[idx, "weather_type"] = "rain"
                    elif weather.get("temp_max", 70) > 95 or weather.get("temp_min", 50) < 20:
                        df.at[idx, "weather_type"] = "extreme_temp"
        
        # Holidays
        df["is_holiday"] = False
        df["holiday_name"] = None
        if holidays:
            holiday_dates = {h["date"]: h["name"] for h in holidays}
            for idx, row in df.iterrows():
                if row["date_str"] in holiday_dates:
                    df.at[idx, "is_holiday"] = True
                    df.at[idx, "holiday_name"] = holiday_dates[row["date_str"]]
        
        # Sports games
        df["has_sports"] = False
        df["sports_event"] = None
        if sports_games:
            sports_dates = {g["date"]: g["name"] for g in sports_games}
            for idx, row in df.iterrows():
                if row["date_str"] in sports_dates:
                    df.at[idx, "has_sports"] = True
                    df.at[idx, "sports_event"] = sports_dates[row["date_str"]]
        
        # Paydays
        df["is_payday"] = False
        if paydays:
            payday_dates = [p["date"] for p in paydays]
            df["is_payday"] = df["date_str"].isin(payday_dates)
        
        return df
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        z_threshold: float = 2.5
    ) -> list:
        """Detect anomalies by comparing within similar groups"""
        
        anomalies = []
        
        # Group by: item + day_of_week + time_block + weather + holiday + sports
        group_columns = ["item_name", "day_of_week", "time_block", "weather_type", "is_holiday", "has_sports"]
        
        # Only use columns that exist
        group_columns = [col for col in group_columns if col in df.columns]
        
        for group_keys, group_data in df.groupby(group_columns):
            
            # Need enough data to compare
            if len(group_data) < 4:
                continue
            
            mean = group_data["quantity"].mean()
            std = group_data["quantity"].std()
            
            # Skip if no variation
            if std == 0:
                continue
            
            for idx, row in group_data.iterrows():
                z_score = abs(row["quantity"] - mean) / std
                
                if z_score > z_threshold:
                    # Build group description
                    if isinstance(group_keys, tuple):
                        group_dict = dict(zip(group_columns, group_keys))
                    else:
                        group_dict = {group_columns[0]: group_keys}
                    
                    anomalies.append({
                        "date": row["date_str"],
                        "item": row["item_name"],
                        "quantity": int(row["quantity"]),
                        "group": group_dict,
                        "group_mean": round(mean, 1),
                        "group_std": round(std, 1),
                        "expected_range": f"{max(0, mean - 2*std):.0f} - {mean + 2*std:.0f}",
                        "z_score": round(z_score, 2),
                        "type": "grouped_outlier",
                        "context": {
                            "day": row.get("day_of_week"),
                            "time_block": row.get("time_block"),
                            "weather": row.get("weather_type"),
                            "holiday": row.get("holiday_name"),
                            "sports": row.get("sports_event"),
                            "payday": row.get("is_payday")
                        }
                    })
        
        # Check for missing dates
        date_range = pd.date_range(df["date"].min(), df["date"].max())
        actual_dates = set(df["date"].dt.date)
        missing_dates = set(date_range.date) - actual_dates
        
        for d in missing_dates:
            anomalies.append({
                "date": str(d),
                "type": "missing_date"
            })
        
        self.anomalies = anomalies
        return anomalies
    
    def analyze(
        self,
        df: pd.DataFrame,
        weather_data: list = None,
        holidays: list = None,
        sports_games: list = None,
        paydays: list = None,
        z_threshold: float = 2.5
    ) -> dict:
        """Full analysis pipeline"""
        
        # Step 1: Add context columns
        enriched_df = self.add_context_columns(
            df,
            weather_data=weather_data,
            holidays=holidays,
            sports_games=sports_games,
            paydays=paydays
        )
        
        # Step 2: Detect anomalies within groups
        anomalies = self.detect_anomalies(enriched_df, z_threshold)
        
        # Step 3: Categorize results
        grouped_outliers = [a for a in anomalies if a["type"] == "grouped_outlier"]
        missing_dates = [a for a in anomalies if a["type"] == "missing_date"]
        
        return {
            "total_anomalies": len(anomalies),
            "grouped_outliers": grouped_outliers,
            "missing_dates": missing_dates,
            "summary": {
                "outlier_count": len(grouped_outliers),
                "missing_date_count": len(missing_dates),
                "data_quality": self._assess_quality(len(anomalies), len(df))
            },
            "recommendation": self._get_recommendation(grouped_outliers, missing_dates, len(df))
        }
    
    def _assess_quality(self, anomaly_count: int, total_rows: int) -> str:
        """Rate overall data quality"""
        
        if total_rows == 0:
            return "no_data"
        
        anomaly_rate = anomaly_count / total_rows
        
        if anomaly_rate < 0.01:
            return "excellent"
        elif anomaly_rate < 0.03:
            return "good"
        elif anomaly_rate < 0.05:
            return "fair"
        else:
            return "poor"
    
    def _get_recommendation(self, outliers: list, missing: list, total_rows: int) -> str:
        """Provide recommendation based on findings"""
        
        if len(missing) > total_rows * 0.1:
            return "Too many missing dates. Fill gaps before training."
        
        if len(outliers) > total_rows * 0.05:
            return "High number of outliers. Review flagged data points before training."
        
        if len(outliers) > 0:
            return f"Found {len(outliers)} outliers. Review them, but okay to proceed with training."
        
        return "Data looks clean. Ready for training."