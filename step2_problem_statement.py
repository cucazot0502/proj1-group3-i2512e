# ============================================================
# AIR POLLUTION DATASET - DATA QUALITY INSPECTION
# Purpose:
# - Identify data quality issues in the raw air pollution dataset
# - Provide evidence for the Problem Statement
# - Support preprocessing, EDA, visualization, and ML preparation
# ============================================================


# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

sns.set_style("whitegrid")


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "air_pollution_dataset_10percent_dirty_100dup_final.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
display(df.head())

print("\nData types:")
display(df.dtypes)


# ============================================================
# 2. BASIC DATASET OVERVIEW
# ============================================================

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nDataset information:")
df.info()

print("\nStatistical summary of available numeric columns:")
display(df.describe())

print("\nStatistical summary of categorical/object columns:")
display(df.describe(include="object"))


# ============================================================
# 3. ISSUE 1 - PRESENCE OF MISSING VALUES
# ============================================================

missing_summary = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percentage": (df.isnull().sum() / len(df)) * 100
})

missing_summary = missing_summary.sort_values(
    by="missing_count",
    ascending=False
)

print("Missing value summary:")
display(missing_summary)

columns_with_missing = missing_summary[missing_summary["missing_count"] > 0]

print("Columns with missing values:")
display(columns_with_missing)

# Visualization: Missing values per column
if len(columns_with_missing) > 0:
    plt.figure(figsize=(10, 5))
    columns_with_missing["missing_count"].plot(kind="bar")
    plt.title("Missing Values per Column")
    plt.xlabel("Column")
    plt.ylabel("Missing Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found.")


# ============================================================
# 4. ISSUE 2 - HIDDEN MISSING VALUES / INVALID VALUES
# ============================================================
# Some columns should be numeric, but may contain invalid text values
# such as "high", "unknown", "error", "bad", "invalid", etc.

expected_numeric_cols = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "AQI", "Temperature", "Humidity", "WindSpeed"
]

# Keep only columns that actually exist in the dataset
expected_numeric_cols = [col for col in expected_numeric_cols if col in df.columns]

invalid_numeric_report = []

for col in expected_numeric_cols:
    converted = pd.to_numeric(df[col], errors="coerce")
    
    # Invalid values are non-null original values that become NaN after conversion
    invalid_mask = converted.isna() & df[col].notna()
    
    invalid_count = invalid_mask.sum()
    invalid_percentage = invalid_count / len(df) * 100
    invalid_examples = df.loc[invalid_mask, col].unique()[:10]
    
    invalid_numeric_report.append({
        "column": col,
        "invalid_count": invalid_count,
        "invalid_percentage": invalid_percentage,
        "invalid_examples": invalid_examples
    })

invalid_numeric_report = pd.DataFrame(invalid_numeric_report)

print("Hidden invalid values inside numeric-like columns:")
display(invalid_numeric_report)

# Show example invalid rows for each numeric-like column
for col in expected_numeric_cols:
    converted = pd.to_numeric(df[col], errors="coerce")
    invalid_mask = converted.isna() & df[col].notna()
    
    if invalid_mask.sum() > 0:
        print(f"\nInvalid values found in column: {col}")
        display(df.loc[invalid_mask, [col]].head(10))


# ============================================================
# 5. ISSUE 3 - MIXED DATA TYPES PREVENTING NUMERICAL COMPUTATION
# ============================================================

mixed_type_check = pd.DataFrame({
    "column": expected_numeric_cols,
    "current_dtype": [df[col].dtype for col in expected_numeric_cols],
    "should_be_numeric": True
})

print("Mixed type check for expected numeric columns:")
display(mixed_type_check)

object_numeric_cols = [
    col for col in expected_numeric_cols
    if df[col].dtype == "object"
]

print("Columns expected to be numeric but stored as object:")
print(object_numeric_cols)

conversion_summary = []

for col in expected_numeric_cols:
    converted = pd.to_numeric(df[col], errors="coerce")
    
    conversion_summary.append({
        "column": col,
        "original_dtype": df[col].dtype,
        "can_be_converted_to_numeric": pd.api.types.is_numeric_dtype(converted),
        "original_missing_values": df[col].isna().sum(),
        "missing_after_conversion": converted.isna().sum(),
        "values_lost_after_conversion": converted.isna().sum() - df[col].isna().sum()
    })

conversion_summary = pd.DataFrame(conversion_summary)

print("Numeric conversion summary:")
display(conversion_summary)


# ============================================================
# 6. ISSUE 4 - INVALID CATEGORICAL VALUES
# ============================================================

categorical_cols = ["Country", "City", "Status", "Station_ID"]
categorical_cols = [col for col in categorical_cols if col in df.columns]

for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].dropna().unique()[:50])
    print("Number of unique values:", df[col].nunique(dropna=False))


# ------------------------------------------------------------
# 6.1 Check invalid Country, City, Status
# ------------------------------------------------------------

# Adjust these lists if your dataset contains more valid countries/cities/statuses
valid_countries = ["USA", "India", "China", "Vietnam", "France"]
valid_cities = ["Hanoi", "HCM", "Delhi", "Beijing", "Paris"]
valid_status = ["Good", "Moderate", "Unhealthy"]

if "Country" in df.columns:
    invalid_country = df[
        ~df["Country"].isin(valid_countries) & df["Country"].notna()
    ]
else:
    invalid_country = pd.DataFrame()

if "City" in df.columns:
    invalid_city = df[
        ~df["City"].isin(valid_cities) & df["City"].notna()
    ]
else:
    invalid_city = pd.DataFrame()

if "Status" in df.columns:
    invalid_status = df[
        ~df["Status"].isin(valid_status) & df["Status"].notna()
    ]
else:
    invalid_status = pd.DataFrame()

print("Invalid Country rows:", len(invalid_country))
if len(invalid_country) > 0:
    display(invalid_country[["Country", "City", "Status"]].head())

print("Invalid City rows:", len(invalid_city))
if len(invalid_city) > 0:
    display(invalid_city[["Country", "City", "Status"]].head())

print("Invalid Status rows:", len(invalid_status))
if len(invalid_status) > 0:
    display(invalid_status[["Country", "City", "Status"]].head())


# ------------------------------------------------------------
# 6.2 Check invalid Station_ID
# ------------------------------------------------------------

if "Station_ID" in df.columns:
    station_id_as_str = df["Station_ID"].astype(str)
    
    invalid_station_id = df[
        df["Station_ID"].notna()
        & ~station_id_as_str.str.fullmatch(r"\d+")
    ]
else:
    invalid_station_id = pd.DataFrame()

print("Invalid Station_ID rows:", len(invalid_station_id))
if len(invalid_station_id) > 0:
    display(invalid_station_id[["Station_ID"]].head(10))


# ============================================================
# 7. ISSUE 5 - INCONSISTENT DATE FORMATS
# ============================================================

if "Date" in df.columns:
    print("Sample Date values:")
    display(df["Date"].dropna().sample(
        min(20, df["Date"].dropna().shape[0]),
        random_state=42
    ).tolist())

    def detect_date_format(x):
        x = str(x)
        
        if pd.isna(x) or x.lower() == "nan":
            return "missing"
        elif pd.Series([x]).str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").iloc[0]:
            return "slash_format_mm_dd_yyyy_or_dd_mm_yyyy"
        elif pd.Series([x]).str.match(r"^\d{1,2}-\d{1,2}-\d{2}$").iloc[0]:
            return "dash_format_short_year"
        elif pd.Series([x]).str.match(r"^\d{4}-\d{1,2}-\d{1,2}$").iloc[0]:
            return "iso_format_yyyy_mm_dd"
        else:
            return "unknown_format"

    df["date_format_type"] = df["Date"].apply(detect_date_format)

    print("Date format summary:")
    display(df["date_format_type"].value_counts())

    parsed_dates = pd.to_datetime(df["Date"], errors="coerce")

    invalid_dates = df[
        parsed_dates.isna() & df["Date"].notna()
    ]

    print("Invalid or unparseable date rows:", len(invalid_dates))
    if len(invalid_dates) > 0:
        display(invalid_dates[["Date"]].head(10))
else:
    print("Date column not found.")
    df["date_format_type"] = "not_available"
    invalid_dates = pd.DataFrame()


# ============================================================
# 8. ISSUE 6 - DUPLICATE RECORDS LEADING TO BIASED ANALYSIS
# ============================================================

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)
print("Duplicate percentage:", duplicate_count / len(df) * 100)

duplicate_rows = df[df.duplicated(keep=False)]

print("All duplicate-related rows shape:", duplicate_rows.shape)
if duplicate_rows.shape[0] > 0:
    display(duplicate_rows.head(20))

df_no_duplicates = df.drop_duplicates()

print("Shape before removing duplicates:", df.shape)
print("Shape after removing duplicates:", df_no_duplicates.shape)


# ============================================================
# 9. ISSUE 7 - POTENTIAL INCONSISTENCY BETWEEN AQI AND STATUS
# ============================================================
# Assumption:
# AQI <= 50          -> Good
# 51 <= AQI <= 100   -> Moderate
# AQI > 100          -> Unhealthy

if "AQI" in df.columns and "Status" in df.columns:
    df_aqi_check = df.copy()

    df_aqi_check["AQI_numeric"] = pd.to_numeric(
        df_aqi_check["AQI"],
        errors="coerce"
    )

    def expected_status_from_aqi(aqi):
        if pd.isna(aqi):
            return np.nan
        elif aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        else:
            return "Unhealthy"

    df_aqi_check["Expected_Status"] = df_aqi_check["AQI_numeric"].apply(
        expected_status_from_aqi
    )

    valid_status_mask = df_aqi_check["Status"].isin(valid_status)

    aqi_status_mismatch = df_aqi_check[
        df_aqi_check["AQI_numeric"].notna()
        & valid_status_mask
        & (df_aqi_check["Status"] != df_aqi_check["Expected_Status"])
    ]

    print("AQI-Status mismatch rows:", len(aqi_status_mismatch))
    print("Mismatch percentage:", len(aqi_status_mismatch) / len(df) * 100)

    if len(aqi_status_mismatch) > 0:
        display(
            aqi_status_mismatch[
                ["AQI", "AQI_numeric", "Status", "Expected_Status"]
            ].head(20)
        )

    print("AQI expected status vs actual status table:")
    comparison_table = pd.crosstab(
        df_aqi_check["Expected_Status"],
        df_aqi_check["Status"],
        rownames=["Expected Status"],
        colnames=["Actual Status"]
    )

    display(comparison_table)
else:
    print("AQI or Status column not found.")
    aqi_status_mismatch = pd.DataFrame()


# ============================================================
# 10. ISSUE 8 - POSSIBLE INCONSISTENCY BETWEEN COUNTRY AND CITY
# ============================================================

if "Country" in df.columns and "City" in df.columns:
    city_country_map = {
        "Hanoi": "Vietnam",
        "HCM": "Vietnam",
        "Delhi": "India",
        "Beijing": "China",
        "Paris": "France"
    }

    df_geo_check = df.copy()

    df_geo_check["Expected_Country"] = df_geo_check["City"].map(city_country_map)

    country_city_mismatch = df_geo_check[
        df_geo_check["Expected_Country"].notna()
        & df_geo_check["Country"].notna()
        & (df_geo_check["Country"] != df_geo_check["Expected_Country"])
    ]

    print("Country-City mismatch rows:", len(country_city_mismatch))
    print("Mismatch percentage:", len(country_city_mismatch) / len(df) * 100)

    if len(country_city_mismatch) > 0:
        display(
            country_city_mismatch[
                ["Country", "City", "Expected_Country"]
            ].head(20)
        )

    print("Country-City cross-tab:")
    country_city_table = pd.crosstab(df["Country"], df["City"])
    display(country_city_table)
else:
    print("Country or City column not found.")
    country_city_mismatch = pd.DataFrame()


# ============================================================
# 11. ISSUE 9 - LACK OF VISUALIZATION CLARITY FOR NON-TECHNICAL USERS
# ============================================================
# The raw table is difficult to interpret because it contains many pollutant,
# weather, date, location, and status fields. Visualizations help make trends clearer.

print("Raw data preview:")
display(df.head(10))

# Create a temporary numeric version for visualization
df_viz = df.copy()

for col in expected_numeric_cols:
    df_viz[col] = pd.to_numeric(df_viz[col], errors="coerce")


# ------------------------------------------------------------
# 11.1 AQI Distribution
# ------------------------------------------------------------

if "AQI" in df_viz.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_viz["AQI"], bins=30, kde=True)
    plt.title("AQI Distribution")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11.2 Air Quality Status Distribution
# ------------------------------------------------------------

if "Status" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Status")
    plt.title("Air Quality Status Distribution")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11.3 AQI by Status
# ------------------------------------------------------------

if "AQI" in df_viz.columns and "Status" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_viz, x=df["Status"], y="AQI")
    plt.title("AQI by Air Quality Status")
    plt.xlabel("Status")
    plt.ylabel("AQI")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11.4 Correlation Heatmap of Pollutants and AQI
# ------------------------------------------------------------

pollutant_cols = [
    col for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
    if col in df_viz.columns
]

if len(pollutant_cols) >= 2:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        df_viz[pollutant_cols].corr(),
        annot=True,
        cmap="coolwarm"
    )
    plt.title("Correlation Heatmap of Pollutants and AQI")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11.5 Average AQI by City
# ------------------------------------------------------------

if "AQI" in df_viz.columns and "City" in df.columns:
    city_aqi = df_viz.groupby(df["City"])["AQI"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    city_aqi.plot(kind="bar")
    plt.title("Average AQI by City")
    plt.xlabel("City")
    plt.ylabel("Average AQI")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11.6 Average AQI by Country
# ------------------------------------------------------------

if "AQI" in df_viz.columns and "Country" in df.columns:
    country_aqi = df_viz.groupby(df["Country"])["AQI"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    country_aqi.plot(kind="bar")
    plt.title("Average AQI by Country")
    plt.xlabel("Country")
    plt.ylabel("Average AQI")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ============================================================
# 12. FINAL ISSUE REPORT
# ============================================================

issue_report = []

# 1. Missing values
total_missing = df.isnull().sum().sum()

issue_report.append({
    "Issue": "Presence of missing values",
    "Detected": total_missing > 0,
    "Evidence": f"{total_missing} total missing values"
})

# 2. Hidden invalid numeric values
total_invalid_numeric = 0

for col in expected_numeric_cols:
    converted = pd.to_numeric(df[col], errors="coerce")
    invalid_mask = converted.isna() & df[col].notna()
    total_invalid_numeric += invalid_mask.sum()

issue_report.append({
    "Issue": "Hidden missing / invalid numeric values",
    "Detected": total_invalid_numeric > 0,
    "Evidence": f"{total_invalid_numeric} invalid numeric-like values"
})

# 3. Mixed data types
mixed_cols = [
    col for col in expected_numeric_cols
    if df[col].dtype == "object"
]

issue_report.append({
    "Issue": "Mixed data types preventing numerical computation",
    "Detected": len(mixed_cols) > 0,
    "Evidence": f"{len(mixed_cols)} expected numeric columns stored as object: {mixed_cols}"
})

# 4. Invalid categorical values
invalid_cat_count = (
    len(invalid_country)
    + len(invalid_city)
    + len(invalid_status)
    + len(invalid_station_id)
)

issue_report.append({
    "Issue": "Invalid categorical values",
    "Detected": invalid_cat_count > 0,
    "Evidence": f"{invalid_cat_count} invalid categorical-related records found"
})

# 5. Inconsistent date formats
if "date_format_type" in df.columns:
    date_format_count = df["date_format_type"].nunique()
else:
    date_format_count = 0

issue_report.append({
    "Issue": "Inconsistent date formats",
    "Detected": date_format_count > 1,
    "Evidence": f"{date_format_count} different date format types detected"
})

# 6. Duplicates
issue_report.append({
    "Issue": "Duplicate records leading to biased analysis",
    "Detected": duplicate_count > 0,
    "Evidence": f"{duplicate_count} duplicate rows found"
})

# 7. AQI vs Status mismatch
issue_report.append({
    "Issue": "Potential inconsistency between AQI and Status",
    "Detected": len(aqi_status_mismatch) > 0,
    "Evidence": f"{len(aqi_status_mismatch)} AQI-Status mismatch rows"
})

# 8. Country vs City mismatch
issue_report.append({
    "Issue": "Possible inconsistency between Country and City",
    "Detected": len(country_city_mismatch) > 0,
    "Evidence": f"{len(country_city_mismatch)} Country-City mismatch rows"
})

# 9. Visualization clarity
issue_report.append({
    "Issue": "Lack of visualization clarity for non-technical users",
    "Detected": True,
    "Evidence": "Dataset contains many pollutant, weather, date, and location fields that require charts for interpretation"
})

issue_report_df = pd.DataFrame(issue_report)

print("Final issue report:")
display(issue_report_df)


# ============================================================
# 13. FINAL CONCLUSION IN VIETNAMESE
# ============================================================

print("""
KẾT LUẬN:

Dựa trên quá trình kiểm tra dữ liệu ban đầu, bộ dữ liệu Air Pollution tồn tại nhiều vấn đề về chất lượng dữ liệu, bao gồm:

1. Giá trị thiếu trong một số cột quan trọng.
2. Giá trị không hợp lệ bị ẩn trong các cột đáng lẽ phải là dữ liệu số.
3. Nhiều cột số đang bị lưu dưới dạng object do có lẫn dữ liệu chữ.
4. Một số giá trị phân loại không hợp lệ trong các cột như Country, City, Status hoặc Station_ID.
5. Cột Date có nhiều định dạng ngày tháng không nhất quán.
6. Dataset tồn tại các bản ghi trùng lặp, có thể làm sai lệch kết quả phân tích.
7. Có khả năng không nhất quán giữa AQI và Status.
8. Có khả năng không nhất quán giữa Country và City.
9. Dữ liệu thô khó hiểu đối với người không chuyên nếu không có biểu đồ trực quan.

Vì vậy, cần thực hiện tiền xử lý dữ liệu trước khi tiến hành EDA, trực quan hóa hoặc xây dựng mô hình machine learning. Dataset cần được làm sạch, chuẩn hóa, chuyển đổi đúng kiểu dữ liệu, loại bỏ trùng lặp và kiểm tra tính hợp lý để đảm bảo kết quả phân tích chất lượng không khí và dự đoán/phân loại AQI đáng tin cậy.
""")