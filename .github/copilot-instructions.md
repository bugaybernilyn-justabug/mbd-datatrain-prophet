# Copilot Instructions: Blood Supply & Demand Forecasting Pipeline

## 1. Context & Architecture
We are building an automated Machine Learning pipeline for a web application. 
The system must predict future blood **Supply** and **Demand** using historical CSV data and live Firebase data. 
The architecture relies on a Python script executed monthly via GitHub Actions, which outputs static JSON files for a Vanilla JS/Tailwind frontend to consume.

## 2. The Datasets
We have three local CSV files that need to be parsed and preprocessed:

**Supply Data 1:** `ByMonthDataset2024.csv`
**Supply Data 2:** `ByMonthDataset2025.csv`
* *Format:* Transactional (One row per donor).
* *Columns:* `Date`, `Location`, `ABO/hr`, `Birthdate`, `Age`, `Gender`, `Address`, `Donor Serial`, `Donor Type`.

**Demand Data:** `Blood-Request-Demand-2024-2025-Blood-Demand-Data.csv`
* *Format:* Aggregated Monthly.
* *Columns:* `Year`, `Month`, `Total`, `O-`, `O+`, `A-`, `A+`, `B-`, `B+`, `AB-`, `AB+`.

## 3. Instructions for Python Script (`generate_forecast.py`)
Write a Python script that accomplishes the following:

**A. Dependency Imports**
* Use `pandas`, `prophet`, `firebase_admin`, `json`, and `os`.

**B. Supply Data Preprocessing**
* Load both 2024 and 2025 Supply CSVs and concatenate them.
* Convert the `Date` column to datetime.
* Group the data by `Date` and count the number of rows to get the daily total donations.
* Rename the grouped date column to `ds` and the total count to `y` (Prophet's required format).

**C. Demand Data Preprocessing**
* Load the Demand CSV.
* Combine the `Year` and `Month` columns into a single datetime column representing the first day of that month (e.g., "2024", "January" -> "2024-01-01").
* Rename this new datetime column to `ds` and the `Total` column to `y`.

**D. Firebase Integration (Live Data)**
* Initialize Firebase using a JSON string from the environment variable `FIREBASE_CREDENTIALS`. Do not use a local file path.
* Fetch recent donation records from the `donations` collection.
* Process the Firebase data into the `ds` and `y` format and append it to the historical Supply dataframe. Gracefully handle errors if the collection is empty.

**E. Model Training & Export**
* Initialize two separate `Prophet()` models: one for Supply, one for Demand.
* Fit the models on their respective dataframes.
* Create a future dataframe for the next 12 months (`freq='M'`).
* Predict future values for both models.
* Export the results to `forecast_supply.json` and `forecast_demand.json`. Ensure the JSON format is an array of objects: `[{"date": "YYYY-MM-DD", "value": 123}]`. Use the `yhat` column for the value, and round it to a whole number.

## 4. Instructions for GitHub Action (`.github/workflows/update_prediction.yml`)
Write the YAML configuration for GitHub Actions:
* **Triggers:** Run on schedule (1st of every month at 00:00 UTC) and manual `workflow_dispatch`.
* **Permissions:** Explicitly grant `contents: write`.
* **Steps:**
    1. Checkout repository.
    2. Setup Python 3.9.
    3. Install dependencies (`pip install pandas prophet firebase-admin`).
    4. Run `generate_forecast.py` (Pass the `FIREBASE_CREDENTIALS` from GitHub Secrets).
    5. Commit and push the updated JSON files back to the repository using standard Git commands.

## 5. Constraints & Best Practices
* **Robustness:** Add print statements in the Python script to log the shape of the dataframes. This helps with debugging in GitHub Actions.
* **Security:** Never hardcode credentials.
* **Clean Code:** Use modular functions in the Python script (e.g., `process_supply_data()`, `train_and_predict()`) rather than a single monolithic block.