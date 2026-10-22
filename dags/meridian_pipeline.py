"""
dags/meridian_pipeline.py
--------------------------
Meridian End-to-End Pipeline — Session 1 + Session 2

Flow:
  extract_customers ──┐
  extract_inventory ──┼──► validate_with_gx ──┬──► [PASS] transform ──► load_to_warehouse
  extract_sales    ──┘                         │
                                               └──► [FAIL] quarantine ──► ai_explain ──► slack_alert

Folder layout:
  dags/meridian_pipeline.py
  include/data/sales.csv
  include/data/customers.csv
  include/data/inventory.csv
  include/gx_checks.py
  include/ai_explainer.py
  include/slack_notifier.py
  include/quarantine/          <- failed rows written here
"""

import sys
import os
sys.path.insert(0, "/usr/local/airflow/include")

from airflow.decorators import dag, task
import pendulum
from datetime import timedelta
import pandas as pd
import duckdb
import json

DATA_DIR       = "/usr/local/airflow/include/data"
WAREHOUSE_DB   = "/usr/local/airflow/include/warehouse_mahima_dalvi.duckdb"
QUARANTINE_DIR = "/usr/local/airflow/include/quarantine"

default_args = {
    "owner": "fde-cohort",
    "retries": 2,
    "retry_delay": timedelta(seconds=30),
}


@dag(
    dag_id="meridian_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 7, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["fde", "meridian", "session1", "session2"],
)
def meridian_pipeline():

    # ── EXTRACT (run in parallel) ─────────────────────────────────────────────

    @task
    def extract_sales():
        df = pd.read_csv(f"{DATA_DIR}/sales.csv")
        print(f"[Extract] {len(df)} sales rows loaded")
        return df.to_json()

    @task
    def extract_customers():
        df = pd.read_csv(f"{DATA_DIR}/customers.csv")
        print(f"[Extract] {len(df)} customer rows loaded")
        return df.to_json()

    @task
    def extract_inventory():
        df = pd.read_csv(f"{DATA_DIR}/inventory.csv")
        print(f"[Extract] {len(df)} inventory rows loaded")
        return df.to_json()

    # ── VALIDATE WITH GREAT EXPECTATIONS ─────────────────────────────────────

    @task
    def validate_with_gx(sales_json, customers_json, inventory_json):
        from gx_checks import run_gx_checks
        result = run_gx_checks(sales_json, customers_json, inventory_json)
        print(f"[GX] Passed: {result['passed_count']} | Failed: {result['failed_count']}")
        return result

    # ── QUARANTINE: write failed rows to file ─────────────────────────────────

    @task
    def quarantine_failed(gx_result: dict):
        os.makedirs(QUARANTINE_DIR, exist_ok=True)

        if gx_result["failed_count"] == 0:
            print("[Quarantine] No failed rows. Skipping.")
            return gx_result

        failed_df = pd.read_json(gx_result["failed"])
        ts = pendulum.now("UTC").strftime("%Y%m%d_%H%M%S")

        # write failed rows CSV
        path = f"{QUARANTINE_DIR}/failed_{ts}.csv"
        failed_df.to_csv(path, index=False)
        print(f"[Quarantine] {gx_result['failed_count']} rows written to {path}")

        # write summary JSON
        summary_path = f"{QUARANTINE_DIR}/summary_{ts}.json"
        with open(summary_path, "w") as f:
            json.dump({
                "failed_count": gx_result["failed_count"],
                "passed_count": gx_result["passed_count"],
                "summary":      gx_result["summary"],
                "checked_at":   gx_result["checked_at"],
            }, f, indent=2)

        return gx_result

    # ── AI EXPLAINER: LangGraph generates plain-English Slack message ──────────

    @task
    def ai_explain(gx_result: dict):
        from ai_explainer import explain_failures
        message = explain_failures(gx_result)
        if message:
            print(f"[AI] Generated alert:\n{message}")
        else:
            print("[AI] All rows passed — no alert needed.")
        return message

    # ── SLACK ALERT ───────────────────────────────────────────────────────────

    @task
    def slack_alert(message: str):
        from airflow.sdk import Variable
        from slack_notifier import send_slack_alert
        if not message:
            print("[Slack] No failures to report.")
            return
        webhook_url = Variable.get("SLACK_WEBHOOK_URL", default=None)
        send_slack_alert(message, webhook_url=webhook_url)

    # ── TRANSFORM: only clean/passed rows ────────────────────────────────────

    @task
    def transform(gx_result: dict):
        if gx_result["passed_count"] == 0:
            print("[Transform] No clean rows to transform.")
            return "{}"

        sales     = pd.read_json(gx_result["passed"])
        customers = pd.read_json(gx_result["customers"])
        inventory = pd.read_json(gx_result["inventory"])

        merged = sales.merge(customers, on="customer_id", how="left")
        merged = merged.merge(inventory, on="product_id", how="left")

        print(f"[Transform] {len(merged)} clean rows ready for warehouse")
        return merged.to_json()

    # ── LOAD TO WAREHOUSE ─────────────────────────────────────────────────────

    @task
    def load_to_warehouse(transformed_json: str):
        if transformed_json == "{}":
            print("[Load] Nothing to load.")
            return

        merged = pd.read_json(transformed_json)

        # Add ingestion timestamp so we know which run each row came from
        merged["ingested_at"] = pendulum.now("UTC").to_iso8601_string()

        con = duckdb.connect(WAREHOUSE_DB)

        # Create table if first run, otherwise append
        con.execute("""
            CREATE TABLE IF NOT EXISTS meridian_daily AS
            SELECT * FROM merged WHERE 1=0
        """)

        # Avoid duplicate order_ids across runs
        existing_ids = con.execute(
            "SELECT order_id FROM meridian_daily"
        ).fetchdf()["order_id"].tolist()

        new_rows = merged[~merged["order_id"].isin(existing_ids)]

        if len(new_rows) > 0:
            con.execute("INSERT INTO meridian_daily SELECT * FROM new_rows")
            print(f"[Load] {len(new_rows)} new rows appended")
        else:
            print("[Load] No new rows to append — all order_ids already exist")

        row_count = con.execute("SELECT COUNT(*) FROM meridian_daily").fetchone()[0]
        con.close()
        print(f"[Load] Total rows in warehouse: {row_count}")

    # ── TRAIN MODEL + PREDICT ─────────────────────────────────────────────────

    @task
    def train_and_predict():
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import mean_absolute_error, r2_score

        print("[ML] Loading data from warehouse...")
        con = duckdb.connect(WAREHOUSE_DB)
        df  = con.execute("SELECT * FROM meridian_daily").fetchdf()

        if len(df) < 5:
            print("[ML] Not enough data to train. Skipping.")
            con.close()
            return

        # ── Feature engineering ───────────────────────────────────────────
        df["order_date"]  = pd.to_datetime(df["order_date"], unit="ms")
        df["day_of_week"] = df["order_date"].dt.dayofweek
        df["month"]       = df["order_date"].dt.month

        le = LabelEncoder()
        df["region_enc"]   = le.fit_transform(df["region"])
        df["category_enc"] = le.fit_transform(df["category"])

        features = ["day_of_week", "month", "region_enc", "category_enc"]
        X = df[features]
        y = df["revenue"]

        # ── Train ─────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)

        mae = mean_absolute_error(y_test, model.predict(X_test))
        r2  = r2_score(y_test, model.predict(X_test))
        print(f"[ML] Trained on {len(df)} rows | MAE: £{mae:.2f} | R²: {r2:.2f}")

        # ── Predict next 7 days ───────────────────────────────────────────
        next_week = pd.DataFrame({
            "day_of_week":  list(range(7)),
            "month":        [df["month"].max()] * 7,
            "region_enc":   [0] * 7,
            "category_enc": [0] * 7,
        })
        preds       = model.predict(next_week)
        weekly_total = float(round(preds.sum(), 2))
        run_date    = pendulum.now("UTC").to_date_string()

        print(f"[ML] Predicted weekly revenue: £{weekly_total}")

        # ── Save prediction to DuckDB ─────────────────────────────────────
        pred_df = pd.DataFrame([{
            "run_date":           run_date,
            "rows_trained_on":    len(df),
            "predicted_weekly_revenue": weekly_total,
            "mae":                round(mae, 2),
            "r2":                 round(r2, 2),
        }])

        con.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                run_date                  VARCHAR,
                rows_trained_on           INTEGER,
                predicted_weekly_revenue  DOUBLE,
                mae                       DOUBLE,
                r2                        DOUBLE
            )
        """)
        con.execute("INSERT INTO ml_predictions SELECT * FROM pred_df")

        all_preds = con.execute("SELECT * FROM ml_predictions ORDER BY run_date").fetchdf()
        print(f"\n[ML] Prediction history:\n{all_preds.to_string(index=False)}")
        con.close()

    # ── DAG WIRING ────────────────────────────────────────────────────────────
    #
    #   extract_sales ──┐
    #   extract_customers ──┼──► validate_with_gx
    #   extract_inventory ──┘         │
    #                          ┌──────┴──────┐
    #                          │             │
    #                   quarantine_failed  transform
    #                          │             │
    #                     ai_explain    load_to_warehouse
    #                          │
    #                     slack_alert
    #
    # ─────────────────────────────────────────────────────────────────────────

    sales     = extract_sales()
    customers = extract_customers()
    inventory = extract_inventory()

    gx_result = validate_with_gx(sales, customers, inventory)

    # Failure branch → quarantine → AI → Slack
    quarantined = quarantine_failed(gx_result)
    message     = ai_explain(quarantined)
    slack_alert(message)

    # Success branch → transform → warehouse → ML model
    transformed  = transform(gx_result)
    loaded       = load_to_warehouse(transformed)
    loaded >> train_and_predict()


meridian_pipeline()
