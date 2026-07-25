import os
import sqlite3
import pandas as pd

DB_PATH = "mlflow.db"
OUTPUT = "results/metrics.csv"

os.makedirs("results", exist_ok=True)

conn = sqlite3.connect(DB_PATH)

# -------------------------------------------------------
# Все успешные run с dataset_size
# -------------------------------------------------------

runs = pd.read_sql("""
SELECT
    r.run_uuid,
    r.start_time,
    p.value AS dataset_size
FROM runs r
JOIN params p
    ON r.run_uuid = p.run_uuid
WHERE
    r.status = 'FINISHED'
    AND p.key = 'dataset_size'
""", conn)

runs["dataset_size"] = runs["dataset_size"].astype(int)

# -------------------------------------------------------
# Берём последний FINISHED run для каждого размера
# -------------------------------------------------------

runs = (
    runs.sort_values("start_time")
        .groupby("dataset_size")
        .tail(1)
)

# Оставляем только нужные размеры

runs = runs[runs["dataset_size"].isin([128, 256, 512])]

print("Используем следующие run:")
print(runs[["dataset_size", "run_uuid"]])

# -------------------------------------------------------
# Загружаем метрики
# -------------------------------------------------------

dfs = []

for _, row in runs.iterrows():

    metrics = pd.read_sql("""
    SELECT
        step,
        key,
        value
    FROM metrics
    WHERE
        run_uuid = ?
        AND key IN ('overall_score', 'overall_nrmse')
    """, conn, params=(row.run_uuid,))

    metrics = metrics.pivot(
        index="step",
        columns="key",
        values="value"
    ).reset_index()

    metrics.rename(columns={"step": "global_step"}, inplace=True)

    metrics["dataset_size"] = row.dataset_size
    metrics["run_uuid"] = row.run_uuid

    dfs.append(metrics)

df = pd.concat(dfs, ignore_index=True)

df = df[
    [
        "dataset_size",
        "run_uuid",
        "global_step",
        "overall_score",
        "overall_nrmse",
    ]
]

df = df.sort_values(
    [
        "dataset_size",
        "global_step",
    ]
)

df.to_csv(OUTPUT, index=False)

print("\nКоличество точек:")
print(df.groupby("dataset_size").size())

print(f"\nФайл сохранён: {OUTPUT}")