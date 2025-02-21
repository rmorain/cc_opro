import os

import pandas as pd

domain = "story"
method = "r1"
df = pd.read_csv(f"artifacts/r1/{domain}_eval.csv")

best_artifacts = []
for i in range(10):
    group = df.iloc[10 * i : 10 * i + 10]
    best_artifact = group.loc[group["score"].idxmax()]
    best_artifacts.append(best_artifact)
best_artifacts_df = pd.DataFrame(best_artifacts)
best_artifacts_df.to_csv(
    os.path.join("artifacts", method, f"best_{domain}.csv"), index=False
)
