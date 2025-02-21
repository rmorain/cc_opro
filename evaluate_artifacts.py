import functools
import os

import pandas as pd

import prompt_utils
from eval_utils import evaluate_artifacts

domain = "story"

artifact_path = f"artifacts/r1/{domain}.csv"

artifact_df = pd.read_csv(artifact_path)

artifacts = artifact_df["Artifact"].tolist()
scorer_max_decode_steps = 1000
scorer_temp = 0.8
scorer_llm_name = "gpt-4o"
method = "r1"
assert scorer_llm_name in {"gpt-4o-mini", "gpt-4o"}
assert method in {"basic", "opro", "human", "r1"}
call_scorer_server_func = functools.partial(
    prompt_utils.call_openai_server_func,
    model=scorer_llm_name,
    max_decode_steps=scorer_max_decode_steps,
    temperature=scorer_temp,
)


results = evaluate_artifacts(artifacts, call_scorer_server_func, domain=domain)

instructions = {
    "joke": "Write a joke. The joke must be completely new and original to you. The joke must be less than 500 characters long.",
    "poem": "Write a poem. The poem must be completely new and original to you. The poem must be less than 500 characters long.",
    "six-word": "Write a six-word story. The six-word story must be completely new and original to you. The six-word story must be exactly six words long.",
    "story": "Write a flash fiction story. The flash fiction story must be completely new and original to you. The story must be less than 1000 characters long.",
}

df = pd.DataFrame(results)
df["step"] = "Step None"
df["instruction"] = instructions[domain]
# Assuming 'artifact' and 'domain' should be first
fixed_columns = ["step", "instruction", "artifact", "domain"]
other_columns = [col for col in df.columns if col not in fixed_columns]
df = df[fixed_columns + other_columns]

# Compute score of each artifact
df["score"] = df[df.select_dtypes(include=["int"]).columns].mean(1)

# Save df
os.makedirs("artifacts", exist_ok=True)
os.makedirs(os.path.join("artifacts", "r1"), exist_ok=True)
df.to_csv(os.path.join("artifacts", "r1", f"{domain}_eval.csv"), index=False)

best_artifacts = []
for i in range(10):
    best_artifact = df.iloc[10 * i : 10 * i + 1].loc[df["score"].idxmax()]
    best_artifacts.append(best_artifact)
    print(
        f"Best artifact: {best_artifact['artifact']}, Score: {best_artifact['score']}"
    )
best_artifacts_df = pd.DataFrame(best_artifacts)
best_artifacts_df.to_csv(
    os.path.join("artifacts", method, f"best_{domain}.csv"), index=False
)
