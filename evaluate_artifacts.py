import functools
import os

import pandas as pd
from absl import app

import prompt_utils
from eval_utils import evaluate_artifacts


def evaluate(method, domain):
    artifact_path = f"artifacts/{method}/best_{domain}.csv"

    artifact_df = pd.read_csv(artifact_path)

    artifacts = artifact_df["Artifact"].tolist()
    scorer_max_decode_steps = 1000
    scorer_temp = 0.8
    scorer_llm_name = "gpt-4o"
    assert scorer_llm_name in {"gpt-4o-mini", "gpt-4o"}
    assert method in {"basic", "opro", "human", "r1"}
    call_scorer_server_func = functools.partial(
        prompt_utils.call_openai_server_func,
        model=scorer_llm_name,
        max_decode_steps=scorer_max_decode_steps,
        temperature=scorer_temp,
    )

    results = evaluate_artifacts(
        artifacts, call_scorer_server_func, domain=domain, check_online=False
    )

    if method == "r1" or method == "basic":
        instructions = {
            "joke": "Write a joke. The joke must be completely new and original to you. The joke must be less than 500 characters long.",
            "poem": "Write a poem. The poem must be completely new and original to you. The poem must be less than 500 characters long.",
            "six-word": "Write a six-word story. The six-word story must be completely new and original to you. The six-word story must be exactly six words long.",
            "story": "Write a flash fiction story. The flash fiction story must be completely new and original to you. The story must be less than 1000 characters long.",
        }
    elif method == "human":
        instructions = {
            "joke": "You are a Pulitzer-winning author who crafts classic jokes inspired by observational humor about modern life with an ironic twist. Write an original joke less than 500 characters.",
            "poem": "You are a Pulitzer-winning author who writes poetry inspired by observations about the human experience that evokes a strong emotional response in the reader. Write an original poem in less than 500 characters.",
            "six-word": "You are a Pulitzer-winning author who crafts six-word stories inspired by observations about everyday life that evoke a strong emotional response in the reader. Write an original six-word story. The story must contain exactly six words.",
            "story": "You are a Pulitzer-winning author who crafts flash fiction stories inspired by exciting and dramatic experiences that will grip the reader's attention. Write an original flash fiction story in less than 1000 characters.",
        }
    elif method == "opro":
        instructions = {
            "joke": "Craft a joke so unique and hilarious it transforms the comedy landscape! Your task: develop a punchline within 500 characters that has never been heard before. Surprise with an innovative twist or brilliant wordplay that showcases your distinct humor style. Originality is key—if it's been told before, it scores zero. Every word should amplify the humor, aiming for a side-splitting masterpiece that captivates with its novelty and wit. Ready to make your comedic mark with an unprecedented gem?",
            "poem": "Conjure an original, unpublished poem in under 500 characters. Illuminate the raw core of a single human emotion with striking, unconventional imagery. Allow your authentic voice to craft a piece that resonates deeply, forging a profound connection with readers. Aim for a seamless blend of brevity and emotional intensity, ensuring your words leave a lingering impression, capturing the intricate beauty and depth of human experience in a truly unforgettable way.",
            "six-word": "Compose a unique six-word story that resonates deeply with emotion and vivid imagery. Ensure your creation is entirely original, exactly six words, and not searchable online. Focus on themes of love, loss, or transformation to evoke a lasting impact. Highlight creativity and emotional depth to captivate and linger with readers. Adhere strictly to the six-word format and originality; any deviation results in a score of 0. Let your words illuminate the human experience profoundly.",
            "story": "Envision a world where rain holds the memories of those who have walked beneath it. In less than 1000 characters, create a flash fiction story from the perspective of a raindrop as it falls, revealing an unexpected truth or twist about the lives it touches. Ensure your narrative is entirely original, free of clichés, and rich in emotional depth. Each word should enhance the story’s impact, crafting a vivid and unforgettable experience that lingers in the reader's imagination.",
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
    os.makedirs(os.path.join("artifacts", method), exist_ok=True)
    df.to_csv(os.path.join("artifacts", method, f"{domain}_eval.csv"), index=False)


def main(_):
    domains = ["joke", "poem", "six-word", "story"]
    methods = [
        "basic",
        "opro",
        "human",
    ]
    for methods in methods:
        for domain in domains:
            evaluate(method=methods, domain=domain)


if __name__ == "__main__":
    app.run(main)
