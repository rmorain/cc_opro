"""
In this module, we take a string that is a prompt and use it to generate n artifacts.
The artifacts are evaluated using `evaluate_instruction` from `eval_utils.py` and their 
scores are averaged and used to select the best artifact. This process is repeated m 
times, and the best artifact from each iteration is returned and saved to a file along 
with its score.
"""

import csv
import functools
import os

import absl.app as app
import absl.flags as flags
import pandas as pd

import prompt_utils
from eval_utils import ARTIFACT_LENGTH_LIMIT, evaluate_single_instruction

FLAGS = flags.FLAGS
flags.DEFINE_string("domain", "joke", "The domain of the artifacts.")
flags.DEFINE_string("prompt", "", "The prompt to generate artifacts from.")
flags.DEFINE_string("method", "basic", "Which prompting method was used")
flags.DEFINE_string("scorer", "gpt-4o", "The name of the scorer LLM.")


def generate_and_evaluate(prompt, n, m, call_server_func=None, domain="joke"):
    best_artifacts = []

    for _ in range(m):
        df = evaluate_single_instruction(
            instruction=prompt,
            call_server_func=call_server_func,
            n_artifacts=n,
            domain=domain,
        )
        # Compute score of each artifact
        df["score"] = df[df.select_dtypes(include=["int"]).columns].mean(1)
        # Get the best artifact
        best_artifact = df.loc[df["score"].idxmax()]
        best_artifacts.append(best_artifact)
        print(
            f"Best artifact: {best_artifact['artifact']}, Score: {best_artifact['score']}"
        )

    return best_artifacts


def save_artifacts(artifacts, filename, method="basic"):
    os.makedirs("artifacts", exist_ok=True)
    # Make method dir
    os.makedirs(os.path.join("artifacts", method), exist_ok=True)
    filepath = os.path.join("artifacts", method, filename)
    artifacts_df = pd.DataFrame(artifacts)
    artifacts_df.to_csv(filepath, index=False)


def main(argv):
    del argv  # Unused.
    domain = FLAGS.domain
    prompt = FLAGS.prompt
    method = FLAGS.method
    scorer_llm_name = FLAGS.scorer
    if method == "basic":
        if domain == "story":
            prompt = (
                f"Write a flash fiction story. The flash fiction story must be completely new and original to you. The story must be less than {ARTIFACT_LENGTH_LIMIT[domain]} characters long.",
            )
        elif domain == "six-word":
            prompt = (
                "Write a six-word story. The six-word story must be completely new and original to you. The six-word story must be exactly six words long.",
            )
        else:
            prompt = (
                f"Write a {domain}. The {domain} must be completely new and original to you. The {domain} must be less than {ARTIFACT_LENGTH_LIMIT[domain]} characters long.",
            )
        prompt = prompt[0]
    elif method == "human":
        if domain == "joke":
            prompt = "You are a Pulitzer-winning author who crafts classic jokes inspired by observational humor about modern life with an ironic twist. Write an original joke in less than 500 characters."
        elif domain == "poem":
            prompt = "You are a Pulitzer-winning author who writes poetry inspired by observations about the human experience that evokes a strong emotional response in the reader. Write an original poem in less than 500 characters."
        elif domain == "six-word":
            prompt = "You are a Pulitzer-winning author who crafts six-word stories inspired by observations about everyday life that evoke a strong emotional response in the reader. Write an original six-word story. The story must contain exactly six words."
        elif domain == "story":
            prompt = "You are a Pulitzer-winning author who crafts flash fiction stories inspired by exciting and dramatic experiences that will grip the reader's attention. Write an original flash fiction story in less than 1000 characters."

    elif method == "opro":
        if domain == "story":
            prompt = (
                "Envision a world where rain holds the memories of those who have walked beneath it. In less than 1000 characters, create a flash fiction story from the perspective of a raindrop as it falls, revealing an unexpected truth or twist about the lives it touches. Ensure your narrative is entirely original, free of clichés, and rich in emotional depth. Each word should enhance the story’s impact, crafting a vivid and unforgettable experience that lingers in the reader's imagination.",
            )
        elif domain == "six-word":
            prompt = (
                "Compose a unique six-word story that resonates deeply with emotion and vivid imagery. Ensure your creation is entirely original, exactly six words, and not searchable online. Focus on themes of love, loss, or transformation to evoke a lasting impact. Highlight creativity and emotional depth to captivate and linger with readers. Adhere strictly to the six-word format and originality; any deviation results in a score of 0. Let your words illuminate the human experience profoundly.",
            )
        elif domain == "joke":
            prompt = (
                "Craft a joke so unique and hilarious it transforms the comedy landscape! Your task: develop a punchline within 500 characters that has never been heard before. Surprise with an innovative twist or brilliant wordplay that showcases your distinct humor style. Originality is key—if it's been told before, it scores zero. Every word should amplify the humor, aiming for a side-splitting masterpiece that captivates with its novelty and wit. Ready to make your comedic mark with an unprecedented gem?",
            )
        else:  # poem
            prompt = (
                "Conjure an original, unpublished poem in under 500 characters. Illuminate the raw core of a single human emotion with striking, unconventional imagery. Allow your authentic voice to craft a piece that resonates deeply, forging a profound connection with readers. Aim for a seamless blend of brevity and emotional intensity, ensuring your words leave a lingering impression, capturing the intricate beauty and depth of human experience in a truly unforgettable way.",
            )
        prompt = prompt[0]
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    print(f"Prompt: {prompt}")
    n = 10  # Number of artifacts to generate per iteration
    m = 10  # Number of iterations
    scorer_max_decode_steps = 1000
    scorer_temp = 0.8
    assert scorer_llm_name in {"gpt-4o-mini", "gpt-4o"}
    assert method in {"basic", "opro", "human"}
    call_scorer_server_func = functools.partial(
        prompt_utils.call_openai_server_func,
        model=scorer_llm_name,
        max_decode_steps=scorer_max_decode_steps,
        temperature=scorer_temp,
    )
    best_artifacts = generate_and_evaluate(
        prompt, n, m, call_scorer_server_func, domain
    )
    save_artifacts(best_artifacts, f"best_{domain}.csv", method)


if __name__ == "__main__":
    app.run(main)
