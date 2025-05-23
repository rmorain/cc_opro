import hashlib
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

RATINGS = {
    "strongly disagree": 0,
    "disagree": 25,
    "neither agree or disagree": 50,
    "agree": 75,
    "strongly agree": 100,
}

PAGE_LENGTH_LIMIT = {
    "joke": 2000,
    "poem": 5000,
    "story": 8000,
    "six-word": 2000,
}

ARTIFACT_LENGTH_LIMIT = {
    "joke": 500,
    "poem": 500,
    "story": 1000,
    "six-word": 100,
}


def evaluate_single_instruction(
    instruction=None,
    call_server_func=None,
    n_artifacts=1,
    domain="joke",
    i_step=None,
):
    # Generate n artifacts
    artifacts = []
    for _ in range(n_artifacts):
        artifact = call_server_func(instruction)[0]
        artifacts.append(artifact)
    # Evaluate the artifacts
    results = evaluate_artifacts(artifacts, call_server_func, domain)
    # Make a dataframe
    df = pd.DataFrame(results)
    df["step"] = f"Step {i_step}"
    df["instruction"] = instruction
    # Assuming 'artifact' and 'domain' should be first
    fixed_columns = ["step", "instruction", "artifact", "domain"]
    other_columns = [col for col in df.columns if col not in fixed_columns]
    df = df[fixed_columns + other_columns]
    return df


def evaluate_artifacts(artifacts, call_server_func, domain, check_online=True):
    results = []
    for artifact in artifacts:
        if domain == "six-word" and len(artifact.split()) != 6:
            print("Six-word artifact must have exactly 6 words, skipped")
            result = get_zero_score_result(artifact, domain, "not six words")
        elif len(artifact) > ARTIFACT_LENGTH_LIMIT[domain]:
            print("Artifact too long, skipped")
            result = get_zero_score_result(artifact, domain, "too long")
        elif check_online and is_artifact_online(artifact, domain):
            result = get_zero_score_result(artifact, domain, "online")
        else:
            result = evaluate_artifact(artifact, call_server_func, domain)
        results.append(result)
    return results


def get_zero_score_result(artifact, domain, reason):
    eval_dir = f"prompts/{domain}/evals"
    evals_files = os.listdir(eval_dir)
    categories = [eval_file.split(".")[0] for eval_file in evals_files]
    category_scores = {category: 0 for category in categories}
    category_scores["artifact"] = artifact
    category_scores["domain"] = domain
    category_scores["valid"] = f"Invalid: {reason}"
    return category_scores


def evaluate_artifact(artifact, call_server_func, domain):
    # Get list of files in prompts/{domain}/evals
    eval_dir = f"prompts/{domain}/evals"
    evals_files = os.listdir(eval_dir)
    categories = [eval_file.split(".")[0] for eval_file in evals_files]
    scores = []
    for eval_file in evals_files:
        # Read the file
        with open(os.path.join(eval_dir, eval_file), "r") as f:
            # Instruction to be a critical judge
            eval_prompt = f.read()
            # Replace <INS> with the artifact
            eval_prompt = eval_prompt.replace("<INS>", artifact)
            result = call_server_func(eval_prompt)[0]
            processed_result = process_result(result)
            if processed_result == -1:
                # Try again
                result = call_server_func(eval_prompt)[0]
                processed_result = process_result(result)
                if processed_result == -1:
                    raise ValueError(f"Invalid result: {result}")
            scores.append(processed_result)
    category_scores = dict(zip(categories, scores))
    category_scores["artifact"] = artifact
    category_scores["domain"] = domain
    category_scores["valid"] = "Valid"
    return category_scores


def process_result(result):
    result = result.lower().strip()
    if result in RATINGS:
        return RATINGS[result]
    else:
        # Check if result has valid substring
        if "disagree" in result:
            return RATINGS["disagree"]
        elif "agree" in result:  # Assume agree
            return RATINGS["agree"]
        print(f"Invalid result: {result}")
        return -1


def instruction_to_filename(instruction):
    """Convert an instruction string to filename."""
    m = hashlib.md5()
    m.update(instruction.encode("ascii", "ignore"))  # ignore non-ASCII characters
    filename = m.hexdigest()
    return filename


def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res.get("items", [])


def is_artifact_online(artifact, domain, num_results=3):
    """
    Search for a artifact online and return True if similar matches are found.
    """
    # Clean the artifact text
    cleaned_artifact = re.sub(r"[^\w\s]", "", artifact.lower())

    try:
        # Search Google
        search_results = google_search(
            cleaned_artifact,
            api_key=os.environ.get("GOOGLE_API"),
            cse_id=os.environ.get("SEARCH_ID"),
            num=num_results,
        )

        # Convert iterator to list to check if any results exist
        if not search_results:
            return False

        # Verify the artifact is actually on the web page
        for result in search_results:
            link = result.get("link")
            if not validate_url(link):
                continue
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                response = requests.get(link, headers=headers, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    page_text = soup.get_text().lower()
                    # Clean and strip new lines, punctuation, and whitespace from page_text
                    clean_page_text = re.sub(
                        r"[^\w\s]", "", page_text.replace("\n", " ")
                    ).replace(" ", "")
                    super_clean_artifact = re.sub(
                        r"[^\w\s]", "", cleaned_artifact.replace("\n", " ")
                    ).replace(" ", "")
                    # Trim string to 2000 characters
                    clean_page_text = clean_page_text[: PAGE_LENGTH_LIMIT[domain]]
                    if super_clean_artifact in clean_page_text:
                        return True
            except Exception:  # Check next search result
                pass
        return False

    except Exception as e:
        print(f"Error during search: {e}")
        return False


def validate_url(url):
    if not url:
        return False
    if not url.startswith(("http://", "https://")):
        return False
    return True
