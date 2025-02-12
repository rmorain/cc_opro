import datetime
import functools
import os

import openai
import torch
from absl import app, flags
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import opt_utils
import prompt_utils
from eval_utils import ARTIFACT_LENGTH_LIMIT

OPRO_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

_SCORER = flags.DEFINE_string("scorer", "gpt-4o-mini", "The name of the scorer LLM.")

_OPTIMIZER = flags.DEFINE_string(
    "optimizer", "gpt-4o-mini", "The name of the optimizer LLM."
)

_SCORER_TEMPERATURE = flags.DEFINE_float(
    "s_temp", 0.8, "The temperature for the scorer LLM."
)

_OPTIMIZER_TEMPERATURE = flags.DEFINE_float(
    "o_temp", 0.8, "The temperature for the optimizer LLM."
)

_SCORER_MAX_DECODE_STEPS = flags.DEFINE_integer(
    "s_max_decode_steps", 1000, "The maximum decode steps for the scorer LLM."
)

_OPTIMIZER_MAX_DECODE_STEPS = flags.DEFINE_integer(
    "o_max_decode_steps", 1000, "The maximum decode steps for the optimizer LLM."
)

_LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

_DOMAIN = flags.DEFINE_string("domain", "joke", "The domain of the instructions.")

_N_ARTIFACTS = flags.DEFINE_integer(
    "n_artifacts", 1, "The number of artifacts to generate."
)

_N_SEARCH_STEPS = flags.DEFINE_integer("steps", 200, "The number of search steps.")


def main(_):
    print("Optimizing the model")
    # Parse flags
    scorer_llm_name = _SCORER.value
    optimizer_llm_name = _OPTIMIZER.value
    scorer_temp = _SCORER_TEMPERATURE.value
    optimizer_temp = _OPTIMIZER_TEMPERATURE.value
    scorer_max_decode_steps = _SCORER_MAX_DECODE_STEPS.value
    optimizer_max_decode_steps = _OPTIMIZER_MAX_DECODE_STEPS.value
    domain = _DOMAIN.value
    n_artifacts = _N_ARTIFACTS.value
    n_search_steps = _N_SEARCH_STEPS.value

    openai_api_key = os.environ.get("OPENAI_API_KEY")

    assert domain in {"joke", "poem", "story", "six-word"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # make sure the scorer and optimizer models are callable
    if scorer_llm_name in {"gpt-4o-mini", "gpt-4o"}:
        assert openai_api_key, "The OpenAI API key must be provided."
        openai.api_key = openai_api_key
    else:
        assert scorer_llm_name == "llama"

    if optimizer_llm_name in {"gpt-4o-mini", "gpt-4o"}:
        assert openai_api_key, "The OpenAI API key must be provided."
        openai.api_key = openai_api_key
    else:
        assert optimizer_llm_name == "llama"

    # Load the models
    if scorer_llm_name == "llama" or optimizer_llm_name == "llama":
        llama_model = AutoModelForCausalLM.from_pretrained(_LLAMA_MODEL_NAME).to(device)
        llama_tokenizer = AutoTokenizer.from_pretrained(_LLAMA_MODEL_NAME)

    # Set up the optimizer server
    if optimizer_llm_name == "llama":
        pipe = pipeline(
            "text-generation",
            model=llama_model,
            tokenizer=llama_tokenizer,
            temperature=optimizer_temp,
            max_new_tokens=optimizer_max_decode_steps,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            eos_token_id=llama_tokenizer.eos_token_id,
            device=device,
        )
        call_optimizer_server_func = functools.partial(
            prompt_utils.call_llama_server_func,
            pipeline=pipe,
        )
    else:
        assert optimizer_llm_name in {"gpt-4o-mini", "gpt-4o"}
        call_optimizer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=optimizer_llm_name,
            max_decode_steps=optimizer_max_decode_steps,
            temperature=optimizer_temp,
        )

    # Set up the scorer server
    if scorer_llm_name == "llama":
        pipe = pipeline(
            "text-generation",
            model=llama_model,
            tokenizer=llama_tokenizer,
            temperature=scorer_temp,
            max_new_tokens=scorer_max_decode_steps,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            eos_token_id=llama_tokenizer.eos_token_id,
            device=device,
        )
        call_scorer_server_func = functools.partial(
            prompt_utils.call_llama_server_func,
            pipeline=pipe,
        )
    else:
        assert scorer_llm_name in {"gpt-4o-mini", "gpt-4o"}
        call_scorer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=scorer_llm_name,
            max_decode_steps=scorer_max_decode_steps,
            temperature=scorer_temp,
        )

    print("\n======== testing the scorer and optimizer servers ===========")
    scorer_test_output = call_scorer_server_func(
        "Does the sun rise from the north? Just answer yes or no."
    )
    print(f"number of scorer output decodes: {len(scorer_test_output)}")
    print(f"scorer test output: {scorer_test_output}")
    optimizer_test_output = call_optimizer_server_func(
        "Does the sun rise from the north? Just answer yes or no.",
    )
    print(f"number of optimizer output decodes: {len(optimizer_test_output)}")
    print(f"optimizer test output: {optimizer_test_output}")
    print("Finished testing the servers.")

    num_generated_instructions_in_each_step = 4
    num_search_steps = n_search_steps
    old_instruction_score_threshold = 25
    initial_instructions = [
        f"Write a {domain}. The {domain} must be completely new and original to you. The {domain} must be less than {ARTIFACT_LENGTH_LIMIT[domain]} characters long.",
    ]
    max_num_instructions = (
        20  # the maximum number of instructions and scores in the meta-prompt
    )
    # every this number of steps, compute the accuracies of current-step
    # instructions on the validation set
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )
    eval_interval = 1
    save_folder = os.path.join(
        OPRO_ROOT_PATH,
        "outputs",
        domain,
        f"s-{scorer_llm_name}-o-{optimizer_llm_name}-{datetime_str}/",
    )
    result_by_instruction_folder = os.path.join(save_folder, "result_by_instruction")
    os.makedirs(result_by_instruction_folder)
    print(f"result directory:\n{save_folder}")

    evolution_kwargs = {
        "n_artifacts": n_artifacts,
        "domain": domain,
        "num_search_steps": num_search_steps,
        "old_instruction_score_threshold": old_instruction_score_threshold,
        "initial_instructions": initial_instructions,
        "call_scorer_server_func": call_scorer_server_func,
        "call_optimizer_server_func": call_optimizer_server_func,
        "max_num_instructions": max_num_instructions,
        "optimizer_llm_name": optimizer_llm_name,
        "num_generated_instructions_in_each_step": (
            num_generated_instructions_in_each_step
        ),
        "eval_interval": eval_interval,
        "save_folder": save_folder,
        "result_by_instruction_folder": result_by_instruction_folder,
    }

    opt_utils.run_evolution(**evolution_kwargs)


if __name__ == "__main__":
    app.run(main)
