import os
import re
import time  # Add import for time module

import matplotlib.pyplot as plt  # Add import for matplotlib
import pandas as pd

import eval_utils


def run_evolution(**kwargs):
    n_artifacts = kwargs.get("n_artifacts")
    domain = kwargs.get("domain")
    num_search_steps = kwargs.get("num_search_steps")
    old_instruction_score_threshold = kwargs.get("old_instruction_score_threshold")
    initial_instructions = kwargs.get("initial_instructions")
    call_scorer_server_func = kwargs.get("call_scorer_server_func")
    call_optimizer_server_func = kwargs.get("call_optimizer_server_func")
    max_num_instructions = kwargs.get("max_num_instructions")
    num_generated_instructions_in_each_step = kwargs.get(
        "num_generated_instructions_in_each_step"
    )
    save_folder = kwargs.get("save_folder")
    result_by_instruction_folder = kwargs.get("result_by_instruction_folder")
    # format: [(i_step, instruction, detailed_results_df)]
    eval_results = []
    # all generated instructions, format: [(instruction, score, step_index)]
    # the instructions that were skipped have score NaN
    old_instructions_and_scores_raw = []
    instruction_score_dict = dict()  # the dictionary of {instruction: score}

    old_instructions_and_scores = []
    meta_prompts = []  # format: [(meta_prompt, step_index)]
    old_instruction_hashstrings_set = set()

    start_time = time.time()  # Start time for the entire function
    best_scores = []  # List to store the best score for each step

    # evaluate initial instructions
    print("\n============== evaluating initial instructions ===============")
    step_scores = []
    for instruction in initial_instructions:
        print(f"""computing the score of "{instruction}" by prompting""")

        detailed_results_df = eval_utils.evaluate_single_instruction(
            instruction=instruction,
            call_server_func=call_scorer_server_func,
            n_artifacts=n_artifacts,
            domain=domain,
            i_step=-1,
        )
        average_score = (
            detailed_results_df[
                detailed_results_df.select_dtypes(include=["int"]).columns
            ]
            .stack()
            .mean()
            .item()
        )
        step_scores.append(average_score)
        print(f"instruction: {instruction}, score: {average_score}")
        filename = eval_utils.instruction_to_filename(instruction)
        file_path = os.path.join(result_by_instruction_folder, f"{filename}.csv")
        detailed_results_df.to_csv(file_path, index=True, header=True)
        print(f"""saving results of "{instruction}" to {file_path}""")
        old_instructions_and_scores.append((instruction, average_score, -1))
        old_instructions_and_scores_raw.append((instruction, average_score, -1))
        instruction_score_dict[instruction] = average_score
        eval_results.append((-1, instruction, detailed_results_df))
    best_scores.append(max(step_scores))  # Initial best score

    # evolution
    for i_step in range(num_search_steps):
        step_start_time = time.time()  # Start time for each step
        print(f"\n================== Step {i_step} =====================")
        if not i_step % 10:
            print(f"old_instructions_and_scores: {old_instructions_and_scores}")

        # generate new instructions
        meta_prompt = gen_meta_prompt(
            domain=domain,
            old_instructions_and_scores=old_instructions_and_scores,
            old_instruction_score_threshold=old_instruction_score_threshold,
            max_num_instructions=max_num_instructions,
        )
        print(f"meta_prompt: \n\n{meta_prompt}\n")
        meta_prompts.append((meta_prompt, i_step))
        remaining_instructions_to_generate = num_generated_instructions_in_each_step
        generated_instructions = []
        while remaining_instructions_to_generate > 0:
            optimizer_llm_input_text = meta_prompt
            # generate instructions
            raw_outputs = call_optimizer_server_func(optimizer_llm_input_text)[0]

            generated_instructions.append(
                extract_string_in_square_brackets(raw_outputs)
            )
            remaining_instructions_to_generate -= 1
        # Save the generated instructions to a file
        new_generated_instructions = []
        for instruction in generated_instructions:
            instruction_hashstring = eval_utils.instruction_to_filename(instruction)
            if instruction_hashstring not in old_instruction_hashstrings_set:
                new_generated_instructions.append(instruction)
                old_instruction_hashstrings_set.add(instruction_hashstring)
        # Filter generated instructions
        to_evaluate_instructions = []
        for instruction in new_generated_instructions:
            if len(instruction) > 500:
                print(f"Step {i_step}: instruction too long, skipped")
                continue
            if "INS" in instruction:
                print(f"Step {i_step}: instruction contains INS, skipped")
                continue
            to_evaluate_instructions.append(instruction)
        print(f"\nto-evaluate generated instructions: {to_evaluate_instructions}\n")

        # Evaluate the generated instructions
        print("\n============== evaluating generated instructions ===============")
        step_scores = []
        for instruction in to_evaluate_instructions:
            print(f"""computing the score of "{instruction}" by prompting""")

            detailed_results_df = eval_utils.evaluate_single_instruction(
                instruction=instruction,
                call_server_func=call_scorer_server_func,
                n_artifacts=n_artifacts,
                domain=domain,
                i_step=i_step,
            )
            average_score = (
                detailed_results_df[
                    detailed_results_df.select_dtypes(include=["int"]).columns
                ]
                .stack()
                .mean()
                .item()
            )
            step_scores.append(average_score)
            print(f"instruction: {instruction}, score: {average_score}")
            filename = eval_utils.instruction_to_filename(instruction)
            file_path = os.path.join(result_by_instruction_folder, f"{filename}.csv")
            detailed_results_df.to_csv(file_path, index=True, header=True)
            print(f"""saving results of "{instruction}" to {file_path}""")
            old_instructions_and_scores.append((instruction, average_score, i_step))
            old_instructions_and_scores_raw.append((instruction, average_score, i_step))
            instruction_score_dict[instruction] = average_score
            eval_results.append((i_step, instruction, detailed_results_df))
        if step_scores:
            best_scores.append(max(step_scores))
        else:
            print(f"Warning: No valid scores for step {i_step}")
            best_scores.append(0)  # or another suitable default value

        # Save metaprompts to a file
        meta_prompt_filename = os.path.join(save_folder, f"metaprompt_{i_step}.txt")
        with open(meta_prompt_filename, "w") as f:
            f.write(meta_prompt)
        print(f"Meta prompt saved to {meta_prompt_filename}")
        # Save old instructions and scores to a file
        old_instructions_and_scores_filename = os.path.join(
            save_folder, f"old_instructions_and_scores_{i_step}.txt"
        )
        with open(old_instructions_and_scores_filename, "w") as f:
            for instruction, score, step in old_instructions_and_scores:
                f.write(f"{instruction}\t{score}\t{step}\n")
        print(
            f"Old instructions and scores saved to {old_instructions_and_scores_filename}"
        )

        print(
            f"Step {i_step} completed in {time.time() - step_start_time:.2f} seconds"
        )  # Log time for each step

        # Plot the best scores over time
        plt.plot(best_scores)
        plt.xlabel("Step")
        plt.ylabel("Best Score")
        plt.title("Best Score Over Time")
        file_path = os.path.join(save_folder, "best_scores_over_time.png")
        plt.savefig(file_path)
        print(f"\nbest_scores_over_time.png saved to {file_path}\n")

    # Save all instructions and scores to a csv
    sorted_instructions = sorted(
        old_instructions_and_scores, key=lambda x: x[1], reverse=True
    )
    df = pd.DataFrame(sorted_instructions, columns=["instruction", "score", "step"])
    file_path = os.path.join(save_folder, "all_instructions_and_scores.csv")
    df.to_csv(file_path, index=False)
    print(f"\nall_instructions_and_scores.csv saved to {file_path}\n")

    # Aggregate all artifact data
    all_artifacts = pd.concat([result[2] for result in eval_results])
    file_path = os.path.join(save_folder, "all_artifacts.csv")
    all_artifacts.to_csv(file_path, index=False)
    # Print location of all_artifacts.csv
    print(f"\nall_artifacts.csv saved to {file_path}\n")

    print(
        f"Total time for run_evolution: {time.time() - start_time:.2f} seconds"
    )  # Log total time


def extract_string_in_square_brackets(input_string):
    raw_result = re.findall(r"\[.*?\]", input_string)
    if raw_result:
        return raw_result[0][1:-1]
    else:
        return ""


def gen_meta_prompt(
    domain="joke",
    old_instructions_and_scores=None,
    old_instruction_score_threshold=0.3,
    max_num_instructions=10,
):
    old_instructions_and_scores = sorted(
        old_instructions_and_scores, key=lambda x: x[1]
    )[-max_num_instructions:]
    with open(os.path.join(os.getcwd(), f"prompts/{domain}/metaprompt.txt"), "r") as f:
        meta_prompt = f.readline()
        meta_prompt += "\n"
        for instruction, score, _ in old_instructions_and_scores:
            # filter out old instructions with low scores
            # if score < old_instruction_score_threshold:
            #     continue
            meta_prompt += "text:\n"
            meta_prompt += f"{instruction}\n"
            meta_prompt += "score:\n"
            meta_prompt += f"{int(score)}\n"
        meta_prompt += "\n"
        # Read remaining lines from the file
        remaining_lines = f.read()
        meta_prompt += remaining_lines
        return meta_prompt
