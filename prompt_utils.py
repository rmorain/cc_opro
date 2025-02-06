import time

import openai


def call_openai_server_single_prompt(
    prompt, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
    """The function to call OpenAI server with an input string."""
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            max_tokens=max_decode_steps,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    except openai.error.Timeout as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_single_prompt(
            prompt, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_single_prompt(
            prompt, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.APIError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_single_prompt(
            prompt, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.APIConnectionError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"API connection error occurred. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_single_prompt(
            prompt, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except openai.error.ServiceUnavailableError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Service unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_single_prompt(
            prompt, max_decode_steps=max_decode_steps, temperature=temperature
        )

    except OSError as e:
        retry_time = 5  # Adjust the retry time as needed
        print(f"Connection error occurred: {e}. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_openai_server_single_prompt(
            prompt, max_decode_steps=max_decode_steps, temperature=temperature
        )


def call_openai_server_func(
    inputs, model="gpt-3.5-turbo", max_decode_steps=20, temperature=0.8
):
    """The function to call OpenAI server with a list of input strings."""
    if isinstance(inputs, str):
        inputs = [inputs]
    outputs = []
    for input_str in inputs:
        output = call_openai_server_single_prompt(
            input_str,
            model=model,
            max_decode_steps=max_decode_steps,
            temperature=temperature,
        )
        outputs.append(output)
    return outputs


def call_llama_server_func(
    inputs,
    pipeline=None,
):
    """The function to call a local llama model with a list of input strings."""
    if isinstance(inputs, str):
        inputs = [inputs]
    outputs = []
    for input_str in inputs:
        output = call_llama_server_single_prompt(
            input_str,
            pipeline=pipeline,
        )
        outputs.append(output)
    return outputs


def call_llama_server_single_prompt(input_str, pipeline):
    """The function to call a local llama model with an input string."""
    messages = [
        {"role": "user", "content": input_str},
    ]
    output = pipeline(
        messages,
    )
    return output[0]["generated_text"][1]["content"]
