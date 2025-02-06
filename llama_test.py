from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

_LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
llama_model = AutoModelForCausalLM.from_pretrained(_LLAMA_MODEL_NAME).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(_LLAMA_MODEL_NAME)
pipe = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.5,
    max_new_tokens=400,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

messages = [
    {
        "role": "system",
        "content": "You are a creative AI. You only write truly original and novel jokes. You never repeat a joke you have heard before.",
    },
    {"role": "user", "content": "Tell me a joke."},
]
output = pipe(messages)
print(output[0]["generated_text"][-1]["content"])
output = pipe(messages)
print(output[0]["generated_text"][-1]["content"])
