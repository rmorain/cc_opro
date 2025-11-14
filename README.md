# [Is Prompt Engineering the Creativity Knob for Large Language Models?](https://computationalcreativity.net/iccc25/wp-content/uploads/papers/iccc25-morain2025prompt.pdf)

## Abstract

The increasing use of large language models to generate creative artifacts raises questions about effective
methods for guiding their output. While prompt engineering has emerged as a key control mechanism
for LLMs, the impact of different prompting strategies on the quality and novelty of creative artifacts remains underexplored. This paper systematically compares four prompting strategies of increasing methodological complexity: basic prompts, human-engineered
prompts, automatically generated prompts, and chainof-thought (CoT) prompting. We generate ten examples in each of four textual domains, evaluating outputs
through both a human survey and GPT-4o-based automatic evaluations. Our analysis reveals that advanced
prompting techniques such as OPRO and R1 surprisingly do not produce artifacts of significantly higher
quality, greater novelty, or greater creativity than artifacts produced through basic prompting. The results
reveal some limitations of using GPT-4o for automatic
evaluation; provide empirical grounding for selecting
prompting methods for creative text generation; and
raise important questions about the creative limitations
of large language models and prompting.

## OPRO

This code is based on [Google Deepmind's OPRO code](https://github.com/google-deepmind/opro). You may find their paper "[Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)" useful for understanding some of the hyperparameters in this code.

## External resources

### OpenAI API

This code uses the OpenAI API to evaluate prompts and generate artifacts. There is also functionality to use a models running locally via Hugging Face's Transformers library (see `llama_test.py`). If you want to use a different API, please fork this repo, modify `prompt_utils.py` and submit a pull request.

### Google Custom Search API

This code uses [Google's custom search API](https://developers.google.com/custom-search/v1/overview) to ensure the novelty of generated artifacts by filtering out plagiarized artifacts. This requires setting up a custom search engine and including in your envirnoment the API key and search engine ID. 

If you do not want to use this functionality, you can set the `check_online` parameter in `eval_utils.evaluate_artifacts` to `False`.

## Environment

Create a Python environment and install dependencies:

```bash
conda create -n cc_opro python=3.10
pip install -r requirements.txt
```

Set environment variables for OpenAI and Google search:

```bash
export OPEN_AI_KEY=<OPEN_AI_KEY>
export GOOGLE_API=<GOOGLE_API_KEY>
export SEARCH_ID=<SEARCH_ID>
```

## Domains

This code is designed to generate jokes, poems, six-word stories, and short stories. To expand this to new domains you may need to modify the `initial_instructions` prompt in `optimize.py`.

## Training

Use `optimize.py` to perform prompt optimization. 

For example, to generate jokes you can use the command: 

```bash
python optimize.py --steps 100 --n_artifacts 5 --domain joke --scorer gpt-4o --optimizer gpt-4o 
```

This will generate and save optimized prompts for the specified domain.

## Artifact generation

To generate artifacts from an optimized prompt, see examples for how this is done in `generate.py`.

## Citation

```bibtex
@inproceedings{morain2025prompt,
  title={Is Prompt Engineering the Creativity Knob for Large Language Models?},
  author={Morain, Robert and Ventura, Dan},
  booktitle={Proceedings of the International Conference for Computational Creativity},
  year={2025}
}
```