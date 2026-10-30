# Module 10 - LlamaIndex

## Setup

From the project root, install the module dependencies into the project's virtual
environment:

```bash
.venv/bin/python -m pip install -r "Phase 3 - Gen AI Libraries - Copy/Module 10 - LlamaIndex/requirements.txt"
```

Example 10.4 sends both chat and embedding requests through OpenRouter. Its
embedding model is NVIDIA `nvidia/nemotron-3.5-embed-1b:free`; it does not use
the OpenAI API or require OpenAI credits. Set `OPENROUTER_API_KEY` to a key from
<https://openrouter.ai/keys> (typically starts with `sk-or-v1-`). The default
chat model is configured in the script and can be overridden with
`OPENROUTER_MODEL`. Availability and pricing depend on OpenRouter's current
model/provider offerings.

Other examples that call OpenAI still need a valid `OPENAI_API_KEY` in the
project root `.env` file and may incur charges.

## Run

Run an example from the project root using the virtual environment. The explicit
`PYTHONPATH` ensures the interpreter resolves the packages installed in `.venv`:

```bash
PYTHONPATH="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
  .venv/bin/python "Phase 3 - Gen AI Libraries - Copy/Module 10 - LlamaIndex/10.4 Vector Store Index.py"
```

Replace the script name with another `.py` file in this folder to run a different
example. `10.3 Documents and Nodes.py` and `10.9 Module 10 Exercises.py` do not
need OpenAI credentials; the other examples use OpenAI embeddings and/or models.
