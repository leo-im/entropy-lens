# Fixtures

`vllm_response.json` — a vLLM `/v1/chat/completions` response with
`logprobs` for the completion "The capital of France is Paris."
(model `Qwen/Qwen2.5-0.5B-Instruct` schema). The JSON follows the exact
response schema vLLM emits, but the logprob values are synthetically
constructed (no GPU server was involved), chosen to be realistic: function
words concentrated, the first token and final punctuation more spread out.

To replace it with a genuinely captured response, run a live server and:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-logprobs 20
python scripts/verify_live.py --base-url http://localhost:8000/v1 \
    --save-json tests/fixtures/vllm_response.json
```
