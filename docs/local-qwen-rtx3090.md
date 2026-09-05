# Local Qwen + llama.cpp on a single RTX 3090

This fork supports a local-first StudentSim path where API-style LLM calls use
Qwen through llama.cpp, while StudentSim's actual Stage-1/Stage-2 LoRA training
continues to use MS-SWIFT/PyTorch.

## Architecture

- **Teacher / generation / judge:** Qwen3.8-27B GGUF through `llama-server`.
- **OpenAI-compatible endpoint:** `http://127.0.0.1:8081/v1`.
- **Model alias:** `qwen38-code`.
- **Student training:** `Qwen/Qwen3-4B-Instruct-2507` with the RTX-3090 configs
  under `configs/training/rtx3090/`.
- **Azure OpenAI:** still available as an explicit fallback provider.

The llama.cpp client accepts a separate scoring URL for calls that request
`top_logprobs`. On one 24 GB RTX 3090, do **not** try to keep two copies of the
27B model loaded at once. Use the generation and scoring PowerShell scripts as
alternate profiles, normally on the same port.

## 1. Python environment

From PowerShell in the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[math,baselines,dev]"
```

Add other domain extras (`chess`, `l2`) as required.

## 2. Start the normal generation profile

```powershell
.\scripts\windows\start-qwen38-generation.ps1 `
  -LlamaCppDir "C:\AI\llama-cpp-qwen38-b10566" `
  -ModelDir "C:\AI\models\Qwen3.8-27B"
```

The script uses a 32K context by default and enables the MTP/draft model for
higher generation throughput. Override with `-Context 65536` if a workload
really needs the larger context and VRAM permits it.

Confirm the server is reachable:

```powershell
Invoke-RestMethod http://127.0.0.1:8081/v1/models
```

## 3. Point StudentSim at llama.cpp

```powershell
$env:STUDENTSIM_LLM_PROVIDER = "llamacpp"
$env:LLAMACPP_BASE_URL = "http://127.0.0.1:8081/v1"
$env:LLAMACPP_SCORING_BASE_URL = "http://127.0.0.1:8081/v1"
$env:LLAMACPP_MODEL = "qwen38-code"
```

`.env.example` contains the same values as a reference. StudentSim does not
implicitly load `.env`; set the variables in the shell or use your preferred
environment loader.

Smoke-test the provider:

```powershell
python -c "from studentsim.core.llm import Message, open_client; c=open_client('qwen38-code'); print(c.complete([Message('user','Reply with exactly LOCAL_OK')], max_tokens=20).text)"
```

## 4. Logprob-sensitive scoring profile

Some fidelity calculations request top-token log probabilities. For those runs,
stop the generation server and start the non-speculative profile:

```powershell
.\scripts\windows\start-qwen38-scoring.ps1 `
  -LlamaCppDir "C:\AI\llama-cpp-qwen38-b10566" `
  -ModelDir "C:\AI\models\Qwen3.8-27B"
```

Because it uses the same port by default, no environment change is required.
If scoring is hosted on another GPU or machine, set
`LLAMACPP_SCORING_BASE_URL` to that server instead; only calls requesting
`top_logprobs` are routed there.

## 5. RTX 3090 LoRA training profile

The original research configs target eight GPUs. The files under
`configs/training/rtx3090/` are practical single-GPU starting points:

- `world_size: 1`
- `per_device_batch: 1`
- gradient accumulation to recover a useful effective batch
- `max_seq_len: 2048`
- LoRA rank 32 / alpha 64
- BF16 with gradient checkpointing

Start with math Stage 1 after preparing the dataset:

```powershell
pip install -U ms-swift
studentsim-train --config configs/training/rtx3090/stage1_math.yaml
```

Then specialize one student using the matching Stage-2 config:

```powershell
studentsim-train --config configs/training/rtx3090/stage2_math.yaml --student-id <student_id>
```

The RTX-3090 configs intentionally write checkpoints below
`<domain>/rtx3090/` so they do not overwrite the original eight-GPU recipe.
If native-Windows MS-SWIFT/PyTorch dependencies are problematic, keep
`llama-server` on Windows and run the training command in WSL2 with the same
repository/data mounted there.

## 6. Azure fallback

To restore the original hosted provider for a particular run:

```powershell
$env:STUDENTSIM_LLM_PROVIDER = "azure_openai"
$env:AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "<key>"
```

No StudentSim caller needs to change; provider selection stays behind
`studentsim.core.llm.open_client`.
