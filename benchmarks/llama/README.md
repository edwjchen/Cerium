## Running Llama3-8B Inference

This directory contains the encrypted Llama3-8B Implementation. Follow the instructions to run the Llama3-8B inference.


### 1. Build and enter the Cerium environment

Follow the root [README](../../README.md) to build the Cerium framework and envrionment 


### 2. Download the native Llama 3.1-8B checkpoint

Accept the Llama license for [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) on Hugging Face and log
in with and set `HF_TOKEN`. Then download and verify the native checkpoint:

```bash
cd llama_models
HF_TOKEN=<your-token> python3 download_llama.py
```

The downloader writes `consolidated.00.pth`, `params.json`, and
`tokenizer.model` to `llama_models/model_weights/Llama3.1-8B/`.

### 3. Create packed model parameters and prompt inputs

Generate the packed parameter files for all 32 transformer blocks, then create
the five prompt and attention-mask input pairs:

```bash
cd llama_models
MAX_JOBS=1 bash create_params.sh
for sample in 0 1 2 3 4; do
  bash run_packed_inputs.sh --data_sample "$sample"
done
cd ..
```

Increase `MAX_JOBS` when sufficient host memory is available.

### 4. Generate and compile the Cerium Llama program

First generate the Cerium CUDA program sources for the desired topology. The
following commands use the full 32-block, one-GPU configuration:

```bash
python llama_cerium.py --gpus 1 --num_blocks 32
bash make_llama.sh
```

`llama_cerium.py` creates the generated source under `outputs/`.
`make_llama.sh` builds the CUDA shared libraries and stages the default
`llama_1gpu_32blocks` runtime artifacts under `outputs/compiled/`; no
standalone `make` invocation is needed. For another topology, add its
`compile_all` call to `make_llama.sh` and use matching `--gpus` and
`--num_blocks` values when generating the sources.

### 5. Create remappable plaintext inputs

After the Cerium programs and packed parameters exist, create the remappable
plaintexts used by encrypted inference:

```bash
TOTAL_BLOCKS=32 BATCH_SIZE=4 MAX_JOBS=1 bash create_remapables.sh
```

This produces the encoded model weights in `llama_plaintexts/`. Increase `MAX_JOBS` and `BATCH_SIZE` when the system permits.

### 6. Generate ciphertexts and run

The runner needs serialized ciphertexts in `llama_ciphertexts/`. For the first
run, set `GENERATE_SAMPLES = True` in `llama_runner.py`, then run:

```bash
python llama_runner.py --gpus 1 --block_start 0 --num_blocks 32
```

This creates ciphertext inputs for the five prepared samples (and writes
intermediate outputs to `fhe_outputs/`). Set `GENERATE_SAMPLES` back to
`False` and repeat the same command for the timed benchmark run:

```bash
python llama_runner.py --gpus 1 --block_start 0 --num_blocks 32
```

`--block_start 0` is the full-model entry point. A nonzero block start requires
matching intermediate ciphertext outputs from the preceding block.

### Generated directories

- `llama_models/model_weights/Llama3.1-8B/`: downloaded native checkpoint.
- `llama_models/params/`: packed transformer parameters.
- `llama_models/llama_inputs/`: packed prompt and attention-mask inputs.
- `outputs/`: Cerium-generated CUDA source and build products.
- `outputs/compiled/`: runtime-staged Cerium libraries and metadata.
- `llama_plaintexts2/`: remappable plaintext inputs.
- `llama_ciphertexts/`: serialized encrypted inputs.
- `fhe_outputs/`: encrypted-inference intermediate outputs.
