# Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference

This artifact accompanies the paper *Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference*. It contains the Cerium compiler, runtime, benchmark workloads used to evaluate encrypted bootstrapping and neural-network inference.

## Artifact contents and relation to the paper

### Implementation code and support files

| Path | Contents | 
| --- | --- | 
| `cerium/compiler/` | Frontend, optimization passes, partitioning, CUDA-code generation | 
| `cerium/runtime/` | CKKS encryption, evaluation, memory management, CUDA, NCCL runtime | 
| `cerium/python/` | Cerium DSL and Python Bindings | 
| `docker/` | Build-image definition and SEAL installer | 
| `requirements.txt` |  Python dependencies | 

### Benchmarks and evaluation data

| Path | Contents |
| --- | --- |
| `benchmarks/bootstrap_32K/` | Bootstrapping benchmark |
| `benchmarks/resnet20/` | ResNet-20 benchmark  |
| `benchmarks/bert_base/` | BERT-Base benchmark |
| `benchmarks/llama/` | Llama 3-8B benchmark |
| `ablations/` | Multi-GPU scheduling and fusion ablations |

`benchmarks/llama/llama_models` is a submodule with Llama helper code.

## Software environment

The software environment is the Docker image built from `docker/Dockerfile`:

| Component | Version/configuration |
| --- | --- |
| Base OS | Ubuntu 22.04 (container) |
| CUDA toolkit/image | NVIDIA CUDA 13.3.0 development image |
| Compiler/build tools | GCC/G++ 12, CMake, Make |
| CPU HE Backend | Microsoft SEAL v4.1.1 |
| GPU Collectives | NCCL 2.30.7 for CUDA 13.3 |
| Python | 3.12; experiments use a Python 3.12 virtual environment |
| Python packages |  Specified in `requirements.txt` |

The host must run Linux, Docker, NVIDIA Container Toolkit, and an NVIDIA driver compatible with CUDA 13.3. The build defaults to GPU compute capabilities `80;86;89;90;100;120`; override them as the fourth argument to `build_cerium.sh` if necessary. Experiments automatically skip unavailable 1-, 2-, 4-, or 8-GPU configurations.


- Llama3 model weights are not distributed with this repository. To download them, accept the Meta Llama license and download [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) as described in `benchmarks/llama/README.md`.

## Building the Cerium framework

From the repository root:

```bash
git submodule update --init --recursive
docker build -t cerium/build-cu13.3:v1 docker/
bash build_cerium.sh
bash activate_cerium_venv.sh
```

This builds the Cerium compiler/runtime in the Docker environment and creates the Python virtual environment used by the benchmarks.

## Quick start: minimal working example

After building the framework, run:

```bash
cd benchmarks/bootstrap_32K
bash compile.sh
bash run_bootstrap_benchmarks.sh
```

The last command discovers visible GPUs, runs supported configurations among 1/2/4/8 GPUs, and prints average execution time.

## Reproducing paper experiments

Run these after building the framework.

### Bootstrap

```bash
cd benchmarks/bootstrap_32K
bash compile.sh
bash run_bootstrap_benchmarks.sh
```

### ResNet-20

```bash
cd benchmarks/resnet20
bash compile.sh
bash run_resnet_benchmarks.sh
```

### BERT-Base

```bash
cd benchmarks/bert_base
bash compile.sh
bash run_bert_benchmarks.sh
```

### Llama 3.1-8B

Follow [`benchmarks/llama/README.md`](benchmarks/llama/README.md) to obtain the
licensed checkpoint, generate inputs and
run the encrypted-inference workflow. It produces generated artifacts and timed
inference output.


## Reproducing ablations

The ablation scripts compile all required variants, run them, and generate a PDF figure.

### Multi-GPU bootstrapping

This study compares the 1-GPU baseline with no cross-chip optimization, the Cerium schedule, and the full Cerium optimization on 2-, 4-, and 8-GPU configurations.

```bash
cd ablations/bootstrap_multi_gpu
bash compile.sh
bash run_bootstrap_benchmarks.sh
```

The script creates `MultiGPUPlots.pdf` in this directory.

### Fusion ablations

Each fusion study measures no fusion, horizontal fusion, vertical fusion, and vertical fusion with CUDA graphs on one GPU.

#### Keyswitch microbenchmark

```bash
cd ablations/fusion/keyswitch
bash compile.sh
bash run_keyswitch_benchmarks.sh
```

This writes `KeyswitchFusion.pdf`.

#### Bootstrap

```bash
cd ablations/fusion/bootstrap
bash compile.sh
bash run_bootstrap_benchmarks.sh
```

This writes `BootstrapFusion.pdf`.

#### ResNet-20

```bash
cd ablations/fusion/resnet
bash compile.sh
bash run_resnet_benchmarks.sh
```

This writes `ResNetFusion.pdf`. 


## License

Cerium is Apache-2.0; see `LICENSE.txt`. External dependencies and the Llama model have their respective licenses.
