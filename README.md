# LARA-HPC: A LAnguage model-powered Research Assistant for HPC

Electronic-structure codes remain challenging to use, especially on High-Performance Computing (HPC) systems where job submission and post-processing require technical expertise. LLMs are progressing rapidly as assistants, but their effectiveness depends heavily on how code and documentation are structured. Our project investigates **how to write documentation that makes scientific codes “LLM-friendly.”**

We propose to build an LLM-driven research assistant that can run realistic electronic-structure simulations on HPC systems using the **BigDFT** code. Instead of designing a new agent framework, our focus is to explore how **documentation and code structure** affect the ability of LLMs to understand, extend and orchestrate workflows.
By leveraging the **PyBigDFT Python API**, the **remotemanager** library for job submission, and the **BigDFT-school** educational repository, we will test how well models can autonomously perform tasks such as computing atomisation energies or vibrational spectra. Our hypothesis: **a codebase with clean, Pythonic APIs and high-quality documentation will enable LLMs to perform complex tasks with minimal intervention.**

* **For researchers:** Reduce the barrier to using electronic-structure codes on HPC and open new research paradigma enabled by the seamlessly integration of LLM.
* **For developers:** Concrete guidelines on how to structure documentation so that AI agents can effectively interact with complex scientific codes.
* **For the hackathon:** A case study showing that the quality of documentation can be as important as the quality of code in enabling AI-assisted discovery.
* **Long-term vision:** Documentation practices derived here could generalize to other simulation packages (chemistry, materials, physics).

## Quick start

The easiest way to try the code is to run the OCI (Docker) image at
```
docker pull ghcr.io/epolack/llm-hackathon-2025:0.0.2
```
and to run Jupyter from within.
See the [container](container) folder for a bit more details.

If you want to setup your environment, you can look at how the container is [built](container/build_oci.sh).
You will find inside it the simple step to build the image to use GPU instead of only using CPU.
This enables the use of more powerful embedding models for the RAG.


## Installation Instructions

Follow these steps to set up the environment.

---

#### 1. Create and activate the virtual environment

```bash
uv init
```

> **Note:** `setuptools` is required when working with a `uv` virtual environment.

```bash
uv pip install setuptools
```

---

#### 2. Build and configure BigDFT

```bash
git clone https://gitlab.com/luigigenovese/bigdft-suite.git
cd bigdft-suite
mkdir build
cd build
../bundler/jhbuild.py -f ../rcfiles/jhbuildrc build

source install/bin/bigdftvars.sh
```

---

#### 3. Install development dependencies

```bash
uv add --dev   ipython ipykernel jupyter-client jupyter-core   pytest pytest-cov pytest-xdist nbval   sphinx sphinx-rtd-theme nbsphinx nbconvert nbformat   pre-commit ruff rich   mcp remotemanager sse-starlette uvicorn typer
```

---

#### 4. Project-specific dependencies

Navigate into the OntoFlow RAG agent:

```bash
cd 2-aiengine/OntoFlow/agent/Onto_wa_rag
```

Install requirements and add missing dependencies:

```bash
# If your repo includes a requirements.txt here, install it:
uv add -r requirements.txt

# For Fortran parsing & AST
uv add open-fortran-parser
uv add tree_sitter

# If you plan to use local GGUF models via llama.cpp
uv add llama-cpp-python
```

### Notebooks demos

Check the following files to see some demo of the project: 2-aiengine/demo.ipynb

### Agentic Langchain Integration

IN PROGRESS - Would be able to bridge between the RAG and the remotemanager to run code on remote HPC system without any human intervention.



## Contributors
* [Luigi Genovese](https://github.com/luigigenovese) - CEA, France
* [Giuseppe Fisicaro](https://github.com/giuseppefisicaro) - CNR Institute for Microelectronics and Microsystems, Italy
* [Louis Beal](https://github.com/ljbeal), INRIA, France
* [Étienne Polack](https://github.com/epolack) - CEA, France
* [Yoann Curé](https://github.com/Yopla38) - CEA, France
* Damien Caliste - CEA, France
* [William Dawson](https://github.com/william-dawson) - RIKEN Center for Computational Science, Kobe, Japan
* [Jan Janssen](https://github.com/jan-janssen) - Max Planck Institute for Sustainable Materials - MPI-SusMat
* [Cinthya Herrera Contreras](https://github.com/cnherrera) - CEA, France
* Tiffany Abui Degbotse -
* [Youssef Briki](https://github.com/youssefbriki1) - Université de Montréal, Canada
* Leonid Didukh - Kyiv Institute of Nuclear Research, Ukraine
