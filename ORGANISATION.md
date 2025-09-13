# Implementation - Project teams

## Team 1 – Docs + Validation
*Curate tutorials, prepare validation tests, provide ground truth, identify doc gaps.*

* **Members:** Luigi, Giuseppe, Damien, William, Louis

* **Main objectives:**
  * Curate and prepare documentation (BigDFT-school, remotemanager, PyBigDFT docs) for RAG ingestion.
  * Define 3–4 **validation tasks** (HCN frequencies, atomisation energies, vibrational/Raman spectra).
  * Provide **ground truth input/output** for those tasks.
  * Note documentation gaps (type hints, docstrings, examples) → draft **guidelines for LLM-friendly documentation**.

* **Material/ideas:**
  * BigDFT-school tutorials (GitHub repo)
  * remotemanager notebooks & docs - indicate URLs to be parsed
  * Validation tests proposed (HCN molecule frequencies, atomisation energies, vibrational/Raman)
  * Protocol for validation: notebook harness comparing LLM outputs to known reference

* **What we can do:**
  1. Collect existing tutorials + notebooks in the repo under /1-humandoc/. We do not need to copy the files, but it is enough to provide a list of URL with the actual files that will have to be "RAG-ed"
  2. Pick 2–3 initial validation tasks (start with HCN geop example - I will move the directory name).
  3. Write a draft validation notebook with input/output pairs (expected results, dry-run,...) .

* **GitHub Repo Materials:**
  * BigDFT-school tutorials (curated, chunked as needed) - provide URL of the jupyter notebooks
  * remotemanager example notebooks - URL as above
  * PyBigDFT documentaitons - same story
  * Draft validation tasks (HCN, atomisation, vibrational/Raman) with expected input/output - Some notebooks that a human would write - they should be used for comparison/validation

## Team 2 – AI / Agent
*Build RAG pipeline, define agents (domain, HPC, validator), link with validation. This group will design and test the **Immediate Augmented Knowledge Agent (IAKA)**, focusing on RAG pipelines and agent roles.*

* **Members:** Louis, Yoann, Etienne, Youssef, William, Tiffany, Jan

* **Main objectives:**
  * Design a **user interface** (e.g. Jupyter notebook + ipython magics).
  * Build a **RAG pipeline** (retrieve docs → feed LLM → grounded code generation).
  * Implement an **augmentation loop** so the LLM justifies answers with docs.
  * Define **agent roles**:
    * *Domain (DFT) agent* → interprets user queries, produces PyBigDFT functions.
    * *HPC agent* → wraps functions into remotemanager jobs & runs them.
    * *Validator agent* → checks code vs. ground truth (works with Team 1).
  * Benchmark models: baseline vs. RAG vs. agentic loop.

* **Material/ideas:**
  * Self-RAG (Shuster et al. 2023), Toolformer, Meta AI “Agents” papers (Tiffany)
  * LangChain / LlamaIndex approaches
  * Existing RAG prototype for BigDFT (Yoann)
  * LangSim ipython magics for interface (Giuseppe)
  * Regex-based doc parsing, indexing methods (Alberto)
  * MCP wrapper for remotemanager (Louis)

* **What we can do:**
  * Tiffany → Leading RAG pipeline design (chunking, metadata, doc prep strategies, enforcing doc-based reasoning).
  * Yoann → Providing agentic RAG prototype with ontology recognition (from Fortran BigDFT experience).
  * Alberto → Tools for regex-based parsing, indexing, differentiating code vs. text.
  * Giuseppe  -> Put the user interface of jupyter magic based on langsim project
  * Louis -> MCP server for remotemanager (also linked with 3)

  1. Create a `/2-aiengine/` folder in repo for RAG experiments and agent prototypes. Feel free to subfolder it
  2. Upload/document any existing RAG code or parsers (Yoann, Alberto).
  3. Upload the MCP server information
  4. Start designing *mock* user case examples - Hello world cases
  5. Move to define 1–2 **validation queries** to test on (in collaboration with Team 1).

* **GitHub Repo Materials:**
  * First RAG experiments (scripts or notebooks)
  * User Interface prototypes (LangSim magic experiments, notebooks)
  * Any doc-chunking/indexing code (regex, ontology parsers, etc.)
  * Notes/papers/resources URL (Self-RAG, Toolformer, Agents)

## Team 3 – HPC & Workflow
*Wrap PyBigDFT tasks, test remotemanager runs, run validation case end-to-end.*

* **Members:** Luigi, Louis, Giuseppe, William, Youssef, Jan

* **Main objectives:**
  * Wrap **PyBigDFT workflows** into remotemanager functions.
  * Test **remotemanager job submission & retrieval** (`%%sanzu` Jupyter magic, dataset API, MCP server, agentic approach).
  * Run at least one **validation case end-to-end** (starting with HCN  geopt/frequencies).
  * Support the AI/Agent team by executing “production” cases (vibrational spectra, atomisation energies).
  * Explore optional **multi-agent scheduling** (domain agent → HPC agent).

* **Material/ideas:**
  * Example PyBigDFT workflow (Luigi)
  * remotemanager infrastructure & notebooks
  * Focus on HCN test, remotemanager job submission, result retrieval

* **What we can do:**
  * Luigi → Example PyBigDFT workflows (local workstation + HPC reference + lab resources).
  * Giuseppe → Experience with LangSim ipython magics create pybigdft-API functions
  * Louis → MCP wrapper for remotemanager (already demonstrated remote execution of simple tasks).

  1. Set up a `/3-hpcjobs/` folder in repo.
  2. Upload Luigi’s PyBigDFT workflow example (local run).
  3. Test Louis’ MCP wrapper on a simple PyBigDFT function.
  4. Prototype remotemanager submission of a basic validation case (HCN frequencies).
  5. Secure small "HPC" resources to showcase the approach

* **GitHub Repo Materials:**
  * Example PyBigDFT workflows (local + remotemanager runs - could go in team 1 too)
  * remotemanager wrapper prototypes (Louis’ MCP wrapper, etc.)
  * Scripts for job submission/retrieval on cluster - secure some remote resources


## Team 4 – Dissemination
*Build storyboard, slides, demo, and final pitch. Here we should make sure the **story of the project is clear, engaging, and compelling** for the hackathon jury. We’ll turn the technical work of the other teams into a narrative with visuals, demos, and a strong pitch.*

* **Members:** Cinthya, Giuseppe, Louis, Leonid, Luigi

* **Main objectives:**
  * Complete and maintain the **Miro board** (Motivation / Problem / Hackathon Target / Teams).
  * Build **slides and video** for the final pitch.
  * Prepare the **demo notebook(s)** (with simple natural language query → HPC run).
  * Record a short **demo video** of the assistant running.
  * Rehearse the **final presentation**.

* **Material/ideas:**
  * Miro? storyboard draft with Motivation / Problem / Hackathon Target / Teams - project narrative
  * Demo notebook + optional short video
  * Wrap-up guidelines for LLM-friendly documentation

* **What we can do:**
  * Create `/4-dissemination/` folder in repo (for slides, assets, demo script).
  * Correct/complete first version of the Miro board (based on hackathon submission format).
  * Write down some prose defining the project narrative to orient the presentation
  * Collect **figures, diagrams, and snippets** from Teams 1–3 as they progress.
  * Define who will present each part during the pitch.
  * Build a **slide skeleton** (intro → problem → solution → demo → impact).

* **GitHub Repo Materials:**
  * Draft of storyboard sections - narrative of the project
  * Early slide drafts / visual assets
  * Demo video snippets if available - technique for recording? Prepare working example?

## Coordination
* Coordination tasks (repo, comms, integration) will be shared across all, with Luigi and Giuseppe as main anchors.
* Everyone is welcome to contribute in more than one place.
 * The **key integration points** will be:
  * Docs+Validation  <-> AI/Agent (to align on test cases + doc ingestion)
  * AI/Agent <-> HPC (to ensure generated code actually runs with remotemanager)
