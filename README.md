# CuraSyn: Knowledge-Guided Synthesis of Rare-Disease EHRs

Curasyn generates rare-disease electronic health records using disease conditions, clinical knowledge, and feedback from accepted samples. The implementation combines three components:

- **Static and dynamic knowledge.** Disease-specific clinical knowledge is supplemented by disease–HPO frequencies extracted from the training data.
- **Coverage feedback.** Accepted records guide condition scheduling, example retrieval, and remaining phenotype targets.
- **Calibrated selection.** Schema and numerical checks are followed by Table-calibrated statistical screening.

The main workflow starts with an EHR panel: normalize and split EHR data → load static knowledge → extract dynamic knowledge → generate and curate records. The repository includes separate tools for MIMIC extraction, Chinese-note HPO mapping, and literature knowledge preparation. These tools have their own input and model requirements.

## Repository Structure

```text
Curasyn/
├── main.py                       # Generation entry point
├── config.yaml                   # Model, data paths, and method parameters
├── requirements.txt              # Dependencies for all four stages
├── src/                          # Dynamic knowledge and synthesis pipeline
├── constraintKB/
│   ├── create_kb/                # Literature extraction, export, and KB build entry point
│   ├── api.py                    # Static knowledge query interface
│   ├── build_kb.py                # Optional database construction
│   ├── data/constraints.json     # Source knowledge
│   ├── disease_mapping.csv       # Disease aliases
│   ├── kb/clinical_knowledge.sqlite
│   └── ontology/
│       ├── hp.obo
│       └── ehr_hpo_vocabulary.json # Fixed reference-EHR HPO vocabulary
└── datasets/
    ├── README.md                 # Data inputs and preparation instructions
    ├── mimic-process/            # MIMIC extraction and HPO mapping scripts
    ├── examples/                 # Fictional input-format examples
    ├── prepare_ehr.py            # Normalization and fixed split
    └── validate_ehr.py            # Input validation
```

Patient records, processed datasets, and generated outputs are excluded from version control. Prepare model weights locally.

## Installation

Use **Python 3.10**. Run all project commands from the repository root.

```bash
conda create -n curasyn python=3.10 -y
conda activate curasyn
# Optional: install the reference CUDA wheel before the requirements.
# python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

The requirements cover the main method, Excel input, and BioLORD-based HPO mapping. The reference GPU environment used PyTorch 2.11.0 with CUDA 13.0; use a wheel compatible with your hardware and driver. Main synthesis communicates with an LLM service through its API. BioLORD mapping can run on CPU or CUDA.

## Dataset Preparation

Start with a locally available EHR panel following the [input contract](datasets/README.md#ehr-input-contract). The [fictional CSV example](datasets/examples/ehr_format_example.csv) demonstrates the format and can be used to check validation and preparation. It is separate from the default dataset and the reference experiment.

MIMIC-IV and MIMIC-IV Note must be obtained by the user. The independent extraction and HPO tools are described in [datasets/README.md](datasets/README.md#independent-data-preparation-tools). The HPO tool accepts Chinese notes; preparing text inputs and joining its output into an EHR panel are separate steps. The documented main synthesis route starts with the completed panel.

If the extracted EHR panel is already available, place `0813iei_final.csv` in `datasets/ehr/raw/`, then run:

```bash
python datasets/validate_ehr.py --raw-csv datasets/ehr/raw/0813iei_final.csv
python datasets/prepare_ehr.py --raw-csv datasets/ehr/raw/0813iei_final.csv \
  --output-dir datasets/ehr --train-fraction 0.7
```

This produces `processed/ehr.csv`, `splits/train.csv`, `splits/test.csv`, and `splits/split_ids.json`. The reference input contains 159 source rows; normalization produces 163 records, split into 113 training and 50 test records. Preserve this fixed split across generation runs.

Any source filename can be supplied through `--raw-csv`. For custom data, choose a dataset output directory and set the corresponding paths in `config.yaml`. The data guide includes an isolated [format example](datasets/README.md#check-the-fictional-format-example) and all supported examination fields.

## Knowledge Preparation

To build new static knowledge from paper Markdown, use [constraintKB/create_kb/README.md](constraintKB/create_kb/README.md). Run its preprocessing, extraction, export, and build commands from `constraintKB/create_kb/`. The Extraction input paths in example configuration need to be replaced. The workflow produces a candidate SQLite database for the existing `--knowledge-source` interface.

The prepared static database and HPO ontology are included. The default configuration reads them directly; no rebuild is needed. See [constraintKB/README.md](constraintKB/README.md) for the resources and query interface.

To rebuild the same static database from the supplied knowledge and disease mapping:

```bash
python constraintKB/build_kb.py
```

Dynamic disease–HPO knowledge is mined automatically from `train.csv` when synthesis starts. It is combined with static knowledge in memory, and remaining HPO probabilities are updated from accepted records. The static database is preserved.

## Configuration

Fill in the following fields in [config.yaml](config.yaml), or set their environment overrides:

| Field | Meaning | Environment override |
| --- | --- | --- |
| `generation.model` | Model ID exposed by the service | `CURASYN_LLM_MODEL` |
| `generation.api.base_url` | Endpoint including `/v1` | `CURASYN_LLM_BASE_URL` |
| `generation.api.api_key` | Service key; `EMPTY` for an unauthenticated service | `CURASYN_LLM_API_KEY` |

These fields are intentionally blank. Configuration precedence is **file → environment → command line**.

Use `python main.py --help` for supported command-line overrides. Omitted options retain the file or environment settings. Relative data and output paths use the repository root when following the documented commands; an external configuration file does not change this working directory. Absolute paths are also supported.

Optional paths retain the original layout when left null:

| Field | Default |
| --- | --- |
| `paths.dataset_csv` | `datasets/<dataset>/processed/ehr.csv` |
| `paths.train_csv` / `paths.test_csv` | `datasets/<dataset>/splits/train.csv` / `test.csv` |
| `paths.results_dir` | `results/<dataset>` |
| `knowledge.hpo_path` | `constraintKB/ontology/hp.obo` |

`knowledge.source` already points to `constraintKB/kb/clinical_knowledge.sqlite`; retain this setting for static knowledge guidance. `paths.dataset_csv` defines the field layout and Table labels. The HPO vocabulary is selected separately through `data.hpo_vocabulary_files`. Custom train/test files must follow the same EHR field contract. Generation size, sampling settings, retrieval limits, knowledge support, and curation thresholds are in `config.yaml`.

## Fixed HPO Vocabulary

The default EHR configuration loads [ehr_hpo_vocabulary.json](constraintKB/ontology/ehr_hpo_vocabulary.json), containing the same 650 codes in the same order as the original full-panel vocabulary:

```yaml
data:
  data_seed: 0
  hpo_vocabulary_files:
    ehr: constraintKB/ontology/ehr_hpo_vocabulary.json
    private_ehr: null
```

This is a fixed, dataset-derived task vocabulary. It was frozen from the complete normalized reference EHR panel, including the records assigned to the test split; it is not an independently predefined ontology subset. Saving it as a file preserves the original vocabulary and does not change the information scope of historical experiments. Disease-HPO frequencies and coverage statistics continue to come from the training split.

The file records `format_version`, `dataset`, `origin`, `source_description`, `hpo_ontology_version`, and the ordered `codes` array. The loader preserves code order and checks the format, dataset, code syntax, and duplicate entries. Relative vocabulary paths resolve against the repository root. A configured missing or invalid file causes an error; it does not fall back to a different vocabulary. Startup output identifies the vocabulary source, path, origin, and code count.

A fixed vocabulary file is required for the selected dataset. An entry set to `null`, omitted, or pointing to a missing file causes an error; the program never constructs a vocabulary from EHR records. The `private_ehr: null` entry is an unfilled resource setting: provide a Private EHR vocabulary before running that dataset. For custom data, supply its task vocabulary and source description. Observed HPO codes must be covered by the chosen vocabulary.

Within each dataset, training, test, and generated records use one shared task vocabulary. There are no separate train/test vocabulary settings. Disease frequencies, coverage statistics, and the curation classifier continue to use training records only. Literature extraction also loads the same file through its `hpo_vocabulary_path` setting. EHR CSVs still supply sample values, field layouts, and Table labels; they do not supply the vocabulary.

Vocabulary contents affect knowledge filtering, feature encoding, exemplar retrieval, and output validation. Retain both contents and order when reproducing existing runs. Existing checkpoints already include the ordered vocabulary in their identity check. A genuinely independent vocabulary can be selected for a new experiment, with its definition and provenance recorded in the file.

## Running Curasyn

### 1. Start the model service

Install **vLLM 0.26.0** in a separate serving environment:

```bash
conda create -n curasyn-vllm python=3.10 -y
conda activate curasyn-vllm
python -m pip install vllm==0.26.0
```

Prepare the **Qwen2.5-7B-Instruct** weights locally. Fill in the empty values and keep the service running:

```bash
MODEL_PATH=""        # Local model directory
MODEL_ID=""          # Served model ID, e.g. Qwen2.5-7B-Instruct
GPU_ID=""            # GPU index
VLLM_PORT=""         # Available port

: "${MODEL_PATH:?Set MODEL_PATH}" "${MODEL_ID:?Set MODEL_ID}"
: "${GPU_ID:?Set GPU_ID}" "${VLLM_PORT:?Set VLLM_PORT}"

CUDA_VISIBLE_DEVICES="$GPU_ID" VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve "$MODEL_PATH" --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 --port "$VLLM_PORT" --dtype bfloat16 \
  --max-model-len 32768 --generation-config vllm \
  --gpu-memory-utilization 0.8 --max-num-seqs 16
```

The template serves on localhost. An existing reachable OpenAI-compatible service can also be used by setting its endpoint and model ID.

### 2. Connect and generate

In a second terminal, return to the repository root, activate the Curasyn environment, and fill in the endpoint settings:

```bash
conda activate curasyn
export CURASYN_LLM_BASE_URL=""  # http://<host>:<port>/v1
export CURASYN_LLM_MODEL=""     # Same served model ID
export CURASYN_LLM_API_KEY=""   # Use EMPTY for the service template above

: "${CURASYN_LLM_BASE_URL:?Set CURASYN_LLM_BASE_URL}"
: "${CURASYN_LLM_MODEL:?Set CURASYN_LLM_MODEL}" "${CURASYN_LLM_API_KEY:?Set CURASYN_LLM_API_KEY}"
curl -fsS "$CURASYN_LLM_BASE_URL/models" -H "Authorization: Bearer $CURASYN_LLM_API_KEY"

python main.py --dataset ehr --data-seed 0 --n-samples 1000 --artifact-tag v1
```

When using file-only endpoint settings, verify the configured URL separately and run the final `python main.py` command. Use `--config /path/to/config.yaml` for a separate configuration file. `--n-samples` specifies the final accepted count; `--artifact-tag` distinguishes output names.

## Outputs and Reproduction

A successful reference EHR run writes:

```text
results/ehr/
├── curasyn_s0_n1000_v1_llm.csv                 # Structurally valid candidates
├── curasyn_s0_n1000_v1_final.csv               # Exactly 1,000 accepted records
├── curasyn_s0_n1000_v1_empirical_hpo.csv       # Training-derived relations
└── curasyn_s0_n1000_v1_empirical_hpo_manifest.json
```

Completed rounds are checkpointed. After interruption, rerun the same command and settings to resume; successful completion removes the checkpoint files. Progress is printed to the terminal and can be redirected to a user-chosen log file. Use a new tag or seed for a separate run.

The reference synthesis settings are:

| Setting | EHR | Private EHR |
| --- | --- | --- |
| Model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| Accepted records | 1,000 | 2,000 |
| Generation seeds | 0, 1, 2 | 0, 1, 2 |
| Temperature / top-p | 0.9 / 0.95 | 0.9 / 0.95 |
| Maximum completion tokens | 8,000 | 8,000 |
| Empirical HPO minimum support | 7 | 10 |
| Exemplars / retrieval theta | 5 / 5 | 5 / 5 |
| Curation estimators | 100 | 100 |
| Standardized confidence / uncertainty thresholds | -2.0 / -2.0 | -2.0 / -2.0 |

The shared `config.yaml` defaults to 1,000 accepted records. For Private EHR, prepare its local data and use:

```bash
python main.py --dataset private_ehr --data-seed 0 --n-samples 2000 --artifact-tag v1
```

First set `data.hpo_vocabulary_files.private_ehr` to that dataset's fixed vocabulary file. The repository does not include a Private EHR vocabulary.

`--data-seed` sets the generation seed when `run.seed` is null; an explicit `run.seed` takes precedence. It does not regenerate the fixed data split. Keep the same prepared files across seeds.

The supplied static database and ontology identify the bundled knowledge resources. Exact model weight revisions and the original patient data are not recorded or distributed here. To reproduce a local run, retain the source data, split files, effective configuration, model revision, and serving command. HPO extraction and literature extraction settings are documented separately in the [data guide](datasets/README.md#hpo-tool-settings) and [knowledge guide](constraintKB/create_kb/README.md#extraction-settings).

Use the same source data, split, knowledge resources, model weights, and parameter settings for comparable results. Stochastic LLM sampling can produce variation across runs. The repository generates synthetic records; experiment plots and benchmark tooling are omitted.

## Acknowledgements

The implementation draws on CLLM for generation and learning-dynamics curation, and TabGen-ICL for residual example retrieval. We thank these projects and the IUIS and Human Phenotype Ontology communities for their resources.
