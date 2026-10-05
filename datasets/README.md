# EHR Inputs and Data Preparation

The main synthesis workflow starts with a completed EHR panel supplied by the user. Validate the panel, normalize it, and create the fixed split before running `main.py`. The MIMIC and Chinese-note HPO tools below are independent preparation utilities.

Run the commands in this guide from the repository root. Install the root requirements in the environment described in the [project README](../README.md#installation).

## EHR input contract

A raw EHR CSV uses the following fields. Column names are case-sensitive. The complete allowed field list is defined in `src/ehr_contract.py`; extra columns must be removed or renamed before validation.

| Field | Required | Format |
| --- | --- | --- |
| `inborn_error_of_immunity_diagnosis` | Column required | Disease name, or a serialized JSON array of disease names. Some rows may be blank, but at least one disease name must be present for conditional generation. |
| `hpo_codes` | Yes | Array of seven-digit `HP:` codes, e.g. `["HP:0002719","HP:0001945"]`. Use `[]` for no observed HPO terms. JSON arrays are recommended; raw preparation also accepts Python list literals. |
| `sex` | Yes | `F` or `M`. |
| `age_in_years` | Yes | Finite numeric value in years. |
| `IUIS_table` | Yes | Nonempty Table label. Composite source labels are separated by `;` and expanded during preparation. Validation expects at least two distinct Table classes. |
| `patient_identifier` | Optional | Nonempty patient ID when supplied; repeated IDs keep one patient's records in the same split. Omission assumes one source row per patient. |
| `hospital_admission_identifier` | Optional | Source admission ID; excluded from model features. |
| `discharge_note_identifier` | Optional | Source note ID; excluded from model features. |
| Supported examination fields | At least one observed field | Numeric values with blank cells for missing observations. See the field and unit table below. |

Quote CSV cells containing arrays according to CSV escaping rules. The [fictional example](examples/ehr_format_example.csv) shows the on-disk representation. Do not add `record_id` to the raw input: preparation creates anonymous record IDs. Processed and split files use JSON HPO arrays, include `record_id`, and retain the active examination fields. Leave Table labels and sex values in their semantic form when supplying prepared files; the main program encodes them internally.

Main synthesis requires a fixed HPO vocabulary through `data.hpo_vocabulary_files`; see the [fixed vocabulary configuration](../README.md#fixed-hpo-vocabulary). Training, test, and generated records share that vocabulary. The supplied EHR vocabulary contains the 650 codes frozen from the complete reference panel. Its source is recorded as `dataset_derived`. The CSV example demonstrates formatting; it does not redefine this task vocabulary. CSV records are never used to construct a vocabulary at runtime.

## Examination fields and units

These are the output scales of the supplied MIMIC extraction tool. `prepare_ehr.py` parses numeric values and preserves missing values; it does not convert units. Normalize your source values to the intended scale before preparation. Unit names belong in a separate dictionary, not in numeric CSV cells or additional unit columns.

| Fields | MIMIC output unit |
| --- | --- |
| `white_blood_cell_count`, `platelet_count`, `absolute_lymphocyte_count`, `absolute_neutrophil_count` | 10^9/L |
| `hemoglobin_concentration`, `mean_corpuscular_hemoglobin_concentration`, `serum_albumin` | g/L |
| `red_blood_cell_count` | 10^12/L |
| `mean_corpuscular_volume` | fL |
| `mean_corpuscular_hemoglobin` | pg |
| `red_cell_distribution_width`, `neutrophil_percentage`, `lymphocyte_percentage`, `monocyte_percentage` | % |
| `serum_sodium`, `serum_potassium`, `serum_chloride` | mmol/L |
| `serum_glucose`, `blood_urea_nitrogen`, `total_calcium`, `serum_magnesium`, `serum_phosphate` | mmol/L |
| `serum_creatinine`, `total_bilirubin` | micromol/L |
| `alanine_aminotransferase`, `aspartate_aminotransferase`, `alkaline_phosphatase` | U/L |
| `prothrombin_time`, `activated_partial_thromboplastin_time` | seconds |
| `immunoglobulin_g`, `immunoglobulin_m` | g/L |

All 31 examination fields are optional individually. At least one examination column must contain observed values. Entirely missing examination columns are dropped during preparation. Keep a consistent scale for each field across the raw panel, training data, knowledge dictionary, and generation runs. For literature extraction, the [dictionary example](examples/examination_dictionary_example.csv) covers exactly the three examination columns in the fictional CSV.

## Check the fictional format example

The eight rows in `examples/ehr_format_example.csv` are invented to illustrate columns, arrays, missing values, and grouping. They are not patient records or the reference experiment dataset. The example is not selected by any default configuration.

```bash
python datasets/validate_ehr.py --raw-csv datasets/examples/ehr_format_example.csv
python datasets/prepare_ehr.py --raw-csv datasets/examples/ehr_format_example.csv \
  --output-dir results/format-example --train-fraction 0.7
```

This writes example preparation artifacts under `results/format-example/`, independently of `datasets/ehr/`. The small fixture demonstrates validation and preparation only; use your actual study data for synthesis.

## Normalize and split the study EHR panel

Place your locally available EHR panel under `datasets/ehr/raw/`, or pass its own path through `--raw-csv`. The historical reference filename is `0813iei_final.csv`.

```bash
python datasets/validate_ehr.py --raw-csv datasets/ehr/raw/0813iei_final.csv
python datasets/prepare_ehr.py --raw-csv datasets/ehr/raw/0813iei_final.csv \
  --output-dir datasets/ehr --train-fraction 0.7
```

Preparation preserves missing values, serializes HPO arrays, creates anonymous record IDs, expands composite Table labels, and splits by patient and source Table stratum with fixed random state 0. The same patient's source rows must have the same source Table stratum. Preserve row order, patient IDs, source labels, and the training fraction when reproducing a split.

The reference panel has 159 source rows and produces 163 normalized rows, with 113 training and 50 test rows. These counts describe the reference input, not arbitrary user data.

```text
datasets/ehr/
├── raw/0813iei_final.csv
├── processed/ehr.csv
└── splits/
    ├── train.csv
    ├── test.csv
    └── split_ids.json
```

Patient-level source and derived files are excluded from version control. A custom output directory can be connected through `config.yaml`:

```yaml
paths:
  dataset_csv: /path/to/prepared/processed/ehr.csv
  train_csv: /path/to/prepared/splits/train.csv
  test_csv: /path/to/prepared/splits/test.csv
  results_dir: /path/to/generated-results
```

These entries replace the corresponding values in the existing full configuration. Relative paths resolve from the repository root when using the documented commands. Keep the prepared files fixed across generation seeds; `--data-seed` does not rebuild the split.

When using your own study data, also choose its vocabulary explicitly. Replace the corresponding entries in the full configuration, for example:

```yaml
data:
  hpo_vocabulary_files:
    ehr: /path/to/your/ehr_hpo_vocabulary.json
    private_ehr: null
```

The file is a JSON object with `format_version: 1`, the selected `dataset` name, nonempty `origin`, `source_description`, and `hpo_ontology_version` strings, and a nonempty `codes` array of unique `HP:` codes. Code order is preserved. A missing, empty, or `null` vocabulary setting causes an error for the selected dataset. No patient records are stored in the vocabulary file.

After configuring the model service, run:

```bash
python main.py --dataset ehr --data-seed 0 --n-samples 1000 --artifact-tag v1
```

For Private EHR, use the same field contract, prepare under `datasets/private_ehr/`, and supply its fixed file through `data.hpo_vocabulary_files.private_ehr`. Then pass `--dataset private_ehr --n-samples 2000` for its reference generation size. The main workflow mines training-derived HPO knowledge and writes synthetic outputs under the configured result directory. See the [reference synthesis settings](../README.md#outputs-and-reproduction).

## Independent data preparation tools

MIMIC extraction and HPO mapping use separate inputs and model configurations. They do not automatically form a complete MIMIC-to-synthesis pipeline. MIMIC records, terminology workbooks, BioLORD weights, and concept vectors are supplied locally by the user; `mimic-process/resources/` can hold local resources and is excluded from version control.

### Required resources

| Tool | Resource | Preparation contract |
| --- | --- | --- |
| MIMIC extraction | MIMIC-IV v3.1 hospital data | `hosp/` with patients, diagnoses_icd, d_icd_diagnoses, labevents, and d_labitems CSV files; plain and gzip files are supported. |
| MIMIC extraction | MIMIC-IV Note v2.2 | Discharge notes under `note/`. |
| MIMIC extraction | IUIS IEI list, July 2024 V2 | Supply the Excel reference with `ICD9` and `ICD10` columns from the [IUIS IEI Committee](https://iuis.org/committees/iei/). |
| MIMIC extraction | Checked IUIS terminology mapping | Supply a reviewed Excel mapping with `IUIS_name` and `IUIS_table_category` columns on its first sheet. The original checked workbook is not included. |
| MIMIC extraction | HPO helper, optional | CSV with `hadm_id` and optional `hpo_codes`/`hpo_descriptions`. A helper with patient/admission keys can constrain the retained cohort. Retain the same helper to reproduce an existing cohort. |
| HPO mapping | Source notes | CSV or XLSX containing Chinese note text and patient/admission IDs. |
| HPO mapping | BioLORD model | Prepare the local model and tokenizer files, e.g. a BioLORD-2023-C directory. Record the actual model revision. |
| HPO mapping | `concept2id.json` | JSON object mapping HPO concept names to HPO IDs. Retain its key order. |
| HPO mapping | `concept_embeddings.pt` | Torch tensor with one vector per concept, in exactly the JSON key order and with an embedding dimension matching the query encoder. |
| HPO mapping | Qwen endpoint, optional | `/completions` service and its model ID. Omission or `--offline_only` selects the local Chinese rule fallback. |

To prepare a checked terminology workbook, use the selected IUIS reference and retain the reviewed disease names and Table assignments in the two required columns. To reproduce the original extraction, obtain the original checked workbook and helper; creating a new mapping can change the cohort.

The concept mapping and tensor must be exported together using the same ontology terms, BioLORD weights, ordering, and encoding convention. The script encodes query text with the first-token hidden state and compares normalized vectors. The reference concept-vector construction recipe, model revision, and concept files are not included, so this guide does not substitute an unverified builder. Use matching existing assets or record your own construction recipe. The included `constraintKB/ontology/hp.obo` supplies ontology terms; it is not a replacement for the concept tensor.

### MIMIC extraction

Copy the supplied example configuration and fill in local paths:

```bash
cp datasets/mimic-process/mimic_ehr_config.example.json \
  datasets/mimic-process/mimic_ehr_config.json
python datasets/mimic-process/iei_ehr_extractor.py \
  --config datasets/mimic-process/mimic_ehr_config.json
```

Relative resource and output paths resolve from the configuration file's directory. The example explicitly sets `hpo_helper_csv` to `null` for extraction without a precomputed helper; specify a file to reuse one. Omitting this key uses `lab_prIGM.csv` from the selected output directory as the helper if available. `reference_final` is optional; use `--compare-reference` with an existing reference file to print the comparison, or `--skip-reference-compare` to disable it.

Outputs include admission tables, `lab_prIGM.csv`, a laboratory mapping, and `0813iei_final.csv`. The method and its existing data rules are retained. Exact cohort size depends on the local inputs and optional helper.

### Chinese-note HPO mapping

The tool retains its Chinese clinical-note prompt and local rules. For the default `patient_id`/`inpat_id` aliases:

```bash
python datasets/mimic-process/hpo_extractor_discharge.py \
  --input_csv /path/to/chinese_notes.csv \
  --output_csv /path/to/hpo_results.csv \
  --text_column original_txt_content \
  --model_path /path/to/BioLORD-2023-C \
  --concept2id_path /path/to/concept2id.json \
  --concept_embeddings_path /path/to/concept_embeddings.pt \
  --api_base_url http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --model_name YOUR_SERVED_QWEN_MODEL \
  --temperature 0.0 --top_p 0.95 --max_tokens 2048 \
  --similarity_threshold 0.5
```

Create the output parent directory before running. All paths use the current working directory. Use `--verbose` to show note excerpts, raw model responses, and per-phenotype mapping details. `--limit N` processes the first N rows. Omission of the endpoint makes no LLM requests; BioLORD and concept resources are still required for mapping.

For custom identifier columns, add:

```text
--patient_id_column subject_id --inpat_id_column hadm_id
```

These parameters select input columns and preserve the output names `patient_id` and `inpat_id`. Without them, the existing identifier aliases are used. Explicit column names are checked before loading models. Identifier selection does not change the supported note language.

### HPO tool settings

| Setting | Command-line default |
| --- | --- |
| Qwen model ID | `Qwen3.6-35B-A3B` |
| Qwen temperature / top-p | 0.0 / 0.95 |
| Maximum completion tokens | 2,048 |
| BioLORD similarity threshold | 0.5 |
| Endpoint | Unset; uses local rule fallback |
| Text column | `original_txt_content`, followed by existing text-column fallbacks |
| Identifier columns | Existing `patient_id` / `inpat_id` aliases |

These are interface defaults, not confirmed reference experiment settings. Direct Python mapping helpers retain their existing similarity threshold default of 0.8; the CLI passes its own 0.5 default. The reference HPO service model revision, similarity threshold, and concept-vector build parameters are not recorded in the repository. Retain them with your local preparation configuration for comparable data.

### Connecting preparation outputs

MIMIC intermediate tables use `subject_id`/`hadm_id`; the final panel uses `patient_identifier`/`hospital_admission_identifier`. The HPO tool writes `patient_id`/`inpat_id`. Preserve identifier values and provide an appropriate Chinese-note input for that tool. Joining HPO results back into the EHR panel remains a separate user preparation step; this repository does not include an automatic merge command. Validate the completed panel against the contract above before synthesis.
