#!/usr/bin/env python3
"""
HPO extraction pipeline for pediatric discharge diagnosis or first-course notes.

Pipeline:
1. Read a CSV/XLSX input file.
2. Resolve the source text column, such as original_txt_content, first-course notes,
   or a discharge-diagnosis text column.
3. Extract the diagnosis-related source text used for phenotype inference.
4. Extract Chinese phenotype phrases through a Qwen /completions endpoint when
   --api_base_url is provided; otherwise use the local rule fallback.
5. Convert Chinese phenotype phrases to BioLORD query terms, then map them to HPO
   codes using local BioLORD embeddings and HPO concept resources.
6. Save patient_id, inpat_id, source diagnosis text, Chinese phenotype phrases,
   and mapped HPO codes.

Prerequisites:
- Download the BioLORD model files from HuggingFace in advance and place them in a
  local directory, for example BioLORD-2023-C.
- Prepare concept2id.json and concept_embeddings.pt locally before running this script.
"""
import argparse
import os
import ast
import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Load the BioLORD model from local files.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger(__name__)


DEFAULT_TEXT_COLUMN = "original_txt_content"
EXCEL_I_COLUMN_INDEX = 8
DISCHARGE_DIAGNOSIS_LABEL = "\u51fa\u9662\u8bca\u65ad"
DISCHARGE_DIAGNOSIS_OUTPUT_COLUMN = "\u51fa\u9662\u8bca\u65ad\u6587\u672c"


class HospitalQwenApi:
    """Qwen /completions client configured through --api_base_url."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 0.95,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def get_completion(self, system_prompt: str, prompt: str) -> Optional[str]:
        url = f"{self.base_url}/completions"
        parsed_url = urllib.parse.urlparse(url)
        full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": ["\n\n\n"],
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Content-Length": str(len(data)),
        }

        try:
            if parsed_url.scheme == "https":
                context = urllib.request.HTTPSHandler()
            else:
                context = urllib.request.HTTPHandler()

            opener = urllib.request.build_opener(context)
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with opener.open(req, timeout=180) as response:
                response_text = response.read().decode("utf-8")

            result = json.loads(response_text)
            choices = result.get("choices", [])
            if not choices:
                print("API response has no choices.")
                LOGGER.debug("API response: %s", response_text[:500])
                return None

            first_choice = choices[0]
            if "text" in first_choice:
                return str(first_choice["text"])
            if "message" in first_choice:
                return str(first_choice["message"].get("content", ""))

            print("API response has an unknown choice format.")
            LOGGER.debug("API choice: %s", first_choice)
            return None
        except Exception as e:
            print(f"API request failed: {e}")
            return None


def read_csv_file(file_path: str) -> Optional[pd.DataFrame]:
    """Read CSV/XLSX/XLS with simple format and encoding fallback."""
    try:
        lower_path = file_path.lower()
        if lower_path.endswith((".xlsx", ".xls")):
            return pd.read_excel(file_path)

        encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
        last_error = None
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError as e:
                last_error = e
                continue

        print(f"Error: failed to decode CSV with encodings {encodings}: {last_error}")
        return None
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: There was a parsing error with the file.")
        return None

def resolve_text_column(df: pd.DataFrame, preferred_column: str) -> str:
    candidates = [
        preferred_column,
        DEFAULT_TEXT_COLUMN,
        "original_text",
        "original_txt",
        "original_text_content",
        "首次病程",
        "出院诊断文本",
        "出院诊断",
        "病程记录",
    ]

    for column in candidates:
        if column and column in df.columns:
            return column

    if len(df.columns) > EXCEL_I_COLUMN_INDEX:
        fallback_column = df.columns[EXCEL_I_COLUMN_INDEX]
        print(
            f"Warning: text column '{preferred_column}' not found. "
            f"Using Excel I column instead: '{fallback_column}'"
        )
        return fallback_column

    raise ValueError(
        f"Cannot find text column '{preferred_column}', and the input file "
        "does not have an Excel I column."
    )


def resolve_required_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Cannot find {label} column. Tried: {', '.join(candidates)}")


def normalize_note_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    normalized = str(text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_discharge_diagnosis(text: Any) -> str:
    note = normalize_note_text(text)
    if not note:
        return ""

    start_match = re.search(rf"{DISCHARGE_DIAGNOSIS_LABEL}\s*[:\uff1a]\s*", note)
    if not start_match:
        # If the source column is already a narrative field such as 首次病程,
        # there may be no 出院诊断 label. Use the whole cell text.
        return note

    tail = note[start_match.end():]
    stop_labels = [
        "\u5165\u9662\u60c5\u51b5",
        "\u5165\u9662\u65f6\u60c5\u51b5",
        "\u8bca\u7597\u7ecf\u8fc7",
        "\u6cbb\u7597\u7ecf\u8fc7",
        "\u75c5\u7a0b\u7ecf\u8fc7",
        "\u51fa\u9662\u533b\u5631",
        "\u51fa\u9662\u60c5\u51b5",
        "\u51fa\u9662\u65e5\u671f",
        "\u7b7e\u540d",
        "\u4ee5\u4e0a\u60c5\u51b5",
        "\u9aa8\u9ad3\u7247\u53f7",
        "\u5fc3\u5f69\u8d85\u53f7",
        "\u7b2c\u9875",
    ]
    stop_pattern = r"(?:^|\n)\s*(?:" + "|".join(map(re.escape, stop_labels)) + r")\s*[:\uff1a]"
    stop_match = re.search(stop_pattern, tail)
    section = tail[: stop_match.start()] if stop_match else tail
    return re.sub(r"\s+", " ", section).strip()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_hpo_resources(
    model_path: str,
    concept2id_path: str,
    concept_embeddings_path: str,
) -> Tuple[Any, Any, Dict, torch.Tensor, List[str]]:
    print("Loading BioLORD model and HPO resources...")

    # Load BioLORD strictly from the local directory prepared before runtime.
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)

    with open(concept2id_path, "r", encoding="utf-8") as f:
        concept2id = json.load(f)

    concept_embeddings = torch.load(concept_embeddings_path, map_location="cpu")
    concept_keys = list(concept2id.keys())

    print("BioLORD and HPO resources loaded successfully.")
    return model, tokenizer, concept2id, concept_embeddings, concept_keys


def topk_similarity(
    query_embeddings: torch.Tensor,
    concept_embeddings: torch.Tensor,
    k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    concept_embeddings = torch.nn.functional.normalize(concept_embeddings, p=2, dim=1)
    similarities = torch.matmul(query_embeddings, concept_embeddings.T)
    topk_values, topk_indices = torch.topk(similarities, k, dim=1)
    return topk_indices, topk_values


def map_phenotypes_to_hpo(
    phenotypes: List[str],
    eval_model: Any,
    eval_tokenizer: Any,
    concept2id: Dict,
    concept_embeddings: torch.Tensor,
    concept_keys: List[str],
    similarity_threshold: float = 0.8,
) -> List[Dict]:
    if not phenotypes:
        return []

    biolord_queries = [chinese_phenotype_to_biolord_query(p) for p in phenotypes]

    device = get_device()
    try:
        eval_model = eval_model.to(device)
        concept_embeddings = concept_embeddings.to(device)
    except Exception as e:
        print(f"Failed to move resources to {device}: {e}")
        device = torch.device("cpu")
        eval_model = eval_model.to(device)
        concept_embeddings = concept_embeddings.to(device)

    phenotype_embeddings_list = []
    batch_size = 30
    for i in range(0, len(biolord_queries), batch_size):
        batch_phenotypes = biolord_queries[i:i + batch_size]
        try:
            inputs = eval_tokenizer(
                batch_phenotypes,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = eval_model(**inputs)
            phenotype_embeddings_list.append(outputs.last_hidden_state[:, 0, :])
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == "cuda":
                print("GPU OOM, switching to CPU")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                eval_model = eval_model.to(device)
                concept_embeddings = concept_embeddings.to(device)
                phenotype_embeddings_list = [
                    emb.to(device) for emb in phenotype_embeddings_list
                ]
                inputs = eval_tokenizer(
                    batch_phenotypes,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = eval_model(**inputs)
                phenotype_embeddings_list.append(outputs.last_hidden_state[:, 0, :])
            else:
                raise e

    phenotype_embeddings = torch.cat(phenotype_embeddings_list, 0)
    topk_indices, topk_values = topk_similarity(phenotype_embeddings, concept_embeddings, k=1)
    topk_indices = topk_indices.cpu().numpy().tolist()
    topk_values = topk_values.cpu().numpy().tolist()

    concept_values = list(concept2id.values())
    mapped_results = []
    seen_hpo_codes = set()

    for i, phenotype in enumerate(phenotypes):
        mapping_query = biolord_queries[i]
        best_match_idx = topk_indices[i][0]
        similarity_score = topk_values[i][0]

        if similarity_score < similarity_threshold:
            LOGGER.debug(
                "Dropping phenotype %r due to low similarity score: %.3f",
                phenotype, similarity_score,
            )
            mapped_results.append(
                {
                    "original_phenotype": phenotype,
                    "mapping_query": mapping_query,
                    "hpo_code": None,
                    "hpo_term": None,
                    "similarity_score": similarity_score,
                    "status": "low_similarity",
                }
            )
            continue

        mapped_hpo_code = concept_values[best_match_idx]
        mapped_concept_name = concept_keys[best_match_idx]
        if mapped_hpo_code in seen_hpo_codes:
            mapped_results.append(
                {
                    "original_phenotype": phenotype,
                    "mapping_query": mapping_query,
                    "hpo_code": mapped_hpo_code,
                    "hpo_term": mapped_concept_name,
                    "similarity_score": similarity_score,
                    "status": "duplicate",
                }
            )
            continue

        seen_hpo_codes.add(mapped_hpo_code)
        LOGGER.debug(
            "Mapping phenotype %r to HPO code %r with similarity score %.3f",
            phenotype, mapped_hpo_code, similarity_score,
        )
        mapped_results.append(
            {
                "original_phenotype": phenotype,
                "mapping_query": mapping_query,
                "hpo_code": mapped_hpo_code,
                "hpo_term": mapped_concept_name,
                "similarity_score": similarity_score,
                "status": "mapped",
            }
        )

    return mapped_results


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


CHINESE_PHENOTYPE_QUERY_RULES = [
    ("反复发生感染", "反复感染", "Recurrent infections"),
    ("反复感染", "反复感染", "Recurrent infections"),
    ("肺炎", "肺炎", "Pneumonia"),
    ("真菌感染", "真菌感染", "Fungal infections"),
    ("皮肤感染", "皮肤感染", "Skin infection"),
    ("脓肿", "脓肿", "Abscess"),
    ("腹泻", "腹泻", "Diarrhea"),
    ("营养不良", "营养不良", "Malnutrition"),
    ("低蛋白血症", "低蛋白血症", "Hypoproteinemia"),
    ("ANC降低", "中性粒细胞减少", "Neutropenia"),
    ("中性粒细胞计数下降", "中性粒细胞减少", "Neutropenia"),
    ("中性粒细胞减少", "中性粒细胞减少", "Neutropenia"),
    ("血小板减少", "血小板减少", "Thrombocytopenia"),
    ("贫血", "贫血", "Anemia"),
    ("IgM升高", "IgM升高", "Elevated IgM level"),
    ("高IgM血症", "IgM升高", "Elevated IgM level"),
    ("免疫球蛋白水平降低", "免疫球蛋白降低", "Decreased immunoglobulin level"),
    ("免疫球蛋白降低", "免疫球蛋白降低", "Decreased immunoglobulin level"),
    ("肝大", "肝大", "Hepatomegaly"),
    ("脾大", "脾大", "Splenomegaly"),
    ("淋巴结肿大", "淋巴结肿大", "Lymphadenopathy"),
    ("浅表淋巴结增大", "淋巴结肿大", "Lymphadenopathy"),
    ("湿疹", "湿疹", "Eczema"),
    ("鼻衄", "鼻衄", "Epistaxis"),
    ("肺出血", "肺出血", "Pulmonary hemorrhage"),
    ("癫痫", "癫痫", "Seizure"),
    ("听力下降", "听力下降", "Hearing impairment"),
    ("蛋白尿", "蛋白尿", "Proteinuria"),
    ("血尿", "血尿", "Hematuria"),
    ("低钾血症", "低钾血症", "Hypokalemia"),
]


def chinese_phenotype_to_biolord_query(phenotype: str) -> str:
    for trigger, _chinese_phenotype, english_query in CHINESE_PHENOTYPE_QUERY_RULES:
        if trigger in phenotype:
            return english_query
    return phenotype


def fallback_phenotypes_from_chinese_text(text: str) -> List[str]:
    """Extract Chinese phenotype keywords when no usable model output is available."""
    return unique_preserve_order(
        chinese_phenotype
        for trigger, chinese_phenotype, _english_query in CHINESE_PHENOTYPE_QUERY_RULES
        if trigger in text
    )


def parse_phenotype_response(response: Optional[str]) -> List[str]:
    if response is None:
        return []

    extracted_phenotypes = []
    cleaned = response.strip()

    candidates = []
    if cleaned.startswith("[") and cleaned.endswith("]"):
        candidates.append(cleaned)
    candidates.extend(
        line.strip().strip(",")
        for line in response.splitlines()
        if line.strip().startswith("{") and line.strip().rstrip(",").endswith("}")
    )

    for item in candidates:
        try:
            parsed = ast.literal_eval(item)
        except (SyntaxError, ValueError):
            continue

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            continue

        for phenotype_data in parsed:
            if not isinstance(phenotype_data, dict):
                continue
            phenotype = phenotype_data.get("Phenotype") or phenotype_data.get("phenotype")
            if phenotype:
                extracted_phenotypes.append(str(phenotype).strip())

    if not extracted_phenotypes:
        regex_matches = re.findall(
            r"(?:Phenotype|phenotype)\s*[:：]\s*['\"]?([^'\"\n,，;；}]+)",
            response,
        )
        extracted_phenotypes.extend(match.strip() for match in regex_matches)

    return unique_preserve_order(extracted_phenotypes)

def extract_phenotypes_from_discharge_diagnosis(
    text: str,
    api: Optional[HospitalQwenApi],
) -> List[str]:
    if not text:
        return []

    system_prompt = (
        "You are a medical expert specialized in rare disease and Human "
        "Phenotype Ontology phenotype extraction."
    )
    prompt = (
        "The input is a Chinese pediatric discharge diagnosis or first-course diagnosis note. "
        "Extract only Chinese clinical phenotype phrases that can be mapped to HPO. "
        "Do not extract disease names themselves, treatments, departments, dates, IDs, or administrative information. "
        "Ignore negated, absent, excluded, suspected, uncertain, pending, or question-mark findings.\n\n"
        "For each phenotype, output exactly one Python-dict-like object per line:\n"
        "{'Phenotype': 'Chinese phenotype phrase'}\n\n"
        "Only output these objects. Do not output explanations or markdown.\n\n"
        f"Source text: {text}"
    )
    if api is None:
        phenotypes = fallback_phenotypes_from_chinese_text(text)
        LOGGER.debug("Offline fallback phenotypes: %s", phenotypes)
        return phenotypes

    response = api.get_completion(system_prompt, prompt)
    LOGGER.debug("Raw LLM response: %s", response)
    phenotypes = parse_phenotype_response(response)
    if not phenotypes:
        phenotypes = fallback_phenotypes_from_chinese_text(text)
        LOGGER.debug("Fallback phenotypes: %s", phenotypes)
    return phenotypes


def process_discharge_diagnosis_list(
    discharge_diagnosis_list: List[str],
    api_key: str,
    api_base_url: str,
    hpo_model: Any,
    hpo_tokenizer: Any,
    concept2id: Dict,
    concept_embeddings: torch.Tensor,
    concept_keys: List[str],
    model: str = "Qwen3.6-35B-A3B",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    similarity_threshold: float = 0.8,
    offline_only: bool = False,
) -> List[Dict]:
    if offline_only:
        api = None
        print("Offline-only extraction: no LLM API calls.")
    elif api_base_url:
        api = HospitalQwenApi(api_key, model, api_base_url, temperature, max_tokens, top_p)
        print(f"Using Qwen API: {api_base_url}")
    else:
        api = None
        print("No --api_base_url supplied: using local rule fallback without Qwen API.")
    results = []

    for i, diagnosis_text in enumerate(discharge_diagnosis_list):
        print(f"Processing case {i + 1}/{len(discharge_diagnosis_list)}")
        LOGGER.debug("Discharge diagnosis text: %s", diagnosis_text[:200])

        extracted_phenotypes = extract_phenotypes_from_discharge_diagnosis(
            diagnosis_text,
            api,
        )
        print(f"Extracted {len(extracted_phenotypes)} phenotypes")

        if extracted_phenotypes:
            hpo_mappings = map_phenotypes_to_hpo(
                extracted_phenotypes,
                hpo_model,
                hpo_tokenizer,
                concept2id,
                concept_embeddings,
                concept_keys,
                similarity_threshold,
            )
        else:
            hpo_mappings = []

        mapped_count = len([m for m in hpo_mappings if m["status"] == "mapped"])
        print(f"Successfully mapped {mapped_count} HPO terms")
        print("-" * 80)

        results.append(
            {
                "discharge_diagnosis_text": diagnosis_text,
                "extracted_phenotypes": extracted_phenotypes,
                "hpo_mappings": hpo_mappings,
                "status": "success",
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="HPO extraction from pediatric discharge diagnosis or first-course notes"
    )
    parser.add_argument("--input_csv", required=True, help="Path to input CSV/XLSX file")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV file; its parent directory must exist")
    parser.add_argument(
        "--patient_id_column",
        help="Input patient identifier column; omitted values use the existing patient_id aliases",
    )
    parser.add_argument(
        "--inpat_id_column",
        help="Input admission identifier column; omitted values use the existing inpat_id aliases",
    )
    parser.add_argument(
        "--text_column",
        default=DEFAULT_TEXT_COLUMN,
        help="Column containing original note text. Default: original_txt_content",
    )
    parser.add_argument("--api_key", default="", help="Qwen API key. Not needed with --offline_only.")
    parser.add_argument(
        "--api_base_url",
        default=None,
        help="Qwen base URL, for example http://127.0.0.1:8000/v1. If omitted, no API call is made.",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen3.6-35B-A3B",
        help="Model ID exposed by the Qwen service (default: Qwen3.6-35B-A3B)",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local BioLORD model directory downloaded from HuggingFace in advance, for example BioLORD-2023-C.",
    )
    parser.add_argument("--concept2id_path", required=True, help="Path to local concept2id.json generated from HPO resources")
    parser.add_argument(
        "--concept_embeddings_path",
        required=True,
        help="Path to local concept_embeddings.pt generated with the BioLORD model",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Qwen sampling temperature (default: 0.0)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum Qwen completion tokens (default: 2048)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Qwen top-p sampling parameter (default: 0.95)")
    parser.add_argument("--similarity_threshold", type=float, default=0.5, help="Minimum BioLORD cosine similarity for a mapping (CLI default: 0.5)")
    parser.add_argument(
        "--offline_only",
        action="store_true",
        help="Extract phenotypes with local Chinese rules without calling an LLM API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows for validation or testing.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show source text, raw model responses, and phenotype mapping details.",
    )

    args = parser.parse_args()
    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    if not LOGGER.handlers:
        LOGGER.addHandler(logging.StreamHandler())
    LOGGER.propagate = False

    print(f"Reading input file: {args.input_csv}")
    df = read_csv_file(args.input_csv)
    if df is None:
        return
    if args.limit is not None:
        df = df.head(args.limit).copy()
        print(f"Processing first {len(df)} rows (--limit).")

    patient_id_column = resolve_required_column(
        df,
        [args.patient_id_column] if args.patient_id_column else
        ["patient_id", "patientId", "patientID", "患者编号", "患者ID", "病人ID"],
        "patient_id",
    )
    inpat_id_column = resolve_required_column(
        df,
        [args.inpat_id_column] if args.inpat_id_column else
        ["inpat_id", "inpatId", "inpatID", "就诊编号", "住院号", "住院ID"],
        "inpat_id",
    )

    text_column = resolve_text_column(df, args.text_column)
    print(f"Using source text column: {text_column}")

    df["discharge_diagnosis_text"] = df[text_column].apply(extract_discharge_diagnosis)
    empty_count = int((df["discharge_diagnosis_text"] == "").sum())
    if empty_count:
        print(f"Warning: {empty_count} rows have no extracted discharge diagnosis.")

    hpo_model, hpo_tokenizer, concept2id, concept_embeddings, concept_keys = (
        load_hpo_resources(
            args.model_path,
            args.concept2id_path,
            args.concept_embeddings_path,
        )
    )

    results = process_discharge_diagnosis_list(
        df["discharge_diagnosis_text"].tolist(),
        args.api_key,
        args.api_base_url,
        hpo_model,
        hpo_tokenizer,
        concept2id,
        concept_embeddings,
        concept_keys,
        model=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        similarity_threshold=args.similarity_threshold,
        offline_only=args.offline_only,
    )

    hpo_codes_list = [
        [m["hpo_code"] for m in result["hpo_mappings"] if m["status"] == "mapped"]
        for result in results
    ]

    output_df = pd.DataFrame(
        {
            "patient_id": df[patient_id_column],
            "inpat_id": df[inpat_id_column],
            "基于的首次病程诊断记录": df["discharge_diagnosis_text"],
            "中文phenotype": [str(result["extracted_phenotypes"]) for result in results],
            "hpo_codes": [str(codes) for codes in hpo_codes_list],
        }
    )

    print(f"Saving results to: {args.output_csv}")
    output_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    total_cases = len(results)
    total_mapped_hpo = sum(
        len([m for m in result["hpo_mappings"] if m["status"] == "mapped"])
        for result in results
    )
    print("\n=== Processing Summary ===")
    print(f"Total cases processed: {total_cases}")
    print(f"Rows without discharge diagnosis: {empty_count}")
    print(f"Total HPO terms mapped: {total_mapped_hpo}")
    print(f"Average HPO terms per case: {total_mapped_hpo / total_cases:.2f}")


if __name__ == "__main__":
    main()
