"""Disease-conditioned prompt construction for Curasyn."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import pandas as pd

from src.data import dataframe_to_pid_records, pid_record_to_generation_fields
from src.retrieval import PhenotypeProfile


def build_pid_generation_prompt(
    example_df: pd.DataFrame,
    target_disease_name: str,
    schema: dict[str, Any],
    *,
    hpo_names: Mapping[str, str],
    coverage_profile: PhenotypeProfile,
    background_knowledge: str | None = None,
    background_hpo_codes: Iterable[str] = (),
) -> str:
    disease_column = schema["disease_name_column"]
    from src.knowledge import parse_disease_names

    if example_df.empty or not example_df[disease_column].map(
        lambda value: target_disease_name in parse_disease_names([value])
    ).all():
        raise ValueError(
            "PID prompts may contain exemplars from one target disease only"
        )

    semantic_records = dataframe_to_pid_records(example_df, schema)
    examples = [
        pid_record_to_generation_fields(record, schema)
        for record in semantic_records
    ]
    phenotype_column = schema["phenotype_column"]
    for example in examples:
        example.pop(phenotype_column, None)
    background_codes = set(background_hpo_codes)
    profile_codes = [
        code for code in dict.fromkeys(coverage_profile.codes)
        if code not in background_codes
    ]
    missing_hpo_names = [
        code for code in profile_codes if code not in hpo_names
    ]
    if missing_hpo_names:
        raise ValueError(
            "HPO names are missing for phenotype profile codes: "
            + ", ".join(missing_hpo_names)
        )
    phenotype_profile = "\n".join(
        f"- {code}: {hpo_names[code]}" for code in profile_codes
    )
    if not phenotype_profile:
        if background_codes.intersection(coverage_profile.codes):
            phenotype_profile = (
                "- Disease-HPO frequency guidance is already included in the "
                "clinical background above."
            )
        else:
            phenotype_profile = (
                "- No empirical HPO guidance is available; rely on the clinical "
                "background and disease name."
            )

    def render_number(value: float) -> str:
        return f"{float(value):.6g}"

    demographic_lines = []
    for field in schema["demographic_fields"]:
        name = field["model_column"]
        if field.get("mapping"):
            allowed = json.dumps(list(field["mapping"]), ensure_ascii=False)
            demographic_lines.append(
                f"- {name}: required categorical field; allowed values: {allowed}."
            )
        else:
            bounds = coverage_profile.numerical_bounds.get(name)
            scale = (
                f"; valid scale [{render_number(bounds[0])}, "
                f"{render_number(bounds[1])}]"
                if bounds is not None
                else ""
            )
            demographic_lines.append(
                f"- {name}: required {field['value_type']} field{scale}."
            )
    examination_lines = []
    for field in schema["examination_fields"]:
        name = field["model_column"]
        bounds = coverage_profile.numerical_bounds.get(name)
        if bounds is None and name in coverage_profile.numerical_bounds:
            examination_lines.append(
                f"- {name}: leave null or omit; D_train contains no observed value."
            )
        elif bounds is not None:
            examination_lines.append(
                f"- {name}: optional numeric; valid scale "
                f"[{render_number(bounds[0])}, {render_number(bounds[1])}]."
            )
        else:
            examination_lines.append(f"- {name}: optional numeric field.")
    schema_information = (
        "Schema information:\n"
        "Phenotype:\n"
        f"- {phenotype_column}: required JSON array of HPO codes.\n"
        "Demographics:\n"
        + "\n".join(demographic_lines)
        + "\nExaminations:\n"
        "- All examination fields are optional numeric fields.\n"
        + "\n".join(examination_lines)
    )

    hpo_low, hpo_high = coverage_profile.hpo_count_range
    missing_low, missing_high = coverage_profile.missing_count_range
    examination_targets = "\n".join(
        "- {column}: prioritize the under-covered interval [{lower}, {upper}] "
        "when present.".format(
            column=target["column"],
            lower=render_number(target["lower"]),
            upper=render_number(target["upper"]),
        )
        for target in coverage_profile.examination_targets
    ) or "- No examination interval currently has a positive coverage deficit."
    demographic_targets = []
    for target in coverage_profile.demographic_targets:
        if target["kind"] == "numeric":
            demographic_targets.append(
                "- {column}: prioritize the under-covered interval "
                "[{lower}, {upper}].".format(
                    column=target["column"],
                    lower=render_number(target["lower"]),
                    upper=render_number(target["upper"]),
                )
            )
        else:
            demographic_targets.append(
                f"- {target['column']}: prioritize {target['value']!r}."
            )
    demographic_guidance = "\n".join(demographic_targets) or (
        "- No demographic interval or category currently has a positive "
        "coverage deficit."
    )
    missing_candidates = (
        json.dumps(coverage_profile.missing_columns, ensure_ascii=False)
        if coverage_profile.missing_columns
        else "[]"
    )
    if coverage_profile.coverage_available:
        coverage_guidance = (
            "Accepted-aware clinical coverage guidance. These are aggregate "
            "residual targets for diversity, not mandatory conditions for every "
            "record:\n"
            f"- Typical HPO count target: {hpo_low} to {hpo_high}.\n"
            f"- Typical missing examination count target: {missing_low} to "
            f"{missing_high}.\n"
            "- Under-covered optional fields to leave null in some candidates: "
            f"{missing_candidates}.\n"
            "Under-covered examination intervals:\n"
            f"{examination_targets}\n"
            "Under-covered demographic regions:\n"
            f"{demographic_guidance}"
        )
    else:
        coverage_guidance = (
            "No target-conditioned coverage residual is used for this rare "
            "cohort because its disease and Table groups are below the frozen "
            "support threshold. Follow the target disease, empirical HPO "
            "guidance, valid numeric scales, and examples."
        )

    knowledge_text = background_knowledge or (
        "No disease-specific background knowledge was retrieved."
    )
    return (
        "You are a tabular synthetic data generation model for PID EHR records.\n"
        f"The target IUIS disease name is {target_disease_name}.\n"
        "Generate new P/E/G fields for this target only. Do not output or change "
        "the disease name or record_id.\n"
        "Use the cohort phenotype guidance and same-disease non-phenotype "
        "examples below. Do not reconstruct or copy a real patient.\n\n"
        f"{schema_information}\n\n"
        f"{knowledge_text}\n\n"
        "Privacy-filtered phenotype guidance. Items come only from patterns "
        "supported by multiple real records; this is not a patient-level HPO "
        "profile, and no item is mandatory:\n"
        f"{phenotype_profile}\n"
        f"Use HPO codes, not phenotype names, in the {phenotype_column} output. "
        "Do not invent HPO codes outside the fixed task vocabulary.\n\n"
        f"{coverage_guidance}\n\n"
        "Same-disease non-phenotype examples:\n"
        f"{json.dumps(examples, ensure_ascii=False)}\n\n"
        "Return one or more flat JSON objects."
    )
