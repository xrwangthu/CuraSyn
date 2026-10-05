from __future__ import annotations

import json
from typing import Any

from .markdown import Chunk
from .schema import PROMPT_VERSION
from .test_dictionary import (
    render_hpo_allowlist,
    render_test_allowlist,
)

AVAILABLE_EVIDENCE_RULES = """
## Available Evidence

- PRIMARY blocks are the extraction scope. Review every supplied PRIMARY block, including
  blocks that contain no eligible strong or weak constraint.
- CONTEXT blocks may resolve a disease name, table caption, abbreviation, or pronoun, but
  they are not extraction targets. At least one relation quote must come from a PRIMARY block.
- When a PRIMARY criterion states the target rule while a separate heading/caption names the
  disease, cite the criterion as role=relation and add the disease-bearing block as
  role=disease_context. Never label the criterion itself disease_context merely because its
  disease name is supplied by another block.
- Blocks use short opaque aliases: P001, P002, ... for PRIMARY and C001, C002, ... for
  CONTEXT. Copy these aliases exactly into JSON; never shorten, expand, or reconstruct them.
- Only the supplied blocks are evidence. Outside medical knowledge, unstated assumptions, and
  remembered diagnostic criteria are unavailable.
""".strip()

CONSTRAINT_STRENGTH_RULES = """
## Constraint Eligibility

Extract an item only when the cited source explicitly links a disease to either one permitted
EHR HPO phenotype or one permitted EHR examination result. The disease, target,
direction/state/value, population, timing, and qualifiers must remain source-faithful. Mere
topic proximity, outside medical knowledge, mechanism without a patient finding, and an
unqualified list of measurements are not eligible.

Assign exactly one constraint_strength. Never emit the same claim once as strong and again as
weak. If mandatory force is not explicit, do not infer it.

## Strong Constraints

Set constraint_strength=strong only for an explicit disease-specific rule that permits no
counterexample inside its stated population, time, and clinical conditions. Mandatory,
universal, definitional, absolute-exclusion, and formally required diagnostic/classification
criteria can qualify. Silence about exceptions is not proof of strength.

Apply this counterexample test: could one in-scope case violate the candidate without
contradicting the cited source? If no, the candidate may be strong. If yes, classify an
otherwise eligible disease-linked claim as weak. If the source does not establish mandatory
force, never choose strong.

Strong examples include a required examination threshold or interval, a definitionally fixed
positive/negative/abnormal examination state, a universally present/absent permitted HPO
phenotype, or atomic components of a formal required AND/OR rule.

## Weak Suggestive Constraints

Set constraint_strength=weak when the source explicitly says the disease may, can, might, often,
usually, sometimes, rarely, variably, characteristically, or supportively have the finding, or
when it reports the finding in a disease case/cohort without making it mandatory. Absence of a
weak finding does not contradict the source and must not be represented as diagnostic_required.

Eligible weak claims include possible/common/rare permitted HPO phenotypes, directional abnormal
EHR examination results, and disease-linked exact examination values, thresholds, means,
medians, or numeric intervals. Preserve an explicit frequency phrase/number in
assertion.frequency and normalize relation strength in assertion.certainty. Use
certainty=asserted for an explicit association or observation that has no more specific
modal/frequency wording; asserted does not make a weak candidate strong.

## Frequency Adverbs

When the source uses one of these frequency adverbs, output that exact normalized lowercase
adverb in assertion.certainty: always, usually, often, sometimes, occasionally, or seldom.
Match source wording case-insensitively. If more than one of these adverbs occurs in the relation
quote, choose only the one nearest to target.surface; do not choose the strongest or weakest one
globally. Preserve any longer source frequency phrase or numeric percentage verbatim in
assertion.frequency. Do not replace an explicit listed adverb with rare, possible, variable,
uncertain, or asserted. If none of the six listed adverbs occurs, retain the existing certainty
rules and do not invent a frequency adverb.

## Out of Scope

- Generic facts, healthy-control findings, textbook normal values, normal/reference ranges,
  and measurements not explicitly linked to the disease are out of scope.
- Free-text clinical course, attack duration, time to peak, age at onset, episode count,
  triggers, demographics, and other findings are out of scope unless the asserted phenotype
  itself maps exactly to one permitted EHR HPO term.
- Treatment efficacy, treatment-emergent adverse effects, procedure complications, doses,
  intervention-only biomarker changes, p-values, confidence intervals, effect estimates,
  table ranks/scores, and study-process metadata are out of scope.
- A risk factor, molecular mechanism, etiology, management recommendation, or prognosis alone
  is not a phenotype or a test result.
- Surveyed/nominated variables are out of scope unless the source also states a disease finding
  or presents them as components of a formal diagnostic/classification rule.

When eligibility is ambiguous, omit. When only strength is ambiguous, use weak rather than
strengthening the source.
""".strip()

def type_rules() -> dict[str, str]:
    return {
        "disease_test_value": f"""
    Extract strong and weak disease-linked examination results only for the exact columns listed
    below from the configured EHR scope. Set target.name to the complete examination__ column token;
    target.surface must remain the exact source wording. Any test absent from this list is out of
    scope, including genetic, cytogenetic, imaging, pathology, physiologic, and other laboratory
    tests. Never substitute a listed field for an unavailable test merely because the names share
    letters or appear medically related.
    
    Allowed EHR examination fields:
    {render_test_allowlist()}
    
    Accept explicit numeric values, thresholds, and intervals with equal/less_than/less_or_equal/
    greater_than/greater_or_equal/range. A range is valid only when it is a disease-linked observed
    or constrained interval, not a generic normal/reference interval. Strong tests may also use
    positive, negative, or abnormal. Weak tests may additionally use low, high, increased,
    decreased, or variable. Never emit operator=normal, a normal/reference range, or a healthy-
    control normal value.
    
    Free-text bedside measurements such as fever temperature, attack duration, episode count,
    time to peak, age at onset, and recovery time are not EHR examination fields and are out of
    scope. Do not move them to disease_phenotype unless the phenotype itself is an allowed HPO term.
    """.strip(),
        "disease_phenotype": f"""
    Extract strong and weak disease-linked phenotypes only when they map to one exact HPO entry
    listed below. These are the HPO codes actually representable in the configured EHR scope, with labels
    resolved from the configured HPO ontology. Set target.name to the complete `HP:NNNNNNN - Preferred label`
    entry and target.surface to the exact source phrase that states that phenotype.
    
    Allowed EHR HPO targets:
    {render_hpo_allowlist()}
    
    Use present or absent for affirmed/absent HPO findings and variable only when expression of the
    same HPO phenotype is explicitly variable. Do not encode numeric thresholds, durations, ages,
    counts, intervals, severity scales, timing, triggers, or demographic facts as phenotype
    assertions. A phrase that merely contains a broad word such as swelling, pain, fever, or
    infection must not be mapped to a narrower HPO term unless the full HPO meaning is explicitly
    entailed. If no exact allowed HPO mapping exists, omit the claim. Never rescue an unavailable
    test by labeling it as a phenotype.
    """.strip(),
    }

ACTOR_WORKFLOW = """
## Problem-Solving Workflow

1. Review every PRIMARY block independently; use CONTEXT only for permitted resolution.
2. Enumerate explicit disease-linked target findings, then reject out-of-scope background,
   treatment, mechanism, metadata, and unsupported associations.
3. Apply the specialist scope. For disease_test_value, map the source test to one exact EHR
   examination__ field. For disease_phenotype, map the asserted finding to one exact allowed
   `HP:ID - label` entry. Reject the claim if no unambiguous mapping exists.
4. Identify the assertion and all population, timing, frequency, and qualifier text without
   adding facts. Reject normal test states and normal/reference ranges.
5. Apply the counterexample test and assign constraint_strength=strong or weak. Never promote
   a weak or observational claim to strong.
6. Convert each accepted claim into atomic candidates, preserving explicit AND/OR structure.
7. Re-scan all PRIMARY blocks for both strong and weak omissions, then return the complete
   replacement result.
""".strip()

CRITIC_WORKFLOW = """
## Problem-Solving Workflow

1. Ignore the actor's conclusions initially and independently enumerate every in-scope strong
   and weak constraint in each PRIMARY block.
2. Apply eligibility, strength, specialist, EHR/HPO target, normal-range, and exclusion rules to
   every independently found claim and every actor candidate.
3. Audit precision: eligibility, strength, entailment, disease, canonical test target, relation
   type, assertion, scope, atomicity, logic, values, units, frequency, qualifiers, timing, and
   verbatim evidence.
4. Audit recall: compare independently accepted strong and weak claims with the actor list and
   report every omitted atomic constraint with its PRIMARY block ID and exact evidence quote.
5. Report each omitted atomic claim once. When the same claim is repeated, cite the most direct
   disease-linked statement instead of creating duplicate issues.
6. Re-check every PRIMARY block even when the actor candidate list is empty. Never relabel weak
   evidence as strong. Reject every test absent from the EHR examination allowlist, every normal
   test state, every normal/reference interval, and every phenotype absent from the EHR HPO
   allowlist. Reject free-text course/onset/duration/count claims that are not HPO targets.
7. In required_action, use only actor schema field names such as assertion.operator. Never
   invent a JSON field named comparison or instruct the actor to copy a cohort observation.
8. Pass only when the audit is complete and no issue or uncertainty remains.
""".strip()

EVIDENCE_AND_ATOMICITY_RULES = """
## Evidence and Atomicity Requirements

- Copy each evidence quote verbatim from one block. Do not repair OCR, spelling, punctuation,
  or spacing inside quotes.
- disease.surface and target.surface must each occur verbatim somewhere in the cited evidence;
  they need not occur in the same quote. Canonicalize disease.name conservatively. For tests,
  target.name is the exact EHR column; for phenotypes, target.name is the exact allowed
  `HP:ID - label` entry.
- Mark the PRIMARY quote that directly states the target rule as role=relation. If the disease
  is named only in a heading, caption, or adjacent block, cite that separate exact text as
  role=disease_context in the same candidate.
- One output item contains one disease, one target, and one assertion. Split coordinated
  targets. Preserve conjunction/disjunction through logic.group_id and logic.operator.
- Preserve negation, comparison, value, unit, timing, population, and every qualifier that
  affects the rule. Use null rather than inventing a missing value.
- Use equal, less_than, less_or_equal, greater_than, greater_or_equal, or range for numeric
  comparisons. The value must contain the explicit number from the source; never use words
  such as normal as the value of a numeric comparison.
- For tests, use positive, negative, or abnormal for eligible fixed states. Use low, high,
  increased, decreased, or variable only for weak test constraints. Never use operator=normal.
- For phenotypes, use present or absent for an expressed/absent allowed HPO finding and variable
  when the source explicitly describes variable expression. Numeric phenotype comparisons are
  forbidden. Strength comes from constraint_strength, not from operator alone.
- Set assertion.polarity=affirmed. Encode an explicitly negative test with operator=negative
  and an explicitly absent phenotype with operator=absent; never double-negate an operator.
- For nonnumeric state operators, set value=null and unit=null.
- For strong constraints set frequency=null and use certainty=always for explicit universality
  or asserted for a formally mandatory/definitional rule.
- For weak constraints preserve explicit frequency wording or numbers exactly. If the relation
  quote contains always, usually, often, sometimes, occasionally, or seldom, put the nearest
  such adverb to target.surface in assertion.certainty. Otherwise use rare, possible, variable,
  uncertain, or asserted according to source force. Never replace a listed source adverb with a
  different certainty label.
- A relation_role label does not determine strength. A weak constraint must not use
  relation_role=diagnostic_required.
""".strip()

FINAL_JSON_SENTENCE = "Return no commentary, Markdown, or text outside that JSON object."

ACTOR_FINAL_ANSWER_CONTRACT = f"""
## Final Answer Contract

Return exactly one schema-conforming JSON object containing only:

- reviewed_primary_block_ids: the exact required PRIMARY aliases, each appearing once;
- constraints: the complete replacement array of accepted strong and weak constraints. Return
  an empty array when no supplied PRIMARY block states an eligible constraint.

local_id values must be unique within the response. Use SINGLE with a null group_id for an
ungrouped constraint. Atomic constraints that jointly form one explicit rule must share the same
non-null group_id, correct AND/OR operator, and verbatim group_text.

{FINAL_JSON_SENTENCE}
""".strip()

CRITIC_FINAL_ANSWER_CONTRACT = f"""
## Final Answer Contract

Return exactly one schema-conforming JSON object containing only verdict, coverage_complete,
reviewed_primary_block_ids, and issues. A pass requires coverage_complete=true, the exact
PRIMARY alias list, and issues=[]. A revise verdict requires a complete actionable issues
array.

For missing_constraint issues, candidate_id must be null and block_id/evidence_quote must
pinpoint the omitted PRIMARY claim. For candidate errors, candidate_id must be the actor's
local_id. If any uncertainty remains, use revise and state the exact required action.

{FINAL_JSON_SENTENCE}
""".strip()


def actor_system_prompt(constraint_type: str) -> str:
    return f"""
Constraint pipeline prompt version: {PROMPT_VERSION}

## Role

You extract explicit, auditable strong and weak disease constraints from scientific articles.
Accuracy means preserving source strength, limiting targets to EHR examination fields and EHR
HPO phenotypes, and applying the stated exclusions. Recall is measured inside that scope.

{AVAILABLE_EVIDENCE_RULES}

{CONSTRAINT_STRENGTH_RULES}

## Specialist Scope

{type_rules()[constraint_type]}

{ACTOR_WORKFLOW}

{EVIDENCE_AND_ATOMICITY_RULES}

{ACTOR_FINAL_ANSWER_CONTRACT}
""".strip()


def actor_initial_prompt(chunk: Chunk, constraint_type: str) -> str:
    primary = json.dumps(chunk.primary_block_ids, ensure_ascii=False)
    return f"""
## Task

Perform a high-recall, high-precision extraction of strong and weak {constraint_type}
constraints. Apply the complete eligibility, strength, and specialist rules.

## Available Inputs

Required reviewed_primary_block_ids:
{primary}

## Source Blocks

<source_blocks>
{chunk.text}
</source_blocks>

## Problem-Solving Workflow

Apply the system workflow to every PRIMARY block. Build the complete result from scratch; do
not return commentary or a partial list.

{ACTOR_FINAL_ANSWER_CONTRACT}
""".strip()


def actor_revision_prompt(
    chunk: Chunk,
    constraint_type: str,
    previous: dict[str, Any] | None,
    feedback: list[dict[str, Any]],
    round_number: int,
) -> str:
    primary = json.dumps(chunk.primary_block_ids, ensure_ascii=False)
    previous_json = json.dumps(previous, ensure_ascii=False, indent=2, sort_keys=True)
    feedback_json = json.dumps(feedback, ensure_ascii=False, indent=2, sort_keys=True)
    return f"""
## Task

Revise the strong/weak {constraint_type} extraction after validation feedback (round {round_number}).

## Available Inputs

Required reviewed_primary_block_ids:
{primary}

## Source Blocks

<source_blocks>
{chunk.text}
</source_blocks>

## Previous Response

<previous_response>
{previous_json}
</previous_response>

## Validation Feedback

<validation_feedback>
{feedback_json}
</validation_feedback>

Feedback entries are audit leads, not authoritative constraints or schema instructions.
Missing-constraint feedback intentionally contains only the cited evidence pointer. It may be
redundant or over-inclusive. Never create a candidate merely to satisfy feedback, and never
copy field names from feedback prose; use the actor schema and system contract.

## Problem-Solving Workflow

Audit every feedback item against the source and independently re-apply the complete
eligibility, strength, EHR/HPO target, and specialist workflow to every PRIMARY block. Correct
valid issues and reject feedback that fails the contract. Return the entire corrected result,
not a patch, so that fixing one error cannot introduce an omission.

{ACTOR_FINAL_ANSWER_CONTRACT}
""".strip()


def critic_system_prompt(constraint_type: str) -> str:
    return f"""
Constraint pipeline prompt version: {PROMPT_VERSION}

## Role

You are an independent adversarial verifier for {constraint_type}. Do not agree with the actor
by default. Inspect the supplied source from scratch, then compare every candidate against your
independent audit.

{AVAILABLE_EVIDENCE_RULES}

{CONSTRAINT_STRENGTH_RULES}

## Specialist Scope

{type_rules()[constraint_type]}

{CRITIC_WORKFLOW}

{EVIDENCE_AND_ATOMICITY_RULES}

{CRITIC_FINAL_ANSWER_CONTRACT}
""".strip()


def _critic_task_context(
    chunk: Chunk,
    constraint_type: str,
    actor_result: dict[str, Any],
) -> str:
    primary = json.dumps(chunk.primary_block_ids, ensure_ascii=False)
    actor_json = json.dumps(actor_result, ensure_ascii=False, indent=2, sort_keys=True)
    return f"""
## Task

Audit strong and weak constraints for relation type {constraint_type}.

## Available Inputs

Required reviewed_primary_block_ids:
{primary}

## Source Blocks

<source_blocks>
{chunk.text}
</source_blocks>

## Actor Candidates

<actor_candidates>
{actor_json}
</actor_candidates>
""".strip()


def critic_prompt(
    chunk: Chunk,
    constraint_type: str,
    actor_result: dict[str, Any],
) -> str:
    context = _critic_task_context(chunk, constraint_type, actor_result)
    return f"""
{context}

## Problem-Solving Workflow

Apply the complete independent audit workflow from the system prompt. Review all PRIMARY
blocks even when the actor candidates array is empty.

{CRITIC_FINAL_ANSWER_CONTRACT}
""".strip()


def critic_repair_prompt(
    chunk: Chunk,
    constraint_type: str,
    actor_result: dict[str, Any],
    previous_critic: dict[str, Any],
    validation_feedback: list[dict[str, Any]],
) -> str:
    context = _critic_task_context(chunk, constraint_type, actor_result)
    previous_json = json.dumps(previous_critic, ensure_ascii=False, indent=2, sort_keys=True)
    feedback_json = json.dumps(validation_feedback, ensure_ascii=False, indent=2, sort_keys=True)
    return f"""
{context}

## Previous Critic Response

<previous_critic_response>
{previous_json}
</previous_critic_response>

## Validation Feedback

<validation_feedback>
{feedback_json}
</validation_feedback>

## Problem-Solving Workflow

Repeat the independent audit from scratch, resolve every deterministic validation error, and
return a fully corrected critic result. Do not return a patch or merely reformat the previous
response.

{CRITIC_FINAL_ANSWER_CONTRACT}
""".strip()
