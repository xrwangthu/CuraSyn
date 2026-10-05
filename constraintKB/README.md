# Static Clinical Knowledge

This directory contains the static disease knowledge used by Curasyn.

- `data/constraints.json`: source knowledge records.
- `disease_mapping.csv`: disease alias mapping.
- `build_kb.py`: SQLite construction script.
- `kb/clinical_knowledge.sqlite`: prepared knowledge database.
- `ontology/hp.obo`: HPO identifiers and names.
- `ontology/ehr_hpo_vocabulary.json`: ordered 650-code reference-EHR task vocabulary and its source description.

The vocabulary is dataset-derived: it freezes the HPO union of the complete normalized reference EHR panel. It is configured through `data.hpo_vocabulary_files.ehr` and preserves the vocabulary used by the existing method. It does not establish an independent pre-test vocabulary. Training and test records share one vocabulary per dataset. Literature extraction uses the same file through `hpo_vocabulary_path`. Both stages require file configuration and never infer a vocabulary from CSV records. Private EHR requires its own file through the corresponding configuration entry. See the [vocabulary configuration](../README.md#fixed-hpo-vocabulary).
- `api.py`: query interface exposing `ClinicalKnowledgeBase`, `KnowledgeRetrieval`, and `render_knowledge`.

The query implementation currently resides in `src/knowledge.py`; `api.py` exposes it without changing the generation workflow.
Training-derived relations and dynamic HPO probabilities are implemented in `src/knowledge.py` and `src/retrieval.py`.

New knowledge is prepared through [create_kb](create_kb/README.md): paper Markdown → atomic constraints with evidence → grouped JSON → SQLite. Run preparation commands from `constraintKB/create_kb/`; `build_kb.py` remains the shared database builder. The builder uses examination patterns to identify target codes, and reports unmapped examination nodes.
