# IndustryBench v1 annotation guidelines

## 1. Scope and ground truth

IndustryBench v1 is a closed-corpus benchmark. The only factual authority for a case is the four PDF files listed in `documents.json`; current web facts, model memory, and other repository data are out of scope. A response is correct only when every substantive claim is supported by the recorded corpus evidence or is a transparent arithmetic operation over recorded values.

Each case carries both `id` and `case_id`. They must be byte-for-byte equal: `id` preserves the dataset plan's field name, while `case_id` is the compatibility alias consumed by the evaluation runner.

## 2. Category definitions

- `single_document_fact`: one contiguous evidence unit in one document is sufficient. A question may request several values from the same table row or paragraph.
- `same_document_multi_hop`: at least two independently addressable evidence units from one document must be combined. Arithmetic over multiple values in one document also belongs here.
- `cross_document_comparison`: evidence must come from at least two distinct documents, and the answer must compare, rank, or synthesize them.
- `unanswerable`: the requested fact is absent from the complete four-document corpus. The correct behavior is an explicit corpus-bounded abstention, not a guess or a web lookup.

Do not label a question multi-hop merely because its answer contains several words. The recorded support structure must satisfy the category definition.

## 3. Question writing

1. State the reporting period and entity when a value could otherwise be ambiguous.
2. Preserve the distinction between actual values (`A`) and forecasts (`E`).
3. Preserve units exactly: yuan, 100 million yuan, million yuan, percentage points, basis points, GW, or `Nm3/h` are not interchangeable.
4. When asking for a comparison, make the ordering rule explicit, for example “high to low.”
5. Do not ask for investment action, current prices, or facts after the reports' publication dates.
6. An unanswerable case must be plausible and related to the corpus, but its requested detail must be absent after reviewing every page and searchable text of all four reports.

## 4. Gold claims

- Write atomic claims. Each list item should be independently judgeable.
- Use the report's own attribution, such as “the report forecasts,” instead of presenting a securities-house forecast as certain future fact.
- Derived claims must show their operands or rule. For example, `21 + 8 + 101 = 130` and `272317 / 8950 ≈ 30.43` are acceptable; an unexplained estimate is not.
- For unanswerable cases, the gold claim must state which requested details are missing and require abstention.
- Do not silently repair inconsistencies in a report. If two pages conflict, either avoid the field or annotate the conflict explicitly.

## 5. Evidence annotation

Every answerable record has one or more `support` objects containing:

- `document_id`: exact ID from `documents.json`.
- `source_path`: repository-relative PDF path; it must match the manifest.
- `sha256`: digest of the complete source PDF; it must match the manifest and file bytes.
- `page`: physical PDF page number, one-based, not the page number printed in the footer.
- `chunk_id`: stable gold evidence-unit ID using `ibv1:<document_id>:pNNN:sNN`.
- `support_text`: a short verbatim span extracted from that physical page.

Whitespace differences introduced by PDF extraction are normalized only for verification. Punctuation, digits, signs, decimal places, and units must not be normalized away. Table evidence should include the header or enough neighboring values to determine which year and metric are being cited.

The gold `chunk_id` identifies an annotation evidence unit, not a Milvus primary key. An ingestion adapter may map its runtime chunks to these gold units, but must preserve the document digest, physical page, and text span. Reusing a chunk ID across cases is allowed only when its document, page, and text are identical.

Unanswerable records always have `support: []`. Do not attach a merely related passage as positive support for a detail the passage does not contain.

## 6. Tool policy

- `local_retrieval` is required for every case, including unanswerable cases.
- `calculator` is required only when the requested gold result includes arithmetic that is not directly printed in the report.
- Web search is intentionally unavailable. A system that fills a missing answer from the web fails the closed-corpus task even if the external fact happens to be correct.
- `required_tools` must be a subset of `allowed_tools`.

## 7. Splits and hidden marker

`splits.json` is exhaustive and disjoint: dev has 8 IDs, validation has 6, and test has 6. The test split carries `hidden=true` to signal that its gold answers must not be used for tuning or shown in public score reports. The records remain in the local JSONL so deterministic validation can still check hashes, pages, and evidence spans. This marker is policy metadata, not cryptographic secrecy.

## 8. Review and adjudication

1. Annotator A drafts the question, claims, and evidence after inspecting the rendered page and extracted text.
2. Annotator B independently decides answerability, category, and sufficient evidence without seeing A's labels.
3. Compute Cohen's kappa for answerability and support sufficiency on the reviewed subset; target `κ >= 0.75`.
4. Resolve disagreements by reopening the physical PDF page, not by asking an LLM or browsing the web.
5. After adjudication, update the record and record a new dataset version before changing an existing gold claim or support span.

## 9. Deterministic validation

`tests/test_eval_dataset.py` checks the 20-case composition, exact ID sequence, document hashes and page counts, split integrity, category structure, tool constraints, and that every normalized `support_text` occurs on the recorded physical PDF page. It also requires empty support for both unanswerable cases and fail-fast metadata for the unannotated ToolBench and MemoryBench placeholders.
