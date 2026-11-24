# Answer check of the evaluation questions using KG metadata (2026-11-24)

Evidence for checking the answers of questions 1 to 6 of the evaluation set.

## KG state at run time

- Structural layer: Extractor v0.1 (8,260 nodes, 21,421 relationships), Odoo 18 Community, database `kg_demo`, 66 installed modules.
- Semantic layer: sem-v0.2 (not used in this step).
- Last source commit before the run: `e35ab73`.
- Queries run in Neo4j Browser on 2026-11-24, results exported as CSV.

## Files

| File | Query | Content |
|---|---|---|
| `QA_adds_field.csv` | Q-A | Models and fields added by each module (ADDS_FIELD) |
| `QB_defines_extends.csv` | Q-B | Models defined (DEFINES) or extended (EXTENDS) by each module |
| `QC_depends_on.csv` | Q-C | Direct dependencies (DEPENDS_ON, 1 hop) |
| `QD_sale_order_links.csv` | Q-D | Fields linking `sale.order` to `stock.picking`, `account.move`, and the module that adds them |

Queries: `queries/answer_check.cypher`. Final answers: `data/eval/requirements_v0.csv` (draft kept as history: `data/eval/requirements_draft_20261124.csv`).

## Main findings

1. Sanity check passed: `stock_no_negative` adds `allow_negative_stock` on 4 models, as in R-07.
2. `stock_picking_back2draft` and `sale_stock_picking_invoice_link` add no field; they only EXTEND models. Their gold models must come from Q-B, not only from ADDS_FIELD.
3. `product.product` has fields of `stock_no_negative` and `stock_picking_auto_create_lot` but is not EXTENDED: the fields come from `_inherits` of `product.template`. Classified as "accepted".
4. `stock_no_negative` EXTENDS `stock.quant` (not in the answer of question 1 in R-07).
5. `account.bank.statement.line` has `picking_ids` through `_inherits` of `account.move` (R-20): excluded from the answer of question 2.
6. Direct dependencies: `stock_picking_invoice_link` -> `stock_account` (matches R-21); 4 other modules -> `stock`; `sale_stock_picking_invoice_link` -> `sale_stock`, `stock_picking_invoice_link`.
7. Question 6: standard Odoo already has `sale.order.picking_ids` (module `sale_stock`) and `sale.order.invoice_ids` (module `sale`).

## BA decisions (2026-11-24)

- Question 1: add `stock.quant` to the gold models (where the negative stock check lives); move `product.product` to "accepted". This changes R-07 (progress report of 11/11: 4 models from ADDS_FIELD).
- Question 6: rewritten for the function of `sale_stock_picking_invoice_link` (the old question is covered by standard Odoo, see Q-D). Questions answered by standard Odoo are left to Internship 2, with a separate metric.
- Question 5: rewritten in BA wording to avoid copying the module description.

## Rules for choosing answers

- Gold models are where the main configuration or logic of the function lives (ADDS_FIELD, DEFINES or EXTENDS).
- Gold models are chosen only from the Q-A and Q-B results; no model outside these lists.
- Models whose fields come only from delegation inheritance (`_inherits`) are "accepted" or excluded.
- The semantic layer (Function, INVOLVES) is never used as ground truth (it is a system output).
- Questions are written in BA wording, not copied from module descriptions.
- "Accepted" modules and models are allowed for ambiguous cases.

## Column names of `requirements_v0.csv`

`id, requirement_vi, requirement_en, main_module, accepted_modules, gold_models, accepted_models, excluded_models, direct_depends, split, status, notes, evidence, checked_on`. Lists inside a cell are separated by `;`. `requirement_vi` keeps the Vietnamese wording of the requirement (the evaluation set is bilingual).

## Note on CSV headers

The CSV files were exported on 2026-11-24 with Vietnamese column aliases. On the same day the aliases in `queries/answer_check.cypher` and the CSV headers were renamed to English (values unchanged): `so_field` -> `n_fields`, `quan_he` -> `relation`, `phu_thuoc_truc_tiep` -> `direct_depends`, `kieu` -> `ttype`, `model_dich` -> `target_model`, `module_them_field` -> `adding_module`, `nguon` -> `source`.
