# Answer check of questions 7 to 20 (2026-11-24)

Evidence for the answers of questions 7 to 20 of the evaluation set (`data/eval/requirements_v0.csv`).

## How the answers were chosen

- Questions 7 to 20 were drafted by Claude at the BA's request and approved by the BA (Minh) on 2026-11-24
  (draft kept as history: `data/eval/requirements_draft_20261124b.csv`).
- Same rules as questions 1 to 6 (see `data/eval/answer_check_20261124/README.md`): gold models are where the
  configuration or the main logic of the function lives (ADDS_FIELD, DEFINES or EXTENDS of the module);
  every gold and accepted model must appear in the Q-A or Q-B results of the module.
- Questions 7 to 16 (seen modules) reuse the Q-A, Q-B, Q-C results of `data/eval/answer_check_20261124/`.
- Questions 17 to 20 (unseen module `stock_request`) use the queries Q-A2, Q-B2, Q-C2 in
  `queries/answer_check.cypher`; results in this folder.
- Where the module text is not enough, the behaviour was checked in the OCA source code (branch 18.0):
  - Q11: lots are created only if `auto_create_lot` is set on the operation type and on the product or its category (`stock_picking_auto_create_lot/models/stock_move.py`).
  - Q10: `stock.picking.button_fillwithstock` (`stock_move_location/models/stock_picking.py`).
  - Q13: `stock.picking.type.show_move_onhand` and `action_move_location` (`stock_move_location/models/stock_picking_type.py`).
  - Q15: invoice lines linked to stock moves in `sale.order.line._prepare_invoice_line` (`sale_stock_picking_invoice_link/models/sale_order.py`).
  - Q16: refund lines linked to returned moves in `account.move.reversal.reverse_moves` (`stock_picking_invoice_link/wizards/account_move_reversal.py`).
  - Q19: `res.company.stock_request_check_available_first`, used in `stock.request` with the free quantity at the request location (`stock_request/models/res_company.py`, `stock_request/models/stock_request.py`).

## KG state at run time

- Structural layer: Extractor v0.2 after the incremental update with `stock_request` (8,427 nodes, 21,896 relationships, 67 installed modules, fingerprint `40dec9c306f6`).
- Expected results (computed from `data/snapshots/schema_20261124.json` before the run):
  - Q-A2: 162 fields on 8 models: `res.company` 2, `res.config.settings` 8, `stock.move` 2, `stock.picking` 2, `stock.request` 53, `stock.request.abstract` 35, `stock.request.allocation` 15, `stock.request.order` 45.
  - Q-B2: DEFINES `stock.request`, `stock.request.abstract`, `stock.request.allocation`, `stock.request.order`; EXTENDS `procurement.group`, `res.company`, `res.config.settings`, `stock.location`, `stock.move`, `stock.move.line`, `stock.picking`, `stock.route`, `stock.rule`, `stock.warehouse`.
  - Q-C2: `stock`.

## Result of the run (2026-11-24)

All three queries match the expected values:

- Q-A2: 8 models, 162 fields (same counts per model as above).
- Q-B2: 14 rows, 4 DEFINES and 10 EXTENDS (same models as above). `res.config.settings` is transient (settings wizard).
- Q-C2: `stock`.

Every gold and accepted model of questions 17 to 20 is in these results.

## Files

| File | Query | Content |
|---|---|---|
| `QA_stock_request.csv` | Q-A2 | Models and fields added by `stock_request` |
| `QB_stock_request.csv` | Q-B2 | Models defined or extended by `stock_request` |
| `QC_stock_request.csv` | Q-C2 | Direct dependencies of `stock_request` |

## Split

Development set (parameter tuning): Q08, Q11, Q13, Q16. Test set: the other 16 questions, including Q01 to Q06 and the 4 questions of the unseen module `stock_request` (Q17 to Q20).

## Data leakage notes

Q16 is related to issue #2535 and Q20 to issues #116, #119, #126 selected by Collector v1; the questions were not written from these issues. The semantic layer does not read issues in Internship 1 (decision D18), so this only matters if the Collector is called in the agentic loop.
