# Agentic GraphRAG for Odoo: requirement mapping

- Requirement (Q01, en): Block outgoing shipments when there is not enough stock on hand
- Understood as: Prevent the processing or validation of outgoing shipments when the physical stock on hand is insufficient to fulfill the required quantity.

## Recommended module

- **`stock_no_negative`** (OCA, "Stock Disallow Negative"): verified, verdict **supported**
- Summary: Disallow negative stock levels by default
- Mapping score 0.972 (tau 1.0); LOW CONFIDENCE, Collector asked for more evidence
- Function (KG): Allow negative stock for individual locations: Enables negative stock levels for individual inventory locations by setting the Allow Negative Stock flag on the selected locations.
- Function (KG): Allow negative stock for product categories: Permits negative stock levels for all products within a specific category by enabling the Allow Negative Stock setting on the product category.
- Function (KG): Allow negative stock for specific products: Allows negative stock levels for specific products by enabling the Allow Negative Stock option on the product form.
- Function (KG): Block stock operations resulting in negative stock: Prevents stock operations such as stock moves, pickings, or manufacturing orders from being validated if they would result in a negative stock level for stockable products or locations.

## Where to configure it or where the logic lives (models and fields)

| Model | Relation of the module | Fields the module adds |
|---|---|---|
| `stock.quant` (Quants) | EXTENDS | (none) |
| `product.template` (Product) | ADDS_FIELD, EXTENDS | allow_negative_stock (Allow Negative Stock) * |
| `stock.location` (Inventory Locations) | ADDS_FIELD, EXTENDS | allow_negative_stock (Allow Negative Stock) * |
| `product.category` (Product Category) | ADDS_FIELD, EXTENDS | allow_negative_stock (Allow Negative Stock) * |

Relations: DEFINES = the module creates the model; EXTENDS = the module changes its code; ADDS_FIELD = the module adds fields. * = field named by a business function.

## Modules to install with it

- Direct dependencies: `stock` (Odoo)
- Full dependency chain: 18 modules; all of them Odoo core

## Not covered (needs customization)

- Nothing reported by the Verifier

## Why: path in the knowledge graph

- `stock_no_negative -[EXTENDS]- stock.quant` (from the module to the closest entry point of the requirement)
- Objects in the requirement linked to models: "outgoing shipment" -> `stock.picking` (exact_name); "stock on hand" -> `stock.quant` (exact_name)
- Closest business functions: Block stock operations resulting in negative stock (`stock_no_negative`, 0.689); Request internal product transfers (`stock_request`, 0.609); Auto create lots for incoming stock pickings (`stock_picking_auto_create_lot`, 0.575)

## Verifier

- Verdict: **supported**; reason: The `stock_no_negative` module fulfills the requirement by raising a blocking error whenever a user attempts to validate a picking or stock operation that would drive stock levels below zero. Standard Odoo allows negative stock by default, making this OCA module necessary to enforce strict stock availability checks.
- Quote: "If you disallow negative stock in Odoo with this module, you will be blocked when trying to validate a stock operation that will set the stock level of a product and/or location as negative."
- Quote: "When you validate a stock operation (a stock move, a picking, a manufacturing order, etc.) that will set the stock level of a stockable product as negative, you will be blocked by an error message."
- Quotes found word for word in the KG and documentation given to the Verifier: 2/2
- Round 1: candidates stock_no_negative, stock_picking_back2draft, stock -> supported, chose stock_no_negative (with GitHub issues)
- Collector: stock_no_negative (low_confidence): cache, 5 issue(s)
- Collector: stock_picking_back2draft (low_confidence): cache, 1 issue(s)
- Collector: stock (low_confidence): no_source, 0 issue(s)

## Other candidate modules

| Module | Source | Score | sim | cov | prox | Verifier |
|---|---|---|---|---|---|---|
| `stock_picking_back2draft` | OCA | 0.890 | 0.515 | 0.500 | 0.500 | no: It only allows resetting canceled inventory transfers back to draft state. |
| `stock` | Odoo | 0.869 | 0.657 | 0.174 | 0.500 | no: By default, standard Odoo allows negative stock and does not block pickings from being validated when stock is insufficient. |
| `sale_stock` | Odoo | 0.841 | 0.643 | 0.147 | 0.500 |  |
| `stock_move_location` | OCA | 0.834 | 0.551 | 0.316 | 0.500 |  |

## Expected answer (evaluation set)

- Main module: `stock_no_negative`
- Gold models: product.template, product.category, stock.location, stock.quant; also accepted: product.product

## Run

- Mapping: LLM analysis (gemini-3.6-flash, official run cache, 0 new calls)
- Verifier: round 1: official run cache
- New LLM calls: Mapping 0, Verifier 0, Collector 0 (limit 3, --dry-llm)
- KG: neo4j, 67 modules, 443 models, 14 functions; parameters data/mapping/params_final_en.json; 1.2 s
- demo-v0.1 on pipe-v0.1 (map-v0.1 + ver-v0.1)
