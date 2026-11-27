# Agentic GraphRAG for Odoo: requirement mapping

- Requirement (Q08, en): After rushed deliveries, accounting keeps finding items with on-hand quantity below zero; the system should refuse to validate such transfers, except for the spare-parts product category that is only counted at month end.
- Understood as: The system must prevent the validation of stock transfers that involve products with a negative on-hand quantity. An exception is made for products belonging to the spare-parts product category, which are counted at month end and can be processed despite negative inventory.

## Recommended module

- **`stock_no_negative`** (OCA, "Stock Disallow Negative"): verified, verdict **supported**
- Summary: Disallow negative stock levels by default
- Mapping score 1.094 (tau 1.0)
- Function (KG): Allow negative stock for individual locations: Enables negative stock levels for individual inventory locations by setting the Allow Negative Stock flag on the selected locations.
- Function (KG): Allow negative stock for product categories: Permits negative stock levels for all products within a specific category by enabling the Allow Negative Stock setting on the product category.
- Function (KG): Allow negative stock for specific products: Allows negative stock levels for specific products by enabling the Allow Negative Stock option on the product form.
- Function (KG): Block stock operations resulting in negative stock: Prevents stock operations such as stock moves, pickings, or manufacturing orders from being validated if they would result in a negative stock level for stockable products or locations.

## Where to configure it or where the logic lives (models and fields)

| Model | Relation of the module | Fields the module adds |
|---|---|---|
| `product.category` (Product Category) | ADDS_FIELD, EXTENDS | allow_negative_stock (Allow Negative Stock) * |
| `stock.quant` (Quants) | EXTENDS | (none) |
| `product.template` (Product) | ADDS_FIELD, EXTENDS | allow_negative_stock (Allow Negative Stock) * |
| `stock.location` (Inventory Locations) | ADDS_FIELD, EXTENDS | allow_negative_stock (Allow Negative Stock) * |

Relations: DEFINES = the module creates the model; EXTENDS = the module changes its code; ADDS_FIELD = the module adds fields. * = field named by a business function.

## Modules to install with it

- Direct dependencies: `stock` (Odoo)
- Full dependency chain: 18 modules; all of them Odoo core

## Not covered (needs customization)

- Nothing reported by the Verifier

## Why: path in the knowledge graph

- `stock_no_negative -[ADDS_FIELD|EXTENDS]- product.category` (from the module to the closest entry point of the requirement)
- Objects in the requirement linked to models: "transfer" -> `stock.picking` (exact_name); "product category" -> `product.category` (exact_name); "product" -> `product.product` (exact_name)
- Closest business functions: Block stock operations resulting in negative stock (`stock_no_negative`, 0.710); Allow negative stock for product categories (`stock_no_negative`, 0.680); Allow negative stock for specific products (`stock_no_negative`, 0.663)

## Verifier

- Verdict: **supported**; reason: The stock_no_negative module blocks stock operations that would cause on-hand stock levels to drop below zero upon transfer validation. It provides a configurable setting on product categories to allow negative stock as an exception, satisfying the business requirement for spare parts.
- Quote: "Prevents stock operations such as stock moves, pickings, or manufacturing orders from being validated if they would result in a negative stock level for stockable products or locations."
- Quote: "Permits negative stock levels for all products within a specific category by enabling the Allow Negative Stock setting on the product category."
- Quotes found word for word in the KG and documentation given to the Verifier: 2/2
- Round 1: candidates stock_no_negative, stock_picking_auto_create_lot, stock_account -> supported, chose stock_no_negative

## Other candidate modules

| Module | Source | Score | sim | cov | prox | Verifier |
|---|---|---|---|---|---|---|
| `stock_picking_auto_create_lot` | OCA | 0.960 | 0.528 | 0.612 | 0.500 | no: Automates lot creation for incoming pickings and does not manage stock validation rules. |
| `stock_account` | Odoo | 0.898 | 0.592 | 0.361 | 0.500 | no: Manages inventory accounting valuation rather than preventing negative stock transfers. |
| `stock` | Odoo | 0.882 | 0.650 | 0.213 | 0.500 |  |
| `stock_picking_back2draft` | OCA | 0.857 | 0.528 | 0.408 | 0.500 |  |

## Run

- Mapping: LLM analysis (gemini-3.6-flash, official run cache, 0 new calls)
- Verifier: round 1: official run cache
- New LLM calls: Mapping 0, Verifier 0, Collector 0 (limit 3, --dry-llm)
- KG: neo4j, 67 modules, 443 models, 14 functions; parameters data/mapping/params_final_en.json; 1.9 s
- demo-v0.1 on pipe-v0.1 (map-v0.1 + ver-v0.1)
