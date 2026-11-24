"""Collector v0: collects the description text of OCA modules from the cloned repositories.

Reads __manifest__.py and the readme/ folder of each module, and saves each module
as 1 JSON file (with source, commit and collection time).

v0 uses no LLM: the module list is given by hand.
LLM-based source selection is added in v1 (collect_modules_v1.py).

Usage (in ~/agentic-graphrag, venv activated):
    python src/collector/collect_modules.py
    python src/collector/collect_modules.py --modules stock_request
"""

import argparse
import ast
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# 6 modules of the seen group
SEEN_MODULES = [
    "stock_no_negative",
    "stock_picking_invoice_link",
    "stock_picking_auto_create_lot",
    "stock_move_location",
    "stock_picking_back2draft",
    "sale_stock_picking_invoice_link",
]

# readme/ files with business content (skip CONTRIBUTORS, CREDITS, HISTORY)
README_KEEP = ["DESCRIPTION", "CONTEXT", "USAGE", "CONFIGURE", "INSTALL", "ROADMAP"]

# Manifest keys to keep
MANIFEST_KEEP = ["name", "summary", "description", "version", "category",
                 "depends", "auto_install", "website", "license"]


def find_module(oca_dir: Path, module: str):
    """Find the module folder in the cloned OCA repositories: oca/<repo>/<module>/."""
    for manifest in oca_dir.glob(f"*/{module}/__manifest__.py"):
        return manifest.parent
    return None


def git_info(repo_dir: Path):
    """Get the current commit and branch of the repository (for reproducibility)."""
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo_dir), *args],
                                           text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "remote": run("config", "--get", "remote.origin.url")}


def read_manifest(module_dir: Path):
    text = (module_dir / "__manifest__.py").read_text(encoding="utf-8")
    data = ast.literal_eval(text)  # safe, does not execute code
    return {k: data.get(k) for k in MANIFEST_KEEP if k in data}


def read_readme(module_dir: Path):
    """Read the readme/<SECTION>.md (or .rst) files. Returns a dict section -> text."""
    sections = {}
    readme_dir = module_dir / "readme"
    if not readme_dir.is_dir():
        return sections
    for name in README_KEEP:
        for ext in (".md", ".rst"):
            f = readme_dir / f"{name}{ext}"
            if f.is_file():
                text = f.read_text(encoding="utf-8").strip()
                if text:
                    sections[name] = {"file": f"readme/{f.name}", "text": text}
                break
    return sections


def collect(module: str, oca_dir: Path, out_dir: Path, collected_at: str):
    module_dir = find_module(oca_dir, module)
    if module_dir is None:
        return {"module": module, "status": "not_found"}

    repo_dir = module_dir.parent
    manifest = read_manifest(module_dir)
    readme = read_readme(module_dir)

    # Combined text for the semantic layer (iText2KG)
    parts = [manifest.get("name", ""), manifest.get("summary", "")]
    if manifest.get("description"):
        parts.append(manifest["description"])
    parts += [f"## {k}\n{v['text']}" for k, v in readme.items()]
    full_text = "\n\n".join(p for p in parts if p).strip()

    record = {
        "module": module,
        "source": {
            "repo": repo_dir.name,
            **git_info(repo_dir),
            "path": str(module_dir.relative_to(oca_dir.parent)),
        },
        "collected_at": collected_at,
        "collector_version": "v0",
        "manifest": manifest,
        "readme": readme,
        "text": full_text,
        "stats": {
            "readme_sections": list(readme.keys()),
            "n_chars": len(full_text),
            "has_description": bool(readme.get("DESCRIPTION") or manifest.get("description")),
        },
    }
    (out_dir / f"{module}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"module": module, "status": "ok", "repo": repo_dir.name,
            **record["stats"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oca-dir", default="oca")
    ap.add_argument("--out-dir", default="data/collected")
    ap.add_argument("--modules", nargs="*", default=SEEN_MODULES)
    args = ap.parse_args()

    oca_dir, out_dir = Path(args.oca_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results = [collect(m, oca_dir, out_dir, collected_at) for m in args.modules]

    # Print the summary table
    print(f"Collector v0, {collected_at}, {len(results)} module")
    print(f"{'module':35} {'status':10} {'repo':30} {'chars':>6}  sections")
    for r in results:
        print(f"{r['module']:35} {r['status']:10} {r.get('repo', ''):30} "
              f"{r.get('n_chars', 0):>6}  {','.join(r.get('readme_sections', []))}")
    ok = [r for r in results if r["status"] == "ok"]
    with_desc = [r for r in ok if r["has_description"]]
    print(f"\nFound: {len(ok)}/{len(results)}; with description: {len(with_desc)}/{len(results)}")

    summary = {"collected_at": collected_at, "results": results}
    stamp = collected_at[:10].replace("-", "")
    (out_dir / f"_summary_{stamp}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
