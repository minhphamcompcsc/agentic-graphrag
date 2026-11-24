"""Collector v1: văn bản module OCA (như v0) + GitHub issue, PR do LLM chọn lọc.

So với v0 (collect_modules.py):
  1. Vẫn đọc __manifest__.py và readme/ (dùng lại hàm của v0, kết quả giống hệt v0).
  2. Tìm trên GitHub các issue, pull request của repo OCA có nhắc tên module
     (GitHub Search API, 2 truy vấn/module: is:issue và is:pull-request).
  3. Lọc bằng luật (không tốn LLM): bỏ migration, port, translation, CI.
  4. LLM chọn nguồn liên quan (1 lời gọi/module): giữ tối đa 5 mục mang thông tin
     nghiệp vụ (use case, quy tắc, cấu hình, giới hạn, lỗi chức năng). Đây là phần
     "LLM chọn nguồn liên quan" của Collector Agent (HANDOFF Mục 6, D16).
  5. Ghi data/collected_v1/<module>.json: giữ nguyên khoá của v0 (manifest, readme)
     để Extractor đọc được, thêm khoá github, selection, text_v0.

Tái lập, tiết kiệm quota (D15):
  - Kết quả GitHub lưu ở data/collected_v1/_github/<module>.json; chạy lại dùng lại
    (trừ khi --refresh-github), nên đầu vào LLM không đổi.
  - Kết quả LLM lưu ở data/collected_v1/_llm/<module>.json theo hash prompt; chạy lại
    0 lời gọi (trừ khi --refresh-llm).
  - Không có ứng viên sau khi lọc thì không gọi LLM.
  - Hết quota ngày thì dừng, giữ cache. Giới hạn cứng --max-llm-calls (mặc định 8).

Cách chạy (trong ~/agentic-graphrag, đã activate venv):
  python src/collector/collect_modules_v1.py --no-llm     # thử: GitHub + lọc, 0 lời gọi LLM
  python src/collector/collect_modules_v1.py              # chạy thật (khoảng 7 lời gọi)
  python src/collector/collect_modules_v1.py --modules stock_request

Biến .env tuỳ chọn: GITHUB_TOKEN (token cá nhân, chỉ đọc; không có thì GitHub cho
10 lượt tìm/phút, script tự nghỉ giữa các lượt), GEMINI_MODEL, LLM_TIMEOUT, LLM_PAUSE.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_modules import SEEN_MODULES, find_module, git_info, read_manifest, read_readme  # noqa: E402

load_dotenv(override=True)   # .env thắng biến cũ trong terminal (sự cố 23/11)

COLLECTOR_VERSION = "v1"
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")      # D12
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))
LLM_PAUSE = int(os.getenv("LLM_PAUSE", "15"))                    # giới hạn 5 lời gọi/phút
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GH_PAUSE = 2 if GITHUB_TOKEN else 7                              # không token: 10 lượt tìm/phút

DEFAULT_MODULES = SEEN_MODULES + ["stock_request"]               # 6 seen + 1 unseen (HANDOFF Mục 8)
MAX_CANDIDATES = 20       # số ứng viên tối đa đưa cho LLM
MAX_SELECTED = 5          # số mục LLM được chọn tối đa
PROMPT_DOC_CHARS = 1500   # độ dài tài liệu module đưa vào prompt
PROMPT_BODY_CHARS = 350   # độ dài mỗi nội dung issue trong prompt
TEXT_BODY_CHARS = 1500    # độ dài mỗi issue khi ghi vào văn bản v1

# Luật lọc trước (trên tiêu đề): loại các mục không mang thông tin nghiệp vụ
FILTER_RULES = [
    ("migration", r"^migration to version|\[mig\]|\bmigrat|\bport(ed|ing)? to\b"),
    ("forward_port", r"\[fwp?\]|forward[- ]?port|cherry[- ]?pick|\[bcp\]|backport"),
    ("translation", r"translat|weblate|\.pot?\b|\bi18n\b|\b[a-z]{2}_[A-Z]{2}\b"),
    ("ci_tooling", r"pre-commit|\[ci\]|\blint|bump version|\[upd\] (readme|dotfiles|setup)"),
]


# ============ 1. GitHub ============
class GitHubError(RuntimeError):
    pass


def gh_search(repo: str, module: str, kind: str):
    """Tìm issue hoặc PR (kind: issue | pull-request) trong OCA/<repo> có nhắc tên module."""
    q = f'"{module}" repo:OCA/{repo} is:{kind}'
    url = "https://api.github.com/search/issues?" + urllib.parse.urlencode({"q": q, "per_page": 50})
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "agentic-graphrag-collector"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise GitHubError(f"GitHub {e.code} khi tìm '{q}': {e.read()[:200]!r}")
    except urllib.error.URLError as e:
        raise GitHubError(f"Không kết nối được GitHub: {e}")
    items = []
    for it in data.get("items", []):
        items.append(dict(
            number=it["number"], type="pr" if "pull_request" in it else "issue",
            state=it["state"], title=it["title"], url=it["html_url"],
            created_at=it["created_at"], updated_at=it["updated_at"], closed_at=it.get("closed_at"),
            labels=[lb["name"] for lb in it.get("labels", [])], body=it.get("body") or ""))
    return q, data.get("total_count", 0), items


def get_github_items(module: str, repo: str, cache_dir: Path, refresh: bool):
    """Dùng lại kết quả GitHub đã lưu (để tái lập); --refresh-github thì tìm lại."""
    cache = cache_dir / f"{module}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8")), True
    queries, items = [], []
    for kind in ("issue", "pull-request"):
        q, total, found = gh_search(repo, module, kind)
        queries.append({"q": q, "total_count": total, "returned": len(found)})
        items += found
        time.sleep(GH_PAUSE)
    raw = {"module": module, "repo": f"OCA/{repo}",
           "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "queries": queries, "items": items}
    cache.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw, False


def clean_body(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.S)       # ghi chú HTML (template issue)
    text = re.sub(r"```.*?```", "", text, flags=re.S)              # khối code, log lỗi
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)               # ảnh markdown
    text = re.sub(r"<img[^>]*>", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def prefilter(item: dict, module: str):
    """Trả về None nếu giữ, hoặc lý do loại."""
    title = item["title"]
    if module not in title and module not in (item["body"] or ""):
        return "module_not_mentioned"        # GitHub tách từ nên có thể khớp sai
    for reason, pattern in FILTER_RULES:
        if re.search(pattern, title, flags=re.I):
            return reason
    return None


# ============ 2. LLM chọn nguồn ============
class SelectedItem(BaseModel):
    number: int = Field(description="Issue or pull request number, copied from the candidate list")
    category: str = Field(description="One of: use_case, business_rule, configuration, limitation, functional_bug")
    reason: str = Field(description="One short sentence in English: what business knowledge this item adds")


class Selection(BaseModel):
    selected: List[SelectedItem] = Field(default_factory=list,
                                         description="Selected items, most useful first; empty if none is useful")


PROMPT = """You are an Odoo functional consultant. A collector agent gathers sources that explain what the
OCA module `{module}` ("{label}") does, to build a knowledge graph that maps business requirements
to Odoo modules.

MODULE DOCUMENTATION (excerpt):
{doc}

CANDIDATE GITHUB ITEMS (issues and pull requests of OCA/{repo} that mention the module):
{items}

TASK: Select at most {k} items that add useful BUSINESS knowledge about THIS module: a use case,
a business rule or expected behaviour, a configuration option, a limitation, or a functional bug
that shows how the module behaves in a business process (for example with deliveries, invoices,
lots, manufacturing).
Do NOT select: migrations or ports to another Odoo version, translations, code refactoring,
CI or tests, performance-only changes, or items mainly about another module.
Prefer items that add information NOT already in the documentation.
If no item is useful, return an empty list. Only use numbers from the candidate list.
"""


class DailyQuotaExceeded(RuntimeError):
    pass


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_prompt(module, label, repo, doc, candidates):
    lines = []
    for c in candidates:
        body = clean_body(c["body"]).replace("\n", " ")[:PROMPT_BODY_CHARS]
        lines.append(f"#{c['number']} [{c['type']}, {c['state']}, {c['created_at'][:4]}] {c['title']}\n"
                     f"    {body}")
    return PROMPT.format(module=module, label=label, repo=repo, doc=doc[:PROMPT_DOC_CHARS],
                         items="\n".join(lines), k=MAX_SELECTED)


def call_llm(prompt: str) -> Selection:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, max_retries=0, timeout=LLM_TIMEOUT)
    try:
        out = llm.with_structured_output(Selection).invoke(prompt)
    except Exception as e:
        msg = repr(e)
        if "PerDay" in msg or "per day" in msg.lower():
            raise DailyQuotaExceeded(msg[:300])
        raise
    if out is None:
        raise RuntimeError("LLM trả về kết quả không đọc được (structured output rỗng)")
    return out if isinstance(out, Selection) else Selection(**out)


def get_selection(module, prompt, cache_dir: Path, refresh: bool, state: dict):
    """Dùng cache nếu prompt và model không đổi; ngược lại gọi LLM (1 lần, không tự thử lại)."""
    cache = cache_dir / f"{module}.json"
    if cache.exists() and not refresh:
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c.get("prompt_sha256") == sha(prompt) and c.get("llm") == LLM_MODEL:
            return Selection(**c["output"]), c, "cache"
    if state["llm_calls"] >= state["max_llm_calls"]:
        raise RuntimeError(f"đã chạm giới hạn {state['max_llm_calls']} lời gọi LLM trong lần chạy này")
    print(f"  {module}: gọi LLM ({LLM_MODEL}) ...")
    state["llm_calls"] += 1
    t = time.time()
    out = call_llm(prompt)
    c = {"module": module, "collector_version": COLLECTOR_VERSION, "llm": LLM_MODEL, "temperature": 0,
         "called_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "seconds": round(time.time() - t, 1), "prompt_sha256": sha(prompt), "prompt": prompt,
         "output": out.model_dump()}
    cache.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(LLM_PAUSE)
    return out, c, "new"


# ============ 3. Thu thập 1 module (dùng được từ Mapping Agent trong vòng agentic) ============
def collect_v1(module: str, oca_dir: Path, out_dir: Path, collected_at: str, state: dict,
               use_llm=True, refresh_github=False, refresh_llm=False):
    module_dir = find_module(oca_dir, module)
    if module_dir is None:
        return {"module": module, "status": "not_found"}
    repo_dir = module_dir.parent
    repo = repo_dir.name
    manifest = read_manifest(module_dir)
    readme = read_readme(module_dir)

    # Văn bản v0 (giống hệt collect_modules.py)
    parts = [manifest.get("name", ""), manifest.get("summary", "")]
    if manifest.get("description"):
        parts.append(manifest["description"])
    parts += [f"## {k}\n{v['text']}" for k, v in readme.items()]
    text_v0 = "\n\n".join(p for p in parts if p).strip()

    # GitHub + lọc bằng luật
    raw, gh_cached = get_github_items(module, repo, out_dir / "_github", refresh_github)
    seen, candidates, dropped = set(), [], {}
    for it in raw["items"]:
        if it["number"] in seen:
            continue
        seen.add(it["number"])
        why = prefilter(it, module)
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            it["filter"] = why
        else:
            it["filter"] = "kept"
            candidates.append(it)
    n_kept = len(candidates)
    candidates = candidates[:MAX_CANDIDATES]

    # LLM chọn nguồn
    selection, sel_meta, llm_status, invalid = Selection(), None, "skip_no_candidates", []
    if candidates and use_llm:
        prompt = build_prompt(module, manifest.get("name", module), repo, text_v0, candidates)
        selection, sel_meta, llm_status = get_selection(module, prompt, out_dir / "_llm", refresh_llm, state)
    elif candidates:
        llm_status = "skip_no_llm"
    by_num = {c["number"]: c for c in candidates}
    selected = []
    for s in selection.selected[:MAX_SELECTED]:
        c = by_num.get(s.number)
        if c is None:
            invalid.append(s.number)          # LLM trả số không có trong danh sách
            continue
        selected.append(dict(number=c["number"], type=c["type"], state=c["state"], title=c["title"],
                             url=c["url"], created_at=c["created_at"], category=s.category,
                             reason=s.reason, body_clean=clean_body(c["body"])[:TEXT_BODY_CHARS]))

    # Văn bản v1 = v0 + các issue được chọn
    issue_parts = [f"### #{s['number']} {s['title']} ({s['type']}, {s['state']}, {s['created_at'][:4]})\n"
                   f"{s['body_clean']}" for s in selected]
    issues_text = ("## GITHUB ISSUES (selected)\n" + "\n\n".join(issue_parts)) if issue_parts else ""
    full_text = "\n\n".join(p for p in [text_v0, issues_text] if p)

    record = {
        "module": module,
        "source": {"repo": repo, **git_info(repo_dir), "path": str(module_dir.relative_to(oca_dir.parent))},
        "collected_at": collected_at,
        "collector_version": COLLECTOR_VERSION,
        "manifest": manifest,
        "readme": readme,
        "text_v0": text_v0,
        "text": full_text,
        "github": {
            "repo": f"OCA/{repo}", "fetched_at": raw["fetched_at"], "queries": raw["queries"],
            "n_raw": len(seen), "n_kept": n_kept, "n_to_llm": len(candidates), "dropped": dropped,
            "candidates": [dict(number=c["number"], type=c["type"], state=c["state"], title=c["title"],
                                url=c["url"], filter=c["filter"])
                           for c in raw["items"]],
            "selected": selected,
        },
        "selection": {"llm": LLM_MODEL if sel_meta else None, "status": llm_status,
                      "prompt_sha256": (sel_meta or {}).get("prompt_sha256"),
                      "called_at": (sel_meta or {}).get("called_at"), "invalid_numbers": invalid},
        "stats": {
            "readme_sections": list(readme.keys()),
            "n_chars_v0": len(text_v0),
            "n_chars_issues": len(issues_text),
            "n_chars": len(full_text),
            "n_sources": 1 + len(selected),       # 1 = tài liệu module (manifest + readme)
            "has_description": bool(readme.get("DESCRIPTION") or manifest.get("description")),
        },
    }
    (out_dir / f"{module}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"module": module, "status": "ok", "repo": repo, "gh_cached": gh_cached, "llm": llm_status,
            "n_raw": len(seen), "n_kept": n_kept, "n_selected": len(selected),
            "invalid_numbers": invalid, **record["stats"]}


# ============ 4. Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oca-dir", default="oca")
    ap.add_argument("--out-dir", default="data/collected_v1")
    ap.add_argument("--modules", nargs="*", default=DEFAULT_MODULES)
    ap.add_argument("--no-llm", action="store_true", help="chỉ GitHub + lọc bằng luật, 0 lời gọi LLM")
    ap.add_argument("--refresh-github", action="store_true", help="tìm lại trên GitHub dù đã có kết quả lưu")
    ap.add_argument("--refresh-llm", action="store_true", help="gọi lại LLM dù đã có cache")
    ap.add_argument("--max-llm-calls", type=int, default=8)
    args = ap.parse_args()

    oca_dir, out_dir = Path(args.oca_dir), Path(args.out_dir)
    for d in (out_dir, out_dir / "_github", out_dir / "_llm"):
        d.mkdir(parents=True, exist_ok=True)
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = {"llm_calls": 0, "max_llm_calls": args.max_llm_calls}
    print(f"Collector {COLLECTOR_VERSION}, {collected_at}, {len(args.modules)} module, "
          f"LLM {'TẮT (--no-llm)' if args.no_llm else LLM_MODEL}, GitHub token {'có' if GITHUB_TOKEN else 'không'}")

    t0, results = time.time(), []
    for m in args.modules:
        try:
            r = collect_v1(m, oca_dir, out_dir, collected_at, state, use_llm=not args.no_llm,
                           refresh_github=args.refresh_github, refresh_llm=args.refresh_llm)
        except DailyQuotaExceeded as e:
            print(f"\nDỪNG: hết quota Gemini trong ngày ({LLM_MODEL}). Module đã xong vẫn giữ cache;"
                  f" chạy lại lệnh khi quota làm mới (khoảng 15:00 giờ VN).\n  Chi tiết: {e}")
            break
        except GitHubError as e:
            print(f"  {m}: LỖI GitHub: {e}")
            r = {"module": m, "status": "github_error"}
        except Exception as e:   # lỗi LLM khác (503, timeout...): ghi lại, sang module sau
            print(f"  {m}: LỖI: {repr(e)[:300]}")
            r = {"module": m, "status": "error", "error": repr(e)[:300]}
        results.append(r)

    print(f"\n{'module':33} {'status':12} {'v0':>6} {'raw':>4} {'giữ':>4} {'chọn':>4} {'v1':>6}  LLM")
    for r in results:
        print(f"{r['module']:33} {r['status']:12} {r.get('n_chars_v0', 0):>6} {r.get('n_raw', 0):>4} "
              f"{r.get('n_kept', 0):>4} {r.get('n_selected', 0):>4} {r.get('n_chars', 0):>6}  {r.get('llm', '')}")
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nXong {len(ok)}/{len(args.modules)} module; lời gọi LLM mới: {state['llm_calls']}; "
          f"có ít nhất 1 issue được chọn: {sum(r['n_selected'] > 0 for r in ok)}/{len(ok)}; "
          f"thời gian {time.time() - t0:.1f} giây")
    bad = {r["module"]: r["invalid_numbers"] for r in ok if r.get("invalid_numbers")}
    if bad:
        print(f"Cảnh báo: LLM trả số không có trong danh sách (đã bỏ): {bad}")

    summary = {"collected_at": collected_at, "collector_version": COLLECTOR_VERSION,
               "llm": None if args.no_llm else LLM_MODEL, "no_llm": args.no_llm,
               "llm_calls": state["llm_calls"], "seconds": round(time.time() - t0, 1), "results": results}
    stamp = collected_at[:10].replace("-", "")
    suffix = "_nollm" if args.no_llm else ""
    (out_dir / f"_summary_{stamp}{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Lưu: {out_dir}/<module>.json, {out_dir}/_summary_{stamp}{suffix}.json")


if __name__ == "__main__":
    main()
