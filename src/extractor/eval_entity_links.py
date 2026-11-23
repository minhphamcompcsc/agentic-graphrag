"""
Đánh giá gộp thực thể của lớp ngữ nghĩa (tiêu chí "Gộp thực thể", HANDOFF Mục 7.4).

Đầu vào: file links_YYYYMMDD.csv do extract_semantic.py tạo, đã điền tay cột gold_model:
  - tên model kỹ thuật đúng (vd. stock.picking), hoặc
  - NONE nếu đối tượng không ứng với model nào trong Odoo (đúng ra phải là Concept mới).
Dòng để trống gold_model bị bỏ qua.

Độ đo:
  - Tỷ lệ gắn đúng = số dòng linked_model == gold_model (NONE khớp với không gắn) / số dòng đã gán nhãn
  - Tách theo cách gộp (exact_name, exact_label, embedding, new)

Cách chạy:
  python src/extractor/eval_entity_links.py data/semantic/links_20261119.csv
"""
import csv
import json
import sys
from pathlib import Path


def main(path: str):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    labeled = [r for r in rows if r["gold_model"].strip()]
    if not labeled:
        sys.exit("Chưa có dòng nào điền gold_model")

    def ok(r):
        gold = r["gold_model"].strip()
        pred = r["linked_model"].strip()
        return (pred == "" and gold.upper() == "NONE") or pred == gold

    by_method = {}
    for r in labeled:
        b = by_method.setdefault(r["method"], [0, 0])
        b[0] += ok(r)
        b[1] += 1
    correct = sum(ok(r) for r in labeled)
    report = {
        "file": path,
        "labeled": len(labeled), "total_rows": len(rows),
        "accuracy": round(correct / len(labeled), 3),
        "correct": correct,
        "by_method": {k: {"correct": c, "total": n, "accuracy": round(c / n, 3)} for k, (c, n) in by_method.items()},
        "errors": [{k: r[k] for k in ("module", "mention", "linked_model", "method", "score", "gold_model")}
                   for r in labeled if not ok(r)],
    }
    print(f"Gắn đúng: {correct}/{len(labeled)} = {report['accuracy']}")
    for k, v in report["by_method"].items():
        print(f"  {k:12} {v['correct']}/{v['total']} = {v['accuracy']}")
    for e in report["errors"]:
        print(f"  SAI: [{e['module']}] '{e['mention']}' -> {e['linked_model'] or 'NEW'} ({e['method']}), đúng: {e['gold_model']}")
    out = Path(path).with_name(Path(path).stem + "_eval.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Lưu: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit("Cần đường dẫn file CSV"))
