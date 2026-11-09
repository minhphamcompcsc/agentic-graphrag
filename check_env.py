import os
import sys
import xmlrpc.client
from importlib.metadata import version, PackageNotFoundError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"

PG = dict(host="localhost", port=5432, dbname="kg_demo",
          user=os.getenv("PG_USER", "odoo"), password=os.getenv("PG_PASSWORD", "odoo"))
ODOO_URL, ODOO_DB, ODOO_USER, ODOO_PWD = "http://localhost:8069", "kg_demo", "admin", "admin"
NEO4J_URI, NEO4J_AUTH = "bolt://localhost:7687", ("neo4j", "thesis12345")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBED_MODEL, EMBED_DIM = "bge-m3", 1024

SEEN = ["stock_no_negative", "stock_picking_invoice_link", "stock_picking_auto_create_lot",
        "stock_move_location", "stock_picking_back2draft"]
UNSEEN = ["stock_request"]
BASELINE = {"models": 439, "rel_fields": 2512}
PINNED = {"itext2kg": "1.1.0", "lightrag-hku": "1.5.7", "langchain-core": "0.3.86",
          "langchain-google-genai": "2.1.12", "langchain-ollama": "0.3.10", "neo4j": "5.28.8"}


def check_python():
    in_venv = sys.prefix != sys.base_prefix
    ver = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] != (3, 12):
        return FAIL, f"Python {ver}, cần 3.12"
    if not in_venv:
        return WARN, f"Python {ver} nhưng CHƯA activate venv"
    return OK, f"Python {ver}, venv: {sys.prefix}"


def check_packages():
    wrong = []
    for pkg, want in PINNED.items():
        try:
            got = version(pkg)
            if got != want:
                wrong.append(f"{pkg} {got} (cần {want})")
        except PackageNotFoundError:
            wrong.append(f"{pkg} chưa cài")
    return (FAIL, "; ".join(wrong)) if wrong else (OK, "6 thư viện đúng version đã pin")


def check_postgres():
    import psycopg2
    with psycopg2.connect(**PG) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ir_model")
        models = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM ir_model_fields WHERE relation IS NOT NULL")
        rel = cur.fetchone()[0]
        cur.execute("""SELECT 1 FROM ir_module_module_dependency d
                       JOIN ir_module_module m ON m.id = d.module_id
                       WHERE m.name = 'stock_picking_invoice_link' AND d.name = 'stock_account'""")
        dep = cur.fetchone() is not None
    msg = f"{models} model, {rel} field quan hệ, dependency stock_account: {'có' if dep else 'KHÔNG'}"
    if models != BASELINE["models"] or rel != BASELINE["rel_fields"] or not dep:
        return WARN, msg + f" (baseline {BASELINE['models']}/{BASELINE['rel_fields']})"
    return OK, msg


def check_odoo():
    common = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/common")
    ver = common.version()["server_version"]
    uid = common.authenticate(ODOO_DB, ODOO_USER, ODOO_PWD, {})
    if not uid:
        return FAIL, f"Odoo {ver} chạy nhưng đăng nhập {ODOO_USER} vào {ODOO_DB} thất bại"
    obj = xmlrpc.client.ServerProxy(f"{ODOO_URL}/xmlrpc/2/object")
    rows = obj.execute_kw(ODOO_DB, uid, ODOO_PWD, "ir.module.module", "search_read",
                          [[["name", "in", SEEN + UNSEEN]]], {"fields": ["name", "state"]})
    state = {r["name"]: r["state"] for r in rows}
    missing = [m for m in SEEN if state.get(m) != "installed"]
    leaked = [m for m in UNSEEN if state.get(m) == "installed"]
    if not ver.startswith("18"):
        return FAIL, f"Odoo {ver}, cần 18.0"
    if missing:
        return FAIL, f"Module 'đã thấy' chưa cài: {', '.join(missing)}"
    if leaked:
        return FAIL, f"Module 'unseen' đã bị cài: {', '.join(leaked)}"
    return OK, f"Odoo {ver}, 5/5 module đã thấy installed, stock_request chưa cài"


def check_neo4j():
    from neo4j import GraphDatabase
    with GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH) as drv:
        drv.verify_connectivity()
        apoc = drv.execute_query("RETURN apoc.version() AS v").records[0]["v"]
        nodes = drv.execute_query("MATCH (n) RETURN count(n) AS c").records[0]["c"]
    return OK, f"Kết nối được, APOC {apoc}, hiện có {nodes} node"


def check_ollama():
    from langchain_ollama import OllamaEmbeddings
    vec = OllamaEmbeddings(model=EMBED_MODEL).embed_query("Tự động trừ kho khi xác nhận giao hàng")
    if len(vec) != EMBED_DIM:
        return FAIL, f"{EMBED_MODEL} trả về {len(vec)} chiều, cần {EMBED_DIM}"
    return OK, f"{EMBED_MODEL} embed câu tiếng Việt: {len(vec)} chiều"


def check_gemini():
    if not os.getenv("GOOGLE_API_KEY"):
        return FAIL, "Không thấy GOOGLE_API_KEY trong .env"
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    out = llm.invoke("Trả lời đúng một từ: OK").content
    return OK, f"{GEMINI_MODEL} phản hồi: {str(out).strip()[:40]!r}"


CHECKS = [("1. Python + venv", check_python), ("2. Thư viện pin", check_packages),
          ("3. PostgreSQL", check_postgres), ("4. Odoo + module OCA", check_odoo),
          ("5. Neo4j + APOC", check_neo4j), ("6. Ollama bge-m3", check_ollama),
          ("7. Gemini API", check_gemini)]

if __name__ == "__main__":
    has_fail = False
    for name, fn in CHECKS:
        try:
            status, msg = fn()
        except Exception as e:
            status, msg = FAIL, f"{type(e).__name__}: {e}"
        has_fail |= status == FAIL
        print(f"[{status}] {name:<22} {msg}")
    print("\nKết luận:", "CÓ LỖI, xem các dòng FAIL" if has_fail else "Môi trường sẵn sàng")
    sys.exit(1 if has_fail else 0)
