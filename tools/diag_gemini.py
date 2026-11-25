"""Chẩn đoán lỗi 504 khi gọi Gemini có structured output. Tốn tối đa 4 lời gọi."""
import os, sys, time
from importlib.metadata import version
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL")
print("langchain-google-genai", version("langchain-google-genai"), "| model", MODEL)

class Fn(BaseModel):
    name: str
    odoo_model: str

PROMPT = "The Odoo module stock_picking_back2draft allows to bring canceled transfers back to draft. Extract one business function and its Odoo model."

def run(label, fn):
    t = time.time()
    try:
        out = fn()
        print(f"[{label}] OK sau {time.time()-t:.1f}s: {str(out)[:150]}")
        return True
    except Exception as e:
        print(f"[{label}] LỖI sau {time.time()-t:.1f}s: {repr(e)[:300]}")
        return False

llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, max_retries=0, timeout=90)
ok1 = run("T1 trả lời thường", lambda: llm.invoke("Reply with one word: OK").content)
if not ok1:
    sys.exit("T1 lỗi: vấn đề ở key, mạng hoặc model, không phải structured output")
ok2 = run("T2 structured, function_calling", lambda: llm.with_structured_output(Fn).invoke(PROMPT))
if not ok2:
    ok3 = run("T3 structured, json_mode", lambda: llm.with_structured_output(Fn, method="json_mode").invoke(PROMPT))
    if not ok3:
        llm0 = ChatGoogleGenerativeAI(model=MODEL, temperature=0, max_retries=0, timeout=90, thinking_budget=0)
        run("T4 structured, tắt thinking", lambda: llm0.with_structured_output(Fn).invoke(PROMPT))
