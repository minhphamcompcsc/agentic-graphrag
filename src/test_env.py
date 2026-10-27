import os
from dotenv import load_dotenv
load_dotenv()

print("1. PostgreSQL (Odoo schema)")
import psycopg2
conn = psycopg2.connect(host="localhost", port=5432, dbname="kg_demo", user="odoo", password="odoo")
cur = conn.cursor()
cur.execute("SELECT count(*) FROM ir_model")
print("   So model:", cur.fetchone()[0])
cur.execute("SELECT count(*) FROM ir_model_fields WHERE relation IS NOT NULL")
print("   So field quan he:", cur.fetchone()[0])
cur.execute("""
    SELECT d.name FROM ir_module_module_dependency d
    JOIN ir_module_module m ON m.id = d.module_id
    WHERE m.name = 'stock_picking_invoice_link'""")
print("   Dependency cua stock_picking_invoice_link:", [r[0] for r in cur.fetchall()])

print("2. Neo4j")
from neo4j import GraphDatabase
drv = GraphDatabase.driver(os.getenv("NEO4J_URI"),
                           auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
drv.verify_connectivity()
print("   OK")

print("3. Ollama embedding")
from langchain_ollama import OllamaEmbeddings
v = OllamaEmbeddings(model="bge-m3").embed_query("Tự động trừ kho khi giao hàng")
print("   Vector dim:", len(v))

print("4. Gemini")
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), temperature=0)
print("   ", llm.invoke("Trả lời 1 câu: module Odoo nào quản lý tồn kho?").content)
