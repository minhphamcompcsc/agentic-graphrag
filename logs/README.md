# Logs

Console output of the runs, kept unchanged as evidence.

Logs created before 2026-11-24 were produced by earlier versions of the scripts whose messages were in Vietnamese. On 2026-11-24 all script messages were translated to English (logic unchanged). The logs themselves were not edited, so that they stay exactly as produced.

| File | Run |
|---|---|
| `env_check_2026-11-09.txt` | `check_env.py` |
| `extract_2026-11-09.txt` | `extract_schema.py` v0, first run |
| `extract_2026-11-09_v01.txt` | `extract_schema.py` v0.1 (`--reset`) |
| `semantic_2026-11-17_dryrun.txt` | `extract_semantic.py` sem-v0, first real dry run (failed: 504 then daily quota) |
| `semantic_2026-11-23_dryrun.txt` | `extract_semantic.py` sem-v0.1 dry run |
| `semantic_2026-11-23_dryrun_b.txt` | `extract_semantic.py` sem-v0.1 dry run with `gemini-3.6-flash` |
| `semantic_2026-11-23.txt` | `extract_semantic.py` sem-v0.2, write to Neo4j |
| `semantic_eval_2026-11-23.txt` | `eval_entity_links.py` |
| `lightrag_before_2026-11-23.txt` | `test_lightrag_neo4j.py --inspect` before the test |
| `lightrag_test_2026-11-23.txt` | `test_lightrag_neo4j.py` |

Common Vietnamese words in the older logs: "chức năng" = function, "đối tượng" = object (mention), "bằng chứng" = evidence, "lời gọi" = call, "gắn đúng" = correctly linked, "lớp cấu trúc" = structural layer, "lớp ngữ nghĩa" = semantic layer, "dòng" = rows, "cạnh" = edges, "Lưu" = saved.
