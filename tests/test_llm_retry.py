"""
Offline check of the error handling of the Mapping Agent LLM call (src/mapping/analyze.py), no network, no database.
  1. 503 / 504 / timeout: wait and try once more; a second transient error raises TransientLLMError
  2. daily quota: DailyQuotaExceeded at once, with the full Google message (quota id and limit kept in the logs)
  3. any other error: raised at once, no retry
Usage: python tests/test_llm_retry.py
"""
import os
import sys
from pathlib import Path

os.environ["LLM_TRANSIENT_WAIT"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mapping"))
import analyze  # noqa: E402

analyze.time.sleep = lambda s: None


class Out:
    def model_dump(self):
        return {"ok": 1}


class FakeLLM:
    def __init__(self, errors):
        self.errors, self.n = list(errors), 0

    def invoke(self, prompt):
        self.n += 1
        if self.errors:
            raise self.errors.pop(0)
        return Out()


def analyzer(errors):
    a = analyze.Analyzer("gemini", "test-model", 1, "/tmp/llm_retry_cache", 5)
    a._llm = FakeLLM(errors)
    return a


results = []
a = analyzer([Exception("DeadlineExceeded('Deadline expired before operation could complete.')")])
results.append(("504 then success: 2 attempts", a._call("x") == {"ok": 1} and a._llm.n == 2))

a = analyzer([Exception("504 Deadline"), Exception("503 UNAVAILABLE")])
try:
    a._call("x")
    results.append(("2 transient errors raise TransientLLMError", False))
except analyze.TransientLLMError:
    results.append(("2 transient errors raise TransientLLMError", a._llm.n == 2))

long_msg = ("ResourceExhausted('Quota exceeded for metric: generativelanguage.googleapis.com/"
            "generate_content_free_tier_requests, limit: 20, requests per day " + "x" * 400 + " quota_value: 20')")
a = analyzer([Exception(long_msg)])
try:
    a._call("x")
    results.append(("daily quota: full message, no retry", False))
except analyze.DailyQuotaExceeded as e:
    results.append(("daily quota: full message, no retry", "quota_value: 20" in str(e) and a._llm.n == 1))

a = analyzer([ValueError("bad schema")])
try:
    a._call("x")
    results.append(("other error: no retry", False))
except Exception as e:  # noqa: BLE001
    results.append(("other error: no retry", a._llm.n == 1 and isinstance(e, ValueError)))

for name, ok in results:
    print(f"{'OK  ' if ok else 'FAIL'} {name}")
print("ALL OK" if all(ok for _, ok in results) else "SOME CHECKS FAILED")
sys.exit(0 if all(ok for _, ok in results) else 1)
