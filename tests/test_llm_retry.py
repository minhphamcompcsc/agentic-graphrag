"""
Offline check of the error handling of the Mapping Agent LLM call (src/mapping/analyze.py), no network, no database.
  1. 503 / 504 / timeout: wait and try once more; a second transient error raises TransientLLMError
  2. daily quota: DailyQuotaExceeded at once, with the full Google message (quota id and limit kept in the logs)
  3. any other error: raised at once, no retry
  5. no structured output (model answered without a valid function call): retried once; the raw answers are
     saved in <cache>/errors/; a second failure raises TransientLLMError (clean stop); Mapping and Verifier
  4. no hidden retries: through LangChain and the real Google client (fake key, no network), a 503 reaches
     our code after exactly 1 request (the client alone retries 503 for up to 600 s, each retry uses quota)
Usage: python tests/test_llm_retry.py
"""
import os
import sys
from pathlib import Path

os.environ["LLM_TRANSIENT_WAIT"] = "0"
os.environ["LLM_NO_OUTPUT_WAIT"] = "0"
os.environ["LLM_PAUSE"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mapping"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "verifier"))
import analyze  # noqa: E402
import verifier  # noqa: E402



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

# 5. no structured output (include_raw answers of with_structured_output)
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402


class RawMsg:
    content = ""
    tool_calls = []
    invalid_tool_calls = []
    usage_metadata = {"output_tokens": 0}
    response_metadata = {"finish_reason": "MALFORMED_FUNCTION_CALL"}


BAD = {"raw": RawMsg(), "parsed": None, "parsing_error": None}


class SeqLLM:
    def __init__(self, answers):
        self.answers, self.n = list(answers), 0

    def invoke(self, prompt):
        self.n += 1
        return self.answers.pop(0)


tmp = Path(tempfile.mkdtemp())
a = analyze.Analyzer("gemini", "test-model", 1, tmp / "map", 5)
a._llm = SeqLLM([BAD, {"raw": RawMsg(), "parsed": Out(), "parsing_error": None}])
out, cached = a.analyze("Q09", "en", "some requirement")
errs = list((tmp / "map").glob("**/errors/en_Q09_*.json"))
rec = json.loads(errs[0].read_text()) if errs else {}
results.append(("Mapping, no structured output once: retried, answer cached, raw answer saved",
                out == {"ok": 1} and a._llm.n == 2 and a.calls == 1 and rec.get("recovered") is True
                and rec["attempts"][0]["finish_reason"] == "MALFORMED_FUNCTION_CALL"))
a = analyze.Analyzer("gemini", "test-model", 1, tmp / "map", 5)
a._llm = SeqLLM([BAD, BAD])
try:
    a.analyze("Q10", "en", "another requirement")
    results.append(("Mapping, no structured output twice: TransientLLMError", False))
except analyze.TransientLLMError:
    errs = list((tmp / "map").glob("**/errors/en_Q10_*.json"))
    results.append(("Mapping, no structured output twice: TransientLLMError, nothing cached, raw saved",
                    a._llm.n == 2 and a.calls == 0 and len(errs) == 1
                    and json.loads(errs[0].read_text())["recovered"] is False
                    and not list((tmp / "map").glob("**/en_Q10.json"))))
v = verifier.Verifier("gemini", "test-model", 1, tmp / "ver")
v._llm = SeqLLM([BAD, BAD])
try:
    v.judge("Q09", "en", 1, "prompt")
    results.append(("Verifier, no structured output twice: TransientLLMError", False))
except verifier.TransientLLMError:
    errs = list((tmp / "ver").glob("**/errors/en_Q09_round1_*.json"))
    results.append(("Verifier, no structured output twice: TransientLLMError (clean stop), raw saved",
                    v._llm.n == 2 and v.calls == 0 and len(errs) == 1))
v = verifier.Verifier("gemini", "test-model", 1, tmp / "ver2")
v._llm = SeqLLM([BAD, {"raw": RawMsg(), "parsed": Out(), "parsing_error": None}])
o, c = v.judge("Q09", "en", 1, "prompt")
results.append(("Verifier, no structured output once: retried and cached", o == {"ok": 1} and v.calls == 1
                 and not c and len(list((tmp / "ver2").glob("**/en_Q09_round1.json"))) == 1))
shutil.rmtree(tmp)

# 4. hidden retries of the Google client, end to end (the RPC is replaced, nothing is sent)
import signal  # noqa: E402
from google.api_core import exceptions as gexc, gapic_v1, retry as gretry  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402


def count_requests(fix, seconds=8):
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key="offline-test", max_retries=0, timeout=300)
    if fix:
        analyze.no_hidden_retry(llm)
    n = {"requests": 0}

    def rpc(request, **kw):
        n["requests"] += 1
        raise gexc.ServiceUnavailable("This model is currently experiencing high demand")

    tr = llm.client._transport
    tr._wrapped_methods[tr.generate_content] = gapic_v1.method.wrap_method(  # same default retry as the client
        rpc, default_retry=gretry.Retry(initial=1.0, maximum=10.0, multiplier=1.3,
                                        predicate=gretry.if_exception_type(gexc.ServiceUnavailable), deadline=600.0),
        default_timeout=600.0)

    def stop(*a):
        raise TimeoutError

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        llm.invoke("Reply OK")
        outcome = "answer"
    except TimeoutError:
        outcome = f"still retrying after {seconds} s"
    except Exception as e:  # noqa: BLE001
        outcome = type(e).__name__
    finally:
        signal.alarm(0)
    return n["requests"], outcome


req0, out0 = count_requests(fix=False)
req1, out1 = count_requests(fix=True)
print(f"    without the fix: {req0} requests, {out0}; with no_hidden_retry: {req1} request(s), {out1}")
results.append(("no hidden retries: 1 request per call on 503", req1 == 1 and req0 > 1))

for name, ok in results:
    print(f"{'OK  ' if ok else 'FAIL'} {name}")
print("ALL OK" if all(ok for _, ok in results) else "SOME CHECKS FAILED")
sys.exit(0 if all(ok for _, ok in results) else 1)
