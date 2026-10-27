"""libjinfer through the Python package, against the models in the hub cache. Plain asserts:
`python test_jinfer.py`. Needs LiquidAI/LFM2.5-350M-GGUF:Q8_0 and LFM2.5-Embedding-350M-GGUF:Q8_0
cached (or a network); the optional speech, transcription and reranking checks run when their
model is cached and are skipped otherwise. Checks pin behaviour, not model prose: prompts have
decisive answers, because a near-tie flips under different kernels."""
import asyncio, base64, enum, json, logging, math, os, sys, tempfile, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict
if "JINFER_INSTALLED" not in os.environ:  # the checkout's package; check-wheel tests the installed one
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jinfer
from jinfer import _native as N

LM = "LiquidAI/LFM2.5-350M-GGUF:Q8_0"
EMB = "LiquidAI/LFM2.5-Embedding-350M-GGUF:Q8_0"
checks = 0

def ok(cond, what):
    global checks
    assert cond, what
    checks += 1

# library
jinfer.configure("-Djinfer.offline=true")  # a test never downloads: offline is the process's, as the hub reads it
ok(jinfer.__version__ == "0.3.1", "version")
ok(not hasattr(jinfer, "Context") and not hasattr(jinfer, "Timings") and not hasattr(jinfer.LanguageModel, "decide") and "OpenAI" not in jinfer.__all__, "the surface is the documented one")
try:
    jinfer.load("nobody/does-not-exist-GGUF:Q8_0")
    ok(False, "missing model must raise")
except jinfer.JinferError as e:
    ok(isinstance(e, jinfer.NotFound) and isinstance(e, FileNotFoundError), f"an offline miss is NotFound: {e}")

records = []
handler = logging.Handler(); handler.emit = records.append
logging.getLogger("jinfer").addHandler(handler); logging.getLogger("jinfer").setLevel(logging.DEBUG)

# language model
m = jinfer.load(LM)
ok(isinstance(m, jinfer.LanguageModel) and not hasattr(m, "embed") and not hasattr(m, "kind"), f"{m!r}")
ok("kernels=native" in repr(m) and "threads=" in repr(m), f"repr says what runs: {m!r}")
ok(m.context_capacity == min(4096, m.max_context_length) and m.max_context_length >= m.context_capacity and m.name.endswith(".gguf"), f"properties: capacity {m.context_capacity} of {m.max_context_length}")
with jinfer.load(LM, context_capacity=1024) as small:
    ok(small.context_capacity == 1024 and small.max_context_length == m.max_context_length, "context_capacity=1024 is the state in use, the maximum unchanged")
    try:
        small.chat("word " * 1200, max_tokens=4).text; ok(False, "a prompt over the capacity must be refused")
    except jinfer.InvalidArgument as e:
        ok("1024" in str(e), f"over the capacity: {e}")
with jinfer.load(LM, context_capacity=-1) as whole:
    ok(whole.context_capacity == m.max_context_length, "context_capacity=-1 is the maximum")
try:
    jinfer.load(LM, context_capacity=m.max_context_length + 1); ok(False, "a capacity above the maximum must be refused")
except jinfer.InvalidArgument as e:
    ok(str(m.max_context_length) in str(e), f"above the maximum: {e}")
try:
    jinfer.load(LM, context_capacity="8k"); ok(False, "a string capacity must be a TypeError")
except TypeError as e:
    ok("context_capacity" in str(e), f"typed: {e}")
tk = m.tokenizer
ids = tk.encode("Hello world, how are you?")
ok(4 <= len(ids) <= 12 and all(isinstance(i, int) for i in ids), f"tokenizer.encode {ids}")
ok(tk.decode(ids) == "Hello world, how are you?" and tk.vocabulary_size > 1000 and tk.id(tk.token(ids[0])) == ids[0], "the model's tokenizer is a toknroll.Tokenizer")
ok(type(tk).__module__ == "toknroll" and repr(tk).startswith("Tokenizer("), f"{type(tk).__module__}.{type(tk).__name__}")

r = m.chat("What is the capital of France? Answer in one word.", max_tokens=8, temperature=0)
ok("Paris" in r.text and not hasattr(r, "response") and not hasattr(r, "message"), f"answer: {r.text!r}")
ok(r.finish_reason in ("stop", "length") and r.tool_calls == [] and r.reasoning is None, "reply fields")
ok(r.usage.prompt_tokens > 0 and r.usage.completion_tokens >= 1 and r.usage.prompt_ms > 0 and r.usage.cached_tokens == 0, f"usage {r.usage}")

# determinism is a contract: temperature 0 gives the same tokens every run
texts = {m.chat("Name the largest planet of the solar system.", max_tokens=12, temperature=0).text for _ in range(3)}
ok(len(texts) == 1 and "Jupiter" in next(iter(texts)), f"deterministic at temperature 0: {texts}")

chunks = list(m.chat("Count from one to five, words only, comma separated.", max_tokens=32, temperature=0))
ok(len(chunks) >= 3 and "three" in "".join(chunks).lower(), f"streamed: {''.join(chunks)!r}")
async def collect():
    return [t async for t in m.chat("Count from one to five, words only, comma separated.", max_tokens=32, temperature=0)]
ok("".join(asyncio.run(collect())) == "".join(chunks), "async iteration gives the same stream")

# a system message, and a reply read after partial iteration is the whole reply
r = m.chat("Who are you?", system="You are Zog. Always start your answer with the word Zog.", max_tokens=12, temperature=0)
first = next(iter(r))
ok(first and r.text.startswith(first) and "Zog" in r.text, f"system + partial iteration: {r.text!r}")

ok(str(m.chat("What is the capital of France? One word.", max_tokens=8, temperature=0)).strip().startswith("Paris"), "a reply prints as its text")
try:
    jinfer._check(N.lib.jinfer_model_property(m._h, 999)); ok(False, "an unknown property key must be refused")
except jinfer.InvalidArgument as e:
    ok("999" in str(e), f"unknown property key: {e}")

# the audit's regressions: mistakes a new user makes get a clear answer
ok(m.conversation(messages="Hi").messages == [{"role": "user", "content": "Hi"}], "conversation(messages=str) is one turn")
try:
    m.chat(None); ok(False, "messages=None must be refused")
except TypeError as e:
    ok("messages" in str(e), f"messages=None: {e}")
try:
    m.chat([{"role": "system", "content": "A"}, {"role": "user", "content": "Hi"}], system="B"); ok(False, "two system prompts must be refused")
except jinfer.InvalidArgument as e:
    ok("system" in str(e), f"two system prompts: {e}")
try:
    m.chat("Hi", max_tokens=4).parsed; ok(False, "parsed without schema must explain")
except jinfer.InvalidArgument as e:
    ok("schema=" in str(e), f"parsed without schema: {e}")
try:
    m.chat("x", schema=dict[str, int]); ok(False, "dict[str, int] cannot be enforced")
except jinfer.InvalidArgument as e:
    ok("dict" in str(e), f"typed open object refused: {e}")
ok(isinstance(m.chat("A person named Ada as JSON.", schema=dict, max_tokens=40, temperature=0).parsed, dict), "schema=dict is a free-form object")
ok(m.tokenizer.decode(m.tokenizer.encode("hello world")) == "hello world", "decode inverts encode")
r = m.chat("Hi", max_tokens=4)
before = repr(r); r.text
ok(before == "jinfer.Reply(generating)" and "finish_reason=" in repr(r), f"reply repr never blocks: {before} -> {r!r}")
r.close()
ok("finish_reason=" in repr(r), "a finished reply keeps its text after close")
r = m.chat("Write a long story about the sea.", max_tokens=300); first = next(iter(r)); r.close()
ok(r.finish_reason == "cancelled" and r.text.startswith(first) and r.usage.completion_tokens >= 1 and list(r) == [], f"a closed reply keeps what it produced: {r!r}")
try:
    m.chat("Three fruits.", schema=list[str], max_tokens=3, temperature=0).parsed; ok(False, "a cut-off JSON reply must explain")
except jinfer.InvalidArgument as e:
    ok("max_tokens" in str(e), f"cut-off JSON: {e}")
try:
    m.tokenizer.encode(123); ok(False, "encode non-str must be a TypeError")
except TypeError as e:
    ok("str" in str(e), f"encode non-str: {e}")
try:
    m.tokenizer.decode([10**6]); ok(False, "an out-of-range token id must be refused")
except ValueError as e:  # toknroll's InvalidArgument: the tokenizer's errors are toknroll's
    ok(type(e).__module__ == "toknroll" and "outside the vocabulary" in str(e), f"token id range: {e}")
try:
    m.chat("Hi", tools=[lambda x: x]); ok(False, "a lambda tool must be refused")
except jinfer.InvalidArgument as e:
    ok("lambda" in str(e), f"lambda tool: {e}")
try:
    jinfer.load("/nonexistent/model.gguf"); ok(False, "a missing path must be NotFound")
except jinfer.NotFound as e:
    ok("no such file" in str(e) and isinstance(e, FileNotFoundError), f"missing path: {e}")
try:
    jinfer.configure("-Djinfer.threads=4"); ok(False, "the runtime takes options before the first load only")
except jinfer.InvalidArgument as e:
    ok("once" in str(e), f"configure after a load: {e}")
import subprocess, sys as _sys
fresh = subprocess.run([_sys.executable, "-c", "import sys; sys.path.insert(0, 'python'); import jinfer\n"
                        "jinfer.configure('-Djinfer.threads=3'); m = jinfer.load(%r); print(repr(m))" % LM], capture_output=True, text=True)
ok("threads=3" in fresh.stdout, f"configure('-Djinfer.threads=3') sizes the pool: {fresh.stdout.strip()[-60:]} {fresh.stderr.strip()[-120:]}")
twice = subprocess.run([_sys.executable, "-c", "import sys; sys.path.insert(0, 'python'); import jinfer\n"
                        "jinfer.configure('-Djinfer.threads=3')\n"
                        "try:\n    jinfer.configure('-Djinfer.trace=false')\nexcept jinfer.InvalidArgument as e:\n    print('refused:', e)"], capture_output=True, text=True)
ok("must come once" in twice.stdout, f"configure is one call: {twice.stdout.strip()[-80:]} {twice.stderr.strip()[-120:]}")
import os as _os
env2 = dict(_os.environ, JINFER_OPTIONS="-Djinfer.threads=2")
over = subprocess.run([_sys.executable, "-c", "import sys; sys.path.insert(0, 'python'); import jinfer\n"
                       "jinfer.configure('-Djinfer.threads=3'); m = jinfer.load(%r); print(repr(m))" % LM], capture_output=True, text=True, env=env2)
ok("threads=2" in over.stdout, f"JINFER_OPTIONS overrides configure: {over.stdout.strip()[-60:]} {over.stderr.strip()[-120:]}")
knobs = subprocess.run([_sys.executable, "-c", "import sys; sys.path.insert(0, 'python'); import jinfer\n"
                        "jinfer.configure('-Djinfer.cache.sessions=1', '-Djinfer.cache.budget=512m', '-Djinfer.offline=true')\n"
                        "m = jinfer.load(%r); print(m._info['cache_sessions'], m._info['cache_budget'])" % LM], capture_output=True, text=True)
ok(knobs.stdout.strip() == f"1 {512 << 20}", f"cache knobs reach the engine: {knobs.stdout.strip()} {knobs.stderr.strip()[-120:]}")
bad = subprocess.run([_sys.executable, "-c", "import sys; sys.path.insert(0, 'python'); import jinfer\n"
                      "jinfer.configure('-Djinfer.cache.budget=lots', '-Djinfer.offline=true')\n"
                      "try:\n    jinfer.load(%r)\nexcept jinfer.InvalidArgument as e:\n    print('refused:', e)" % LM], capture_output=True, text=True)
ok("refused: jinfer.cache.budget must be" in bad.stdout and "lots" in bad.stdout, f"a bad budget is refused by name: {bad.stdout.strip()[-80:]} {bad.stderr.strip()[-120:]}")
try:
    m.chat("Hi", max_tokens="8"); ok(False, "a string max_tokens must be a TypeError")
except TypeError as e:
    ok("max_tokens" in str(e), f"option type checked: {e}")
r = m.chat("What is 2+2?", max_tokens=16, temperature=0, schema=int)
ok(json.loads(r.text) == {"value": 4} and r.parsed == 4, f"a scalar schema rides in an object on the wire: {r.text!r}")
ok(json.loads(m.chat("Name three planets.", max_tokens=48, temperature=0, schema=list[str]).text).__class__ is list, "a list schema goes bare")
ok(isinstance(m.chat([{"type": "text", "text": "What is the capital of France? One word."}], max_tokens=8, temperature=0).text, str), "a content list may start with a part dict")
try:
    m.chat("Hi", max_tokens=1, temperature=5).text; ok(False, "temperature 5 must be refused")
except jinfer.InvalidArgument as e:
    ok(not str(e).startswith("Invalid argument:"), f"no doubled prefix: {e}")

# a single function, a function as tool_choice, a dataclass result
def report(city: str) -> dict:
    """Weather report for a city."""
    return {"temp": 18}
r = m.chat("Weather in Oslo?", tools=report, tool_choice=report, max_tokens=48, temperature=0)
ok(r.finish_reason == "tool_calls" and r.tool_calls[0].name == "report", f"tools=fn, tool_choice=fn: {r.tool_calls}")
@dataclass
class Temp:
    celsius: int
def temp(city: str) -> Temp:
    """Temperature in a city."""
    return Temp(18)
cv = m.conversation(tools=[temp], max_tokens=48, temperature=0)
ok("18" in cv.chat("Temperature in Oslo? Use the tool.", tool_choice="temp").text and json.loads(cv.messages[2]["content"]) == {"celsius": 18}, "dataclass tool result is JSON")

# a typo in an option is an error at the call, not a silently ignored key
try:
    m.chat("Hi", temperatur=0); ok(False, "unknown option must be refused")
except TypeError as e:
    ok("temperatur" in str(e), f"typo refused: {e}")

# cancel mid-stream, then the model still answers
s = m.chat("List fifty animals.", max_tokens=200); next(iter(s)); s.close()
after = m.chat("What is the capital of France? Answer in one word.", max_tokens=12, temperature=0).text
ok("Paris" in after, f"usable after cancel: {after!r}")

# schema: the reply is the Python type it was asked for
@dataclass
class Place:
    city: str
    country: str
    continent: Literal["Europe", "Asia", "Africa", "America", "Oceania"]
place = m.chat("Where is the Eiffel Tower?", max_tokens=64, temperature=0, schema=Place).parsed
ok(isinstance(place, Place) and "paris" in place.city.lower() and place.continent == "Europe", f"dataclass schema: {place}")
turn = m.conversation(system="Be brief.", max_tokens=64, temperature=0)
ok(isinstance(turn.chat("Where is the Eiffel Tower?", schema=Place).parsed, Place) and turn.messages[-1]["role"] == "assistant", "a conversation turn takes a schema")
class Colours(TypedDict):
    colours: list[str]
doc = m.chat("Name two colours.", max_tokens=32, temperature=0, schema=Colours).parsed
ok(isinstance(doc, dict) and len(doc["colours"]) == 2, f"TypedDict schema: {doc}")
class Mood(enum.Enum):
    happy = "happy"; sad = "sad"
mood = m.chat("I won the lottery! How do I feel?", max_tokens=8, temperature=0, schema=Mood).parsed
ok(mood is Mood.happy, f"Enum schema: {mood}")
ok(m.chat("Name three planets.", max_tokens=32, temperature=0, schema=list[str]).parsed.__class__ is list, "list[str] schema")
schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
ok("paris" in m.chat("Where is the Eiffel Tower?", max_tokens=32, temperature=0, schema=schema).parsed["city"].lower(), "dict schema")
class Pyd:  # the duck type Pydantic models have, without the dependency
    @staticmethod
    def model_json_schema(): return schema
    @staticmethod
    def model_validate(value): return ("Pyd", value)
ok(m.chat("Where is the Eiffel Tower?", max_tokens=32, temperature=0, schema=Pyd).parsed[0] == "Pyd", "Pydantic-shaped schema")

# stop sequences are honoured
r = m.chat("Say the alphabet separated by spaces.", max_tokens=40, temperature=0, stop=["e"])
ok("e" not in r.text and r.finish_reason == "stop", f"stop: {r.text!r}")

# tools as functions: the schema comes from the signature, a conversation runs them
calls_made = []
def weather(city: str, unit: str = "celsius") -> dict:
    """Current weather for a city.

    Not part of the description."""
    calls_made.append(city)
    return {"city": city, "temperature": 18, "unit": unit, "sky": "clear"}
spec = jinfer._tool(weather)["function"]
ok(spec["name"] == "weather" and spec["description"] == "Current weather for a city." and spec["parameters"]["required"] == ["city"]
   and spec["parameters"]["properties"]["unit"] == {"type": "string"}, f"tool schema from the function: {spec}")
chat = m.conversation(system="You are a helpful assistant.", tools=[weather], max_tokens=64, temperature=0)
r = chat.chat("What's the weather in Zurich right now? Use the tool.")
ok("18" in r.text and calls_made == ["Zurich"] and [x["role"] for x in chat.messages] == ["system", "user", "assistant", "tool", "assistant"], f"tool ran, model answered: {r.text!r}")
# the single-shot form leaves the call to you
r = m.chat("What's the weather in Zurich?", tools=[weather], tool_choice={"type": "function", "function": {"name": "weather"}}, max_tokens=64, temperature=0)
ok(r.finish_reason == "tool_calls" and r.tool_calls[0].name == "weather" and "Zurich" in json.dumps(r.tool_calls[0].arguments), f"single-shot tool call: {r.tool_calls}")
forced = m.conversation(tools=[weather], max_tokens=64, temperature=0)
r = forced.chat("What's the weather in Zurich?", tool_choice={"type": "function", "function": {"name": "weather"}})
# a forced tool_choice applies to the first round only: the function runs once, then the model answers
ok(r.finish_reason == "stop" and r.text and len(calls_made) == 2, f"forced call ran once, then an answer: {r.text!r}")
hand = m.conversation(tools=[jinfer._tool(weather)], max_tokens=64, temperature=0)   # a dict tool has nobody to run it here
try:
    hand.chat("What's the weather in Zurich?"); ok(False, "a conversation must refuse a tool it cannot run")
except jinfer.InvalidArgument as e:
    ok("functions" in str(e) and len(hand.messages) == 0, f"dict tool in a conversation: {e}")

# a conversation remembers, forks and rewinds; the engine serves the shared prefix from its state
chat = m.conversation(system="You are a helpful assistant.", max_tokens=16, temperature=0)
chat.chat("My name is Zoltan. Remember it.")
r = chat.chat("What is my name?")
ok("Zoltan" in r.text and len(chat.messages) == 5 and chat.messages[-1]["role"] == "assistant", f"conversation: {r.text!r} {len(chat.messages)}")
ok(r.usage.cached_tokens > 0, f"second turn served from the cache: {r.usage}")
branch = chat.fork()
branch.chat("Forget that. What is 2+2? Digits only.")
ok(len(branch.messages) == 7 and len(chat.messages) == 5, "fork is independent")
chat.rewind(1)
ok(len(chat.messages) == 3 and chat.messages[-1]["role"] == "assistant", f"rewind drops one exchange: {[x['role'] for x in chat.messages]}")
ok(chat.fork().rewind(10).messages == [{"role": "system", "content": "You are a helpful assistant."}], "rewind past the start keeps the system prompt")
parts = list(chat.chat("Count from one to five, words only, comma separated.", max_tokens=24))
ok(len(parts) >= 3 and "".join(parts) == chat.messages[-1]["content"], f"a conversation turn streams: {parts}")
r = chat.chat("Write a long story about the sea.", max_tokens=300); first = next(iter(r)); r.close()
ok(chat.messages[-1]["content"].startswith(first) and r.finish_reason == "cancelled", "a cancelled turn keeps the words that were said")
r = chat.chat("Write a long story about the sea.", max_tokens=300); r.close()
ok(chat.messages[-1]["role"] == "assistant" and chat.messages[-1]["content"].startswith(first), "a turn cancelled before a word leaves no trace")
secret_calls = []
def secret() -> str:
    """The secret word."""
    secret_calls.append(1); return "pineapple"
r = m.conversation(tools=[secret], max_tokens=32, temperature=0).chat("What is the secret word? Use the tool, then tell me.", tool_choice=secret)
parts = list(r)
ok(secret_calls == [1] and "pineapple" in "".join(parts).lower() and r.usage.completion_tokens > 0, f"a tool turn streams the answer after the call: {parts}")

# prepare computes the prefix now; the next turn finds it cached
warm = m.conversation(system="You are a terse assistant. " * 20)
warm.prepare()
r = warm.chat("Hi", max_tokens=4)
ok(r.usage.cached_tokens > 0, f"prepared prefix cached: {r.usage}")

# the prompt cache file is read and extended: a later model finds what this one prepared
catalog = os.path.join(tempfile.mkdtemp(), "prompts.bin")
with jinfer.load(LM, prompt_cache=catalog) as m1:
    m1.conversation(system="You are a terse assistant. " * 10).prepare()
with jinfer.load(LM, prompt_cache=catalog) as m2:
    r = m2.conversation(system="You are a terse assistant. " * 10).chat("Hi", max_tokens=2)
    ok(r.usage.cached_tokens > 0, f"prompt cache survives the model: {r.usage}")

# media are values: strings, Paths and bytes become content parts
parts = jinfer._content(["What is this?", b"\x89PNG\r\n\x1a\n" + b"\0" * 8])
ok(parts[0] == {"type": "text", "text": "What is this?"} and parts[1]["image_url"]["url"].startswith("data:image/png;base64,"), "parts from text and PNG bytes")
wavpath = Path(tempfile.mkdtemp()) / "a.wav"; wavpath.write_bytes(b"RIFF\0\0\0\0WAVEfmt ")
ok(jinfer._part(wavpath)["input_audio"]["format"] == "wav" and jinfer._part(b"RIFF\0\0\0\0WAVE")["type"] == "input_audio", "audio parts from a Path and bytes")
try:
    jinfer._part(b"not a known format"); ok(False, "unknown bytes must be refused")
except jinfer.InvalidArgument as e:
    ok("Path" in str(e), f"unknown bytes: {e}")

def cached(ref, **kw):
    try:
        return jinfer.load(ref, **kw)
    except jinfer.JinferError:
        print(f"skipped: {ref} is not cached")
        return None

# a corrupt image is refused, never decoded into a frame nobody sent
if (vl := cached("LiquidAI/LFM2.5-VL-450M-GGUF:Q8_0", companions={"media": "LiquidAI/LFM2.5-VL-450M-GGUF/mmproj-LFM2.5-VL-450m-Q8_0.gguf"})) is not None:
    try:
        vl.chat([b"\x89PNG" + b"\0" * 20, "What is this?"], max_tokens=4).text; ok(False, "a corrupt PNG must be refused")
    except jinfer.InvalidArgument as e:
        ok("decoded" in str(e), f"corrupt image: {str(e)[:80]}")
    vl.close()

# a generation that fails raises its own error, not the result accessor's
try:
    m.chat([base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"), "What is this?"], max_tokens=4).text; ok(False, "an image to a text-only model must be refused")  # a valid 1x1 PNG
except jinfer.JinferError as e:
    ok("companion" in str(e), f"image to a text-only model: {str(e)[:70]}")

# the model outlives its handle while a reply is live
s = m.chat("Say hi.", max_tokens=4); m.close(); ok(isinstance(s.text, str), "reply after model close"); s.close()

# OpenAI surface, a migration aid
client = jinfer.OpenAI(LM)
r = client.chat.completions.create(messages=[{"role": "user", "content": "What is the capital of France? Answer in one word."}], max_tokens=8, temperature=0, n=1)
ok("Paris" in r.choices[0].message.content and r.object == "chat.completion", f"openai: {r.choices[0].message.content!r}")
deltas = [c.choices[0].delta.get("content", "") for c in client.chat.completions.create(messages=[{"role": "user", "content": "Count to three."}], max_tokens=16, stream=True)]
ok(len(deltas) >= 2 and deltas[-1] == "", f"openai stream: {deltas}")
r = client.chat.completions.create(messages=[{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "A JSON object with a city."}], response_format={"type": "json_object"}, max_completion_tokens=24, temperature=0)
ok(isinstance(json.loads(r.choices[0].message.content), dict) and r.usage.completion_tokens <= 24, f"openai json_object, developer role, max_completion_tokens: {r.choices[0].message.content!r}")
client.close()

# embeddings
try:
    jinfer.load(EMB).embed([1, 2]); ok(False, "non-strings must be refused")
except TypeError as e:
    ok("str" in str(e), f"embed non-str: {e}")
e = jinfer.load(EMB)
ok(isinstance(e, jinfer.Embedder) and e.dimension > 0 and not hasattr(e, "chat") and e.context_capacity == e.max_context_length > 0, f"{e!r}")
v = e.embed(["AI on the JVM", "the JVM thinks now", "the python shed its skin"])
def cos(a, b): return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
rows = [list(map(float, row)) for row in v]
ok(len(rows) == 3 and len(rows[0]) == e.dimension, "embed shape")
ok(cos(rows[0], rows[1]) > cos(rows[0], rows[2]), f"similar beats unrelated: {cos(rows[0], rows[1]):.3f} vs {cos(rows[0], rows[2]):.3f}")
one = e.embed("AI on the JVM")
# (1e-3: the native int8 kernels quantize activations per batch, so a row embedded alone differs
# from the same row inside a batch by ~1e-4 in cosine)
ok(len(one) == 1 and len(one[0]) == e.dimension and abs(cos(list(map(float, one[0])), rows[0]) - 1) < 1e-3, "one string is a batch of one")
ok(len(e.embed(["word " * 4000] * 3)) == 3, "texts of thousands of tokens embed whole")
try:
    e.embed("a", dimension=-1); ok(False, "a negative dimension must be refused")
except jinfer.InvalidArgument as err:
    ok("negative" in str(err), f"dimension=-1: {err}")
try:
    e.embed("AI on the JVM", dimension=64); ok(False, "a fixed-dimension model must refuse truncation")
except jinfer.InvalidArgument as err:
    ok("fixed" in str(err), f"dimension refused with a reason: {err}")
emb_client = jinfer.OpenAI(e)
ok(len(emb_client.embeddings.create(input=["a", "b"]).data[1].embedding) == e.dimension, "openai embeddings")
e.close()

# the other kinds, when their model is already cached (offline: never download in a test)

if (rr := cached("mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0")) is not None:
    ok(isinstance(rr, jinfer.Reranker), f"{rr!r}")
    scores = rr.rerank("capital of France", ["Paris is the capital of France.", "Bananas are yellow."])
    ok(scores[0] > scores[1], f"rerank order {scores}")
    ok(len(rr.rerank("q", "one document")) == 1, "one document is a list of one")
    rr.close()
wav = None
if (tts := cached("simonfxr/kokoro.cpp-GGUF:Q8_0", companions={"voice": "simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf"})) is not None:
    ok(isinstance(tts, jinfer.Speaker) and tts.sample_rate == 24000, f"{tts!r} {tts.sample_rate}")
    speech = tts.speak("Turns out, matrix multiplications can talk.", speed=1.3)
    ok(repr(speech) == "jinfer.Speech(generating)", "speech repr never blocks")
    chunks = list(speech)
    ok(repr(speech).startswith("jinfer.Speech(") and "Hz)" in repr(speech), f"speech repr: {speech!r}")
    seconds = len(speech.audio) / speech.sample_rate
    ok(len(chunks) >= 1 and 1.0 < seconds < 10 and len(speech.audio) == sum(len(c) for c in chunks), f"speech: {len(chunks)} chunks, {seconds:.1f} s")
    wav = os.path.join(tempfile.mkdtemp(), "speech.wav")
    speech.save(wav)
    ok(os.path.getsize(wav) > 44 + len(speech.audio), f"saved {wav}")
    tts.close()
if (asr := cached("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf")) is not None:
    ok(isinstance(asr, jinfer.Transcriber) and asr.sample_rate == 16000, f"{asr!r}")
    try:
        asr.transcribe([0.0] * 100); ok(False, "pcm without a rate must be refused")
    except jinfer.InvalidArgument as e:
        ok(isinstance(e, ValueError) and isinstance(e, jinfer.JinferError), f"typed invalid argument: {e}")
    if wav:
        t = asr.transcribe(wav)   # a 24 kHz file; the library resamples
        ok("matrix" in t.text.lower() and len(t.words) >= 5 and t.words[0].end > t.words[0].start, f"transcript: {t.text}")
    asr.close()

# the library's lines reached logging.getLogger("jinfer")
ok(any("loaded" in r.getMessage() for r in records) and any(r.levelno == logging.DEBUG for r in records), f"log records {[r.getMessage() for r in records[:3]]}")
# the static libjam is found: prefill runs on the native int8 kernels, not the Java fallback
ok(any("jam providers [native" in r.getMessage() for r in records), "native jam linked in")

# cancel from another thread: the blocked step returns Cancelled, the result still counts what ran
m = jinfer.load(LM)
r = m.chat("Write a very long story about the sea.", max_tokens=400)
seen, outcome = [], []
def consume():
    try:
        for text in r:
            seen.append(text)
    except jinfer.Cancelled:
        outcome.append("cancelled")
t = threading.Thread(target=consume); t.start()
while not seen and t.is_alive(): time.sleep(0.01)
N.lib.jinfer_stream_cancel(r._ptr)
t.join(timeout=30)
ok(outcome == ["cancelled"] and seen, f"cancel from another thread: {outcome} after {len(seen)} chunks")
ok(r.finish_reason == "cancelled" and 0 < r.usage.completion_tokens <= 400 and r.usage.prompt_tokens > 0, f"result after cancel: {r.finish_reason} {r.usage}")
ok(jinfer.__version__ == "0.3.1" == N.lib.jinfer_version().decode(), f"version {jinfer.__version__}")
m.close()
print(f"ok: {checks} checks")
