"""
Amharic response-level distillation -- two-stage data pipeline.

STAGE 1  PROMPTS  (teacher-independent: written once, reusable with ANY teacher)
    prompts          -> data/prompts_raw.jsonl
    filter-prompts   -> data/prompts_clean.jsonl (+ .csv, prompts_dropped.jsonl, prompts_clean.meta.json)

STAGE 2  RESPONSES  (one run per teacher; same prompts every time)
    answer           -> data/responses_<tag>.jsonl
    judge            -> adds judge scores to that file (resumable)
    finalize         -> data/train_<tag>.jsonl
    sample           -> data/sample_check_<tag>.csv   (hand review)

Setup:
    pip install google-genai            # Gemini backend
    pip install openai                  # optional: any OpenAI-compatible backend (OpenAI, vLLM, Ollama, ...)
    pip install transformers            # optional: exact Gemma token lengths (gated model: accept licence + HF login)
    export GEMINI_API_KEY=...           # for --backend gemini
    export OPENAI_API_KEY=...           # for --backend openai   (optional: OPENAI_BASE_URL)

Typical run:
    python generate_data.py evalstats                         # compare your prompt lengths with the 60 eval prompts
    python generate_data.py prompts --jobs-per-task 2         # pilot (test first!)
    python generate_data.py prompts                           # full run (resumable, safe to re-run)
    python generate_data.py filter-prompts                    # needs eval prompts (see below)
    python generate_data.py sample-prompts                    # review prompts by hand BEFORE paying for answers
    python generate_data.py answer --backend gemini --limit-per-task 5     # pilot answers
    python generate_data.py answer --backend gemini                         # full answers
    python generate_data.py judge --backend gemini
    python generate_data.py finalize --backend gemini
    python generate_data.py sample --backend gemini
    python generate_data.py answer-eval --backend gemini      # optional teacher_output column for the 60 eval prompts
    python generate_data.py examples --backend gemini         # README examples: kept / removed by reason / eval duplicates

    # another teacher on the SAME prompts:
    python generate_data.py answer --backend openai --model <model-name>
    python generate_data.py judge  --backend openai --model <model-name>
    python generate_data.py finalize --backend openai --model <model-name>

Eval prompts (trainer's 60): put eval_prompts_60.csv (or eval_prompts.csv) next to this script; it needs a
'prompt' column (an 'eval_id' and a 'task' column are used when present). A plain eval_prompts.txt also works
(prompts separated by a line containing only "-----"). filter-prompts and finalize ABORT unless exactly 60
eval prompts are found, so the contamination check can never be silently skipped. NEVER use reference answers.
Re-run filter-prompts whenever the eval file changes (finalize refuses to run on a stale filter).
Outputs for the submission: data/train_<tag>.csv (prompt, teacher_response, task, source), data/readme_examples_<tag>.md,
data/prompts_clean.meta.json (counts incl. removed_as_eval_duplicates), data/run_log.jsonl (models, seed, versions).
"""
import os, sys, re, csv, json, time, random, hashlib, argparse, unicodedata, datetime
from collections import Counter
from pathlib import Path

# ======================= CONFIG =======================
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": None}   # check current names / free-tier limits
DEFAULT_SLEEP = {"gemini": 4.0, "openai": 0.5}                    # seconds between calls (free-tier friendly)
MAX_RETRIES = 6
MAX_FAIL_STREAK = 5             # stop (resumably) if this many calls in a row fail after all retries
SEED = 42

OUT_DIR = Path("data"); OUT_DIR.mkdir(exist_ok=True)
PROMPTS_RAW = OUT_DIR / "prompts_raw.jsonl"
PROMPT_JOBS_DONE = OUT_DIR / "done_prompt_jobs.txt"
PROMPTS_CLEAN = OUT_DIR / "prompts_clean.jsonl"
PROMPTS_CLEAN_CSV = OUT_DIR / "prompts_clean.csv"
PROMPTS_DROPPED = OUT_DIR / "prompts_dropped.jsonl"
PROMPTS_META = OUT_DIR / "prompts_clean.meta.json"
PROMPTS_CANDIDATES = OUT_DIR / "prompts_candidates.jsonl"   # pre-trim survivors of the rule filters
PROMPT_SCORES = OUT_DIR / "prompt_scores.jsonl"             # judge-prompts cache (resumable)
MIN_PROMPT_SCORE = 4                                        # judge-prompts: keep only if natural & answerable >= this
EVAL_FILE = Path("eval_prompts.txt")
EVAL_CSV_CANDIDATES = [Path("eval_prompts.csv"), Path("eval_prompts_60.csv")]   # first one that exists is used
EXPECTED_EVAL = 60
RUN_LOG = OUT_DIR / "run_log.jsonl"
STUDENT_TOKENIZER = "google/gemma-3-270m"     # exact token counts for the student (same 262k Gemma-3 vocabulary as 1b-it)
# eval task names <-> this script's task names
EVAL_TASK_MAP = {"closed_book_knowledge": "knowledge", "instruction_following": "comprehension",
                 "translation": "translation", "summarization": "summarisation"}
CSV_TASK_NAME = {v: k for k, v in EVAL_TASK_MAP.items()}

TARGETS = {"knowledge": 500, "comprehension": 500, "translation": 500, "summarisation": 500}
ITEMS_PER_CALL = {"knowledge": 10, "comprehension": 3, "translation": 8, "summarisation": 3}
# (sentences, words) bands, rotated over the writer jobs so passage lengths cover the eval range.
# Eval (EthioNLP) text after 'Context:': reading comprehension median 103 words (48-273); summarization median 167 (73-348);
# translation source text median 11 words (7-34).  Models follow sentence counts better than word counts, and (pilot) write ~8-10-word
# sentences unless told otherwise, so the sentence counts below assume ~18 words per sentence and plen() asks for 15-25.
PASSAGE_BANDS = {
    "comprehension": [((3, 5), (55, 90)), ((5, 7), (90, 130)), ((7, 11), (130, 200)), ((9, 13), (180, 260))],
    "summarisation": [((5, 7), (80, 130)), ((8, 10), (130, 190)), ((11, 15), (190, 280)), ((14, 18), (270, 400))],
}
PASSAGE_WORDS = {t: (min(b[1][0] for b in bands), max(b[1][1] for b in bands)) for t, bands in PASSAGE_BANDS.items()}
TRANSLATION_WORDS = (4, 50)     # accepted size of the text to translate (we ask for 8-35 words, mostly one sentence)
P_CONTEXT = {"comprehension": 0.7, "summarisation": 0.7, "translation": 0.75}   # share of prompts in the eval's 'Context:' layout
P_BARE_QUESTION = 0.4           # closed-book eval prompts are bare questions

# Writer sampling temperature per task: long Amharic passages glitch (other scripts leak in) at high temperature
WRITER_TEMPERATURE = {"knowledge": 0.8, "comprehension": 0.4, "summarisation": 0.4, "translation": 0.8}
# Over-generate, because filters drop glitched items (pilot: only ~40% of passage items survived);
# filter-prompts then trims each task back to TARGETS
WRITE_FACTOR = {"knowledge": 1.2, "comprehension": 5.0, "translation": 1.2, "summarisation": 5.0}

ANSWER_TEMPERATURE = 0.3
ANSWER_MAX_WORDS = 120          # told to the teacher; keeps answers inside the token cap
MIN_SCORE = 4                   # judge: keep only if all scores >= this
MAX_ANSWER_TOKENS = 200         # keeps answers safely inside max_new_tokens=256
JACCARD_EVAL = 0.40             # char 4-gram Jaccard: full train prompt vs eval prompt
CONTAIN_EVAL = 0.60             # share of an eval prompt's 4-grams found in the train prompt
CONTAIN_CONTENT = 0.60          # share of the train CONTENT's 4-grams found in an eval prompt
JACCARD_INTERNAL = 0.80         # near-duplicate threshold inside the training prompts (content only)

DOMAINS = {
    "history": ["ancient Ethiopian kingdoms (Aksum)", "Battle of Adwa", "Ethiopian calendar and holidays",
                "African independence movements", "ancient world civilizations", "20th century world events"],
    "health": ["malaria prevention", "maternal and child health", "nutrition and balanced diet",
               "hygiene and sanitation", "vaccination", "common first aid"],
    "science": ["water cycle and weather", "photosynthesis and plants", "human body systems",
                "electricity and energy", "earth and space", "simple chemistry in daily life"],
    "agriculture": ["teff and cereal farming", "coffee production", "soil conservation",
                    "livestock care", "irrigation and water use", "pest and crop disease management"],
    "news": ["local community events", "economy and markets", "education developments",
             "environment and climate", "transport and infrastructure", "sports"],
    "technology": ["mobile phones and mobile money", "internet and social media", "artificial intelligence basics",
                   "renewable energy and solar", "computer basics", "digital safety"],
    "daily_life": ["cooking and food culture", "family and relationships", "markets and shopping",
                   "travel and transport", "school life", "festivals and traditions"],
}

# Rotated instruction wordings (reviewing/editing these Amharic templates yourself is recommended)
T = {
    "knowledge": [
        "የሚከተለውን ጥያቄ በአጭሩ እና በትክክል መልስ።\n\nጥያቄ፡ {q}",
        "{q}",
        "እባክህ ይህን ጥያቄ መልስልኝ፡ {q}",
        "ጥያቄ፡ {q}\n\nመልስ፡",
        "ስለሚከተለው ጥያቄ ግልጽ እና አጭር መልስ ስጥ፡\n{q}",
    ],
    "comprehension": [
        "የተሰጠውን ጽሑፍ በጥንቃቄ ካነበብክ በኋላ የቀረበውን ጥያቄ መልስ።\n\nጽሑፍ፡ {p}\n\nጥያቄ፡ {q}",
        "ከሚከተለው ጽሑፍ በመነሳት ጥያቄውን መልስ።\n\n{p}\n\nጥያቄ፡ {q}",
        "ጽሑፉን አንብበህ ጥያቄውን መልስ፡\n\nጽሑፍ፡ {p}\n\nጥያቄ፡ {q}",
        "ይህን ምንባብ መሠረት በማድረግ ለጥያቄው መልስ ስጥ።\n\nምንባብ፡ {p}\n\nጥያቄ፡ {q}",
    ],
    "EN2AM": [
        'Translate the following English text into clear and natural Amharic: "{t}"',
        "Translate into Amharic:\n\n{t}",
        'የሚከተለውን የእንግሊዝኛ ጽሑፍ ወደ አማርኛ ተርጉም፡ "{t}"',
        "Please translate this to Amharic: {t}",
    ],
    "AM2EN": [
        'የሚከተለውን የአማርኛ ጽሑፍ ወደ እንግሊዝኛ ተርጉም፡ "{t}"',
        'Translate the following Amharic text into English: "{t}"',
        "Translate into English:\n\n{t}",
        "ይህን አማርኛ ወደ እንግሊዝኛ ተርጉምልኝ፡ {t}",
    ],
    "summarisation": [
        "የሚከተለውን ጽሑፍ ዋና ዋና ነጥቦች በመያዝ በ2-3 ዓረፍተ ነገሮች አጠቃልል።\n\nጽሑፍ፡ {p}",
        "ይህን ጽሑፍ በአጭሩ አጠቃልልልኝ።\n\n{p}",
        "ጽሑፉን አንብበህ ዋናውን ሐሳብ በሁለት ወይም ሦስት ዓረፍተ ነገሮች ግለጽ።\n\nጽሑፍ፡ {p}",
        "የሚከተለውን ምንባብ በአጭሩ አጠቃልል፡\n\n{p}",
    ],
}

# The 45 non-closed-book eval prompts all use:   <instruction>\n\nContext:\n<text>   (comprehension: the question follows
# the passage after a blank line; no "ጽሑፍ፡/ጥያቄ፡" labels).  The wordings below are ORIGINAL (not copied from the eval prompts)
# and cover masculine / feminine / polite / plural forms.  Have a native speaker review them.
CTX_INSTR = {
    "comprehension": [
        "ከዚህ በታች ያለውን ምንባብ አንብበህ ጥያቄውን መልስ።",
        "ምንባቡን ካነበብሽ በኋላ ለጥያቄው መልስ ስጪ።",
        "እባክዎን ከታች ያለውን ጽሑፍ አንብበው ለሚከተለው ጥያቄ መልስ ይስጡ።",
        "በሚከተለው ጽሑፍ መሠረት ጥያቄውን በአጭሩ መልስ።",
        "ጽሑፉን በጥንቃቄ አንብባችሁ ጥያቄውን ከጽሑፉ በመነሳት መልሱ።",
        "ከዚህ በታች የቀረበውን መረጃ በመጠቀም ለጥያቄው መልስ ስጥ።",
        "ንባቡን መሠረት በማድረግ ጥያቄውን መልስ።",
    ],
    "summarisation": [
        "የሚከተለውን ጽሑፍ አጠቃልል።",
        "ለዚህ ጽሑፍ አጭር ማጠቃለያ ጻፍ።",
        "የጽሑፉን ዋና ሐሳብ በአጭሩ ግለጽ።",
        "ከዚህ በታች ያለውን ጽሑፍ በ2-3 ዓረፍተ ነገሮች አጠቃልለው።",
        "እባክዎን ይህን ጽሑፍ በአጭሩ ያጠቃልሉ።",
        "ጽሑፉን አንብበሽ ማጠቃለያውን ጻፊ።",
        "ይህ ጽሑፍ ምን እንደሚናገር በአጭሩ አስረዳ።",
        "የሚከተለውን ጽሑፍ ዋና ዋና ነጥቦች ብቻ በመያዝ አጭር አድርገህ ግለጽ።",
    ],
    "AM2EN": [
        "የሚከተለውን የአማርኛ ዓረፍተ ነገር ወደ እንግሊዝኛ ተርጉም።",
        "ይህን አማርኛ ጽሑፍ ወደ እንግሊዝኛ ተርጉምልኝ።",
        "ከአማርኛ ወደ እንግሊዝኛ ተርጉም።",
        "እባክዎን ይህን የአማርኛ ጽሑፍ በእንግሊዝኛ ይተርጉሙ።",
        "ከታች ያለውን የአማርኛ ጽሑፍ ወደ English ተርጉም።",
    ],
    "EN2AM": [
        "የሚከተለውን የእንግሊዝኛ ዓረፍተ ነገር ወደ አማርኛ ተርጉም።",
        "ይህን እንግሊዝኛ ጽሑፍ ወደ አማርኛ ተርጉምልኝ።",
        "ከእንግሊዝኛ ወደ አማርኛ ተርጉም።",
        "እባክዎን ይህን የእንግሊዝኛ ጽሑፍ ወደ አማርኛ ይተርጉሙ።",
        "ከታች ያለውን English ጽሑፍ ወደ አማርኛ ተርጉም።",
    ],
}
T["knowledge"].append("እባክዎን ለሚከተለው ጥያቄ መልስ ይስጡ፡ {q}")      # polite form

CONTENT_KEYS = {"knowledge": ("question",), "comprehension": ("passage", "question"),
                "summarisation": ("passage",), "translation": ("text",)}

# Markdown / chatty preambles the teacher was told not to produce
BAD_RESPONSE = re.compile(r"(\*\*|^\s*#{1,6}\s|```|^\s*(?:here is\b|here's\b|sure\b|certainly\b|of course\b|translation\s*:))",
                          re.I | re.M)


# Refusal-like answers (e.g. "I cannot answer, the information was not provided"); our prompts are always answerable
REFUSAL = re.compile(r"አልችልም|አልተሰጠም|ስላልተካተተ|ስላልተሰጠ|ስላልተገለጸ|ስላልተጠቀሰ")


class Abort(Exception):
    pass

# ======================= LLM BACKENDS =======================
class LLM:
    """Thin wrapper so any teacher / prompt-writer / judge can be swapped in with --backend/--model."""

    def __init__(self, backend, model=None, sleep=None, base_url=None):
        self.backend = backend
        self.model = model or DEFAULT_MODELS.get(backend)
        if not self.model:
            raise Abort(f"--model is required for backend '{backend}'")
        self.sleep = DEFAULT_SLEEP[backend] if sleep is None else sleep
        self.name = f"{backend}:{self.model}"
        self.fail_streak = 0
        self.thinking_level = (os.environ.get("THINKING_LEVEL") or "").strip().upper() or None   # MINIMAL | LOW | MEDIUM | HIGH
        self.json_ok = True          # switched off automatically if the model rejects JSON mode
        if backend == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise Abort("GEMINI_API_KEY is not set")
            from google import genai
            from google.genai import types
            self._types = types
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=480000))
        elif backend == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise Abort("OPENAI_API_KEY is not set")
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)
        else:
            raise Abort(f"unknown backend '{backend}'")

    def __call__(self, prompt, temperature, json_mode=False):
        for attempt in range(MAX_RETRIES):
            try:
                if self.backend == "gemini":
                    kw = {"temperature": temperature}
                    if json_mode and self.json_ok:
                        kw["response_mime_type"] = "application/json"
                    if self.thinking_level:
                        kw["thinking_config"] = self._types.ThinkingConfig(thinking_level=self.thinking_level)
                    r = self._client.models.generate_content(
                        model=self.model, contents=prompt,
                        config=self._types.GenerateContentConfig(**kw))
                    text = r.text
                else:   # openai-compatible (JSON mode not forced: it requires a top-level object)
                    r = self._client.chat.completions.create(
                        model=self.model, temperature=temperature,
                        messages=[{"role": "user", "content": prompt}])
                    text = r.choices[0].message.content
                time.sleep(self.sleep)
                self.fail_streak = 0
                return text or None          # empty (e.g. safety-blocked) is not retried
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                if self.backend == "gemini" and json_mode and self.json_ok and (
                        "json mode" in low or "response_mime_type" in low or "mime type" in low):
                    self.json_ok = False
                    print("  (this model does not support JSON mode: continuing without it)")
                    continue
                if "429" in msg and ("perday" in low or "per day" in low or "limit: 0" in low):
                    raise Abort("quota exhausted for this model/key (per-day limit, or no free tier for it). Progress is saved: use another --model or wait for the reset.")
                wait = min(90, 3 * 2 ** attempt)
                print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {type(e).__name__}: {str(e)[:120]} -> sleep {wait}s")
                time.sleep(wait)
        self.fail_streak += 1
        if self.fail_streak >= MAX_FAIL_STREAK:
            raise Abort(f"{MAX_FAIL_STREAK} calls in a row failed (quota? wrong model name?). "
                        "Progress is saved; fix the cause and re-run.")
        return None


def parse_json(text):
    """Tolerant JSON parse: strips code fences, finds the outermost [..]/{..}, unwraps {"items": [...]}."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    cands = [text]
    for a, b in (("[", "]"), ("{", "}")):
        i, j = text.find(a), text.rfind(b)
        if i != -1 and j > i:
            cands.append(text[i:j + 1])
    for c in cands:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict):
            lists = [v for v in obj.values() if isinstance(v, list)]
            if lists:
                return lists[0]
        return obj
    return None

# ======================= META-PROMPTS (A: write prompts, B: answer, C: judge) =======================
def build_A(task, domain, sub, ctx, n, direction, band=0, style="EXPLAINER"):
    news = style == "NEWS" and task in PASSAGE_BANDS
    if task == "translation" and direction == "EN2AM":
        script_note = ("Write every item in English only (Latin letters); do not write any Amharic or Ethiopic "
                       "characters. Double-check spelling.")
    else:
        script_note = ("Use natural, standard Amharic (Fidel script) wherever Amharic is required. Use ONLY Ethiopic "
                       "(Fidel) characters for Amharic and plain Latin letters for English; never mix scripts inside "
                       "a word; write Amharic prefixes attached to the next word (no space after የ, በ, ለ, ከ); "
                       "separate words with exactly one space: never glue two words together and never use double spaces. "
                         "never use characters from any other writing system. If you do not know the Amharic form of a loanword, "
                       "use a simpler Amharic word or a short Amharic description; never switch to another language "
                       "or script. Double-check spelling.")
    if news:
        facts = ("Write each passage like a short Amharic news report with concrete details: invented names of fictional "
                 "people, organisations and places, dates, numbers and amounts. Every specific detail must be fictional: "
                 "never use real public figures and never describe real events. ")
    else:
        facts = "Never invent real named people, statistics or news events. "
    head = (
        "You are building an Amharic instruction dataset for a small language model.\n"
        f"Task: {task}\nDomain: {domain}\nSubtopic: {sub}\n"
        f"Setting: {'Ethiopian context' if ctx == 'ETHIOPIAN' else 'general / global context'}\n"
        f"Write {n} diverse items. Vary question type and difficulty; no two items may share wording "
        "or structure. " + facts + script_note + "\n"
    )
    def plen(t):
        bands = PASSAGE_BANDS[t]
        (s_lo, s_hi), (lo, hi) = bands[band % len(bands)]
        return (f"{s_lo}-{s_hi} sentences, each a full, information-rich sentence of about 15-25 words "
                f"(so about {lo}-{hi} words in total; never shorter than {lo} words)")
    only = (' The "text" value must contain ONLY the text to be translated: no instruction, no label, '
            'no quotation marks.')
    blocks = {
        "knowledge": "Each item is a self-contained question in Amharic, answerable without any passage. Mix the types: "
                     "about 60% factual questions (what/who/when/where/which) and about 40% questions that ask to explain "
                     "a cause, reason, process or comparison (why/how/'explain ...'), each answerable in one short paragraph. "
                     "Ask only about well-established facts that can be answered reliably; avoid obscure details, "
                     "disputed claims and exact figures unless they are widely known, and never ask for the year of an event. "
                     'Output ONLY JSON: [{"question": "..."}]',
        "comprehension": "Each item has an original Amharic passage of " + plen("comprehension") + " and one "
                         "question answerable ONLY from that passage (vary: who/what/why/how, yes/no, list; at most one question in four may ask for a number or a count; write every question as a complete, natural Amharic sentence). "
                         'Output ONLY JSON: [{"passage": "...", "question": "..."}]',
        "summarisation": "Each item is an original Amharic passage of " + plen("summarisation") + " suitable "
                         'for a 2-3 sentence summary. Output ONLY JSON: [{"passage": "..."}]',
        "EN2AM": 'Each item is an original text written in ENGLISH ONLY (Latin letters; no Amharic): usually ONE sentence of '
                 '8-35 words, sometimes two short sentences; mix formal and everyday styles.' + only + ' Output ONLY JSON: [{"text": "..."}]',
        "AM2EN": 'Each item is an original text written in AMHARIC ONLY (Fidel script; no English words): usually ONE sentence of '
                 '8-35 words, sometimes two short sentences; mix formal and everyday styles.' + only + ' Output ONLY JSON: [{"text": "..."}]',
    }
    key = direction if task == "translation" else task
    return head + blocks[key]


def build_B(prompt):
    return (
        "Respond to the user prompt below.\n"
        "Rules:\n"
        "- Write in natural, standard Amharic (Fidel script only), unless the prompt asks for a translation "
        "into English; then answer in English only.\n"
        "- For translation, output only the translation.\n"
        f"- Maximum {ANSWER_MAX_WORDS} words. Be accurate and concise. If unsure of a fact, be cautious; "
        "never invent facts.\n"
        "- If the prompt contains a passage, answer only from that passage, in one or two sentences.\n"
        "- If the prompt is a plain question with no passage, answer it from your own knowledge. "
        "Never say that information is missing and never refuse.\n"
        "- For a summary, write 2-3 sentences unless the prompt asks for another length.\n"
        "- If you mention a year, say whether it is in the Ethiopian or the Gregorian calendar.\n"
        "- No preamble, no explanations, no markdown.\n\n"
        f"User prompt:\n{prompt}"
    )


def build_C(pairs):
    body = "\n\n".join(f"### {i}\nPROMPT: {p}\nRESPONSE: {a}" for i, (p, a) in enumerate(pairs))
    return (
        "You are a strict reviewer of Amharic training data. Rate each numbered pair from 1 to 5 on "
        "fluency, accuracy and task (task-following). Output ONLY JSON: "
        '[{"i": 0, "fluency": n, "accuracy": n, "task": n}, ...] with one entry per pair.\n\n' + body
    )

# ======================= SMALL HELPERS =======================
def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def pct(vals, p):
    if not vals:
        return 0
    v = sorted(vals)
    return v[min(len(v) - 1, int(p / 100 * len(v)))]


def to_num(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def tag_of(backend, model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{backend}_{model}")


def resolve_tag(args):
    if getattr(args, "tag", None):
        return args.tag
    model = args.model or DEFAULT_MODELS.get(args.backend)
    if not model:
        raise Abort("give --model (or --tag) so the right responses file can be found")
    return tag_of(args.backend, model)


def resp_path(tag):
    return OUT_DIR / f"responses_{tag}.jsonl"


def read_eval_rows():
    """Trainer's eval prompts as [{'id','task','prompt'}]: eval_prompts.csv / eval_prompts_60.csv ('prompt' column) or eval_prompts.txt."""
    for csvp in EVAL_CSV_CANDIDATES:
        if csvp.exists():
            with csvp.open(encoding="utf-8-sig", newline="") as f:
                rd = csv.DictReader(f)
                names = {(c or "").strip().lower(): c for c in (rd.fieldnames or [])}
                if "prompt" not in names:
                    raise Abort(f"{csvp} needs a column named 'prompt'")
                idc, tc = names.get("eval_id") or names.get("id"), names.get("task")
                rows = []
                for n, r in enumerate(rd, 1):
                    p = (r[names["prompt"]] or "").strip()
                    if p:
                        rows.append({"id": ((r.get(idc) if idc else None) or f"E{n:02d}").strip(),
                                     "task": r.get(tc) if tc else None, "prompt": p})
            return rows
    if EVAL_FILE.exists():
        text = EVAL_FILE.read_text(encoding="utf-8")
        if re.search(r"^-{3,}\s*$", text, re.M):
            ps = [p.strip() for p in re.split(r"^-{3,}\s*$", text, flags=re.M) if p.strip()]
        else:
            ps = [l.strip() for l in text.splitlines() if l.strip()]
        return [{"id": f"E{n:02d}", "task": None, "prompt": p} for n, p in enumerate(ps, 1)]
    return []


def read_eval():
    return [r["prompt"] for r in read_eval_rows()]


def require_eval_rows():
    """The contamination check must never be skipped silently: abort unless exactly 60 eval prompts are found."""
    rows = read_eval_rows()
    if not rows:
        raise Abort("no eval prompts found. Copy eval_prompts_60.csv (column 'prompt') next to this script "
                    "(or save it as eval_prompts.csv).")
    if len(rows) != EXPECTED_EVAL:
        raise Abort(f"expected {EXPECTED_EVAL} eval prompts, parsed {len(rows)} (check the file / separator format).")
    return rows


def eval_digest(ev):
    return hashlib.sha256("\x00".join(ev).encode("utf-8")).hexdigest() if ev else None


def pkg_versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for p in ("google-genai", "openai", "transformers", "torch", "peft", "trl"):
        try:
            out[p] = metadata.version(p)
        except Exception:
            pass
    return out


def log_run(stage, llm=None, **extra):
    """Append model / seed / temperature / package versions to data/run_log.jsonl (for the README)."""
    rec = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "stage": stage, "seed": SEED, "versions": pkg_versions(), **extra}
    if llm is not None:
        rec["llm"] = llm.name
    append_jsonl(RUN_LOG, [rec])


# ======================= TEXT NORMALISATION / SIMILARITY =======================
def norm(t):
    t = unicodedata.normalize("NFC", t).casefold()
    t = re.sub(r"[^\w\s]", "", t)          # also strips Ethiopic punctuation (U+1362..)
    return re.sub(r"\s+", " ", t).strip()


def sha(t):
    return hashlib.sha256(norm(t).replace(" ", "").encode("utf-8")).hexdigest()


def grams(t, n=4):
    t = norm(t)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def contain(a, b):
    """Share of a's n-grams that also occur in b."""
    return len(a & b) / len(a) if a else 0.0


def geez_ratio(t):
    letters = [c for c in t if c.isalpha()]
    return sum("\u1200" <= c <= "\u137F" for c in letters) / len(letters) if letters else 0.0


def has_foreign_script(t):
    """True if any letter is neither Ethiopic nor Latin (e.g. Georgian, Cyrillic, CJK, Arabic glitches)."""
    for c in t:
        if c.isalpha():
            n = unicodedata.name(c, "")
            if not (n.startswith("ETHIOPIC") or n.startswith("LATIN")):
                return True
    return False


def has_mixed_script_word(t):
    """True if one whitespace-separated word mixes Fidel and Latin letters (e.g. 'አዳulis')."""
    for w in t.split():
        if any("\u1200" <= c <= "\u139F" for c in w) and any("a" <= c.lower() <= "z" for c in w):
            return True
    return False


def detached_particles(t):
    """Count one-syllable Fidel tokens such as 'የ አኩስም': Amharic prefixes belong attached to the next word."""
    n = 0
    for w in t.split():
        w = w.strip(".,;:!?()\"'«»፡።፣፤፥፦፧")
        if len(w) == 1 and w.isalpha() and "\u1200" <= w <= "\u139F":
            n += 1
    return n


TRANSLATE_LEAK = re.compile(r"translate\b|ተርጉም|ተርጉሙ", re.I)
LABEL_LEAK = re.compile(r"^\s*(?:ጥያቄ|ጽሑፍ|ምንባብ|መልስ|question|passage|text|answer)\s*[:፡]", re.I)


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        AutoTokenizer = None
        err = e
    if AutoTokenizer is not None:
        for name in (STUDENT_TOKENIZER, "google/gemma-3-1b-it"):      # same Gemma-3 vocabulary
            try:
                return AutoTokenizer.from_pretrained(name)
            except Exception as e:
                err = e
    print(f"(no Gemma tokenizer: {type(err).__name__}; using words*3 as a rough token proxy -- "
          "accept the licence + `huggingface-cli login` for exact counts)")
    return None


def count_tokens(tok, text):
    return len(tok(text)["input_ids"]) if tok else int(len(text.split()) * 3)

# ======================= STAGE 1: PROMPTS =======================
def build_jobs():
    combos = [(d, s) for d, subs in DOMAINS.items() for s in subs]
    L = len(combos)
    jobs = []
    for task, target in TARGETS.items():
        n = ITEMS_PER_CALL[task]
        for i in range(-(-int(target * WRITE_FACTOR[task]) // n)):
            d, s = combos[(i + i // L) % L]
            r = random.Random(f"{SEED}-{task}-{i}")
            ctx = r.choice(["ETHIOPIAN", "GLOBAL"])
            passage = task in PASSAGE_BANDS
            style = ("NEWS" if d == "news" else r.choice(["NEWS", "EXPLAINER"])) if passage else None
            jobs.append({
                "id": f"{task}|{i}", "task": task, "n": n, "domain": d, "sub": s, "ctx": ctx,
                "band": i % len(PASSAGE_BANDS[task]) if passage else None, "style": style,
                "direction": ("EN2AM" if i % 2 == 0 else "AM2EN") if task == "translation" else None,
            })
    return jobs


def clean_ws(s):
    """Collapse runs of spaces/tabs/NBSP inside a text (the writer sometimes emits double spaces)."""
    return re.sub(r"[ \t\u00a0]{2,}", " ", s)


def extract_content(task, item):
    c = {k: clean_ws(str(item.get(k) or "").strip()) for k in CONTENT_KEYS[task]}
    return c if all(c.values()) else None


def render_prompt(task, content, direction, r):
    key = direction if task == "translation" else task
    if task in P_CONTEXT and r.random() < P_CONTEXT[task]:      # eval layout: <instruction>\n\nContext:\n<text>[\n\n<question>]
        k = r.randrange(len(CTX_INSTR[key]))
        if task == "comprehension":
            body = f"{content['passage']}\n\n{content['question']}"
        elif task == "summarisation":
            body = content["passage"]
        else:
            body = content["text"]
        return f"{CTX_INSTR[key][k]}\n\nContext:\n{body}", f"{key}-ctx{k}"
    if task == "knowledge" and r.random() < P_BARE_QUESTION:
        k = T["knowledge"].index("{q}")
    else:
        k = r.randrange(len(T[key]))
    fmt = {"q": content.get("question"), "p": content.get("passage"), "t": content.get("text")}
    return T[key][k].format(**fmt), f"{key}-{k}"


def generate_prompts(llm, jobs_per_task=None, bands=None):
    jobs = build_jobs()
    if bands:                       # e.g. --bands 1 2: only passage jobs of those bands
        jobs = [j for j in jobs if j["band"] in bands]
    if jobs_per_task:
        per, picked = Counter(), []
        for j in jobs:
            if per[j["task"]] < jobs_per_task:
                picked.append(j); per[j["task"]] += 1
        jobs = picked
    done = set(PROMPT_JOBS_DONE.read_text().split("\n")) if PROMPT_JOBS_DONE.exists() else set()
    todo = [j for j in jobs if j["id"] not in done]
    print(f"prompt writer: {llm.name} | {len(todo)} jobs to run ({len(jobs) - len(todo)} already done)")
    log_run("prompts", llm, writer_temperature=WRITER_TEMPERATURE, jobs=len(todo))
    for idx, job in enumerate(todo, 1):
        print(f"[{idx}/{len(todo)}] {job['id']} {job['domain']}/{job['sub']} {job['ctx']} band={job['band']} style={job['style']}")
        items = parse_json(llm(build_A(job["task"], job["domain"], job["sub"], job["ctx"], job["n"],
                                       job["direction"], job["band"] or 0, job["style"] or "EXPLAINER"),
                               WRITER_TEMPERATURE[job["task"]], json_mode=True))
        if not isinstance(items, list):
            print("  writer output unusable, will retry on next run"); continue
        r = random.Random(f"{SEED}-{job['id']}")
        rows = []
        for k, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            content = extract_content(job["task"], item)
            if not content:
                continue
            prompt, tid = render_prompt(job["task"], content, job["direction"], r)
            rows.append({
                "id": f"{job['id']}#{k}", "task": job["task"], "direction": job["direction"],
                "domain": job["domain"], "subtopic": job["sub"], "context": job["ctx"],
                "band": job["band"], "style": job["style"],
                "template_id": tid, "content": content, "prompt": prompt,
                "prompt_writer": llm.name,
            })
        append_jsonl(PROMPTS_RAW, rows)
        with PROMPT_JOBS_DONE.open("a", encoding="utf-8") as f:
            f.write(job["id"] + "\n")
    print(f"prompts_raw now has {len(read_jsonl(PROMPTS_RAW))} rows -> {PROMPTS_RAW}")


def filter_prompts():
    """Quality + contamination filtering on PROMPTS only -- done before any answer is paid for."""
    rows = read_jsonl(PROMPTS_RAW)
    if not rows:
        raise Abort("no prompts_raw.jsonl yet: run `prompts` first")
    pscores = {x["id"]: x for x in read_jsonl(PROMPT_SCORES)}      # judge-prompts results (empty on the first pass)
    ev_rows = require_eval_rows()                      # aborts unless exactly 60 eval prompts are found
    ev = [r["prompt"] for r in ev_rows]
    ev_ids = [r["id"] for r in ev_rows]
    ev_hash_id = {}                                    # hash -> eval id; whole prompt AND the text after "Context:"
    for i, r in enumerate(ev_rows):
        ev_hash_id[sha(r["prompt"])] = ev_ids[i]
        if "\n\nContext:\n" in r["prompt"]:
            ev_hash_id[sha(r["prompt"].split("\n\nContext:\n", 1)[1])] = ev_ids[i]
    ev_hashes = set(ev_hash_id)
    ev_grams = [grams(e) for e in ev]
    seen_prompt, seen_content, kept_cgrams = set(), set(), []
    kept, dropped, why = [], [], Counter()

    def drop(x, reason, extra=None):
        why[reason] += 1
        dropped.append({"reason": reason, **(extra or {}), **{k: x[k] for k in ("id", "task", "prompt")}})

    for x in rows:
        x["content"] = {k: clean_ws(v) for k, v in x["content"].items()}
        x["prompt"] = clean_ws(x["prompt"])
        task, c, prompt = x["task"], x["content"], x["prompt"]
        ctext = " ".join(c.values())
        # 1. script of the content
        gr_all = geez_ratio(ctext)
        if task == "translation" and x["direction"] == "EN2AM":
            if geez_ratio(c["text"]) > 0.1:
                drop(x, "script_mismatch"); continue
        elif gr_all < 0.8:
            drop(x, "script_mismatch"); continue
        # 1b. glitches: foreign scripts, scripts mixed inside a word, instructions/labels leaked into the content
        if has_foreign_script(ctext):
            drop(x, "foreign_script"); continue
        if has_mixed_script_word(ctext):
            drop(x, "mixed_script_word"); continue
        if detached_particles(ctext) >= 2:
            drop(x, "detached_particles"); continue
        if task == "translation" and TRANSLATE_LEAK.search(c["text"]):
            drop(x, "instruction_in_text"); continue
        if any(LABEL_LEAK.match(v) for v in c.values()):
            drop(x, "label_in_content"); continue
        # 2. length: passage within its band (tune with `evalstats`); translation source text within TRANSLATION_WORDS
        if "passage" in c:
            bands = PASSAGE_BANDS[task]
            lo, hi = bands[x["band"] % len(bands)][1] if x.get("band") is not None else PASSAGE_WORDS[task]
            wc = len(c["passage"].split())
            if wc < 0.7 * lo or wc > 1.6 * hi:
                drop(x, "passage_length", {"words": wc}); continue
        if task == "translation":
            wc = len(c["text"].split())
            if not TRANSLATION_WORDS[0] <= wc <= TRANSLATION_WORDS[1]:
                drop(x, "text_length", {"words": wc}); continue
        # 3. exact duplicates (against train and eval)
        h, hc = sha(prompt), sha(ctext)
        if h in ev_hashes or hc in ev_hashes:
            drop(x, "exact_duplicate_eval", {"eval_id": ev_hash_id.get(h) or ev_hash_id.get(hc)}); continue
        if h in seen_prompt or hc in seen_content:
            drop(x, "exact_duplicate_train"); continue
        # 4. near-duplicates vs eval (char 4-grams; three complementary tests)
        gf, gc = grams(prompt), grams(ctext)
        hit = None
        for i, e in enumerate(ev_grams):
            j, ce, cc = jaccard(gf, e), contain(e, gf), contain(gc, e)
            if j > JACCARD_EVAL or ce > CONTAIN_EVAL or cc > CONTAIN_CONTENT:
                hit = {"eval_index": i, "eval_id": ev_ids[i], "jaccard": round(j, 3), "eval_in_prompt": round(ce, 3),
                       "content_in_eval": round(cc, 3)}
                break
        if hit:
            drop(x, "near_dup_eval", hit); continue
        # 4b. judge-prompts scores (only prompts that were already scored are affected)
        s = pscores.get(x["id"])
        if s and (to_num(s.get("natural")) < MIN_PROMPT_SCORE or to_num(s.get("answerable")) < MIN_PROMPT_SCORE):
            drop(x, "low_prompt_quality", {"scores": {k: s.get(k) for k in ("natural", "answerable")}}); continue
        # 5. near-duplicates inside the training prompts (content only, so shared templates don't count)
        if any(jaccard(gc, k) > JACCARD_INTERNAL for k in kept_cgrams):
            drop(x, "near_dup_train"); continue
        seen_prompt.add(h); seen_content.add(hc); kept_cgrams.append(gc); kept.append(x)

    write_jsonl(PROMPTS_CANDIDATES, kept)          # pre-trim survivors: this is what judge-prompts scores
    # trim each task (translation: each direction) to its target, spread randomly over topics
    groups = {}
    for x in kept:
        groups.setdefault((x["task"], x["direction"]), []).append(x)
    rnd, keep_ids = random.Random(SEED), set()
    for (task, _), v in groups.items():
        cap = TARGETS[task] // 2 if task == "translation" else TARGETS[task]
        rnd.shuffle(v)
        if task in PASSAGE_BANDS:          # take bands round-robin so long passages are not crowded out
            pools = {}
            for x in v:
                pools.setdefault(x["band"], []).append(x)
            pools, order = list(pools.values()), []
            while any(pools):
                for pool in pools:
                    if pool:
                        order.append(pool.pop())
            v = order
        keep_ids.update(id(x) for x in v[:cap])
    n_trim = len(kept) - len(keep_ids)
    kept = [x for x in kept if id(x) in keep_ids]
    write_jsonl(PROMPTS_CLEAN, kept)
    write_jsonl(PROMPTS_DROPPED, dropped)
    with PROMPTS_CLEAN_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "task", "direction", "domain", "prompt"])
        for x in kept:
            w.writerow([x["id"], x["task"], x["direction"] or "", x["domain"], x["prompt"]])
    meta = {"n_raw": len(rows), "n_kept": len(kept), "n_eval": len(ev), "eval_sha256": eval_digest(ev),
            "dropped": dict(why),
            "removed_as_eval_duplicates": why["exact_duplicate_eval"] + why["near_dup_eval"], "seed": SEED,
            "thresholds": {"JACCARD_EVAL": JACCARD_EVAL, "CONTAIN_EVAL": CONTAIN_EVAL,
                           "CONTAIN_CONTENT": CONTAIN_CONTENT, "JACCARD_INTERNAL": JACCARD_INTERNAL}}
    PROMPTS_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"raw={len(rows)} kept={len(kept)} -> {PROMPTS_CLEAN} (+ .csv)")
    print("dropped:", dict(why), f"(details: {PROMPTS_DROPPED})")
    print(f"REMOVED AS EVAL DUPLICATES: {why['exact_duplicate_eval'] + why['near_dup_eval']} "
          f"(exact {why['exact_duplicate_eval']}, near {why['near_dup_eval']}) of {len(rows)} raw prompts")
    print("raw per task:", dict(Counter(x["task"] for x in rows)), f"| trimmed to target: {n_trim}")
    print("kept per task:", dict(Counter(x["task"] for x in kept)))
    print("translation directions kept:", dict(Counter(x["direction"] for x in kept if x["task"] == "translation")))
    for t in PASSAGE_BANDS:
        print(f"per band ({t}): kept", dict(sorted(Counter(x["band"] for x in kept if x["task"] == t).items())),
              "| raw", dict(sorted(Counter(x["band"] for x in rows if x["task"] == t).items())))
    for t in PASSAGE_WORDS:
        wc = [len(x["content"]["passage"].split()) for x in rows if x["task"] == t]
        if wc:
            print(f"passage words written ({t}, before filtering): median={pct(wc, 50)} min={min(wc)} "
                  f"max={max(wc)} | target {PASSAGE_WORDS[t][0]}-{PASSAGE_WORDS[t][1]}")
    for d in dropped[:6]:
        print(f"  dropped [{d['reason']}] {d['id']}: {d['prompt'][:90]!r}")


def build_P(texts):
    body = "\n\n".join(f"### {i}\n{t}" for i, t in enumerate(texts))
    return (
        "You are a strict native-speaker reviewer of Amharic training prompts. Rate each numbered item from 1 to 5:\n"
        "- natural: is the Amharic grammatical and idiomatic, with correctly spaced words (no glued words, no "
        "meaningless or garbled phrases, no wrongly mixed-in words)? Give 1 or 2 if any sentence is garbled.\n"
        "- answerable: is the task clear and answerable as asked? For a question about a passage: answerable ONLY "
        "from the passage. For a translation: one clean text to translate. For a knowledge question: a well-defined "
        "question with a reliable answer.\n"
        'Output ONLY JSON: [{"i": 0, "natural": n, "answerable": n}, ...] with one entry per item.\n\n' + body
    )


def judge_prompts(llm, batch=6):
    """Stage 1 quality check of the candidate prompts (pre-trim survivors). Resumable; scores cached in PROMPT_SCORES.
    Afterwards re-run `filter-prompts`: it drops low-scored prompts and refills each task from the over-generated pool."""
    if not PROMPTS_CANDIDATES.exists():
        raise Abort("no prompts_candidates.jsonl: run `filter-prompts` first")
    cands = read_jsonl(PROMPTS_CANDIDATES)
    done = {x["id"] for x in read_jsonl(PROMPT_SCORES)}
    todo = [x for x in cands if x["id"] not in done]
    print(f"prompt judge: {llm.name} | {len(todo)} prompts to score ({len(done)} already scored)")
    log_run("judge-prompts", llm, min_score=MIN_PROMPT_SCORE, n=len(todo))
    if any(x.get("prompt_writer") == llm.name for x in todo):
        print("NOTE: some prompts were written by this same model; an independent judge (--model) is better.")
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        sc = parse_json(llm(build_P([x["prompt"] for x in chunk]), 0.0, json_mode=True))
        if not isinstance(sc, list):
            continue
        by_i = {x.get("i"): x for x in sc if isinstance(x, dict)}
        rows = []
        for j, x in enumerate(chunk):
            r = by_i.get(j)
            if r and r.get("natural") is not None and r.get("answerable") is not None:
                rows.append({"id": x["id"], "task": x["task"], "natural": r["natural"],
                             "answerable": r["answerable"], "judge": llm.name})
        append_jsonl(PROMPT_SCORES, rows)
        print(f"  scored {min(s + batch, len(todo))}/{len(todo)}")
    print("done. Now re-run: python generate_data.py filter-prompts")


def evalstats():
    """Compare the trainer's eval prompts with ours (length/script only; no answers involved)."""
    ev = read_eval_rows()
    if ev:
        by = {}
        for r in ev:
            by.setdefault(EVAL_TASK_MAP.get(r["task"], r["task"] or "all"), []).append(r["prompt"])
        print(f"eval prompts: n={len(ev)}; mean Fidel ratio={sum(geez_ratio(r['prompt']) for r in ev) / len(ev):.2f}")
        for task, ps in by.items():
            wc = [len(p.split()) for p in ps]
            cw = [len(p.split("\n\nContext:\n", 1)[1].split()) for p in ps if "\n\nContext:\n" in p]
            extra = f" | words after 'Context:' min/median/max = {min(cw)}/{pct(cw, 50)}/{max(cw)}" if cw else ""
            print(f"  eval {task:<14} n={len(ps):<3} prompt words min/median/max = {min(wc)}/{pct(wc, 50)}/{max(wc)}{extra}")
    else:
        print("no eval prompts found yet")
    rows = read_jsonl(PROMPTS_CLEAN) or read_jsonl(PROMPTS_RAW)
    for task in TARGETS:
        xs = [x for x in rows if x["task"] == task]
        if xs:
            wc = [len(x["prompt"].split()) for x in xs]
            cw = [len(" ".join(x["content"].values()).split()) for x in xs]
            print(f"  ours {task:<14} n={len(xs):<4} prompt words min/median/max = {min(wc)}/{pct(wc, 50)}/{max(wc)}"
                  f" | content words min/median/max = {min(cw)}/{pct(cw, 50)}/{max(cw)}")


def _words(t):
    return [w.strip(".,;:!?()\"'«»፡።፣፤፥፦፧") for w in t.split()]


def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(w for w in _words(t) if w)
    return c


def glued_words(t, vocab, min_part=2, whole_max=1, part_min=5):
    """Tokens that look like two frequent words stuck together (the model sometimes drops a space)."""
    out = []
    for w in _words(t):
        if len(w) < 2 * min_part + 1 or not any("\u1200" <= ch <= "\u137F" for ch in w):
            continue
        if vocab[w] > whole_max:
            continue
        for k in range(min_part, len(w) - min_part + 1):
            if vocab[w[:k]] >= part_min and vocab[w[k:]] >= part_min:
                out.append(w)
                break
    return out


def sample_rows(rows, n=50):
    by = {}
    for x in rows:
        by.setdefault(x["task"], []).append(x)
    rnd = random.Random(SEED)
    for v in by.values():
        rnd.shuffle(v)
    per = -(-n // max(1, len(by)))
    out = [x for v in by.values() for x in v[:per]]
    return out[:n] if len(out) > n else out


def sample_prompts():
    rows = read_jsonl(PROMPTS_CLEAN) or read_jsonl(PROMPTS_RAW)
    path = OUT_DIR / "sample_prompts_check.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "task", "prompt", "my_naturalness_1to5", "notes"])
        for x in sample_rows(rows):
            w.writerow([x["id"], x["task"], x["prompt"], "", ""])
    print(f"wrote {path} (review by hand: are the prompts natural, diverse, correct Amharic?)")

# ======================= STAGE 2: RESPONSES =======================
def parallel_map(fn, items, workers):
    """Yield (item, fn(item)) as each call completes. workers<=1 keeps the old one-at-a-time behaviour."""
    if workers <= 1:
        for it in items:
            yield it, fn(it)
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            yield futs[f], f.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def answer(llm, limit_per_task=None):
    if not PROMPTS_CLEAN.exists():
        raise Abort("no prompts_clean.jsonl: run `prompts` then `filter-prompts` first")
    prompts = read_jsonl(PROMPTS_CLEAN)
    tag = tag_of(llm.backend, llm.model)
    path = resp_path(tag)
    done = {x["id"] for x in read_jsonl(path)}
    todo = [p for p in prompts if p["id"] not in done]
    if limit_per_task:
        per, picked = Counter(), []
        for p in todo:
            if per[p["task"]] < limit_per_task:
                picked.append(p); per[p["task"]] += 1
        todo = picked
    print(f"teacher: {llm.name} | {len(todo)} prompts to answer ({len(done)} already done) -> {path}")
    log_run("answer", llm, temperature=ANSWER_TEMPERATURE, max_words=ANSWER_MAX_WORDS, n=len(todo))
    workers = int(os.environ.get("WORKERS", "1") or 1)
    print(f"workers: {workers}")
    results = parallel_map(lambda p: llm(build_B(p["prompt"]), ANSWER_TEMPERATURE), todo, workers)
    for idx, (p, ans) in enumerate(results, 1):
        print(f"[{idx}/{len(todo)}] {p['id']}")
        if not ans:
            continue
        append_jsonl(path, [{
            "id": p["id"], "task": p["task"], "direction": p["direction"], "domain": p["domain"],
            "subtopic": p["subtopic"], "context": p["context"], "template_id": p["template_id"],
            "prompt": p["prompt"], "response": ans.strip(), "scores": None,
            "teacher": llm.name, "prompt_writer": p["prompt_writer"],
        }])


def judge(llm, tag, batch=8):
    path = resp_path(tag)
    rows = read_jsonl(path)
    if not rows:
        raise Abort(f"{path} is empty or missing: run `answer` first")
    pending = [i for i, x in enumerate(rows) if not x.get("scores")]
    print(f"judge: {llm.name} | {len(pending)} responses to score")
    log_run("judge", llm, tag=tag, min_score=MIN_SCORE)
    if llm.name.split(":", 1)[1] == rows[0]["teacher"].split(":", 1)[1]:
        print("WARNING: the judge is the same model as the teacher (self-judging bias); "
              "use --judge-backend/--judge-model for an independent judge.")
    for s in range(0, len(pending), batch):
        idxs = pending[s:s + batch]
        sc = parse_json(llm(build_C([(rows[i]["prompt"], rows[i]["response"]) for i in idxs]), 0.0, json_mode=True))
        if isinstance(sc, list):
            by_i = {x.get("i"): x for x in sc if isinstance(x, dict)}
            for j, i in enumerate(idxs):
                x = by_i.get(j)
                if x:
                    rows[i]["scores"] = {k: x.get(k) for k in ("fluency", "accuracy", "task")}
                    rows[i]["judge"] = llm.name
        write_jsonl(path, rows)          # progress saved after every batch
        print(f"  scored {min(s + batch, len(pending))}/{len(pending)}")


def finalize(tag, min_score=MIN_SCORE, skip_judge=False):
    if not PROMPTS_META.exists():
        raise Abort("run `filter-prompts` first")
    meta = json.loads(PROMPTS_META.read_text(encoding="utf-8"))
    cur = eval_digest([r["prompt"] for r in require_eval_rows()])
    if meta.get("eval_sha256") != cur:
        raise Abort("eval prompts changed (or were added) since `filter-prompts` ran: "
                    "re-run filter-prompts (and re-answer any newly kept prompts) before finalizing.")
    path = resp_path(tag)
    rows = read_jsonl(path)
    if not rows:
        raise Abort(f"{path} is empty or missing")
    clean_ids = {x["id"] for x in read_jsonl(PROMPTS_CLEAN)}
    tok = load_tokenizer()
    vocab = build_vocab([x["response"] for x in rows] + [x["prompt"] for x in rows])
    kept, dropped, why = [], [], Counter()

    def drop(x, reason):
        why[reason] += 1
        dropped.append({"reason": reason, "id": x["id"], "task": x["task"], "direction": x.get("direction"),
                        "prompt": x["prompt"], "response": x["response"], "scores": x.get("scores")})

    for x in rows:
        if x["id"] not in clean_ids:
            drop(x, "prompt_no_longer_clean"); continue
        if not skip_judge:
            s = x.get("scores")
            if not s or any(to_num(s.get(k)) < min_score for k in ("fluency", "accuracy", "task")):
                drop(x, "low_or_missing_score"); continue
        resp = x["response"]
        if not resp.strip():
            drop(x, "empty"); continue
        if BAD_RESPONSE.search(resp):
            drop(x, "markdown_or_preamble"); continue
        if REFUSAL.search(resp):
            drop(x, "refusal"); continue
        want_latin = x["task"] == "translation" and x["direction"] == "AM2EN"
        gr = geez_ratio(resp)
        if (want_latin and gr > 0.1) or (not want_latin and gr < 0.8):
            drop(x, "script_mismatch"); continue
        if has_foreign_script(resp) or has_mixed_script_word(resp):
            drop(x, "foreign_or_mixed_script"); continue
        if not want_latin and glued_words(resp, vocab):
            drop(x, "glued_words"); continue
        if not want_latin and detached_particles(resp) >= 2:
            drop(x, "detached_particles"); continue
        n_ans = count_tokens(tok, resp)
        if n_ans > MAX_ANSWER_TOKENS:
            drop(x, "answer_too_long"); continue
        x["answer_tokens"] = n_ans
        x["prompt_tokens"] = count_tokens(tok, x["prompt"])
        kept.append(x)
    out = OUT_DIR / f"train_{tag}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for x in kept:
            f.write(json.dumps({
                "messages": [{"role": "user", "content": x["prompt"]},
                             {"role": "assistant", "content": x["response"]}],
                "meta": {k: x[k] for k in ("id", "task", "direction", "domain", "subtopic", "context",
                                           "template_id", "teacher", "prompt_writer")}},
                ensure_ascii=False) + "\n")
    write_jsonl(OUT_DIR / f"responses_dropped_{tag}.jsonl", dropped)
    # submission CSV: exactly prompt, teacher_response, task, source (utf-8 without BOM); task uses the eval's task names
    sub_csv = OUT_DIR / f"train_{tag}.csv"
    with sub_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["prompt", "teacher_response", "task", "source"])
        for x in kept:
            w.writerow([x["prompt"], x["response"], CSV_TASK_NAME[x["task"]],
                        f"synthetic|prompts:{x['prompt_writer']}|answers:{x['teacher']}"])
    with (OUT_DIR / f"train_{tag}_full.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "task", "direction", "domain", "subtopic", "template_id", "prompt", "teacher_response",
                    "prompt_tokens", "answer_tokens", "judge_scores", "teacher", "prompt_writer"])
        for x in kept:
            w.writerow([x["id"], x["task"], x["direction"] or "", x["domain"], x["subtopic"], x["template_id"],
                        x["prompt"], x["response"], x["prompt_tokens"], x["answer_tokens"],
                        json.dumps(x.get("scores")), x["teacher"], x["prompt_writer"]])
    (OUT_DIR / f"train_{tag}.meta.json").write_text(json.dumps({
        "teacher": rows[0]["teacher"], "responses": len(rows), "kept": len(kept), "dropped": dict(why),
        "kept_per_task": dict(Counter(x["task"] for x in kept)), "min_score": None if skip_judge else min_score,
        "prompts_meta": meta, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log_run("finalize", None, tag=tag, kept=len(kept))
    print(f"responses={len(rows)} kept={len(kept)} -> {out} (+ {sub_csv.name}, {sub_csv.stem}_full.csv, "
          f"responses_dropped_{tag}.jsonl)")
    print("dropped:", dict(why))
    print("kept per task:", dict(Counter(x["task"] for x in kept)))
    if kept:
        pt = [x["prompt_tokens"] for x in kept]; at = [x["answer_tokens"] for x in kept]
        print(f"prompt tokens median/p95/max = {pct(pt, 50)}/{pct(pt, 95)}/{max(pt)} | "
              f"answer tokens median/p95/max = {pct(at, 50)}/{pct(at, 95)}/{max(at)}")
        if why["answer_too_long"] > 0.15 * len(rows):
            print("NOTE: >15% of answers exceeded MAX_ANSWER_TOKENS -> lower ANSWER_MAX_WORDS and re-answer, "
                  "otherwise the dataset is biased toward short answers.")


def sample_responses(tag):
    rows = read_jsonl(resp_path(tag))
    path = OUT_DIR / f"sample_check_{tag}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "task", "prompt", "response", "judge_scores", "my_fluency_1to5", "my_accuracy_1to5", "notes"])
        for x in sample_rows(rows):
            w.writerow([x["id"], x["task"], x["prompt"], x["response"], json.dumps(x.get("scores")), "", "", ""])
    print(f"wrote {path} (review by hand; report how often your ratings agree with the judge)")

def answer_eval(llm, predictions=None, out=None):
    """Teacher answers to the 60 eval prompts -> teacher_output column. Written to a SEPARATE file; never used for training."""
    rows = require_eval_rows()
    tag = tag_of(llm.backend, llm.model)
    path = OUT_DIR / f"teacher_eval_{tag}.jsonl"
    done = {x["id"] for x in read_jsonl(path)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"teacher: {llm.name} | {len(todo)} eval prompts to answer ({len(done)} already done) -> {path}")
    log_run("answer-eval", llm, temperature=ANSWER_TEMPERATURE, n=len(todo))
    for idx, r in enumerate(todo, 1):
        print(f"[{idx}/{len(todo)}] {r['id']}")
        ans = llm(build_B(r["prompt"]), ANSWER_TEMPERATURE)
        if ans:
            append_jsonl(path, [{"id": r["id"], "task": r["task"], "teacher_output": ans.strip(), "teacher": llm.name}])
    by_id = {x["id"]: x["teacher_output"] for x in read_jsonl(path)}
    pred = Path(predictions) if predictions else Path("predictions_template.csv")
    if not pred.exists():
        print(f"(no predictions file at {pred}; teacher answers are in {path})"); return
    with pred.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        fields, recs = rd.fieldnames, list(rd)
    if "teacher_output" not in fields or "eval_id" not in fields:
        raise Abort(f"{pred} needs 'eval_id' and 'teacher_output' columns")
    for r in recs:
        if r["eval_id"] in by_id:
            r["teacher_output"] = by_id[r["eval_id"]]
    dest = Path(out) if out else OUT_DIR / "predictions_with_teacher.csv"
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(recs)
    print(f"teacher_output filled for {sum(1 for r in recs if r['teacher_output'])}/{len(recs)} rows -> {dest}")


def clip(s, n=300):
    s = str(s).replace("\n", " \u23ce ")
    return s if len(s) <= n else s[:n] + "..."


def make_examples(tag):
    """Markdown with README examples: generated / removed by reason / removed as eval duplicates / kept."""
    L = ["# Examples for the README", "", "## 1. Generated prompts (raw prompt-writer output), one per task", ""]
    raw = read_jsonl(PROMPTS_RAW)
    for task in TARGETS:
        x = next((r for r in raw if r["task"] == task), None)
        if x:
            L += [f"**{task}** (`{x['id']}`, template `{x['template_id']}`)", "", "```", clip(x["prompt"], 600), "```", ""]
    dropped = read_jsonl(PROMPTS_DROPPED)
    ev = {r["id"]: r["prompt"] for r in read_eval_rows()}
    L += ["## 2. Prompts removed, by reason", ""]
    for reason, n in Counter(d["reason"] for d in dropped).most_common():
        d = next(d for d in dropped if d["reason"] == reason)
        L += [f"**{reason}**: {n} removed. Example (`{d['id']}`):", "", "```", clip(d["prompt"]), "```", ""]
    dups = [d for d in dropped if d["reason"] in ("exact_duplicate_eval", "near_dup_eval")]
    L += ["## 3. Prompts removed as duplicates of evaluation prompts", "", f"Total removed: **{len(dups)}**", ""]
    for d in dups[:6]:
        sims = {k: d[k] for k in ("jaccard", "eval_in_prompt", "content_in_eval") if k in d}
        L += [f"- `{d['id']}` ({d['reason']}) matched eval `{d.get('eval_id')}` {sims}", "",
              "  ```", "  train: " + clip(d["prompt"], 200), "  eval:  " + clip(ev.get(d.get("eval_id"), ""), 200), "  ```", ""]
    rdrop = read_jsonl(OUT_DIR / f"responses_dropped_{tag}.jsonl")
    L += ["## 4. Teacher responses removed, by reason", ""]
    for reason, n in Counter(d["reason"] for d in rdrop).most_common():
        d = next(d for d in rdrop if d["reason"] == reason)
        L += [f"**{reason}**: {n} removed. Example (`{d['id']}`):", "", "```", "PROMPT:   " + clip(d["prompt"], 200),
              "RESPONSE: " + clip(d["response"], 250), "```", ""]
    L += ["## 5. Kept (final training data), one per task", ""]
    kept = read_jsonl(OUT_DIR / f"train_{tag}.jsonl")
    for task in TARGETS:
        x = next((r for r in kept if r["meta"]["task"] == task), None)
        if x:
            L += [f"**{task}** (`{x['meta']['id']}`)", "", "```", "PROMPT:   " + clip(x["messages"][0]["content"], 400),
                  "RESPONSE: " + clip(x["messages"][1]["content"], 400), "```", ""]
    out = OUT_DIR / f"readme_examples_{tag}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


# ======================= CLI =======================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def llm_args(p):
        p.add_argument("--backend", default="gemini", choices=["gemini", "openai"])
        p.add_argument("--model", default=None, help="model name (default depends on backend)")
        p.add_argument("--sleep", type=float, default=None, help="seconds between API calls")
        p.add_argument("--tag", default=None, help="explicit responses tag (overrides backend/model)")
        p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (vLLM, Ollama, OpenRouter, ...)")

    p = sub.add_parser("prompts", help="Stage 1: write prompts (teacher-independent)")
    llm_args(p); p.add_argument("--jobs-per-task", type=int, default=None, help="pilot: jobs per task")
    p.add_argument("--bands", type=int, nargs="+", default=None,
                   help="only run passage jobs of these bands, e.g. --bands 1 2 (skips knowledge/translation jobs)")
    sub.add_parser("filter-prompts", help="Stage 1: dedup + contamination filter -> prompts_clean.jsonl")
    sub.add_parser("evalstats", help="compare prompt lengths with the eval prompts")
    sub.add_parser("sample-prompts", help="50 prompts -> CSV for manual review")

    p = sub.add_parser("judge-prompts", help="Stage 1: LLM quality check of candidate prompts (resumable)")
    llm_args(p)

    p = sub.add_parser("answer", help="Stage 2: teacher answers for every clean prompt")
    llm_args(p); p.add_argument("--limit-per-task", type=int, default=None, help="pilot: prompts per task")

    p = sub.add_parser("judge", help="Stage 2: score responses (resumable)")
    llm_args(p)
    p.add_argument("--judge-backend", default=None, choices=["gemini", "openai"])
    p.add_argument("--judge-model", default=None)

    p = sub.add_parser("finalize", help="Stage 2: filter responses -> train_<tag>.jsonl / .csv")
    llm_args(p)
    p.add_argument("--min-score", type=float, default=MIN_SCORE)
    p.add_argument("--skip-judge", action="store_true", help="ignore judge scores (not recommended)")

    p = sub.add_parser("sample", help="Stage 2: 50 responses -> CSV for manual review")
    llm_args(p)

    p = sub.add_parser("answer-eval", help="teacher answers to the 60 eval prompts -> teacher_output column (never used for training)")
    llm_args(p)
    p.add_argument("--predictions", default=None, help="predictions template CSV (default: predictions_template.csv)")
    p.add_argument("--out", default=None, help="output CSV (default: data/predictions_with_teacher.csv)")

    p = sub.add_parser("examples", help="README examples (kept / removed by reason / eval duplicates) -> markdown")
    llm_args(p)

    a = ap.parse_args()
    try:
        bu = getattr(a, "base_url", None)
        if a.cmd == "prompts":
            generate_prompts(LLM(a.backend, a.model, a.sleep, bu), a.jobs_per_task, a.bands)
        elif a.cmd == "filter-prompts":
            filter_prompts()
        elif a.cmd == "evalstats":
            evalstats()
        elif a.cmd == "judge-prompts":
            judge_prompts(LLM(a.backend, a.model, a.sleep, bu))
        elif a.cmd == "sample-prompts":
            sample_prompts()
        elif a.cmd == "answer":
            answer(LLM(a.backend, a.model, a.sleep, bu), a.limit_per_task)
        elif a.cmd == "judge":
            tag = resolve_tag(a)
            jb = a.judge_backend or a.backend
            jm = a.judge_model or (a.model if jb == a.backend else None)
            judge(LLM(jb, jm, a.sleep, bu), tag)
        elif a.cmd == "finalize":
            finalize(resolve_tag(a), a.min_score, a.skip_judge)
        elif a.cmd == "sample":
            sample_responses(resolve_tag(a))
        elif a.cmd == "answer-eval":
            answer_eval(LLM(a.backend, a.model, a.sleep, bu), a.predictions, a.out)
        elif a.cmd == "examples":
            make_examples(resolve_tag(a))
    except Abort as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()