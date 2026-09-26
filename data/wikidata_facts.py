"""Builds data/facts_wikidata.tsv from Wikidata (CC0, public domain).

    python data/wikidata_facts.py            # uses data/wikidata_cache/ if present
    python data/wikidata_facts.py --refresh  # re-query

Queries go to QLever (qlever.dev), a fast SPARQL engine over the full
Wikidata graph; the public Wikidata endpoint times out on queries this size.
Items are ranked by sitelinks (how many Wikipedias have an article on them),
a good proxy for "well known".

Everything here is filtered for reliability rather than volume:
  - dates respect Wikidata's precision (a month-precise date never becomes a
    made-up day) and nothing before 1583 (calendar and certainty issues)
  - birthplaces must be towns or cities, not hospitals
  - quantities must be in the expected unit (metres, kilometres)
  - a phrasing that two different items share (two films called "Crash")
    goes to the better known one only
  - any phrasing already answered by another data/facts*.tsv file is
    dropped: hand-written facts win
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from common import MAX_A, STRICT_FILES, TIERS, load_facts, normalize  # noqa: E402
import gate_index  # noqa: E402

ENDPOINT = "https://qlever.dev/api/wikidata"
CTX = 96  # model context, train/model.py
UA = "local-esp32-ai/1.0 (offline Q&A dataset; https://github.com/l3afzy/local-esp32-ai)"
CACHE = os.path.join(HERE, "wikidata_cache")

PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
"""

REFRESH = False
TIER = "small"
# How many of the best-known items of each kind to query, per knowledge tier.
LIMITS = {
    "small": dict(people=3000, cities=1500, mountains=400, rivers=300, landmarks=600, books=1200,
                  films=1500, songs=700, albums=500, paintings=250, companies=700, universities=300,
                  events=500, compounds=400, taxa=2500, languages=100),
    "max4mb": dict(people=4800, cities=2500, mountains=700, rivers=500, landmarks=1000, books=2000,
                   films=2300, songs=1100, albums=800, paintings=400, companies=1100, universities=550,
                   events=850, compounds=700, taxa=3700, languages=170),
    "large": dict(people=7000, cities=3500, mountains=1000, rivers=700, landmarks=1500, books=2800,
                  films=3200, songs=1500, albums=1100, paintings=600, companies=1500, universities=800,
                  events=1200, compounds=1000, taxa=5000, languages=250),
}


def query(name, body):
    """Run a SPARQL query (cached in data/wikidata_cache/, keyed by the query)."""
    path = os.path.join(CACHE, f"{name}-{hashlib.sha1(body.encode()).hexdigest()[:10]}.json")
    legacy = os.path.join(CACHE, name + ".json")  # caches from before the key had a hash
    if not REFRESH and os.path.exists(path):
        return json.load(open(path))
    if not REFRESH and TIER == "small" and os.path.exists(legacy):
        # One-time migration: the old name-only cache holds the small tier's
        # queries. Re-save it under the hashed name, then it is never used again.
        rows = json.load(open(legacy))
        json.dump(rows, open(path, "w"))
        os.remove(legacy)
        return rows
    data = urllib.parse.urlencode({"query": PREFIXES + body}).encode()
    for attempt in range(8):
        try:
            req = urllib.request.Request(ENDPOINT, data=data, headers={
                "User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                res = json.load(r)
            break
        except Exception as e:  # network hiccup or rate limit (429): back off and retry
            wait = 30 * (attempt + 1) if "429" in str(e) else 5 * (attempt + 1)
            print(f"  {name}: {e}, retrying in {wait} s", file=sys.stderr)
            time.sleep(wait)
    else:
        raise RuntimeError(f"query {name} failed")
    rows = [{k: v["value"] for k, v in b.items()} for b in res["results"]["bindings"]]
    os.makedirs(CACHE, exist_ok=True)
    json.dump(rows, open(path, "w"))
    print(f"  {name}: {len(rows)} rows", file=sys.stderr)
    time.sleep(10)  # be gentle with the public endpoint
    return rows


def label(var):
    """OPTIONAL English and language-neutral ("mul") labels for ?var."""
    return (f'OPTIONAL {{ ?{var} rdfs:label ?{var}En FILTER(lang(?{var}En)="en") }} '
            f'OPTIONAL {{ ?{var} rdfs:label ?{var}Mul FILTER(lang(?{var}Mul)="mul") }} ')


def top(cls, n, path="wdt:P31"):
    """Subquery: the n best-known items of class `cls`."""
    return (f"{{ SELECT ?item ?sl WHERE {{ ?item {path} {cls} ; wikibase:sitelinks ?sl . }} "
            f"ORDER BY DESC(?sl) LIMIT {n} }}")


# ---------------------------------------------------------------- text helpers

TRANSLIT = str.maketrans({
    "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "ß": "ss", "æ": "ae",
    "Æ": "AE", "đ": "d", "Đ": "D", "þ": "th", "Þ": "Th", "œ": "oe",
    "Œ": "OE", "ı": "i", "ð": "d", "–": "-", "—": "-", "’": "'",
    "‘": "'"})


def ascii_text(s):
    s = unicodedata.normalize("NFKD", s.translate(TRANSLIT)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip()


def name_of(r, var):
    v = r.get(var + "En") or r.get(var + "Mul")
    if not v:
        return None
    a = ascii_text(v)
    # reject names that lost most of their letters in the ASCII fold
    if not a or len(a) < 0.8 * len(v) or re.search(r"[^A-Za-z0-9 .,'&:!?()/+-]", a):
        return None
    return a


def with_period(s):
    s = s.strip()
    return s if s.endswith((".", "!", "?")) else s + "."


def cap(s):
    return s[:1].upper() + s[1:] if s else s


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def fmt_date(value, precision):
    """Wikidata time + precision (9 year, 10 month, 11 day) -> text, or None."""
    m = re.match(r"^(-?)(\d+)-(\d\d)-(\d\d)", value or "")
    if not m or m.group(1):
        return None
    y, mo, d = int(m.group(2)), int(m.group(3)), int(m.group(4))
    p = int(precision or 0)
    if y < 1583 or y > 2100:
        return None
    if p >= 11:
        return f"{MONTHS[mo - 1]} {d}, {y}"
    if p == 10:
        return f"{MONTHS[mo - 1]} {y}"
    if p == 9:
        return str(y)
    return None


def year_of(value, precision=9):
    m = re.match(r"^(-?)(\d+)-", value or "")
    if not m or m.group(1) or int(precision or 0) < 9:
        return None
    y = int(m.group(2))
    return y if 1 <= y <= 2100 else None


def group(rows):
    """Rows of one query -> {item: [rows]} in sitelink order."""
    out = {}
    for r in rows:
        out.setdefault(r["item"], []).append(r)
    return sorted(out.values(), key=lambda v: -int(v[0].get("sl", 0)))


def one(vals):
    """The single distinct value, or None if there are none or several."""
    vals = {v for v in vals if v}
    return vals.pop() if len(vals) == 1 else None


def distinct(vals, limit):
    seen = []
    for v in vals:
        if v and v not in seen:
            seen.append(v)
    return seen if 0 < len(seen) <= limit else None


def join_and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------- fact collection

def signature(q):
    """What the knowledge gate sees: question type + content words. Two
    phrasings with the same signature are the same question to it."""
    return (gate_index.qtype(q), tuple(sorted(set(gate_index.content_words(q)))))


class Facts:
    def __init__(self):
        # content words -> question types already answered. A phrasing with no
        # question word ("google founded") matches questions of every type.
        self.taken = defaultdict(set)
        root = os.path.join(HERE, "..")
        # Every file of the tier except the Wikidata output itself (whatever its
        # name: facts_wikidata.tsv, facts_wikidata_max4mb.tsv), which is rebuilt.
        hand = [os.path.join(root, f) for f in TIERS[TIER] if os.path.basename(f) not in STRICT_FILES]
        for qs, a in [fact for f in hand for fact in load_facts(f)]:
            for q in qs:
                t, w = signature(normalize(q))
                self.taken[w].add(t)
        self.claimed = defaultdict(set)  # same, for earlier (better known) items
        self.facts = []
        self.counts = defaultdict(int)

    @staticmethod
    def clashes(sig, owners):
        t, w = sig
        types = owners.get(w)
        return bool(types) and (t in types or "" in types or t == "")

    def add(self, section, questions, answer, aliases=()):
        """`aliases` are extra phrasings the gate accepts ("einstein birthday" for
        "albert einstein birthday"); they are never the fact's key."""
        if not answer or len(answer) > MAX_A:
            return
        keep, sigs = [], []
        for q in questions:
            n = self.clean(q)
            sig = signature(n)
            if not n or not sig[1] or n in keep:
                continue
            if self.clashes(sig, self.taken) or self.clashes(sig, self.claimed):
                continue  # hand-written facts and better-known items win
            keep.append(n)
            sigs.append(sig)
        # The key is the shortest phrasing kept; the model must fit key + answer.
        if not keep or min(map(len, keep)) + len(answer) + 2 > CTX:
            return
        for t, w in sigs:
            self.claimed[w].add(t)
        self.facts.append((section, keep, answer, list(aliases)))
        self.counts[section] += 1

    @staticmethod
    def clean(q):
        return re.sub(r"\bthe the\b", "the", normalize(q))  # "where is the The Scream"

    def finish(self):
        """Resolve aliases once every fact has its phrasings: an alias may only
        take a question nothing else answers, so it can't change any key."""
        out = []
        for section, keep, answer, aliases in self.facts:
            extra = []
            for q in aliases:
                n = self.clean(q)
                sig = signature(n)
                if not n or not sig[1] or n in keep or n in extra:
                    continue
                if self.clashes(sig, self.taken) or self.clashes(sig, self.claimed):
                    continue
                self.claimed[sig[1]].add(sig[0])
                extra.append(n)
            out.append((section, keep + ["~" + n for n in extra], answer))
        return out


# ---------------------------------------------------------------- names

SUFFIX = re.compile(r",? (Jr\.?|Sr\.?|II|III|IV)$")
PARTICLES = {"van", "von", "da", "de", "di", "del", "della", "du", "la", "le", "der", "den", "bin", "ibn", "al"}
NOT_NAMES = {"born", "birth", "birthday", "die", "died", "death", "date", "place", "year", "president"}
NICKNAMES = {"John F. Kennedy": ["JFK"], "Franklin Delano Roosevelt": ["FDR"], "Lyndon B. Johnson": ["LBJ"],
             "Martin Luther King Jr.": ["MLK"], "Elvis Presley": ["Elvis"], "Oprah Winfrey": ["Oprah"],
             "Galileo Galilei": ["Galileo"]}


def name_key(name):
    """A name as the gate sees it: its content words."""
    return frozenset(gate_index.content_words(normalize(name)))


def name_variants(name):
    """Shorter ways to say a name: without "Jr.", first + last, and the surname.
    First names only from NICKNAMES: "venus" or "paris" alone is not a person."""
    base = SUFFIX.sub("", name)
    w = base.split()
    out = [base]
    if len(w) >= 3:
        out.append(f"{w[0]} {w[-1]}")
    if len(w) >= 2:
        if len(w) >= 3 and w[-2].lower() in PARTICLES:
            out.append(f"{w[-2]} {w[-1]}")  # "van gogh", "da vinci"
        if len(w[-1]) >= 4 and w[-1].isalpha() and w[-1].lower() not in NOT_NAMES:
            out.append(w[-1])
    return [v for v in dict.fromkeys(out + NICKNAMES.get(name, [])) if v != name]


def blind(name):
    """True if the gate can't see all of a name: a capitalized word it treats as
    filler ("Will Smith", "Charles I"), or a single word ("Ronaldo")."""
    words = re.findall(r"[A-Za-z']+", name)
    return len(name_key(name)) < 2 or any(
        x[0].isupper() and gate_index.SYNONYMS.get(x.lower(), x.lower()) in STOPWORDS for x in words)


def pick_aliases(items):
    """items: [(id, name, sitelinks)] -> ({id: [short names]}, {ids to skip}).

    A short name goes to the one item people mean by it: every other item it
    could name ("obama": Barack, Michelle) must have under half the sitelinks.
    An item is skipped when its own name is ambiguous to the gate: two items
    with that name, or a name the gate only partly sees ("Will Smith" is just
    "smith", since "will" is a filler word) that also shortens another,
    better-known name. Answering for the wrong Smith is worse than refusing."""
    owners, full = defaultdict(list), defaultdict(list)
    for i, name, sl in items:
        full[name_key(name)].append((sl, i))
        for k in {name_key(name)} | {name_key(v) for v in name_variants(name)}:
            if k:
                owners[k].append((sl, i))

    def dominant(pool, k, i):
        o = sorted(pool[k], key=lambda x: -x[0])
        return o[0][1] == i and (len(o) == 1 or o[0][0] >= 2 * o[1][0])

    aliases, skip = {}, set()
    for i, name, sl in items:
        k = name_key(name)
        if not k or not dominant(full, k, i) or (blind(name) and not dominant(owners, k, i)):
            skip.add(i)
        aliases[i] = [v for v in name_variants(name)
                      if name_key(v) and name_key(v) != k and dominant(owners, name_key(v), i)]
    return aliases, skip


# ---------------------------------------------------------------- people

def people_rows(n=None):
    n = n or LIMITS[TIER]["people"]
    return query("people", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?desc ?birth ?bprec ?death ?dprec ?bpEn ?bpMul ?bpcEn ?bpcMul WHERE {{
  {top("wd:Q5", n)}
  {label("item")}
  OPTIONAL {{ ?item schema:description ?desc FILTER(lang(?desc)="en") }}
  OPTIONAL {{ ?item p:P569/psv:P569 [ wikibase:timeValue ?birth ; wikibase:timePrecision ?bprec ] }}
  OPTIONAL {{ ?item p:P570/psv:P570 [ wikibase:timeValue ?death ; wikibase:timePrecision ?dprec ] }}
  OPTIONAL {{ ?item wdt:P19 ?bp . ?bp wdt:P31/wdt:P279* wd:Q486972 .
             {label("bp")}
             OPTIONAL {{ ?bp wdt:P17 ?bpc . {label("bpc")} }} }}
}}""")


def people(F, n=None):
    rows = people_rows(n)
    groups = [(rs, name_of(rs[0], "item")) for rs in group(rows)]
    groups = [(rs, name) for rs, name in groups if name and (len(name.split()) >= 2 or len(name) >= 4)]
    aliases, skip = pick_aliases([(rs[0]["item"], name, int(rs[0].get("sl", 0))) for rs, name in groups])
    for rs, name in groups:
        item = rs[0]["item"]
        if item in skip:
            continue
        short = aliases[item]

        def add(templates, answer):
            F.add("people", [t.format(n=name) for t in templates], answer,
                  [t.format(n=a) for a in short for t in templates])

        desc = one(r.get("desc") for r in rs)
        if desc:
            d = ascii_text(desc)
            if len(d) + 1 > MAX_A:
                d = re.sub(r"\s*\([^)]*\)\s*$", "", d)  # drop "(1879-1955)"
            if d and len(d) + 1 <= MAX_A:
                add(["who was {n}", "who is {n}", "who's {n}"], with_period(cap(d)))
        born = one(fmt_date(r.get("birth"), r.get("bprec")) for r in rs)
        if born:
            add(["when was {n} born", "what year was {n} born", "{n} birthday", "{n} date of birth"], born + ".")
        died = one(fmt_date(r.get("death"), r.get("dprec")) for r in rs)
        if died:
            add(["when did {n} die", "what year did {n} die", "{n} date of death"], died + ".")
        place = one(name_of(r, "bp") for r in rs)
        country = one(name_of(r, "bpc") for r in rs)
        if place:
            where = place if not country or country == place else f"{place}, {country}"
            add(["where was {n} born", "{n} birthplace", "{n} place of birth"], where + ".")


def approx(n):
    """2143512 -> 'About 2.1 million'."""
    n = float(n)
    if n >= 1e9:
        return f"About {n / 1e9:.1f}".rstrip("0").rstrip(".") + " billion"
    if n >= 1e6:
        return f"About {n / 1e6:.1f}".rstrip("0").rstrip(".") + " million"
    if n >= 1e4:
        return f"About {round(n, -3):,.0f}"
    return f"About {n:,.0f}"


def latest(rs, value, when):
    """The value with the most recent point in time (ties -> None)."""
    best = {}
    for r in rs:
        if r.get(value) and r.get(when):
            best.setdefault(r[when][:10], set()).add(r[value])
    if not best:
        return None, None
    t = max(best)
    vals = best[t]
    return (vals.pop(), t[:4]) if len(vals) == 1 else (None, None)


def valid_title(t):
    """Distinctive titles only. "Hold Me" or "Crash" could be any of a dozen
    songs or films, and answering for the wrong one is worse than refusing:
    at least two real words, or one real word of 6+ letters."""
    words = [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOPWORDS]
    return len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 6)


# the gate's stop words (train/gate_index.py) -- a title made only of these has no key
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from gate_index import STOP as STOPWORDS  # noqa: E402


# ---------------------------------------------------------------- places

def countries(F):
    from more_facts import COUNTRY_CODES, COUNTRY_EXTRA
    rows = query("countries", f"""
SELECT ?item ?iso ?pop ?popt ?area ?contEn ?contMul ?langEn ?langMul ?hpEn ?hpMul ?driveEn ?driveMul ?anthemEn ?anthemMul WHERE {{
  ?item wdt:P297 ?iso .
  ?item wdt:P31 wd:Q3624078 .
  OPTIONAL {{ ?item p:P1082 [ ps:P1082 ?pop ; pq:P585 ?popt ] }}
  OPTIONAL {{ ?item p:P2046/psv:P2046 [ wikibase:quantityAmount ?area ; wikibase:quantityUnit wd:Q712226 ] }}
  OPTIONAL {{ ?item wdt:P30 ?cont . {label("cont")} }}
  OPTIONAL {{ ?item wdt:P37 ?lang . {label("lang")} }}
  OPTIONAL {{ ?item wdt:P610 ?hp . {label("hp")} }}
  OPTIONAL {{ ?item wdt:P1622 ?drive . {label("drive")} }}
  OPTIONAL {{ ?item wdt:P85 ?anthem . {label("anthem")} }}
}}""")
    by_iso = defaultdict(list)
    for r in rows:
        by_iso[r["iso"]].append(r)
    for country, code in COUNTRY_CODES.items():
        if country in ("scotland", "wales", "northern ireland", "england") or code not in by_iso:
            continue
        rs = by_iso[code]
        names = [country] + COUNTRY_EXTRA.get(country, [])
        if country.startswith("the "):
            names.append(country[4:])
        each = lambda *ts: [t.format(n=n) for n in names for t in ts]
        pop, year = latest(rs, "pop", "popt")
        if pop:
            F.add("countries", each("what is the population of {n}", "how many people live in {n}",
                                    "{n} population"), f"{approx(pop)} ({year}).")
        area = one(r.get("area") for r in rs)
        if area:
            F.add("countries", each("how big is {n}", "what is the area of {n}", "{n} area"),
                  f"About {float(area):,.0f} square km.")
        conts = distinct([name_of(r, "cont") for r in rs], 2)
        if conts:
            F.add("countries", each("what continent is {n} in", "{n} continent"), join_and(conts) + ".")
        langs = distinct([name_of(r, "lang") for r in rs], 4)
        if langs and len(join_and(langs)) < MAX_A:
            F.add("countries", each("what language is spoken in {n}", "what is the official language of {n}",
                                    "what language do they speak in {n}", "{n} language"), join_and(langs) + ".")
        hp = one(name_of(r, "hp") for r in rs)
        if hp:
            F.add("countries", each("what is the highest point in {n}", "highest mountain in {n}"), hp + ".")
        drive = one(name_of(r, "drive") for r in rs)
        if drive in ("left", "right"):
            F.add("countries", each("what side of the road do they drive on in {n}",
                                    "which side do they drive on in {n}"), cap(drive) + ".")
        anthem = one(name_of(r, "anthem") for r in rs)
        if anthem:
            F.add("countries", each("what is the national anthem of {n}", "{n} national anthem"),
                  with_period(anthem))


def cities(F, n):
    rows = query("cities", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?cEn ?cMul ?pop ?popt WHERE {{
  {top("wd:Q515", n, "wdt:P31/wdt:P279*")}
  {label("item")}
  OPTIONAL {{ ?item wdt:P17 ?c . {label("c")} }}
  OPTIONAL {{ ?item p:P1082 [ ps:P1082 ?pop ; pq:P585 ?popt ] }}
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        if not name:
            continue
        country = one(name_of(r, "c") for r in rs)
        if country and country != name:
            F.add("cities", [f"what country is {name} in", f"where is {name}", f"{name} country"], country + ".")
        pop, year = latest(rs, "pop", "popt")
        if pop and float(pop) >= 1000:
            F.add("cities", [f"what is the population of {name}", f"how many people live in {name}",
                             f"{name} population"], f"{approx(pop)} ({year}).")


def mountains(F, n):
    rows = query("mountains", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?elev ?cEn ?cMul WHERE {{
  {top("wd:Q8502", n, "wdt:P31/wdt:P279*")}
  {label("item")}
  OPTIONAL {{ ?item p:P2044/psv:P2044 [ wikibase:quantityAmount ?elev ; wikibase:quantityUnit wd:Q11573 ] }}
  OPTIONAL {{ ?item wdt:P17 ?c . {label("c")} }}
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        if not name:
            continue
        short = [re.sub(r"^Mount ", "", name)] if re.match(r"^Mount \w{4,}$", name) else []
        elev = one(r.get("elev") for r in rs)
        if elev and float(elev) > 0:
            m = float(elev)
            qs = ["how tall is {n}", "how high is {n}", "{n} height", "what is the elevation of {n}"]
            F.add("mountains", [q.format(n=name) for q in qs], f"{m:,.0f} m ({m / 0.3048:,.0f} ft).",
                  [q.format(n=a) for a in short for q in qs])
        cs = distinct([name_of(r, "c") for r in rs], 2)
        if cs and cs[0] != name:
            qs = ["where is {n}", "what country is {n} in"]
            F.add("mountains", [q.format(n=name) for q in qs], join_and(cs) + ".",
                  [q.format(n=a) for a in short for q in qs])


def rivers(F, n):
    rows = query("rivers", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?len WHERE {{
  {top("wd:Q4022", n)}
  {label("item")}
  OPTIONAL {{ ?item p:P2043/psv:P2043 [ wikibase:quantityAmount ?len ; wikibase:quantityUnit wd:Q828224 ] }}
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        ln = one(r.get("len") for r in rs)
        if not name or not ln or float(ln) <= 0:
            continue
        base = re.sub(r" [Rr]iver$", "", name)
        km = float(ln)
        qs = [f"how long is the {base} river", f"how long is the {base}", f"length of the {base} river",
              f"{base} river length"]
        F.add("rivers", qs, f"About {km:,.0f} km ({km / 1.609344:,.0f} miles).")


def landmarks(F, n):
    rows = query("landmarks", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?cityEn ?cityMul ?cEn ?cMul ?built ?bprec WHERE {{
  {top("wd:Q811979", n, "wdt:P31/wdt:P279*")}
  {label("item")}
  OPTIONAL {{ ?item wdt:P131 ?city . ?city wdt:P31/wdt:P279* wd:Q486972 . {label("city")} }}
  OPTIONAL {{ ?item wdt:P17 ?c . {label("c")} }}
  OPTIONAL {{ ?item p:P571/psv:P571 [ wikibase:timeValue ?built ; wikibase:timePrecision ?bprec ] }}
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        if not name or not valid_title(name):
            continue
        city = one(name_of(r, "city") for r in rs)
        country = one(name_of(r, "c") for r in rs)
        where = ", ".join(x for x in dict.fromkeys((city, country)) if x and x != name)
        if where:
            F.add("landmarks", [f"where is {name}", f"where is the {name}", f"what city is {name} in"],
                  where + ".")
        y = one(year_of(r.get("built"), r.get("bprec")) for r in rs)
        if y and y >= 1000:
            F.add("landmarks", [f"when was {name} built", f"when was the {name} built", f"{name} year built"],
                  f"{y}.")


# ---------------------------------------------------------------- works

def creative(F, section, cls, n, prop, verbs, when_qs, path="wdt:P31", max_who=2):
    rows = query(section, f"""
SELECT ?item ?sl ?itemEn ?itemMul ?whoEn ?whoMul ?date ?dprec WHERE {{
  {top(cls, n, path)}
  {label("item")}
  OPTIONAL {{ ?item wdt:{prop} ?who . {label("who")} }}
  OPTIONAL {{ ?item p:P577/psv:P577 [ wikibase:timeValue ?date ; wikibase:timePrecision ?dprec ] }}
}}""")
    for rs in group(rows):
        title = name_of(rs[0], "item")
        if not title or not valid_title(title) or len(title) > 40:
            continue
        who = distinct([name_of(r, "who") for r in rs], max_who)
        if who:
            F.add(section, [v.format(t=title) for v in verbs], join_and(who) + ".")
        years = [year_of(r.get("date"), r.get("dprec")) for r in rs]
        years = [y for y in years if y]
        if years:
            F.add(section, [v.format(t=title) for v in when_qs], f"{min(years)}.")


def paintings(F, n):
    rows = query("paintings", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?byEn ?byMul ?atEn ?atMul WHERE {{
  {top("wd:Q3305213", n)}
  {label("item")}
  OPTIONAL {{ ?item wdt:P170 ?by . {label("by")} }}
  OPTIONAL {{ ?item wdt:P276 ?at . {label("at")} }}
}}""")
    for rs in group(rows):
        title = name_of(rs[0], "item")
        if not title or not valid_title(title):
            continue
        by = one(name_of(r, "by") for r in rs)
        if by:
            F.add("paintings", [f"who painted {title}", f"who painted the {title}", f"{title} painter"], by + ".")
        at = one(name_of(r, "at") for r in rs)
        if at:
            F.add("paintings", [f"where is {title}", f"where is the {title}", f"where is {title} displayed"],
                  at + ".")


# ---------------------------------------------------------------- organizations

def companies(F, n):
    rows = query("companies", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?fEn ?fMul ?inc ?iprec ?hqEn ?hqMul WHERE {{
  {top("wd:Q4830453", n, "wdt:P31/wdt:P279*")}
  {label("item")}
  OPTIONAL {{ ?item wdt:P112 ?f . ?f wdt:P31 wd:Q5 . {label("f")} }}
  OPTIONAL {{ ?item p:P571/psv:P571 [ wikibase:timeValue ?inc ; wikibase:timePrecision ?iprec ] }}
  OPTIONAL {{ ?item wdt:P159 ?hq . ?hq wdt:P31/wdt:P279* wd:Q515 . {label("hq")} }}
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        if not name or not valid_title(name):
            continue
        short = campus_names(name)  # Wikidata files some universities as organizations

        def add(templates, answer):
            F.add("companies", [t.format(n=name) for t in templates], answer,
                  [t.format(n=a) for a in short for t in templates])

        fs = distinct([name_of(r, "f") for r in rs], 3)
        if fs and len(join_and(fs)) < MAX_A:
            add(["who founded {n}", "who started {n}", "{n} founder"], join_and(fs) + ".")
        y = one(year_of(r.get("inc"), r.get("iprec")) for r in rs)
        if y:
            add(["when was {n} founded", "what year was {n} founded", "{n} founded"], f"{y}.")
        hq = one(name_of(r, "hq") for r in rs)
        if hq:
            add(["where is {n} headquartered", "where is {n} based", "{n} headquarters"], hq + ".")


def campus_names(name):
    """"Harvard University" / "University of Oxford" -> ["Harvard"] / ["Oxford"]."""
    if name == "Massachusetts Institute of Technology":
        return ["MIT"]
    base = re.sub(r"^University of |^The University of | University$", "", name)
    return [base] if base != name and re.fullmatch(r"[A-Z][a-z]{3,}", base) else []


def universities(F, n):
    rows = query("universities", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?inc ?iprec ?cityEn ?cityMul ?cEn ?cMul WHERE {{
  {top("wd:Q3918", n)}
  {label("item")}
  OPTIONAL {{ ?item p:P571/psv:P571 [ wikibase:timeValue ?inc ; wikibase:timePrecision ?iprec ] }}
  OPTIONAL {{ ?item wdt:P131 ?city . ?city wdt:P31/wdt:P279* wd:Q486972 . {label("city")} }}
  OPTIONAL {{ ?item wdt:P17 ?c . {label("c")} }}
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        if not name:
            continue
        short = campus_names(name)
        y = one(year_of(r.get("inc"), r.get("iprec")) for r in rs)
        if y:
            qs = ["when was {n} founded", "how old is {n}", "{n} founded"]
            F.add("universities", [q.format(n=name) for q in qs], f"{y}.", [q.format(n=a) for a in short for q in qs])
        city = one(name_of(r, "city") for r in rs)
        country = one(name_of(r, "c") for r in rs)
        where = ", ".join(x for x in dict.fromkeys((city, country)) if x and x != name)
        if where:
            F.add("universities", [f"where is {name}", f"where is the {name}"], where + ".",
                  [f"where is {a} university" for a in short])


# ---------------------------------------------------------------- history, science, nature

def events(F, n):
    rows = query("events", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?when ?wprec ?start ?sprec ?end ?eprec WHERE {{
  {{ SELECT ?item ?sl WHERE {{ {{ ?item wdt:P31 wd:Q178561 }} UNION {{ ?item wdt:P31 wd:Q198 }}
     ?item wikibase:sitelinks ?sl . }} ORDER BY DESC(?sl) LIMIT {n} }}
  {label("item")}
  OPTIONAL {{ ?item p:P585/psv:P585 [ wikibase:timeValue ?when ; wikibase:timePrecision ?wprec ] }}
  OPTIONAL {{ ?item p:P580/psv:P580 [ wikibase:timeValue ?start ; wikibase:timePrecision ?sprec ] }}
  OPTIONAL {{ ?item p:P582/psv:P582 [ wikibase:timeValue ?end ; wikibase:timePrecision ?eprec ] }}
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        if not name or not valid_title(name):
            continue
        w = one(year_of(r.get("when"), r.get("wprec")) for r in rs)
        s = one(year_of(r.get("start"), r.get("sprec")) for r in rs)
        e = one(year_of(r.get("end"), r.get("eprec")) for r in rs)
        if w:
            ans = f"{w}."
        elif s and e:
            ans = f"{s}." if s == e else f"{s} to {e}."
        else:
            continue
        F.add("history", [f"when was the {name}", f"when was {name}", f"when did the {name} happen",
                          f"{name} year"], ans)


SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def compounds(F, n):
    rows = query("compounds", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?formula WHERE {{
  {{ SELECT ?item ?sl WHERE {{ {{ ?item wdt:P31 wd:Q11173 }} UNION {{ ?item wdt:P31 wd:Q113145171 }}
     ?item wdt:P274 [] ; wikibase:sitelinks ?sl . }} ORDER BY DESC(?sl) LIMIT {n} }}
  {label("item")}
  ?item wdt:P274 ?formula .
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        f = one(r.get("formula") for r in rs)
        if not name or not f:
            continue
        f = f.translate(SUB)
        # Organic formulas only: Wikidata uses Hill notation, which is the
        # familiar form for organics but writes NaCN as "CNNa".
        if not re.fullmatch(r"C\d*H[A-Za-z0-9()]*", f):
            continue
        F.add("chemistry", [f"what is the chemical formula for {name}", f"what is the formula for {name}",
                            f"{name} formula"], f + ".")


def taxa(F, n):
    rows = query("taxa", f"""
SELECT ?item ?sl ?itemEn ?sci WHERE {{
  {top("wd:Q16521", n)}
  ?item rdfs:label ?itemEn FILTER(lang(?itemEn)="en")
  ?item wdt:P225 ?sci .
}}""")
    for rs in group(rows):
        common = rs[0].get("itemEn")
        sci = one(r.get("sci") for r in rs)
        if not common or not sci:
            continue
        common, sci = ascii_text(common), ascii_text(sci)
        if common.lower() == sci.lower() or not re.fullmatch(r"[A-Z][a-z]+ [a-z]+( [a-z]+)?", sci) \
                or not re.fullmatch(r"[A-Za-z .'-]+", common):
            continue  # a species (not a hybrid or genus) with a real common name
        F.add("nature", [f"what is the scientific name of {common}", f"what is the scientific name for a {common}",
                         f"{common} scientific name", f"latin name for {common}"], sci + ".")


def languages(F, n):
    rows = query("proglangs", f"""
SELECT ?item ?sl ?itemEn ?itemMul ?byEn ?byMul ?inc ?iprec WHERE {{
  {top("wd:Q9143", n)}
  {label("item")}
  OPTIONAL {{ ?item wdt:P287 ?by . {label("by")} }}
  OPTIONAL {{ ?item p:P571/psv:P571 [ wikibase:timeValue ?inc ; wikibase:timePrecision ?iprec ] }}
}}""")
    for rs in group(rows):
        name = name_of(rs[0], "item")
        if not name:
            continue
        name = re.sub(r" \(programming language\)$", "", name)
        by = distinct([name_of(r, "by") for r in rs], 2)
        if by:
            F.add("computing", [f"who created {name}", f"who designed {name}", f"who invented {name}"],
                  join_and(by) + ".")
        y = one(year_of(r.get("inc"), r.get("iprec")) for r in rs)
        if y:
            F.add("computing", [f"when was {name} created", f"when was {name} released", f"how old is {name}"],
                  f"{y}.")


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def presidents(F):
    rows = query("presidents", """
SELECT ?item ?itemEn ?itemMul ?ord ?start ?end WHERE {
  ?item p:P39 ?st . ?st ps:P39 wd:Q11696 ; pq:P1545 ?ord .
  OPTIONAL { ?st pq:P580 ?start } OPTIONAL { ?st pq:P582 ?end }
""" + label("item") + "}")
    terms = defaultdict(dict)  # person -> ordinal -> (start, end)
    names = {}
    for r in rows:
        name = name_of(r, "item")
        if not r.get("ord", "").isdigit() or not name:
            continue
        names[r["item"]] = name
        terms[r["item"]][int(r["ord"])] = (year_of(r.get("start")), year_of(r.get("end")))
    by_ord = defaultdict(set)
    for item, ts in terms.items():
        for k in ts:
            by_ord[k].add(item)
    for k in sorted(by_ord):
        if len(by_ord[k]) == 1:
            name = names[next(iter(by_ord[k]))]
            F.add("presidents", [f"who was the {ordinal(k)} president", f"who was the {ordinal(k)} us president",
                                 f"who was the {ordinal(k)} president of the united states"], name + ".")
    # Two-term presidents (Cleveland, Trump) get every term, not just the first.
    # "lincoln" is enough among presidents; "roosevelt" or "bush" is not
    fame = {r["item"]: int(r.get("sl", 0)) for r in people_rows()}
    aliases, _ = pick_aliases([(i, n, fame.get(i, 1)) for i, n in names.items()])
    for item, ts in sorted(terms.items(), key=lambda x: min(x[1])):
        name, ks = names[item], sorted(ts)
        qs = ["what number president was {n}", "which president was {n}"]
        F.add("presidents", [q.format(n=name) for q in qs], "The " + join_and([ordinal(k) for k in ks]) + ".",
              [q.format(n=a) for a in aliases[item] for q in qs])
        spans = [f"{ts[k][0]} to {ts[k][1]}" if ts[k][1] else f"since {ts[k][0]}" for k in ks if ts[k][0]]
        if len(spans) == len(ks):
            qs = ["when was {n} president", "what years was {n} president"]
            F.add("presidents", [q.format(n=name) for q in qs], cap(join_and(spans)) + ".",
                  [q.format(n=a) for a in aliases[item] for q in qs])


def main():
    global REFRESH
    ap = argparse.ArgumentParser()
    global TIER
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--tier", default="small", choices=["small", "max4mb", "large"])
    ap.add_argument("--out", default=None, help="default: the tier's facts_wikidata.tsv")
    args = ap.parse_args()
    REFRESH, TIER = args.refresh, args.tier
    n = LIMITS[TIER]
    out = args.out or os.path.join(HERE, "..", TIERS[TIER][-1])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sys.path.insert(0, HERE)
    F = Facts()
    presidents(F)
    countries(F)
    people(F, n["people"])
    cities(F, n["cities"])
    mountains(F, n["mountains"])
    rivers(F, n["rivers"])
    landmarks(F, n["landmarks"])
    creative(F, "books", "wd:Q7725634", n["books"], "P50", ["who wrote {t}", "who is the author of {t}", "{t} author"],
             ["when was {t} published", "when was {t} written", "{t} publication year"])
    creative(F, "films", "wd:Q11424", n["films"], "P57", ["who directed {t}", "who is the director of {t}", "{t} director"],
             ["when did {t} come out", "what year did {t} come out", "{t} release year"])
    creative(F, "songs", "wd:Q7366", n["songs"], "P175", ["who sang {t}", "who sings {t}", "{t} singer"],
             ["when did {t} come out", "what year did {t} come out", "{t} release year"], max_who=1)
    creative(F, "albums", "wd:Q482994", n["albums"], "P175", ["who made the album {t}", "whose album is {t}", "{t} album artist"],
             ["when did the album {t} come out", "{t} album release year"], max_who=1)
    paintings(F, n["paintings"])
    companies(F, n["companies"])
    universities(F, n["universities"])
    events(F, n["events"])
    compounds(F, n["compounds"])
    taxa(F, n["taxa"])
    languages(F, n["languages"])
    with open(out, "w") as f:
        f.write("# Generated by data/wikidata_facts.py from Wikidata (CC0). Do not edit.\n")
        section = None
        for sec, qs, a in F.finish():
            if sec != section:
                f.write(f"# {sec}\n")
                section = sec
            f.write("|".join(qs) + "\t" + a + "\n")
    print(dict(F.counts), sum(F.counts.values()), "facts", file=sys.stderr)


if __name__ == "__main__":
    main()
