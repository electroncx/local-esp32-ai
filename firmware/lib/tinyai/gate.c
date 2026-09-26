// Knowledge gate. A 600k-parameter model has no idea what it does not know:
// ask it for the capital of Wakanda and it will confidently say something.
// So before generating, check that the question's content words match a
// question the model was trained on. If not, answer "I don't know." —
// a direct answer beats a wrong one.
//
// The phrasings are indexed at export time (train/gate_index.py): each input
// word is fuzzy-matched once against a dictionary of known words, then every
// phrasing is scored with integer compares. content_words() and qtype() here
// define the rules; gate_index.py mirrors them and train/evaluate.py checks
// that the two agree.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyai.h"

#define MAX_WORDS 16
#define MAX_WLEN 24

static const char *const STOP[] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "of", "in", "on",
    "at", "to", "for", "from", "by", "with", "and", "or", "what", "whats", "who", "whos", "how",
    "when", "where", "which", "why", "me", "please", "you", "u", "your", "can", "could",
    "would", "will", "i", "im", "want", "wondering", "question", "quick", "hey", "hi",
    "yo", "ok", "so", "exactly", "roughly", "approximately", "about", "again", "quickly",
    "much", "many", "there", "it", "its", "that", "this", "we", "my", "have", "has", "called",
    "just", "like", "um", "uh", "hmm", "now", "actually", "really", "some", "any", "all",
    "should", "need", "needs", "must", "supposed", "ought", "make", "makes", "whens", "wheres",
    "hows", "whys", "gonna", "wanna", "take", "takes", "once", "after", "if", "am", "using",
};

// Words that add context but never change which fact is meant
// ("capital *city* of france", "tallest mountain on *earth*"). The input may
// contain them without the known question having them.
static const char *const SOFT[] = {
    "city", "country", "planet", "earth", "world", "our", "one", "book", "language", "man",
    "woman", "person", "people", "human", "solar", "system", "away", "number", "total",
    "average", "normal", "typical", "usually", "generally", "name", "thing", "kind", "type",
    "exact", "approx", "approximate", "grown", "adult", "full", "whole", "known",
    "today", "tomorrow", "tonight", "currently", "right", "app", "application", "program",
    "treatment", "treat", "min", "minimum", "recommended", "best", "way", "ways", "tip",
    "tips", "safely", "survival", "wild", "wilderness", "woods", "outdoors", "stranded",
    "okay", "idea", "safe", "true", "being", "only", "but", "close", "near", "nearby",
    "coming", "approaching", "got", "their", "his", "her", "they", "them", "he", "she", "him",
    "tell", "know", "get", "getting", "survive", "surviving",
};

// Same meaning, one spelling.
static const char *const SYNONYMS[][2] = {
    {"begin", "start"}, {"began", "start"}, {"begins", "start"}, {"started", "start"},
    {"starts", "start"}, {"ended", "end"}, {"ends", "end"}, {"finish", "end"},
    {"finished", "end"}, {"biggest", "largest"}, {"quickest", "fastest"},
    {"temp", "temperature"}, {"temps", "temperature"}, {"refrigerator", "fridge"},
    {"cooking", "cook"}, {"cooked", "cook"}, {"boiling", "boil"}, {"boiled", "boil"},
    {"baking", "bake"}, {"baked", "bake"}, {"defrost", "thaw"}, {"reboot", "restart"},
    {"hrs", "hours"}, {"hr", "hours"}, {"mins", "minutes"}, {"stay", "last"}, {"keep", "last"},
    {"replace", "change"}, {"isnt", "not"}, {"arent", "not"}, {"doesnt", "not"}, {"dont", "not"},
    {"cant", "not"}, {"wont", "not"}, {"opened", "open"}, {"opening", "open"},
    {"spaghetti", "pasta"}, {"noodles", "pasta"}, {"detector", "alarm"}, {"detectors", "alarms"},
    {"frostbitten", "frostbite"}, {"forest", "woods"}, {"forests", "woods"}, {"elevation", "altitude"},
    {"no", "without"}, {"dehydrated", "dehydration"}, {"bit", "bite"}, {"bitten", "bite"},
    {"bites", "bite"}, {"symptoms", "signs"}, {"symptom", "signs"}, {"broke", "broken"},
    {"breaks", "broken"}, {"break", "broken"},
};

static int in_list(const char *w, const char *const *list, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!strcmp(w, list[i])) return 1;
    return 0;
}

// Words that change an answer even as an aside: who it is for, or a
// negation. A lenient phrasing never lets one of these go unmatched.
static const char *const QUALIFIERS[] = {
    "baby", "babies", "infant", "infants", "child", "children", "kid", "kids", "toddler",
    "dog", "dogs", "cat", "cats", "pet", "pets", "someone", "somebody", "friend", "friends",
    "buddy", "not", "without", "never",
};

static int is_stop(const char *w) { return in_list(w, STOP, sizeof STOP / sizeof *STOP); }
static int is_soft(const char *w) { return in_list(w, SOFT, sizeof SOFT / sizeof *SOFT); }
static int is_qualifier(const char *w) {
    for (const char *c = w; *c; c++)
        if (*c >= '0' && *c <= '9') return 1;  // "5 gallons" is not "a gallon"
    return in_list(w, QUALIFIERS, sizeof QUALIFIERS / sizeof *QUALIFIERS);
}

// Splits normalized text into content words (letters/digits only, lowercased,
// apostrophes dropped, stop words removed).
static int content_words(const char *s, char w[MAX_WORDS][MAX_WLEN]) {
    int n = 0;
    while (*s && n < MAX_WORDS) {
        while (*s && !((*s >= 'a' && *s <= 'z') || (*s >= '0' && *s <= '9'))) s++;
        int len = 0;
        while (*s && ((*s >= 'a' && *s <= 'z') || (*s >= '0' && *s <= '9') || *s == '\'')) {
            if (*s != '\'' && len < MAX_WLEN - 1) w[n][len++] = *s;
            s++;
        }
        w[n][len] = 0;
        for (size_t i = 0; i < sizeof SYNONYMS / sizeof *SYNONYMS; i++)
            if (!strcmp(w[n], SYNONYMS[i][0])) strcpy(w[n], SYNONYMS[i][1]);
        if (len && !is_stop(w[n])) n++;
    }
    return n;
}

static int edit_distance(const char *a, const char *b) {
    int la = (int)strlen(a), lb = (int)strlen(b);
    int row[MAX_WLEN + 1];
    if (lb > MAX_WLEN) return MAX_WLEN;  // longer than row[] holds: no match
    for (int j = 0; j <= lb; j++) row[j] = j;
    for (int i = 1; i <= la; i++) {
        int diag = row[0];
        row[0] = i;
        for (int j = 1; j <= lb; j++) {
            int up = row[j];
            int best = diag + (a[i - 1] != b[j - 1]);
            if (row[j] + 1 < best) best = row[j] + 1;
            if (row[j - 1] + 1 < best) best = row[j - 1] + 1;
            row[j] = best;
            diag = up;
        }
    }
    return row[lb];
}

static int has_digit(const char *s) {
    for (; *s; s++)
        if (*s >= '0' && *s <= '9') return 1;
    return 0;
}

// Same word, allowing for typos and plurals. Numbers must match exactly.
static int word_match(const char *a, const char *b) {
    if (!strcmp(a, b)) return 1;
    if (has_digit(a) || has_digit(b)) return 0;
    int la = (int)strlen(a), lb = (int)strlen(b);
    int shorter = la < lb ? la : lb;
    if (shorter <= 4) {
        // cat/cats: only a trailing 's' may differ. Short words are too close
        // to each other for typo tolerance (bake/cake/take).
        return (la == lb + 1 && a[la - 1] == 's' && !strncmp(a, b, (size_t)lb)) ||
               (lb == la + 1 && b[lb - 1] == 's' && !strncmp(a, b, (size_t)la));
    }
    // emergencies/emergency, batteries/battery
    if (la >= 5 && lb >= 5 && la != lb) {
        const char *ies = la > lb ? a : b, *y = la > lb ? b : a;
        size_t li = strlen(ies), ly = strlen(y);
        if (li == ly + 2 && !strcmp(ies + li - 3, "ies") && y[ly - 1] == 'y' &&
            !strncmp(ies, y, ly - 1))
            return 1;
    }
    int d = edit_distance(a, b);
    return d <= (shorter >= 8 ? 2 : 1);
}

#define MAX_HITS 24

typedef struct {
    uint16_t id;
    uint8_t q;  // 2 = exact, 1 = typo/plural
} hit;

typedef struct {
    hit h[MAX_HITS];
    int n;
} hits;

static int find_word(const tai_model *m, const char *w) {  // binary search, -1 if absent
    int lo = 0, hi = m->n_words - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2, c = strcmp(w, m->words + m->word_off[mid]);
        if (!c) return mid;
        if (c < 0) hi = mid - 1;
        else lo = mid + 1;
    }
    return -1;
}

static void add_hit(hits *out, int id, uint8_t q) {
    if (id >= 0 && out->n < MAX_HITS) out->h[out->n++] = (hit){(uint16_t)id, q};
}

// Every dictionary word matching `w`. A word the dictionary knows exactly is
// taken as spelled right: only its plural forms are added (cheap lookups). An
// unknown word is a likely typo, so the whole dictionary is scanned for
// near matches. This keeps "iceland" from also matching "ireland".
static void lookup(const tai_model *m, const char *w, hits *out) {
    out->n = 0;
    int lw = (int)strlen(w), exact = find_word(m, w);
    if (exact >= 0) {
        char v[MAX_WLEN + 2];
        add_hit(out, exact, 2);
        if (lw + 1 < MAX_WLEN) {
            snprintf(v, sizeof v, "%ss", w);
            add_hit(out, find_word(m, v), 1);
        }
        if (lw > 3 && w[lw - 1] == 's') {
            snprintf(v, sizeof v, "%.*s", lw - 1, w);
            add_hit(out, find_word(m, v), 1);
        }
        if (lw > 4 && !strcmp(w + lw - 3, "ies")) {
            snprintf(v, sizeof v, "%.*sy", lw - 3, w);
            add_hit(out, find_word(m, v), 1);
        }
        if (lw > 3 && w[lw - 1] == 'y' && lw + 2 < MAX_WLEN) {
            snprintf(v, sizeof v, "%.*sies", lw - 1, w);
            add_hit(out, find_word(m, v), 1);
        }
        return;
    }
    for (int i = 0; i < m->n_words && out->n < MAX_HITS; i++) {
        int d = lw - (m->word_len[i] & 0x7f);
        if (d > 2 || d < -2) continue;  // no fuzzy match spans more than 2 letters
        const char *k = m->words + m->word_off[i];
        if (strcmp(w, k) && word_match(w, k)) out->h[out->n++] = (hit){(uint16_t)i, 1};
    }
}

static int quality(const hits *h, uint16_t id) {
    int q = 0;
    for (int i = 0; i < h->n; i++)
        if (h->h[i].id == id && h->h[i].q > q) q = h->h[i].q;
    return q;
}

// Question type code, in the order of QTYPES in train/gate_index.py:
// "", what, who, when, where, why, how, how many, how much, ...
// A "how many" question must never be answered by a "what" fact:
// "how many moons does jupiter have" is not "what is jupiter's largest moon".
static const char *const HOW[] = {"many", "much", "long", "far", "fast", "old", "big", "tall",
                                  "heavy", "hot", "cold", "deep", "high", "often", "smart"};

// True if the words "way to" or "ways to" occur in s.
static int way_to(const char *s) {
    for (const char *p = s; (p = strstr(p, "way")) != NULL; p += 3) {
        if (p != s && p[-1] != ' ') continue;
        const char *q = p + 3 + (p[3] == 's');
        if (!strncmp(q, " to", 3) && (q[3] == ' ' || q[3] == 0)) return 1;
    }
    return 0;
}

static int qtype(const char *s) {
    char w[MAX_WLEN];
    while (*s) {
        int len = 0;
        while (*s == ' ') s++;
        while (*s && *s != ' ') {
            if (*s != '\'' && len < MAX_WLEN - 1) w[len++] = *s;
            s++;
        }
        w[len] = 0;
        while (*s == ' ') s++;
        int next_year = !strncmp(s, "year", 4) && (s[4] == ' ' || s[4] == 0);
        if ((!strcmp(w, "what") || !strcmp(w, "which")) && next_year) return 3;  // "what year" asks when
        // "what's the best way to purify water" asks how
        if ((!strcmp(w, "what") || !strcmp(w, "whats") || !strcmp(w, "which")) && way_to(s)) return 6;
        if (!strcmp(w, "what") || !strcmp(w, "whats") || !strcmp(w, "which")) return 1;
        if (!strcmp(w, "who") || !strcmp(w, "whos") || !strcmp(w, "whose")) return 2;
        if (!strcmp(w, "when") || !strcmp(w, "whens")) return 3;
        if (!strcmp(w, "where") || !strcmp(w, "wheres")) return 4;
        if (!strcmp(w, "why") || !strcmp(w, "whys")) return 5;
        if (!strcmp(w, "how") || !strcmp(w, "hows")) {
            for (int i = 0; i < (int)(sizeof HOW / sizeof *HOW); i++) {
                size_t n = strlen(HOW[i]);
                if (!strncmp(s, HOW[i], n) && (s[n] == ' ' || s[n] == 0)) return 7 + i;
            }
            return 6;
        }
    }
    return 0;
}

// "what date is X" and "when is X" ask the same thing. That is the only
// cross-type pair: "how far is pluto" must never match "what is pluto". A
// cross-type match must cover every word of the known phrasing, and loses to
// any same-type match.
static int compatible(int a, int b) { return (a == 1 && b == 3) || (a == 3 && b == 1); }

// Character-bigram overlap (Dice coefficient) of two strings, 0..1000.
static int dice(const char *a, const char *b) {
    int la = (int)strlen(a), lb = (int)strlen(b), common = 0;
    if (la < 2 || lb < 2) return strcmp(a, b) ? 0 : 1000;
    unsigned char used[256] = {0};
    for (int i = 0; i + 1 < la; i++)
        for (int j = 0; j + 1 < lb && j < 255; j++)
            if (!used[j] && a[i] == b[j] && a[i + 1] == b[j + 1]) {
                used[j] = 1;
                common++;
                break;
            }
    return 2000 * common / (la - 1 + lb - 1);
}

// Index of the first content word after "with" or "using" ("purify water
// with a lifestraw"), or MAX_WORDS if there is none.
static int method_start(const char *norm) {
    char w[MAX_WLEN];
    int words = 0;
    for (const char *s = norm; *s;) {
        int len = 0;
        while (*s == ' ') s++;
        while (*s && *s != ' ') {
            if (*s != '\'' && len < MAX_WLEN - 1) w[len++] = *s;
            s++;
        }
        w[len] = 0;
        if (!len) break;
        if (!strcmp(w, "with") || !strcmp(w, "using")) return words;
        static char one[MAX_WORDS][MAX_WLEN];
        words += content_words(w, one);
    }
    return MAX_WORDS;
}

// Picks the known phrasing closest to the input. Every content word of the
// input must be accounted for, the question types must agree, and the known
// phrasing must be at least half present in the input. Among those, exact
// words beat typo matches and leftover unmatched words cost points. Ties go
// to the phrasing with the same word order ("fahrenheit to celsius" is not
// "celsius to fahrenheit"), then the closest length.
// 0 = refuse, else 1 + fact index.
int tai_gate(const tai_model *m, const char *norm) {
    char in[MAX_WORDS][MAX_WLEN];
    int n = content_words(norm, in), tin = qtype(norm), len = (int)strlen(norm);
    // Only numbers ("42"): they name nothing on their own, so they may not
    // stand in for "element 42".
    int wordy = 0;
    for (int i = 0; i < n; i++)
        for (const char *c = in[i]; *c; c++) wordy |= *c >= 'a' && *c <= 'z';
    // One dictionary lookup per input word, and per adjacent pair joined
    // ("humming bird" -> "hummingbird"). Static: too big for a small stack.
    static hits word[MAX_WORDS], joined[MAX_WORDS];
    int soft[MAX_WORDS], qualifier[MAX_WORDS], can_join[MAX_WORDS];
    int method = method_start(norm);
    for (int i = 0; i < n; i++) {
        lookup(m, in[i], &word[i]);
        soft[i] = is_soft(in[i]);
        // Not context: who it's for, a number, a negation, the method
        // ("with a flint"), or a word some other advice is about ("freezer").
        qualifier[i] = is_qualifier(in[i]) || i >= method;
        for (int h = 0; h < word[i].n; h++) qualifier[i] |= m->word_len[word[i].h[h].id] >> 7;
        can_join[i] = i + 1 < n && strlen(in[i]) + strlen(in[i + 1]) < MAX_WLEN;
        if (can_join[i]) {
            char j[MAX_WLEN];
            strcpy(j, in[i]);
            strcat(j, in[i + 1]);
            lookup(m, j, &joined[i]);
        }
    }

    // key = score without the length tie-break (0..59, always below one step of
    // the key), so the best score always has the top key. If two different
    // facts reach the top key, the question is ambiguous.
    long best_score = -1, top_key = -1;
    int best = 0, top_fact = 0, tied = 0;
    // Every dictionary word any input word matched. A phrasing needs half its
    // words in here to pass, so most are rejected after a few bit lookups.
    static uint8_t hit_bits[65536 / 8];
    memset(hit_bits, 0, (size_t)(m->n_words + 7) / 8);
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < word[i].n; h++) hit_bits[word[i].h[h].id >> 3] |= 1u << (word[i].h[h].id & 7);
        if (can_join[i])
            for (int h = 0; h < joined[i].n; h++) hit_bits[joined[i].h[h].id >> 3] |= 1u << (joined[i].h[h].id & 7);
    }
    const uint16_t *ids = m->known_words;
    const char *talk = m->small_talk;
    // phrasings are stored grouped by fact: f is the fact of phrasing p
    int f = -1, left = 0;
    for (int p = 0; p < m->n_known; p++) {
        while (left == 0 && f + 1 < m->n_facts) left = m->fact_count[++f];
        left--;
        int nk = m->known_nwords[p];
        const uint16_t *k = ids;
        ids += nk;
        const char *text = NULL;
        if (nk == 0) {
            text = talk;
            talk += strlen(talk) + 1;
        }
        int tk = m->known_qtype[p] & 0x3f, strict = m->known_qtype[p] >> 7,
            lenient = (m->known_qtype[p] >> 6) & 1;
        int cross = tin && tk && tin != tk;
        if (cross && !compatible(tin, tk)) continue;
        long score, key = -1;
        if (n == 0 || nk == 0) {
            // Small talk ("hi", "how are you"): nothing to fact-check, so
            // match the whole phrase instead.
            if (n != nk || cross) continue;
            int d = dice(norm, text);
            if (d < 500) continue;
            score = d;
        } else {
            int present = 0;
            for (int j = 0; j < nk; j++) present += (hit_bits[k[j] >> 3] >> (k[j] & 7)) & 1;
            if (2 * present < nk) continue;
            // input side: every word found, or excused as soft
            int a_quality = 0, hard_missing = 0, extra = 0, in_order = 0, last_pos = -1;
            for (int i = 0; i < n && !hard_missing; i++) {
                int f = 0, pos = -1;
                for (int j = 0; j < nk; j++) {
                    int q = quality(&word[i], k[j]);
                    if (q > f) {
                        f = q;
                        pos = j;
                    }
                }
                if (pos >= 0) {
                    in_order += pos > last_pos;
                    last_pos = pos;
                }
                if (!f && can_join[i]) {
                    for (int j = 0; j < nk; j++) {
                        int q = quality(&joined[i], k[j]);
                        if (q > f) f = q;
                    }
                    if (f) {
                        a_quality += 2 * f;  // both halves count
                        i++;
                        continue;
                    }
                }
                if (f) a_quality += f;
                else if (soft[i]) continue;
                // Advice phrasings (survival) excuse one word of context:
                // "how to survive in *extreme* heat". Checked below.
                else if (lenient && !extra && !qualifier[i]) extra = 1;
                else hard_missing = 1;
            }
            if (hard_missing) continue;
            // known side: at least half of the phrasing's words present
            int b_quality = 0, b_matched = 0;
            for (int j = 0; j < nk; j++) {
                int f = 0;
                for (int i = 0; i < n; i++) {
                    int q = quality(&word[i], k[j]);
                    if (q > f) f = q;
                    if (can_join[i] && (q = quality(&joined[i], k[j])) > f) f = q;
                }
                if (f) {
                    b_matched++;
                    b_quality += f;
                }
            }
            // A strict phrasing names something ("michelle obama"): all of it
            // must be in the question, or "obama" alone would reach it.
            if (2 * b_matched < nk || ((cross || strict || !wordy) && b_matched < nk)) continue;
            // ...and only when every word of a 2+ word phrasing is there, so
            // the extra word is context, not a different subject.
            if (extra && (b_matched < nk || nk < 2)) continue;
            int d = abs(len - m->known_len[p]);
            // A known word the question never mentions costs more than a typo:
            // that phrasing asks something more specific ("... batteries").
            // A phrasing with the question's own type beats one with no type:
            // "where is the painter's studio" is not "the painter's studio painter".
            int untyped = tin && !tk;
            key = 1000L * (a_quality + b_quality - 3 * (nk - b_matched) - 3 * cross - 2 * untyped -
                           4 * extra) +
                  60L * in_order;
            score = key + (d > 59 ? 0 : 59 - d);
        }
        int fact = f + 1;
        if (key > top_key) {
            top_key = key;
            top_fact = fact;
            tied = 0;
        } else if (key >= 0 && key == top_key && fact != top_fact) {
            tied = 1;  // two facts match equally well: don't guess
        }
        if (score > best_score) {
            best_score = score;
            best = fact;
        }
    }
    return tied ? 0 : best;
}

// For tests: "<qtype> <content words...>" of a string, as the gate sees it.
int tai_gate_words(const char *norm, char *out, int out_len) {
    char w[MAX_WORDS][MAX_WLEN];
    int n = content_words(norm, w), o = snprintf(out, (size_t)out_len, "%d", qtype(norm));
    for (int i = 0; i < n && o < out_len; i++) o += snprintf(out + o, (size_t)(out_len - o), " %s", w[i]);
    return n;
}
