// Arithmetic. A tiny language model is bad at math; a parser is perfect at
// it. "what is 12 times 7" -> "84".
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "tinyai.h"

typedef struct {
    const char *p;
    int err;
    int pct;    // the last number read had a % sign
    int depth;  // nesting, capped: the ESP32's loop task has an 8 KB stack
} parser;

static void skip(parser *ps) {
    while (*ps->p == ' ') ps->p++;
}

static double expr(parser *ps);

static double power(parser *ps);

static double atom(parser *ps) {
    skip(ps);
    if (*ps->p == '#') {  // square root
        ps->p++;
        double v = power(ps);
        if (v < 0) ps->err = 3;
        return sqrt(v);
    }
    if (*ps->p == '(') {
        ps->p++;
        double v = expr(ps);
        skip(ps);
        if (*ps->p != ')') ps->err = 1;
        else ps->p++;
        return v;
    }
    if (!((*ps->p >= '0' && *ps->p <= '9') || *ps->p == '.')) {
        ps->err = 1;
        return 0;
    }
    double v = 0, frac = 0.1;
    int dot = 0, digits = 0;
    for (; (*ps->p >= '0' && *ps->p <= '9') || *ps->p == '.' || *ps->p == ','; ps->p++) {
        if (*ps->p == ',') continue;  // 1,000
        if (*ps->p == '.') {
            if (dot++) ps->err = 1;
            continue;
        }
        digits++;
        if (dot) {
            v += (*ps->p - '0') * frac;
            frac /= 10;
        } else {
            v = v * 10 + (*ps->p - '0');
        }
    }
    if (!digits) ps->err = 1;
    skip(ps);
    if (*ps->p == '%') {
        ps->p++;
        v /= 100;
        ps->pct = 1;
    }
    return v;
}

// Unary minus binds looser than ^, as in math: -2^2 = -4, 2^-1 = 0.5.
static double power(parser *ps) {
    if (++ps->depth > 16) {  // "((((((...": refuse rather than overflow the stack
        ps->err = 1;
        return 0;
    }
    double v;
    skip(ps);
    if (*ps->p == '-') {
        ps->p++;
        v = -power(ps);
    } else {
        v = atom(ps);
        skip(ps);
        if (*ps->p == '^') {
            ps->p++;
            v = pow(v, power(ps));  // right associative
        }
    }
    ps->depth--;
    return v;
}

// *is_pct: the term is a bare percentage ("10%"), not "10% * 80".
static double term(parser *ps, int *is_pct) {
    ps->pct = 0;
    double v = power(ps);
    *is_pct = ps->pct;
    for (;;) {
        skip(ps);
        char op = *ps->p;
        if (op != '*' && op != '/') return v;
        *is_pct = 0;
        ps->p++;
        double r = power(ps);
        if (op == '*') v *= r;
        else if (r == 0) ps->err = 2;
        else v /= r;
    }
}

// "50 + 10%" means 55, as on any calculator, not 50.1.
static double expr(parser *ps) {
    int pct;
    double v = term(ps, &pct);
    for (;;) {
        skip(ps);
        char op = *ps->p;
        if (op != '+' && op != '-') return v;
        ps->p++;
        double r = term(ps, &pct);
        if (pct) r *= v;
        v = op == '+' ? v + r : v - r;
    }
}

// Replaces whole words in `s` (in place, result never longer than input).
static void replace_word(char *s, const char *word, const char *with) {
    size_t lw = strlen(word), lr = strlen(with);
    char *p = s;
    while ((p = strstr(p, word)) != NULL) {
        int left = p == s || p[-1] == ' ';
        int right = p[lw] == 0 || p[lw] == ' ';
        if (left && right) {
            memcpy(p, with, lr);
            memmove(p + lr, p + lw, strlen(p + lw) + 1);
            p += lr;
        } else {
            p += lw;
        }
    }
}

static int strip_prefix(char *s, const char *pre) {
    size_t n = strlen(pre);
    if (strncmp(s, pre, n)) return 0;
    memmove(s, s + n, strlen(s + n) + 1);
    return 1;
}

// Returns 1 and writes the result if `norm` is an arithmetic question.
int tai_calc(const char *norm, char *out, int out_len) {
    char words[128];
    tai_number_words(norm, words, sizeof words);  // "twelve times seven" -> "12 times 7"
    if (tai_everyday(words, out, out_len) || tai_convert(words, out, out_len)) return 1;
    char s[128];
    strncpy(s, words, sizeof s - 1);
    s[sizeof s - 1] = 0;

    static const char *const pre[] = {"what is ", "whats ", "what's ", "calculate ",
                                      "compute ", "how much is ", "solve ", "evaluate "};
    for (size_t i = 0; i < sizeof pre / sizeof *pre; i++)
        if (strip_prefix(s, pre[i])) break;
    replace_word(s, "the square root of", "#");
    replace_word(s, "square root of", "#");
    replace_word(s, "sqrt of", "#");
    replace_word(s, "sqrt", "#");
    replace_word(s, "squared", "^2");
    replace_word(s, "cubed", "^3");
    replace_word(s, "multiplied by", "*");
    replace_word(s, "divided by", "/");
    replace_word(s, "to the power of", "^");
    replace_word(s, "plus", "+");
    replace_word(s, "minus", "-");
    replace_word(s, "times", "*");
    replace_word(s, "x", "*");
    replace_word(s, "over", "/");
    replace_word(s, "of", "*");  // 15% of 80
    size_t n = strlen(s);
    while (n && (s[n - 1] == '=' || s[n - 1] == ' ')) s[--n] = 0;

    // Only digits and operators may remain, with at least one operator
    // between two numbers.
    int ops = 0, digits = 0;
    for (const char *c = s; *c; c++) {
        if (*c >= '0' && *c <= '9') digits++;
        else if (strchr("+-*/^%#", *c)) ops++;
        else if (!strchr(" .,()", *c)) return 0;
    }
    if (!digits || !ops) return 0;

    parser ps = {s, 0, 0, 0};
    double v = expr(&ps);
    skip(&ps);
    if (*ps.p) ps.err = 1;
    if (ps.err == 1) return 0;
    if (ps.err == 3) {
        snprintf(out, (size_t)out_len, "Not a real number.");
        return 1;
    }
    if (ps.err == 2) {
        snprintf(out, (size_t)out_len, "Undefined, division by zero.");
        return 1;
    }
    if (isnan(v) || isinf(v)) {
        snprintf(out, (size_t)out_len, "Undefined.");
        return 1;
    }
    if (fabs(v) < 1e15 && fabs(v - llround(v)) < 1e-9)
        snprintf(out, (size_t)out_len, "%lld", (long long)llround(v));
    else
        snprintf(out, (size_t)out_len, "%.6g", v);
    return 1;
}
