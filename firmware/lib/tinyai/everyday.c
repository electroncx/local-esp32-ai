// Everyday questions that are computed, not memorized: money, percentages,
// times of day, dates. Also rewrites number words so "twelve times seven"
// reaches the calculator as "12 times 7".
//
//   "15% tip on 42.50"                 -> "Tip 6.38, total 48.88."
//   "20 percent off 80"                -> "64"
//   "what percent is 12 of 48"         -> "25%"
//   "split 90 between 4 people"        -> "22.50 each."
//   "average of 3, 5 and 10"           -> "6"
//   "3pm in 24 hour time"              -> "15:00"
//   "how long from 9am to 5:30pm"      -> "8 hours 30 minutes"
//   "what day was july 20 1969"        -> "Sunday."
//   "days between march 3 and june 10" -> "99 days"
//   "30 days after march 3 2025"       -> "Wednesday, April 2, 2025."
//   "8% tax on 50"                     -> "Tax 4, total 54."
//   "percent change from 50 to 75"     -> "+50%"
//   "3pm eastern in pacific"           -> "12:00 PM"
//   "bmi 70 kg 175 cm"                 -> "BMI 22.9: healthy weight."
//   "10 factorial"                     -> "3628800"
//   "flip a coin"                      -> "Heads."
//   "water for 4 people for 3 days"    -> "12 gallons (45 L) for 3 days."
//   "thunder 10 seconds after lightning" -> "About 2.1 miles (3.4 km). Go indoors."
//   "how much bleach for 4 gallons"    -> "32 drops of 6% bleach; wait 30 min."
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyai.h"

#define MAX_TOK 32
#define BUF 128

static int split(char *s, char *tok[MAX_TOK]) {
    int n = 0;
    for (char *p = strtok(s, " "); p && n < MAX_TOK; p = strtok(NULL, " ")) tok[n++] = p;
    return n;
}

static int eq(const char *a, const char *b) { return !strcmp(a, b); }

// "42.50", "1,000", "-3" -> value. Whole token must be a number.
static int num(const char *t, double *v) {
    char buf[32];
    int n = 0, digits = 0;
    for (const char *c = t; *c; c++) {
        if (*c == ',') continue;
        if (n >= (int)sizeof buf - 1) return 0;
        if (*c >= '0' && *c <= '9') digits++;
        else if (!(*c == '.' || (*c == '-' && c == t))) return 0;
        buf[n++] = *c;
    }
    buf[n] = 0;
    if (!digits) return 0;
    char *end;
    *v = strtod(buf, &end);
    return !*end;
}

// "15%" or "15" followed by "%": returns tokens used.
static int percent(char **tok, int n, int i, double *v) {
    char buf[32];
    size_t l = i < n ? strlen(tok[i]) : 0;
    if (l > 1 && l < sizeof buf && tok[i][l - 1] == '%') {
        memcpy(buf, tok[i], l - 1);
        buf[l - 1] = 0;
        return num(buf, v) ? 1 : 0;
    }
    if (i + 1 < n && eq(tok[i + 1], "%") && num(tok[i], v)) return 2;
    return 0;
}

// Money: cents when there are any. 64 -> "64", 6.375 -> "6.38".
static void money(char *out, size_t len, double v) {
    double r = round(v * 100 * (1 + 1e-12)) / 100;  // 48.875 is 48.87499.. in binary
    if (fabs(r - round(r)) < 1e-9) snprintf(out, len, "%.0f", r);
    else snprintf(out, len, "%.2f", r);
}

static void plain(char *out, size_t len, double v) {
    if (fabs(v - round(v)) < 1e-9 && fabs(v) < 1e15) snprintf(out, len, "%.0f", v);
    else snprintf(out, len, "%.6g", v);
}

static int is_currency(const char *t) {
    return eq(t, "dollars") || eq(t, "dollar") || eq(t, "bucks") || eq(t, "euros") ||
           eq(t, "euro") || eq(t, "usd") || eq(t, "eur");
}

// ---------------------------------------------------------------- number words

static const char *const SMALL[] = {"zero", "one", "two", "three", "four", "five", "six",
                                    "seven", "eight", "nine", "ten", "eleven", "twelve",
                                    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                                    "eighteen", "nineteen"};
static const char *const TENS[] = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                                   "eighty", "ninety"};

static int word_value(const char *w, long long *v) {  // 1 = unit/ten, 2 = scale
    for (int i = 0; i < 20; i++)
        if (eq(w, SMALL[i])) return (*v = i), 1;
    for (int i = 0; i < 8; i++)
        if (eq(w, TENS[i])) return (*v = 20 + 10 * i), 1;
    if (eq(w, "hundred")) return (*v = 100), 2;
    if (eq(w, "thousand")) return (*v = 1000), 2;
    if (eq(w, "million")) return (*v = 1000000), 2;
    if (eq(w, "billion")) return (*v = 1000000000), 2;
    return 0;
}

// "twenty-one" -> two words
static int word_or_hyphenated(const char *w, long long *v) {
    const char *h = strchr(w, '-');
    if (!h) return word_value(w, v);
    char a[24], b[24];
    size_t la = (size_t)(h - w);
    if (la >= sizeof a || strlen(h + 1) >= sizeof b) return 0;
    memcpy(a, w, la);
    a[la] = 0;
    strcpy(b, h + 1);
    long long x, y;
    if (word_value(a, &x) == 1 && x >= 20 && word_value(b, &y) == 1 && y < 10) return (*v = x + y), 1;
    return 0;
}

void tai_number_words(const char *in, char *out, int out_len) {
    char s[BUF], *tok[MAX_TOK];
    strncpy(s, in, sizeof s - 1);
    s[sizeof s - 1] = 0;
    int n = split(s, tok), o = 0;
    out[0] = 0;
#define PUT(str)                                                                           \
    do {                                                                                   \
        o += snprintf(out + o, (size_t)(out_len - o), "%s%s", o ? " " : "", (str));       \
        if (o >= out_len) o = out_len - 1;                                                 \
    } while (0)
    for (int i = 0; i < n; i++) {
        // "a dozen" -> 12, "3 dozen" -> 3 * 12
        if ((eq(tok[i], "a") || eq(tok[i], "one")) && i + 1 < n && eq(tok[i + 1], "dozen")) {
            PUT("12");
            i++;
            continue;
        }
        if (eq(tok[i], "dozen")) {
            double prev;
            PUT(i > 0 && num(tok[i - 1], &prev) ? "* 12" : "12");
            continue;
        }
        // A run of number words -> one number: "one hundred and five", "a thousand".
        // "and" only joins after hundred/thousand/...: "one and ten" stays two numbers.
        long long total = 0, cur = 0, w;
        int j = i, any = 0, big = 0;
        for (; j < n; j++) {
            if (cur > 1000000000LL || total > 1000000000000000LL) break;  // no overflow
            int k = word_or_hyphenated(tok[j], &w);
            if (!k && j == i && (eq(tok[j], "a") || eq(tok[j], "an")) && j + 1 < n &&
                word_value(tok[j + 1], &w) == 2) {
                cur = 1;  // "a hundred"
                continue;
            }
            if (k == 1) cur += w;
            else if (k == 2 && w == 100) cur = (cur ? cur : 1) * 100, big = 1;
            else if (k == 2) total += (cur ? cur : 1) * w, cur = 0, big = 1;
            else if (big && eq(tok[j], "and") && j + 1 < n && word_or_hyphenated(tok[j + 1], &w)) continue;
            else break;
            any = 1;
        }
        if (any) {
            char buf[24];
            snprintf(buf, sizeof buf, "%lld", total + cur);
            PUT(buf);
            i = j - 1;
            continue;
        }
        if ((eq(tok[i], "half") || eq(tok[i], "quarter")) && i + 1 < n && eq(tok[i + 1], "of")) {
            PUT(eq(tok[i], "half") ? "0.5" : "0.25");
            continue;
        }
        if ((eq(tok[i], "double") || eq(tok[i], "twice") || eq(tok[i], "triple")) && i + 1 < n) {
            double d;
            long long w;
            if (num(tok[i + 1], &d) || word_or_hyphenated(tok[i + 1], &w)) {
                PUT(eq(tok[i], "triple") ? "3 *" : "2 *");
                continue;
            }
        }
        if (eq(tok[i], "percent") || eq(tok[i], "pct")) {
            PUT("%");
            continue;
        }
        if (eq(tok[i], "per") && i + 1 < n && eq(tok[i + 1], "cent")) {
            PUT("%");
            i++;
            continue;
        }
        PUT(tok[i]);
    }
#undef PUT
}

// ---------------------------------------------------------------- money & percent

static int skip_prefix(char **tok, int n) {
    int i = 0;
    if (n > 1 && eq(tok[0], "what") && eq(tok[1], "is")) i = 2;
    else if (n > 0 && (eq(tok[0], "whats") || eq(tok[0], "what's") || eq(tok[0], "calculate"))) i = 1;
    else if (n > 2 && eq(tok[0], "how") && eq(tok[1], "much") && eq(tok[2], "is")) i = 3;
    if (i < n && (eq(tok[i], "a") || eq(tok[i], "the"))) i++;
    return i;
}

static int money_question(char **tok, int n, char *out, int out_len) {
    double p = 0, x, y;
    int has_tip = 0, has_tax = 0, has_change = 0, pct_at = -1, pct_len = 0;
    for (int i = 0; i < n; i++) {
        if (eq(tok[i], "tip") || eq(tok[i], "tips") || eq(tok[i], "gratuity")) has_tip = 1;
        if (eq(tok[i], "tax") || eq(tok[i], "vat") || eq(tok[i], "gst")) has_tax = 1;
        if (eq(tok[i], "change") || eq(tok[i], "increase") || eq(tok[i], "decrease") ||
            eq(tok[i], "difference"))
            has_change = 1;
        int l = percent(tok, n, i, &x);
        if (l && pct_at < 0) {
            pct_at = i;
            pct_len = l;
            p = x;
        }
    }
    // collect plain numbers (not the percentage)
    double nums[8];
    int nn = 0;
    for (int i = 0; i < n && nn < 8; i++) {
        if (i >= pct_at && i < pct_at + pct_len) continue;
        if (num(tok[i], &x)) nums[nn++] = x;
    }
    char a[24], b[24], c[24];

    if (has_tip) {  // "15% tip on 42.50", "tip on 60", "what's a 20 percent tip for 85 dollars"
        if (nn != 1 || nums[0] <= 0) return 0;
        x = nums[0];
        if (pct_at >= 0) {
            money(a, sizeof a, x * p / 100);
            money(b, sizeof b, x * (1 + p / 100));
            snprintf(out, (size_t)out_len, "Tip %s, total %s.", a, b);
        } else {
            money(a, sizeof a, x * 0.15);
            money(b, sizeof b, x * 0.18);
            money(c, sizeof c, x * 0.20);
            snprintf(out, (size_t)out_len, "15%%: %s, 18%%: %s, 20%%: %s.", a, b, c);
        }
        return 1;
    }

    if (has_tax && pct_at >= 0 && nn == 1 && nums[0] > 0) {  // "8% tax on 50", "sales tax 7.25% on 19.99"
        money(a, sizeof a, nums[0] * p / 100);
        money(b, sizeof b, nums[0] * (1 + p / 100));
        snprintf(out, (size_t)out_len, "Tax %s, total %s.", a, b);
        return 1;
    }

    // "percent change from 50 to 75", "percentage increase from 80 to 100"
    if (has_change && pct_at < 0 && nn == 2) {
        int asks_pct = 0;
        for (int i = 0; i < n; i++) asks_pct |= eq(tok[i], "%") || eq(tok[i], "percentage");
        if (asks_pct && nums[0] != 0) {
            plain(a, sizeof a, (nums[1] - nums[0]) / fabs(nums[0]) * 100);
            snprintf(out, (size_t)out_len, "%s%s%%", nums[1] >= nums[0] ? "+" : "", a);
            return 1;
        }
    }

    // "20% off 80", "80 with 20% off"
    for (int i = 0; i < n; i++)
        if (eq(tok[i], "off") && pct_at >= 0 && nn == 1) {
            money(out, (size_t)out_len, nums[0] * (1 - p / 100));
            return 1;
        }

    // "increase 50 by 10%", "decrease 80 by 25%"
    if (n >= 4 && (eq(tok[0], "increase") || eq(tok[0], "decrease")) && pct_at >= 0 && nn == 1) {
        plain(out, (size_t)out_len, nums[0] * (eq(tok[0], "increase") ? 1 + p / 100 : 1 - p / 100));
        return 1;
    }

    // "what percent is 12 of 48", "12 is what percent of 48", "what percent of 48 is 12"
    for (int i = 0; i + 1 < n; i++) {
        if (!eq(tok[i], "what") || !(eq(tok[i + 1], "%") || eq(tok[i + 1], "percentage"))) continue;
        int j = i + 2;
        if (nn != 2) return 0;
        if (i >= 2 && eq(tok[i - 1], "is")) x = nums[0], y = nums[1];       // X is what % of Y
        else if (j < n && eq(tok[j], "is")) x = nums[0], y = nums[1];       // what % is X of Y
        else if (j < n && eq(tok[j], "of")) x = nums[1], y = nums[0];       // what % of Y is X
        else return 0;
        if (y == 0) return 0;
        plain(a, sizeof a, x / y * 100);
        snprintf(out, (size_t)out_len, "%s%%", a);
        return 1;
    }

    // "split 90 between 4 people", "split 90 4 ways", "divide 90 among 3"
    if (n >= 3 && (eq(tok[0], "split") || eq(tok[0], "divide") || eq(tok[0], "share"))) {
        if (nn != 2) return 0;
        if (eq(tok[0], "divide") && n == 4 && eq(tok[2], "by")) {  // "divide 90 by 4"
            if (nums[1] == 0) return 0;
            plain(out, (size_t)out_len, nums[0] / nums[1]);
            return 1;
        }
        if (nums[1] < 1 || nums[1] != floor(nums[1])) return 0;
        int ok = 0;
        for (int i = 1; i < n; i++)
            if (eq(tok[i], "between") || eq(tok[i], "among") || eq(tok[i], "ways") ||
                eq(tok[i], "way") || eq(tok[i], "people") || eq(tok[i], "persons"))
                ok = 1;
        if (!ok) return 0;
        money(a, sizeof a, nums[0] / nums[1]);
        snprintf(out, (size_t)out_len, "%s each.", a);
        return 1;
    }

    // "average of 3, 5 and 10", "mean of 1 2 3"
    int i = skip_prefix(tok, n);
    if (i + 1 < n && (eq(tok[i], "average") || eq(tok[i], "mean")) && eq(tok[i + 1], "of")) {
        double sum = 0;
        int k = 0;
        for (int j = i + 2; j < n; j++) {
            char t[32];
            strncpy(t, tok[j], sizeof t - 1);
            t[sizeof t - 1] = 0;
            size_t l = strlen(t);
            if (l && t[l - 1] == ',') t[l - 1] = 0;
            if (num(t, &x)) {
                sum += x;
                k++;
            } else if (!eq(t, "and") && !eq(t, ",") && !is_currency(t)) {
                return 0;
            }
        }
        if (k < 2) return 0;
        plain(out, (size_t)out_len, sum / k);
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------- times of day

// "3pm", "3:30pm", "3 pm", "15:30", "noon", "midnight" -> minutes after midnight.
// *ampm = 1 when am/pm was given. Returns tokens used.
static int parse_time(char **tok, int n, int i, int *mins, int *ampm, int allow_hhmm) {
    if (i >= n) return 0;
    *ampm = 0;
    if (eq(tok[i], "noon") || eq(tok[i], "midday")) return (*mins = 12 * 60), (*ampm = 1), 1;
    if (eq(tok[i], "midnight")) return (*mins = 0), (*ampm = 1), 1;
    const char *t = tok[i];
    int h = 0, m = 0, digits = 0, used = 1;
    while (*t >= '0' && *t <= '9' && digits < 5) h = h * 10 + (*t++ - '0'), digits++;
    if (!digits || digits > 2) {
        if (digits == 4 && !*t && allow_hhmm) {  // "1530"
            m = h % 100;
            h /= 100;
        } else {
            return 0;
        }
    } else if (*t == ':') {
        t++;
        if (!(t[0] >= '0' && t[0] <= '9' && t[1] >= '0' && t[1] <= '9')) return 0;
        m = (t[0] - '0') * 10 + (t[1] - '0');
        t += 2;
    }
    const char *suffix = t;
    if (!*suffix && i + 1 < n && (eq(tok[i + 1], "am") || eq(tok[i + 1], "pm") ||
                                  eq(tok[i + 1], "a.m") || eq(tok[i + 1], "p.m"))) {
        suffix = tok[i + 1];
        used = 2;
    }
    if (*suffix) {
        int pm;
        if (eq(suffix, "am") || eq(suffix, "a.m")) pm = 0;
        else if (eq(suffix, "pm") || eq(suffix, "p.m")) pm = 1;
        else return 0;
        if (h < 1 || h > 12) return 0;
        h = h % 12 + (pm ? 12 : 0);
        *ampm = 1;
    } else if (digits <= 2 && t == tok[i] + digits) {
        return 0;  // a bare "9" is not a time
    }
    if (h > 23 || m > 59) return 0;
    *mins = h * 60 + m;
    return used;
}

static void fmt_12h(char *out, size_t len, int mins) {
    int h = mins / 60, m = mins % 60, h12 = h % 12 ? h % 12 : 12;
    snprintf(out, len, "%d:%02d %s", h12, m, h < 12 ? "AM" : "PM");
}

static void fmt_duration(char *out, size_t len, int mins) {
    int h = mins / 60, m = mins % 60;
    if (h && m) snprintf(out, len, "%d hour%s %d minute%s", h, h == 1 ? "" : "s", m, m == 1 ? "" : "s");
    else if (h) snprintf(out, len, "%d hour%s", h, h == 1 ? "" : "s");
    else snprintf(out, len, "%d minute%s", m, m == 1 ? "" : "s");
}

static int time_question(char **tok, int n, char *out, int out_len) {
    int a, b, ampm_a, ampm_b;
    // "how long from 9am to 5pm", "hours between 9:15 and 17:40", "time from noon to 3:30pm"
    for (int i = 0; i < n; i++) {
        if (!(eq(tok[i], "from") || eq(tok[i], "between"))) continue;
        int la = parse_time(tok, n, i + 1, &a, &ampm_a, 0);
        if (!la) continue;
        int j = i + 1 + la;
        if (j >= n || !(eq(tok[j], "to") || eq(tok[j], "and") || eq(tok[j], "until") || eq(tok[j], "till")))
            continue;
        int lb = parse_time(tok, n, j + 1, &b, &ampm_b, 0);
        if (!lb || j + 1 + lb != n) continue;
        int d = b - a;
        if (d <= 0) d += 24 * 60;  // overnight
        fmt_duration(out, (size_t)out_len, d);
        return 1;
    }
    // "3pm in 24 hour time", "15:30 in 12 hour", "what is 1400 in standard time"
    int i = skip_prefix(tok, n), t;
    int l = parse_time(tok, n, i, &t, &ampm_a, 1);
    if (!l || i + l >= n) return 0;
    int j = i + l;
    if (!(eq(tok[j], "in") || eq(tok[j], "to") || eq(tok[j], "into"))) return 0;
    j++;
    int to24 = -1;
    if (j < n && eq(tok[j], "24") && j + 1 < n && (eq(tok[j + 1], "hour") || eq(tok[j + 1], "hr"))) to24 = 1, j += 2;
    else if (j < n && eq(tok[j], "12") && j + 1 < n && (eq(tok[j + 1], "hour") || eq(tok[j + 1], "hr"))) to24 = 0, j += 2;
    else if (j < n && eq(tok[j], "military")) to24 = 1, j++;
    else if (j < n && (eq(tok[j], "standard") || eq(tok[j], "am/pm"))) to24 = 0, j++;
    if (to24 < 0) return 0;
    if (j < n && (eq(tok[j], "time") || eq(tok[j], "format") || eq(tok[j], "clock"))) j++;
    if (j != n) return 0;
    if (to24) snprintf(out, (size_t)out_len, "%02d:%02d", t / 60, t % 60);
    else fmt_12h(out, (size_t)out_len, t);
    return 1;
}

// ---------------------------------------------------------------- dates

static const char *const MONTHS[] = {"january", "february", "march", "april", "may", "june",
                                     "july", "august", "september", "october", "november",
                                     "december"};
static const char *const MONTH_NAMES[] = {"January", "February", "March", "April", "May", "June",
                                          "July", "August", "September", "October", "November",
                                          "December"};
static const char *const WEEKDAYS[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                                       "Friday", "Saturday"};

static int month_of(const char *t) {  // "jan", "sept" or "january" -> 1
    size_t l = strlen(t);
    for (int i = 0; i < 12; i++)
        if (eq(t, MONTHS[i]) || (l >= 3 && l < strlen(MONTHS[i]) && !strncmp(t, MONTHS[i], l) &&
                                 (l == 3 || (i == 8 && l == 4))))
            return i + 1;
    return 0;
}

static int leap(long y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

static int days_in(int month, long year) {
    static const int d[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return month == 2 && leap(year) ? 29 : d[month - 1];
}

// Days since 1970-01-01 (proleptic Gregorian), and back.
static long days_from_civil(long y, int m, int d) {
    y -= m <= 2;
    long era = (y >= 0 ? y : y - 399) / 400;
    long yoe = y - era * 400;
    long doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

static void civil_from_days(long z, long *y, int *m, int *d) {
    z += 719468;
    long era = (z >= 0 ? z : z - 146096) / 146097;
    long doe = z - era * 146097;
    long yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    long mp = (5 * doy + 2) / 153;
    *d = (int)(doy - (153 * mp + 2) / 5 + 1);
    *m = (int)(mp < 10 ? mp + 3 : mp - 9);
    *y = yoe + era * 400 + (*m <= 2);
}

// "20", "20th", "20," -> 20
static int day_number(const char *t, int *d) {
    char *end;
    long v = strtol(t, &end, 10);
    if (end == t) return 0;
    if (eq(end, "st") || eq(end, "nd") || eq(end, "rd") || eq(end, "th") || eq(end, ",") || !*end) {
        *d = (int)v;
        return v >= 1 && v <= 31;
    }
    return 0;
}

static int year_number(const char *t, long *y) {
    char *end;
    long v = strtol(t, &end, 10);
    if (end == t || *end || v < -9999 || v > 9999) return 0;
    *y = v;
    return 1;
}

// "july 20 1969", "20 july 1969", "1969-07-20", or without a year ("july 20").
// *y = 0 when no year was given. Returns tokens used.
static int parse_date(char **tok, int n, int i, long *y, int *mo, int *d) {
    int yy, mm, dd;
    *y = 0;
    // ISO only: the year has 4 digits ("12-25-2024" is not year 12).
    if (i < n && strspn(tok[i], "0123456789") == 4 && sscanf(tok[i], "%4d-%2d-%2d", &yy, &mm, &dd) == 3) {
        *y = yy, *mo = mm, *d = dd;
        return 1;
    }
    if (i + 1 >= n) return 0;
    int used = 0;
    if ((*mo = month_of(tok[i])) && day_number(tok[i + 1], d)) used = 2;
    else if (day_number(tok[i], d) && (*mo = month_of(tok[i + 1]))) used = 2;
    else if (i + 2 < n && day_number(tok[i], d) && eq(tok[i + 1], "of") && (*mo = month_of(tok[i + 2]))) used = 3;
    if (!used) return 0;
    if (i + used < n && year_number(tok[i + used], y)) used++;
    return used;
}

static int valid(long y, int mo, int d) { return mo >= 1 && mo <= 12 && d >= 1 && d <= days_in(mo, y ? y : 2001); }

static int dates_question(char **tok, int n, char *out, int out_len) {
    long y;
    // "is 2024 a leap year"
    if (n >= 4 && eq(tok[0], "is") && year_number(tok[1], &y) && eq(tok[n - 1], "year") &&
        eq(tok[n - 2], "leap") && (n == 4 || (n == 5 && eq(tok[2], "a")))) {
        snprintf(out, (size_t)out_len, "%s", y > 0 && leap(y) ? "Yes." : "No.");
        return 1;
    }
    // "how many days [are] [there] in <month> [year]"
    if (n >= 5 && eq(tok[0], "how") && eq(tok[1], "many") && eq(tok[2], "days")) {
        int i = 3, mo;
        if (i < n && (eq(tok[i], "are") || eq(tok[i], "is"))) i++;
        if (i < n && eq(tok[i], "there")) i++;
        if (i + 1 < n && eq(tok[i], "in") && (mo = month_of(tok[i + 1]))) {
            i += 2;
            if (i == n) {
                if (mo == 2) snprintf(out, (size_t)out_len, "28 days, or 29 in a leap year.");
                else snprintf(out, (size_t)out_len, "%d days", days_in(mo, 2001));
                return 1;
            }
            if (i + 1 == n && year_number(tok[i], &y) && y > 0) {
                snprintf(out, (size_t)out_len, "%d days", days_in(mo, y));
                return 1;
            }
            return 0;
        }
    }
    // "[how many] days between|from <date> and|to <date>"
    for (int i = 0; i < n; i++) {
        if (!(eq(tok[i], "between") || eq(tok[i], "from"))) continue;
        int has_days = 0;
        for (int k = 0; k < i; k++) has_days |= eq(tok[k], "days");
        long y1, y2;
        int m1, d1, m2, d2;
        int l1 = parse_date(tok, n, i + 1, &y1, &m1, &d1);
        if (!l1 || !has_days) continue;
        int j = i + 1 + l1;
        if (j >= n || !(eq(tok[j], "and") || eq(tok[j], "to") || eq(tok[j], "until"))) continue;
        int l2 = parse_date(tok, n, j + 1, &y2, &m2, &d2);
        if (!l2 || j + 1 + l2 != n) continue;
        if (!y1 && !y2) y1 = y2 = 2001;  // no years: a common year
        else if (!y1) y1 = y2;
        else if (!y2) y2 = y1;
        if (!valid(y1, m1, d1) || !valid(y2, m2, d2)) break;
        long diff = days_from_civil(y2, m2, d2) - days_from_civil(y1, m1, d1);
        if (diff < 0 && y1 == 2001 && y2 == 2001) diff += 365;  // "dec 20 to jan 5": next year
        snprintf(out, (size_t)out_len, "%ld day%s", labs(diff), labs(diff) == 1 ? "" : "s");
        return 1;
    }
    // "[what date is] 30 days after|from|before <date>"
    for (int i = 1; i + 1 < n; i++) {
        double k;
        if (!(eq(tok[i], "days") || eq(tok[i], "day") || eq(tok[i], "weeks") || eq(tok[i], "week"))) continue;
        if (!num(tok[i - 1], &k) || k != floor(k) || fabs(k) > 100000) continue;
        int sign;
        if (eq(tok[i + 1], "after") || eq(tok[i + 1], "from")) sign = 1;
        else if (eq(tok[i + 1], "before")) sign = -1;
        else continue;
        long yy;
        int mo, d;
        int l = parse_date(tok, n, i + 2, &yy, &mo, &d);
        if (!l || i + 2 + l != n || !yy) continue;
        if (!valid(yy, mo, d)) break;
        long days = (long)k * (tok[i][0] == 'w' ? 7 : 1);
        long z = days_from_civil(yy, mo, d) + sign * days, ry;
        int rm, rd;
        civil_from_days(z, &ry, &rm, &rd);
        if (ry < 1 || ry > 9999) return 0;  // no year 0 or 10000 on any calendar people use
        snprintf(out, (size_t)out_len, "%s, %s %d, %ld.", WEEKDAYS[((z % 7) + 11) % 7], MONTH_NAMES[rm - 1], rd, ry);
        return 1;
    }
    // "what day [of the week] was|is|will be <date>"
    int i = 0, mo, d;
    if (n >= 3 && eq(tok[0], "what") && eq(tok[1], "day")) i = 2;
    else if (n >= 2 && (eq(tok[0], "whats") || eq(tok[0], "what's") || eq(tok[0], "which")) && eq(tok[1], "day")) i = 2;
    else return 0;
    if (i + 2 < n && eq(tok[i], "of") && eq(tok[i + 1], "the") && eq(tok[i + 2], "week")) i += 3;
    if (i < n && (eq(tok[i], "was") || eq(tok[i], "is") || eq(tok[i], "falls"))) i++;
    else if (i + 1 < n && eq(tok[i], "will") && eq(tok[i + 1], "be")) i += 2;
    if (i < n && eq(tok[i], "on")) i++;
    int used = parse_date(tok, n, i, &y, &mo, &d);
    if (!used || i + used != n || !y) return 0;
    if (!valid(y, mo, d)) {
        snprintf(out, (size_t)out_len, "That date doesn't exist.");
        return 1;
    }
    if (y < 1583 || y > 9999) {
        snprintf(out, (size_t)out_len, "Only Gregorian dates, 1583 to 9999.");
        return 1;
    }
    long z = days_from_civil(y, mo, d);
    snprintf(out, (size_t)out_len, "%s.", WEEKDAYS[((z % 7) + 11) % 7]);  // 1970-01-01 was a Thursday
    return 1;
}

// ---------------------------------------------------------------- time zones

typedef struct {
    const char *name;
    int offset;  // minutes from UTC
    int us;      // generic US zone ("eastern"): offset is standard time
} zone;

static const zone ZONES[] = {
    {"utc", 0, 0}, {"gmt", 0, 0}, {"est", -300, 0}, {"edt", -240, 0}, {"cst", -360, 0},
    {"cdt", -300, 0}, {"mst", -420, 0}, {"mdt", -360, 0}, {"pst", -480, 0}, {"pdt", -420, 0},
    {"akst", -540, 0}, {"akdt", -480, 0}, {"hst", -600, 0}, {"bst", 60, 0}, {"cet", 60, 0},
    {"cest", 120, 0}, {"eet", 120, 0}, {"eest", 180, 0}, {"msk", 180, 0}, {"ist", 330, 0},
    {"sgt", 480, 0}, {"hkt", 480, 0}, {"jst", 540, 0}, {"kst", 540, 0}, {"awst", 480, 0},
    {"aest", 600, 0}, {"aedt", 660, 0}, {"nzst", 720, 0}, {"nzdt", 780, 0},
    {"eastern", -300, 1}, {"et", -300, 1}, {"central", -360, 1}, {"ct", -360, 1},
    {"mountain", -420, 1}, {"mt", -420, 1}, {"pacific", -480, 1}, {"pt", -480, 1},
};

static const zone *find_zone(const char *t) {
    for (size_t i = 0; i < sizeof ZONES / sizeof *ZONES; i++)
        if (eq(t, ZONES[i].name)) return &ZONES[i];
    return NULL;
}

// "3pm est in pst", "what is 10:30 utc in jst", "noon eastern to pacific time".
// Generic US zones ("eastern") only convert to each other: their difference
// never changes, but their offset from UTC does with daylight saving.
static int zone_question(char **tok, int n, char *out, int out_len) {
    int i = skip_prefix(tok, n), t, ampm;
    int l = parse_time(tok, n, i, &t, &ampm, 0);
    if (!l) return 0;
    i += l;
    const zone *a = i < n ? find_zone(tok[i]) : NULL;
    if (!a) return 0;
    i++;
    if (i < n && eq(tok[i], "time")) i++;
    if (!(i < n && (eq(tok[i], "in") || eq(tok[i], "to") || eq(tok[i], "into")))) return 0;
    i++;
    const zone *b = i < n ? find_zone(tok[i]) : NULL;
    if (!b) return 0;
    i++;
    if (i < n && eq(tok[i], "time")) i++;
    if (i != n || a->us != b->us) return 0;
    int m = t + b->offset - a->offset, day = 0;
    while (m < 0) m += 1440, day--;
    while (m >= 1440) m -= 1440, day++;
    char buf[16];
    if (ampm) fmt_12h(buf, sizeof buf, m);
    else snprintf(buf, sizeof buf, "%02d:%02d", m / 60, m % 60);
    snprintf(out, (size_t)out_len, "%s%s", buf, day > 0 ? ", next day" : day < 0 ? ", previous day" : "");
    return 1;
}

// ---------------------------------------------------------------- health, math, chance

// "bmi 70 kg 175 cm", "what is my bmi 150 pounds 5'9", "bmi for 180 lbs and 6 feet"
static int bmi_question(char **tok, int n, char *out, int out_len) {
    int asks = 0;
    for (int i = 0; i < n; i++) asks |= eq(tok[i], "bmi");
    if (!asks) return 0;
    double kg = 0, m = 0, v, v2;
    for (int i = 0; i < n; i++) {
        const char *u = i + 1 < n ? tok[i + 1] : "";
        int f, in;
        if (strlen(tok[i]) <= 5 && sscanf(tok[i], "%d'%d", &f, &in) == 2 && f > 0 && f < 10 && in >= 0 && in < 12) {  // 5'9
            m = (f * 12 + in) * 0.0254;
            continue;
        }
        if (!num(tok[i], &v)) continue;
        if (eq(u, "kg") || eq(u, "kgs") || eq(u, "kilograms") || eq(u, "kilos")) kg = v;
        else if (eq(u, "lb") || eq(u, "lbs") || eq(u, "pounds")) kg = v * 0.45359237;
        else if (eq(u, "cm") || eq(u, "centimeters")) m = v / 100;
        else if (eq(u, "m") || eq(u, "meters") || eq(u, "metres")) m = v;
        else if (eq(u, "feet") || eq(u, "foot") || eq(u, "ft")) {
            m = v * 0.3048;
            if (i + 2 < n && num(tok[i + 2], &v2)) m += v2 * 0.0254;  // "5 feet 9"
        }
    }
    if (kg <= 0 || m <= 0.5 || m > 2.8) return 0;
    double bmi = kg / (m * m);
    const char *cat = bmi < 18.5 ? "underweight" : bmi < 25 ? "healthy weight" : bmi < 30 ? "overweight" : "obese";
    snprintf(out, (size_t)out_len, "BMI %.1f: %s.", bmi, cat);
    return 1;
}

// "10 factorial", "factorial of 5"
static int factorial_question(char **tok, int n, char *out, int out_len) {
    int i = skip_prefix(tok, n);
    double v;
    int ok = (i + 2 == n && num(tok[i], &v) && eq(tok[i + 1], "factorial")) ||
             (i + 3 == n && eq(tok[i], "factorial") && eq(tok[i + 1], "of") && num(tok[i + 2], &v));
    if (!ok) return 0;
    if (v < 0 || v != floor(v) || v > 170) return 0;
    double f = 1;
    for (int k = 2; k <= (int)v; k++) f *= k;
    if (v <= 20) snprintf(out, (size_t)out_len, "%.0f", f);  // exact up to 20!
    else snprintf(out, (size_t)out_len, "%.6g", f);
    return 1;
}

// ---------------------------------------------------------------- survival math

static int unit(const char *t, const char *const *names) {
    for (; *names; names++)
        if (eq(t, *names)) return 1;
    return 0;
}

static const char *const PEOPLE[] = {"people", "persons", "person", "adults", NULL};
static const char *const DAYS[] = {"days", "day", NULL};
static const char *const WEEKS[] = {"weeks", "week", NULL};
static const char *const GALLONS[] = {"gallons", "gallon", "gal", NULL};
static const char *const LITERS[] = {"liters", "liter", "litres", "litre", "l", NULL};
static const char *const SECONDS[] = {"seconds", "second", "secs", "sec", "s", NULL};

// Emergency water (Ready.gov: 1 gallon per person per day), thunder distance
// (sound covers 343 m a second), and bleach for water (CDC: 8 drops of 6%
// bleach per gallon, 2 per liter).
static int survival_question(char **tok, int n, char *out, int out_len) {
    int water = 0, amount = 0, without = 0, storm = 0, bleach = 0;
    double people = 0, days = 0, secs = 0, gallons = 0, liters = 0, v;
    for (int i = 0; i < n; i++) {
        const char *t = tok[i], *u = i + 1 < n ? tok[i + 1] : "";
        water |= eq(t, "water");
        amount |= eq(t, "much") || unit(t, GALLONS) || unit(t, LITERS);
        without |= eq(t, "without");
        storm |= eq(t, "thunder") || eq(t, "lightning");
        bleach |= eq(t, "bleach");
        int a_week = eq(t, "a") && unit(u, WEEKS);
        if (!num(t, &v) && !a_week) {
            if ((eq(t, "family") || eq(t, "group") || eq(t, "household")) && eq(u, "of") && i + 2 < n &&
                num(tok[i + 2], &v))
                people = v;
            continue;
        }
        if (a_week) v = 1;
        if (unit(u, PEOPLE)) people = v;
        else if (unit(u, DAYS)) days = v;
        else if (unit(u, WEEKS)) days = 7 * v;
        else if (unit(u, SECONDS)) secs = v;
        else if (unit(u, GALLONS)) gallons = v;
        else if (unit(u, LITERS)) liters = v;
    }
    if (storm && secs > 0 && secs <= 120) {
        double km = secs * 0.343;
        snprintf(out, (size_t)out_len, "About %.1f miles (%.1f km). Go indoors.", km / 1.609344, km);
        return 1;
    }
    if (bleach && (gallons > 0 || liters > 0) && gallons <= 1000 && liters <= 4000) {
        double drops = gallons > 0 ? 8 * gallons : 2 * liters;
        if (drops <= 64) snprintf(out, (size_t)out_len, "%.0f drops of 6%% bleach; wait 30 min.", ceil(drops));
        else snprintf(out, (size_t)out_len, "About %.1f tsp of 6%% bleach; wait 30 min.", drops / 96);
        return 1;
    }
    if (water && amount && !without && !bleach && (people > 0 || days > 0) && people <= 1000 && days <= 365) {
        double g = (people > 0 ? people : 1) * (days > 0 ? days : 1);
        char gs[24], ds[24], what[40];
        plain(gs, sizeof gs, g);
        plain(ds, sizeof ds, days);
        if (people > 0 && days > 0) snprintf(what, sizeof what, "for %s day%s", ds, days == 1 ? "" : "s");
        else snprintf(what, sizeof what, "%s", people > 0 ? "a day" : "per person");
        snprintf(out, (size_t)out_len, "%s gallon%s (%.0f L) %s.", gs, g == 1 ? "" : "s", g * 3.785, what);
        return 1;
    }
    return 0;
}

static uint32_t rng_state = 2463534242u;
void tai_seed(uint32_t seed) { rng_state = seed ? seed : 2463534242u; }

static uint32_t rnd(uint32_t n) {  // xorshift32, 0..n-1
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (uint32_t)(((uint64_t)rng_state * n) >> 32);
}

// "flip a coin", "roll a die", "roll 2 dice", "roll a d20", "random number between 1 and 10"
static int chance_question(char **tok, int n, char *out, int out_len) {
    double a, b;
    if (n >= 2 && (eq(tok[0], "flip") || eq(tok[0], "toss")) && (eq(tok[n - 1], "coin"))) {
        snprintf(out, (size_t)out_len, "%s", rnd(2) ? "Heads." : "Tails.");
        return 1;
    }
    if (n >= 2 && eq(tok[0], "roll")) {
        int count = 1, sides = 6;
        const char *d = tok[n - 1];
        if (d[0] == 'd' && d[1] >= '1' && d[1] <= '9') sides = strspn(d + 1, "0123456789") > 4 ? 0 : atoi(d + 1);
        else if (!(eq(d, "die") || eq(d, "dice"))) return 0;
        if (n >= 3 && num(tok[1], &a) && a >= 1 && a <= 6 && a == floor(a)) count = (int)a;
        if (sides < 2 || sides > 1000) return 0;
        int o = 0, total = 0;
        for (int k = 0; k < count; k++) {
            int r = 1 + (int)rnd((uint32_t)sides);
            total += r;
            if (o < out_len) o += snprintf(out + o, (size_t)(out_len - o), "%s%d", k ? (k == count - 1 ? " and " : ", ") : "", r);
        }
        if (o >= out_len) return 1;  // truncated, still terminated
        if (count > 1) snprintf(out + o, (size_t)(out_len - o), " (%d).", total);
        else snprintf(out + o, (size_t)(out_len - o), ".");
        return 1;
    }
    int asks = 0, lo = -1;
    for (int i = 0; i < n; i++) {
        asks |= eq(tok[i], "random") || (eq(tok[i], "pick") && i + 1 < n && eq(tok[i + 1], "a"));
        if ((eq(tok[i], "between") || eq(tok[i], "from")) && i + 3 < n && num(tok[i + 1], &a) &&
            (eq(tok[i + 2], "and") || eq(tok[i + 2], "to")) && num(tok[i + 3], &b))
            lo = i;
    }
    if (!asks || lo < 0 || a != floor(a) || b != floor(b) || b - a > 1e9) return 0;
    if (a > b) {
        double t = a;
        a = b;
        b = t;
    }
    snprintf(out, (size_t)out_len, "%.0f.", a + rnd((uint32_t)(b - a + 1)));
    return 1;
}

// Returns 1 and writes the answer if `norm` (number words already rewritten)
// is an everyday computable question.
int tai_everyday(const char *norm, char *out, int out_len) {
    char s[BUF], *tok[MAX_TOK];
    strncpy(s, norm, sizeof s - 1);
    s[sizeof s - 1] = 0;
    int n = split(s, tok);
    // "$" is dropped by normalization; drop currency words too
    int k = 0;
    for (int i = 0; i < n; i++)
        if (!is_currency(tok[i])) tok[k++] = tok[i];
    n = k;
    return dates_question(tok, n, out, out_len) || zone_question(tok, n, out, out_len) ||
           time_question(tok, n, out, out_len) || bmi_question(tok, n, out, out_len) ||
           factorial_question(tok, n, out, out_len) || chance_question(tok, n, out, out_len) ||
           survival_question(tok, n, out, out_len) ||
           money_question(tok, n, out, out_len);
}
