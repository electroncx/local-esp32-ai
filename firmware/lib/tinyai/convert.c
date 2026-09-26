// Facts that are cheaper to compute than to memorize: unit conversions,
// number bases, Roman numerals, primes. A few KB of code here answers more
// questions than the whole model can store.
//
//   "10 km in miles"            -> "6.21371 miles"
//   "how many feet in a mile"   -> "5,280 feet"
//   "100 f in celsius"          -> "37.7778 C"
//   "255 in hex"                -> "FF"
//   "1994 in roman numerals"    -> "MCMXCIV"
//   "is 91 prime"               -> "No, 91 = 7 x 13."
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyai.h"

#define MAX_TOK 24

enum { LEN, MASS, VOL, TIME, SPEED, AREA, ENERGY, POWER, PRESSURE, TEMP };

typedef struct {
    const char *names;  // '|'-separated spellings, may contain spaces
    int dim;
    double factor, offset;  // si = value * factor + offset
    const char *one, *many;
} unit;

static const unit UNITS[] = {
    {"mm|millimeter|millimeters|millimetre|millimetres", LEN, 0.001, 0, "millimeter", "millimeters"},
    {"cm|centimeter|centimeters|centimetre|centimetres", LEN, 0.01, 0, "centimeter", "centimeters"},
    {"m|meter|meters|metre|metres", LEN, 1, 0, "meter", "meters"},
    {"km|kms|kilometer|kilometers|kilometre|kilometres", LEN, 1000, 0, "kilometer", "kilometers"},
    {"in|inch|inches", LEN, 0.0254, 0, "inch", "inches"},
    {"ft|foot|feet", LEN, 0.3048, 0, "foot", "feet"},
    {"yd|yard|yards", LEN, 0.9144, 0, "yard", "yards"},
    {"mi|mile|miles", LEN, 1609.344, 0, "mile", "miles"},
    {"nautical mile|nautical miles|nmi", LEN, 1852, 0, "nautical mile", "nautical miles"},
    {"light year|light years", LEN, 9.4607304725808e15, 0, "light year", "light years"},
    {"mg|milligram|milligrams", MASS, 1e-6, 0, "milligram", "milligrams"},
    {"g|gram|grams|gramme|grammes", MASS, 1e-3, 0, "gram", "grams"},
    {"kg|kgs|kilo|kilos|kilogram|kilograms", MASS, 1, 0, "kilogram", "kilograms"},
    {"lb|lbs|pound|pounds", MASS, 0.45359237, 0, "pound", "pounds"},
    {"oz|ounce|ounces", MASS, 0.028349523125, 0, "ounce", "ounces"},
    {"stone|stones", MASS, 6.35029318, 0, "stone", "stone"},
    {"short ton|short tons|us ton|us tons", MASS, 907.18474, 0, "US ton", "US tons"},
    {"tonne|tonnes|metric ton|metric tons", MASS, 1000, 0, "tonne", "tonnes"},
    {"ml|milliliter|milliliters|millilitre|millilitres", VOL, 0.001, 0, "milliliter", "milliliters"},
    {"l|liter|liters|litre|litres", VOL, 1, 0, "liter", "liters"},
    {"gallon|gallons|gal", VOL, 3.785411784, 0, "US gallon", "US gallons"},
    {"quart|quarts|qt", VOL, 0.946352946, 0, "quart", "quarts"},
    {"pint|pints|pt", VOL, 0.473176473, 0, "pint", "pints"},
    {"cup|cups", VOL, 0.2365882365, 0, "cup", "cups"},
    {"fluid ounce|fluid ounces|fl oz", VOL, 0.0295735295625, 0, "fluid ounce", "fluid ounces"},
    {"tablespoon|tablespoons|tbsp", VOL, 0.01478676478125, 0, "tablespoon", "tablespoons"},
    {"teaspoon|teaspoons|tsp", VOL, 0.00492892159375, 0, "teaspoon", "teaspoons"},
    {"ms|millisecond|milliseconds", TIME, 0.001, 0, "millisecond", "milliseconds"},
    {"s|sec|secs|second|seconds", TIME, 1, 0, "second", "seconds"},
    {"min|mins|minute|minutes", TIME, 60, 0, "minute", "minutes"},
    {"h|hr|hrs|hour|hours", TIME, 3600, 0, "hour", "hours"},
    {"day|days", TIME, 86400, 0, "day", "days"},
    {"week|weeks", TIME, 604800, 0, "week", "weeks"},
    {"km/h|kph|kmh|kilometers per hour|kilometres per hour", SPEED, 1 / 3.6, 0, "km/h", "km/h"},
    {"mph|miles per hour", SPEED, 0.44704, 0, "mph", "mph"},
    {"m/s|meters per second|metres per second", SPEED, 1, 0, "m/s", "m/s"},
    {"knot|knots|kn", SPEED, 1852.0 / 3600, 0, "knot", "knots"},
    {"m2|sq m|square meter|square meters|square metre|square metres", AREA, 1, 0,
     "square meter", "square meters"},
    {"km2|sq km|square kilometer|square kilometers|square kilometre|square kilometres", AREA,
     1e6, 0, "square kilometer", "square kilometers"},
    {"ft2|sq ft|square foot|square feet", AREA, 0.09290304, 0, "square foot", "square feet"},
    {"sq mi|square mile|square miles", AREA, 2589988.110336, 0, "square mile", "square miles"},
    {"acre|acres", AREA, 4046.8564224, 0, "acre", "acres"},
    {"ha|hectare|hectares", AREA, 10000, 0, "hectare", "hectares"},
    {"j|joule|joules", ENERGY, 1, 0, "joule", "joules"},
    {"kj|kilojoule|kilojoules", ENERGY, 1000, 0, "kilojoule", "kilojoules"},
    {"cal|calorie|calories", ENERGY, 4.184, 0, "calorie", "calories"},
    {"kcal|kilocalorie|kilocalories", ENERGY, 4184, 0, "kilocalorie", "kilocalories"},
    {"wh|watt hour|watt hours", ENERGY, 3600, 0, "watt hour", "watt hours"},
    {"kwh|kilowatt hour|kilowatt hours", ENERGY, 3.6e6, 0, "kilowatt hour", "kilowatt hours"},
    {"w|watt|watts", POWER, 1, 0, "watt", "watts"},
    {"kw|kilowatt|kilowatts", POWER, 1000, 0, "kilowatt", "kilowatts"},
    {"hp|horsepower", POWER, 745.69987158227022, 0, "horsepower", "horsepower"},
    {"pa|pascal|pascals", PRESSURE, 1, 0, "pascal", "pascals"},
    {"hpa|hectopascal|hectopascals", PRESSURE, 100, 0, "hectopascal", "hectopascals"},
    {"kpa|kilopascal|kilopascals", PRESSURE, 1000, 0, "kilopascal", "kilopascals"},
    {"bar|bars", PRESSURE, 1e5, 0, "bar", "bar"},
    {"mbar|millibar|millibars", PRESSURE, 100, 0, "millibar", "millibars"},
    {"psi", PRESSURE, 6894.757293168361, 0, "psi", "psi"},
    {"atm|atmosphere|atmospheres", PRESSURE, 101325, 0, "atmosphere", "atmospheres"},
    {"mmhg|torr", PRESSURE, 133.322387415, 0, "mmHg", "mmHg"},
    {"c|celsius|centigrade|degrees c|degrees celsius|degree celsius", TEMP, 1, 273.15, "C", "C"},
    {"f|fahrenheit|degrees f|degrees fahrenheit|degree fahrenheit", TEMP, 5.0 / 9,
     273.15 - 32 * 5.0 / 9, "F", "F"},
    {"k|kelvin|kelvins", TEMP, 1, 0, "K", "K"},
};

// ---------------------------------------------------------------- tokens

static int split(char *s, char *tok[MAX_TOK]) {
    int n = 0;
    for (char *p = strtok(s, " "); p && n < MAX_TOK; p = strtok(NULL, " ")) tok[n++] = p;
    return n;
}

// Length in tokens of `phrase` if tok[i..] starts with it, else 0.
static int match_phrase(char **tok, int n, int i, const char *phrase, size_t len) {
    int used = 0;
    const char *p = phrase, *end = phrase + len;
    while (p < end) {
        const char *sp = memchr(p, ' ', (size_t)(end - p));
        size_t wl = sp ? (size_t)(sp - p) : (size_t)(end - p);
        if (i + used >= n || strlen(tok[i + used]) != wl || strncmp(tok[i + used], p, wl)) return 0;
        used++;
        p += wl + (sp ? 1 : 0);
    }
    return used;
}

// Longest unit spelling at tok[i]; returns tokens used, 0 if none.
static int find_unit(char **tok, int n, int i, const unit **out) {
    int best = 0;
    for (size_t u = 0; u < sizeof UNITS / sizeof *UNITS; u++) {
        const char *p = UNITS[u].names;
        while (*p) {
            const char *bar = strchr(p, '|');
            size_t len = bar ? (size_t)(bar - p) : strlen(p);
            int used = match_phrase(tok, n, i, p, len);
            if (used > best) {
                best = used;
                *out = &UNITS[u];
            }
            p += len + (bar ? 1 : 0);
        }
    }
    return best;
}

static int parse_number(const char *t, double *v) {
    if (!strcmp(t, "a") || !strcmp(t, "an") || !strcmp(t, "one")) {
        *v = 1;
        return 1;
    }
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
    return *end == 0;
}

// 5280 -> "5,280", 1609.344 -> "1,609.34", 0.6213712 -> "0.621371".
// Six significant digits, thousands separators, no exponent in normal ranges.
static void fmt_num(char *out, size_t len, double v) {
    char raw[48];
    double a = fabs(v);
    if (a >= 1e15 || (a > 0 && a < 1e-4)) {
        snprintf(out, len, "%.6g", v);
        return;
    }
    if (fabs(v - llround(v)) < 1e-9 * (a > 1 ? a : 1)) {
        snprintf(raw, sizeof raw, "%lld", (long long)llround(v));
    } else {
        int decimals = 5 - (int)floor(log10(a));
        if (decimals < 0) decimals = 0;
        snprintf(raw, sizeof raw, "%.*f", decimals, v);
        char *e = raw + strlen(raw) - 1;  // trim trailing zeros
        while (*e == '0') *e-- = 0;
        if (*e == '.') *e = 0;
    }
    const char *digits = raw + (raw[0] == '-');
    size_t int_len = strcspn(digits, "."), o = 0;
    if (raw[0] == '-') out[o++] = '-';
    for (size_t i = 0; i < int_len && o + 2 < len; i++) {
        if (i && (int_len - i) % 3 == 0) out[o++] = ',';
        out[o++] = digits[i];
    }
    for (const char *c = digits + int_len; *c && o + 1 < len; c++) out[o++] = *c;
    out[o] = 0;
}

// ---------------------------------------------------------------- units

static int convert(double qty, const unit *from, const unit *to, char *out, int out_len) {
    if (from->dim != to->dim || from == to) return 0;
    double v = ((qty * from->factor + from->offset) - to->offset) / to->factor;
    char num[48];
    fmt_num(num, sizeof num, v);
    snprintf(out, (size_t)out_len, "%s %s", num, strcmp(num, "1") ? to->many : to->one);
    return 1;
}

// Skips "what is", "whats", "convert", "how much is"; returns the next index.
static int skip_prefix(char **tok, int n) {
    if (n > 0 && (!strcmp(tok[0], "convert") || !strcmp(tok[0], "whats") ||
                  !strcmp(tok[0], "what's") || !strcmp(tok[0], "calculate")))
        return 1;
    if (n > 1 && !strcmp(tok[0], "what") && !strcmp(tok[1], "is")) return 2;
    if (n > 2 && !strcmp(tok[0], "how") && !strcmp(tok[1], "much") && !strcmp(tok[2], "is")) return 3;
    return 0;
}

static int is_prep(const char *t) { return !strcmp(t, "in") || !strcmp(t, "to") || !strcmp(t, "into"); }

static int units_question(char **tok, int n, char *out, int out_len) {
    int i;
    const unit *a, *b;
    double qty;
    // "how many <unit> [are|is] [there] in [qty] <unit>", "how many <unit> is <qty> <unit>"
    if (n >= 4 && !strcmp(tok[0], "how") && !strcmp(tok[1], "many")) {
        int ub = find_unit(tok, n, 2, &b);
        if (!ub) return 0;
        i = 2 + ub;
        int verb = i < n && (!strcmp(tok[i], "are") || !strcmp(tok[i], "is"));
        i += verb;
        if (i < n && !strcmp(tok[i], "there")) i++;
        if (i < n && (!strcmp(tok[i], "in") || !strcmp(tok[i], "per"))) i++;
        else if (!verb) return 0;  // "how many pounds is 70 kg" needs no "in"
        qty = 1;
        if (i < n && parse_number(tok[i], &qty)) i++;
        int ua = find_unit(tok, n, i, &a);
        if (!ua || i + ua != n) return 0;
        return convert(qty, a, b, out, out_len);
    }
    // "[convert|what is|how much is] <qty> <unit> in|to|into <unit>"
    i = skip_prefix(tok, n);
    if (i >= n || !parse_number(tok[i], &qty)) return 0;
    i++;
    int ua = find_unit(tok, n, i, &a);
    if (!ua) return 0;
    i += ua;
    if (i < n && is_prep(tok[i])) i++;
    else return 0;
    int ub = find_unit(tok, n, i, &b);
    if (!ub || i + ub != n) return 0;
    return convert(qty, a, b, out, out_len);
}

// ---------------------------------------------------------------- numbers

static int parse_int(const char *t, long long *v) {
    double d;
    if (!parse_number(t, &d) || d != floor(d) || fabs(d) > 9e15) return 0;
    *v = (long long)d;
    return 1;
}

static void to_base(unsigned long long v, int base, char *out, int out_len) {
    char tmp[72];
    int n = 0;
    do {
        tmp[n++] = "0123456789ABCDEF"[v % (unsigned)base];
        v /= (unsigned)base;
    } while (v && n < (int)sizeof tmp);
    int o = 0;
    while (n && o < out_len - 1) out[o++] = tmp[--n];
    out[o] = 0;
}

static void to_roman(int v, char *out) {
    static const int val[] = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    static const char *const sym[] = {"M", "CM", "D", "CD", "C", "XC", "L", "XL",
                                      "X", "IX", "V", "IV", "I"};
    out[0] = 0;
    for (int i = 0; i < 13; i++)
        while (v >= val[i]) {
            strcat(out, sym[i]);
            v -= val[i];
        }
}

static int from_roman(const char *s, int *v) {
    int total = 0, prev = 0;
    for (int i = (int)strlen(s) - 1; i >= 0; i--) {
        int d;
        switch (s[i]) {
            case 'i': d = 1; break;
            case 'v': d = 5; break;
            case 'x': d = 10; break;
            case 'l': d = 50; break;
            case 'c': d = 100; break;
            case 'd': d = 500; break;
            case 'm': d = 1000; break;
            default: return 0;
        }
        total += d < prev ? -d : d;
        if (d > prev) prev = d;
    }
    // accept only the canonical spelling
    char back[32];
    if (total <= 0 || total > 3999) return 0;
    to_roman(total, back);
    for (int i = 0; s[i]; i++)
        if (back[i] != s[i] - 'a' + 'A') return 0;
    *v = total;
    return 1;
}

static long long smallest_factor(long long n) {
    if (n % 2 == 0) return 2;
    for (long long f = 3; f * f <= n; f += 2)
        if (n % f == 0) return f;
    return n;
}

static int numbers_question(char **tok, int n, char *out, int out_len) {
    long long v;
    // "is 97 prime", "is 97 a prime number", "is 12 even"
    if (n >= 3 && !strcmp(tok[0], "is") && parse_int(tok[1], &v)) {
        int j = 2;
        if (!strcmp(tok[j], "a") || !strcmp(tok[j], "an")) j++;
        if (j + 1 == n && (!strcmp(tok[j], "even") || !strcmp(tok[j], "odd"))) {
            snprintf(out, (size_t)out_len, "%s", (tok[j][0] == 'e') == (v % 2 == 0) ? "Yes." : "No.");
            return 1;
        }
        int prime_q = j < n && !strcmp(tok[j], "prime") &&
                      (j + 1 == n || (j + 2 == n && !strcmp(tok[j + 1], "number")));
        if (!prime_q || v > 1000000000000LL) return 0;  // bounded trial division
        long long f = v < 2 ? 0 : smallest_factor(v);
        if (v < 2) snprintf(out, (size_t)out_len, "No.");
        else if (f == v) snprintf(out, (size_t)out_len, "Yes.");
        else snprintf(out, (size_t)out_len, "No, %lld = %lld x %lld.", v, f, v / f);
        return 1;
    }

    // "<n> in binary|hex|hexadecimal|octal|roman [numerals]", "<roman> in numbers"
    int i = skip_prefix(tok, n);
    if (i + 2 >= n || !is_prep(tok[i + 1])) return 0;
    const char *target = tok[i + 2];
    int words = n - (i + 2);  // 1, or 2 for "roman numerals"
    int r;
    if (!parse_int(tok[i], &v)) {
        if (words == 1 && from_roman(tok[i], &r) &&
            (!strcmp(target, "numbers") || !strcmp(target, "number") ||
             !strcmp(target, "decimal") || !strcmp(target, "arabic") || !strcmp(target, "digits"))) {
            snprintf(out, (size_t)out_len, "%d", r);
            return 1;
        }
        return 0;
    }
    if (v < 0) return 0;
    if (!strcmp(target, "roman") &&
        (words == 1 || (words == 2 && (!strcmp(tok[n - 1], "numerals") || !strcmp(tok[n - 1], "numeral"))))) {
        if (v < 1 || v > 3999) snprintf(out, (size_t)out_len, "Roman numerals only go from 1 to 3999.");
        else {
            char r[16];  // longest is MMMDCCCLXXXVIII, 15 letters
            to_roman((int)v, r);
            snprintf(out, (size_t)out_len, "%s", r);
        }
        return 1;
    }
    if (words != 1) return 0;
    int base = !strcmp(target, "binary") ? 2
             : (!strcmp(target, "hex") || !strcmp(target, "hexadecimal")) ? 16
             : !strcmp(target, "octal") ? 8 : 0;
    if (!base) return 0;
    to_base((unsigned long long)v, base, out, out_len);
    return 1;
}

// Returns 1 and writes the answer if `norm` is a computable question.
int tai_convert(const char *norm, char *out, int out_len) {
    char s[128], *tok[MAX_TOK];
    strncpy(s, norm, sizeof s - 1);
    s[sizeof s - 1] = 0;
    int n = split(s, tok);
    return numbers_question(tok, n, out, out_len) || units_question(tok, n, out, out_len);
}
