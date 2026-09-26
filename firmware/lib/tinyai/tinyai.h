// tinyai: a direct-answer language model small enough for an ESP32.
//
//   tai_model m;
//   tai_load(&m, model_blob, model_blob_len);
//   char out[64];
//   tai_ask(&m, "how much does a hummingbird weigh", out, sizeof out);
//   // out = "Between 2 and 20 grams."
//
// Pure C99, no dependencies. Builds for ESP32 (Arduino / ESP-IDF) and for a PC.
#ifndef TINYAI_H
#define TINYAI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TAI_MAGIC 0x32494154u  // "TAI2"
#define TAI_VERSION 4          // int4 weights
#define TAI_VERSION_TERNARY 5  // ternary weights (-1, 0, +1), 1.6 bits each
#define TAI_VERSION_INT2 6     // int2 weights (-1.5, -0.5, 0.5, 1.5), 2.5 bits each
#define TAI_GROUP 32           // int4 weights: one scale per 32 inputs
#define TAI_MAX_Q 64           // question chars kept, matches train/common.py
#define TAI_MAX_A 48           // answer chars, hard cap
#define TAI_EOS 0
#define TAI_SEP 1

typedef struct {        // int8 matrix, one scale per row (embeddings)
    const int8_t *q;
    const float *s;
    int rows, cols;
} tai_q8;

typedef struct {        // quantized weight matrix, fp16 (IEEE half) scales
    const uint8_t *q;   // int4: two weights per byte, low nibble first, stored +8
                        // int2: four per byte, lowest bits first, stored +2
                        // ternary: five per byte in base 3, first weight in the
                        // lowest digit, stored +1; rows padded to whole bytes
    const uint16_t *s;  // int4, int2: one per row per 32 columns; ternary: one per row
    int rows, cols, kind;  // kind: file version (TAI_VERSION, _TERNARY, _INT2)
} tai_q4;

typedef struct {
    const float *n1, *n2;
    tai_q4 wq, wk, wv, wo, w1, w2;
} tai_layer;

// Optional multi-core hook: run job(ctx, lo, hi) over [0, n), split however
// the platform likes, and return when all of it is done. The firmware sets
// one on dual-core ESP32s; without it everything runs on the calling core.
typedef void (*tai_job)(void *ctx, int lo, int hi);
typedef void (*tai_parallel_fn)(tai_job job, void *ctx, int n);

typedef struct {
    int vocab, ctx, dim, layers, heads, kv_heads, hidden;
    tai_q8 tok, pos;
    tai_layer *layer;
    const float *norm;
    int n_facts;
    // Gate index (see train/gate_index.py): a sorted dictionary of content
    // words, and every accepted phrasing, grouped by fact (fact_count per
    // fact), with its question type (bit 7: every word must match; bit 6:
    // one extra question word is allowed), length and content-word ids.
    int n_words, n_known;
    const char *words;
    const uint32_t *word_off;
    const uint8_t *word_len;
    const uint8_t *fact_count;
    const uint8_t *known_qtype, *known_nwords, *known_len;
    const uint16_t *known_words;
    const char *small_talk;  // text of phrasings with no content words
    // Each fact's canonical question (the model's prompt), coded with the
    // gate's dictionary plus a few extra words (see tai_fact).
    int n_extra;
    const uint16_t *extra_off;
    const char *extras;
    const uint8_t *keys;
    // Errata: the few facts the int4 model answers wrong, stored as text
    // (found by train/export.py running this engine on every fact).
    int n_errata;
    const uint16_t *errata_fact;  // sorted fact indices
    const char *errata;           // their answers, in the same order
    // runtime state (heap)
    float *x, *xb, *hb, *q, *k, *v, *att, *logits;
    int8_t *xq;
    int32_t *xsum;       // per-group sums of xq, for the int4 kernel
    float xs;
    int8_t **kc, **vc;   // [layer][ctx * kv_dim]
    float **ks, **vs;    // [layer][ctx * kv_heads]
    tai_parallel_fn parallel;
} tai_model;

// Borrow `blob` (must stay valid and 4-byte aligned). Returns 0 on success.
int tai_load(tai_model *m, const uint8_t *blob, size_t len);
void tai_free(tai_model *m);
// Split matmuls across cores (see tai_parallel_fn). NULL = single core.
void tai_set_parallel(tai_model *m, tai_parallel_fn fn);

// Full pipeline: calculator -> knowledge gate -> model. Always writes a
// short, direct answer to `out`. Returns 0 = model, 1 = calculator,
// 2 = refused (unknown topic).
int tai_ask(tai_model *m, const char *question, char *out, int out_len);

// The stored answer for fact i (0-based) if it is on the errata list, else
// NULL. "" = a fact the model gets wrong that is refused instead.
const char *tai_errata(const tai_model *m, int fact);

// Model only, no gate: generate from `question` as typed. For experiments.
void tai_generate(tai_model *m, const char *question, char *out, int out_len);

// Building blocks (exposed for tests).
int tai_normalize(const char *in, char *out, int out_len);
int tai_calc(const char *normalized, char *out, int out_len);     // arithmetic + tai_convert
int tai_convert(const char *normalized, char *out, int out_len);  // units, bases, primes
int tai_everyday(const char *normalized, char *out, int out_len); // money, %, times, dates
void tai_number_words(const char *in, char *out, int out_len);    // "twelve" -> "12"
// Seed for "flip a coin" / "roll a die" / "random number" (e.g. esp_random()).
void tai_seed(uint32_t seed);
// 0 = unknown topic, else 1 + index of the matched fact.
int tai_gate(const tai_model *m, const char *normalized);
// "<qtype code> <content words>" as the gate parses `normalized` (for tests).
int tai_gate_words(const char *normalized, char *out, int out_len);
// Canonical question of fact `i` (0-based), decoded into out; returns its length.
int tai_fact(const tai_model *m, int i, char *out, int out_len);

#ifdef __cplusplus
}
#endif
#endif
