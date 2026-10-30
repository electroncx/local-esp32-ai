// Offline survival and everyday answers on an ESP32. Type a question over serial:
//
//   you: how do i purify water
//   esp: Boil it for 1 minute (3 above 5,000 ft).
#include <Arduino.h>

#include "model_data.h"
#include "tinyai.h"

#define SHOW_TIMING 0  // 1 = print how long each answer took

static tai_model model;
static char line[256];
static size_t line_len = 0;

#if !CONFIG_FREERTOS_UNICORE
// Dual-core ESP32 / ESP32-S3: every matmul is split in half and the second
// half runs on the other core. Rows are independent, so the answers are
// bit-identical to single-core; only the time changes.
static TaskHandle_t caller, worker;
static volatile tai_job job;
static void *volatile job_ctx;
static volatile int job_lo, job_hi;

static void worker_task(void *) {
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        job(job_ctx, job_lo, job_hi);
        xTaskNotifyGive(caller);
    }
}

static void run_on_both_cores(tai_job fn, void *ctx, int n) {
    int mid = n / 2;
    caller = xTaskGetCurrentTaskHandle();
    job = fn;
    job_ctx = ctx;
    job_lo = mid;
    job_hi = n;
    xTaskNotifyGive(worker);
    fn(ctx, 0, mid);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}
#endif

void setup() {
    Serial.begin(115200);
    delay(500);
    int rc = tai_load(&model, model_data, model_data_len);
    if (rc) {
        Serial.printf("model load failed (%d)\n", rc);
        for (;;) delay(1000);
    }
    tai_seed(esp_random());  // hardware RNG, for coin flips and dice
    int cores = 1;
#if !CONFIG_FREERTOS_UNICORE
    // the loop task runs on ARDUINO_RUNNING_CORE; put the worker on the other
    if (xTaskCreatePinnedToCore(worker_task, "tinyai", 4096, NULL, 2, &worker,
                                1 - xPortGetCoreID()) == pdPASS) {
        tai_set_parallel(&model, run_on_both_cores);
        cores = 2;
    }
#endif
    // Spare RAM makes answers faster: the question's characters go through
    // the model in blocks, reading each weight row from flash once per block
    // (same answers, bit for bit). Keep RESERVE free for everything else; the
    // block's biggest single buffer (dim x 4 bytes per position) must also
    // fit in the heap's largest free piece.
    const size_t RESERVE = 48 * 1024;
    size_t heap = ESP.getFreeHeap(), largest = ESP.getMaxAllocHeap();
    size_t per = tai_block_bytes(&model), widest = (size_t)model.dim * 4;
    size_t want = heap > RESERVE + per ? (heap - RESERVE) / per : 1;
    if (want > largest / widest) want = largest / widest;
    int block = tai_set_block(&model, want < 1 ? 1 : (int)want);
    if (!block) {
        Serial.println("out of memory");
        for (;;) delay(1000);
    }
    Serial.printf("\ntinyai ready: %d facts, %u KB model, %d cores, %d-character blocks, %u KB heap free\n",
                  model.n_facts, model_data_len / 1024, cores, block, (unsigned)(ESP.getFreeHeap() / 1024));
    Serial.println("Offline survival and everyday answers. It might make a mistake:");
    Serial.println("double-check anything important. In an emergency, call for help first.");
    Serial.print("you: ");
}

void loop() {
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\r' || c == '\n') {
            if (!line_len) continue;
            line[line_len] = 0;
            line_len = 0;
            char out[64];
            uint32_t t0 = millis();
            tai_ask(&model, line, out, sizeof out);
            uint32_t ms = millis() - t0;
            Serial.printf("esp: %s\n", out);
            if (SHOW_TIMING) Serial.printf("     (%u ms)\n", (unsigned)ms);
            Serial.print("you: ");
        } else if (line_len < sizeof line - 1) {
            line[line_len++] = c;
        } else {  // too long: keep the end, like the engine does (the question is usually last)
            memmove(line, line + 1, sizeof line - 2);
            line[sizeof line - 2] = c;
        }
    }
}
