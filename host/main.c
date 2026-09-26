// Runs the exact firmware engine on a PC.
//
//   make -C host
//   ./host/tinyai firmware/data/model.bin               # chat
//   echo "how fast is a cheetah" | ./host/tinyai firmware/data/model.bin
//   ./host/tinyai firmware/data/model.bin --raw < qs    # skip the gate
//   ./host/tinyai firmware/data/model.bin --why         # show the gate match
//   ./host/tinyai firmware/data/model.bin --route       # 0 model, 1 computed, 2 refused
//   ./host/tinyai firmware/data/model.bin --words       # how the gate parses each line
//   ./host/tinyai firmware/data/model.bin --gate        # routing only: computed answer, fact key, or "-"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __unix__
#include <unistd.h>
#endif

#include "tinyai.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.bin [--raw] [--time] [--why] [--route] [--words] [--gate]\n", argv[0]);
        return 1;
    }
    int raw = 0, timing = 0, why = 0, route = 0, words = 0, gate = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--raw")) raw = 1;
        if (!strcmp(argv[i], "--time")) timing = 1;
        if (!strcmp(argv[i], "--why")) why = 1;
        if (!strcmp(argv[i], "--route")) route = 1;
        if (!strcmp(argv[i], "--words")) words = 1;
        if (!strcmp(argv[i], "--gate")) gate = 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = (uint8_t *)malloc((size_t)n);  // malloc is 4-byte aligned
    if (!blob || fread(blob, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "read failed\n");
        return 1;
    }
    fclose(f);

    tai_model m;
    int rc = tai_load(&m, blob, (size_t)n);
    tai_seed((uint32_t)time(NULL));
    if (rc) {
        fprintf(stderr, "load failed: %d\n", rc);
        return 1;
    }
    int tty = 0;
#ifdef __unix__
    tty = isatty(0);
#endif
    char line[512], out[128];
    for (;;) {
        if (tty) {
            printf("you: ");
            fflush(stdout);
        }
        if (!fgets(line, sizeof line, stdin)) break;
        // A line longer than the buffer: keep its end, as the engine keeps
        // the end of a long question, instead of splitting it in two.
        for (size_t n = strlen(line); n && line[n - 1] != '\n';) {
            int c = getchar();
            if (c == EOF) break;
            memmove(line, line + 1, n - 1);
            line[n - 1] = (char)c;
        }
        line[strcspn(line, "\r\n")] = 0;
        if (words) {  // the line is already normalized (as stored in the index)
            char w[512];
            tai_gate_words(line, w, sizeof w);
            printf("%s\n", w);
            continue;
        }
        if (gate) {  // no model: what would answer this line?
            char norm[TAI_MAX_Q + 1];
            tai_normalize(line, norm, sizeof norm);
            if (!norm[0]) printf("-\n");
            else if (tai_calc(norm, out, sizeof out)) printf(strlen(out) > TAI_MAX_A ? "-\n" : "=%s\n", out);
            else {
                int g = tai_gate(&m, norm);
                char key[TAI_MAX_Q + 1] = "-";
                // a fact refused by design (empty errata) has an empty key
                if (g && !tai_fact(&m, g - 1, key, sizeof key)) strcpy(key, "-");
                printf("%s\n", key);
            }
            continue;
        }
        clock_t t0 = clock();
        int r = 0;
        if (raw) tai_generate(&m, line, out, sizeof out);
        else r = tai_ask(&m, line, out, sizeof out);
        double ms = 1000.0 * (double)(clock() - t0) / CLOCKS_PER_SEC;
        if (tty) printf("esp: ");
        printf("%s", out);
        if (timing) printf("\t%.1f ms", ms);
        if (route) printf("\t%d", r);
        if (why) {
            char norm[TAI_MAX_Q + 1];
            tai_normalize(line, norm, sizeof norm);
            int g = tai_gate(&m, norm);
            char key[TAI_MAX_Q + 1] = "-";
            if (g && !tai_fact(&m, g - 1, key, sizeof key)) strcpy(key, "-");
            printf("\t[%s] -> [%s]", norm, key);
        }
        printf("\n");
        fflush(stdout);
    }
    tai_free(&m);
    free(blob);
    return 0;
}
