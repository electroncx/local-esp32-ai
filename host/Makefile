LIB := ../firmware/lib/tinyai
# -ffp-contract=off (always): the same floats as the ESP32 build, bit for bit
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -pedantic

tinyai: main.c $(LIB)/tinyai.c $(LIB)/gate.c $(LIB)/calc.c $(LIB)/convert.c $(LIB)/everyday.c $(LIB)/tinyai.h
	$(CC) $(CFLAGS) -ffp-contract=off -D_POSIX_C_SOURCE=200809L -I$(LIB) -o $@ main.c $(LIB)/tinyai.c $(LIB)/gate.c $(LIB)/calc.c $(LIB)/convert.c $(LIB)/everyday.c -lm

clean:
	rm -f tinyai

.PHONY: clean
