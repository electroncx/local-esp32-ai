// The model file, embedded in flash: by embed_model.py with PlatformIO, or as
// the generated model_data.c in the Arduino sketch (make_arduino.py).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
extern const uint8_t model_data[];
extern const uint32_t model_data_size;
#ifdef __cplusplus
}
#endif

#define model_data_len ((unsigned)model_data_size)
