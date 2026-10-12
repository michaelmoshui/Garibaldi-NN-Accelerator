// CV-Wally replacement for tflm/.../riscv32_generic/debug_log.cc. The
// testbench has no UART, so MicroPrintf output is dropped before formatting.
// Error strings stay compiled in (no TF_LITE_STRIP_ERROR_STRINGS) because that
// define also removes the per-op ScopedMicroProfiler that main.cc relies on.

#include "tensorflow/lite/micro/debug_log.h"

#include "eyalroz_printf/src/printf/printf.h"

extern "C" void DebugLog(const char* /*format*/, va_list /*args*/) {}

extern "C" int DebugVsnprintf(char* buffer, size_t buf_size, const char* format,
                              va_list vlist) {
  return vsnprintf_(buffer, buf_size, format, vlist);
}
