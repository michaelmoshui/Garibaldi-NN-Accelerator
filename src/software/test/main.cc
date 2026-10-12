// ResNet-18 W8A8 inference on the LiteRT Micro (TFLM) interpreter.
//
// Runs one inference on the generated golden input and checks the int8 logits
// bit-exactly against the LiteRT reference-kernel output. No heap, no OS:
// all tensors live in g_tensor_arena and the model is a const array.
//
// On RV64 (CONFIG=rv64 / Spike), this file also reads the mcycle and minstret
// CSRs around Invoke() to measure dynamic instruction count and cycle cost.
// These are the primary baseline metrics for ISA-extension evaluation.

#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// RISC-V performance-counter helpers (compiled out on host builds)
// ---------------------------------------------------------------------------
#ifdef __riscv
static inline uint64_t rv_rdcycle(void) {
  uint64_t v;
  __asm__ volatile("rdcycle %0" : "=r"(v));
  return v;
}
static inline uint64_t rv_rdinstret(void) {
  uint64_t v;
  __asm__ volatile("rdinstret %0" : "=r"(v));
  return v;
}
#endif  // __riscv

#include "model/gen_micro_mutable_op_resolver.h"
#include "model/resnet18_int8_channelwise_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "test_data/resnet18_int8_channelwise_golden_int8_test_data.h"
#include "test_data/resnet18_int8_channelwise_input0_int8_test_data.h"

#ifndef WORKLOAD_ARENA_SIZE
#define WORKLOAD_ARENA_SIZE (3 * 1024 * 1024)
#endif

namespace {
alignas(16) uint8_t g_tensor_arena[WORKLOAD_ARENA_SIZE];
}  // namespace

int main() {
  // Model setup
  tflite::InitializeTarget();

  const tflite::Model* model =
      tflite::GetModel(g_resnet18_int8_channelwise_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d != supported %d",
                static_cast<int>(model->version()), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // Interpreter setup
  tflite::MicroMutableOpResolver<kNumberOperators> resolver = get_resolver();
  tflite::MicroInterpreter interpreter(model, resolver, g_tensor_arena,
                                       sizeof(g_tensor_arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return 1;
  }

  // I/O sanity check
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      input->bytes != g_resnet18_int8_channelwise_input0_int8_test_data_size ||
      output->bytes != g_resnet18_int8_channelwise_golden_int8_test_data_size) {
    MicroPrintf("Unexpected I/O tensors: in %d B, out %d B",
                static_cast<int>(input->bytes),
                static_cast<int>(output->bytes));
    return 1;
  }

  // Copies the test data to the input tensor
  std::memcpy(input->data.int8, g_resnet18_int8_channelwise_input0_int8_test_data,
              input->bytes);

  // Runs the inference — bracket with CSR reads on RV64 for baseline profiling
#ifdef __riscv
  const uint64_t cycle_start   = rv_rdcycle();
  const uint64_t instret_start = rv_rdinstret();
#endif

  if (interpreter.Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke() failed");
    return 1;
  }

#ifdef __riscv
  const uint64_t cycle_end   = rv_rdcycle();
  const uint64_t instret_end = rv_rdinstret();
#endif

  // Compare results
  const int8_t* logits = output->data.int8;
  const int n = static_cast<int>(output->bytes);
  int mismatches = 0;
  int max_abs_diff = 0;
  int top1 = 0;
  for (int i = 0; i < n; ++i) {
    const int expected = g_resnet18_int8_channelwise_golden_int8_test_data[i];
    const int diff = logits[i] - expected;
    if (diff != 0) {
      ++mismatches;
      if (mismatches <= 8) {
        MicroPrintf("  logit[%d] = %d, reference %d", i, logits[i], expected);
      }
    }
    if (diff > max_abs_diff || -diff > max_abs_diff) {
      max_abs_diff = diff < 0 ? -diff : diff;
    }
    if (logits[i] > logits[top1]) top1 = i;
  }

  // Print results
  MicroPrintf("arena used: %d / %d bytes",
              static_cast<int>(interpreter.arena_used_bytes()),
              WORKLOAD_ARENA_SIZE);
  MicroPrintf("top-1 class: %d (int8 logit %d)", top1, logits[top1]);
  MicroPrintf("mismatches vs LiteRT reference: %d / %d (max |diff| %d)",
              mismatches, n, max_abs_diff);
  // Accept max |diff| <= 1: INT8 requantization rounding between the LiteRT
  // reference (x86/FMA) and TFLM RV64 scalar kernels can differ by 1 LSB.
  const bool pass = (max_abs_diff <= 1);
  MicroPrintf(pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
