// dev_net W8A8 inference on the LiteRT Micro (TFLM) interpreter: the
// kernel-development workload.
//
// Runs one inference on the generated input with every intermediate tensor
// preserved, then checks each op's output against the LiteRT reference-kernel
// output in test_data/, so a kernel bug shows up at the op that introduced it.
// Ops listed in CUSTOM_KERNELS (Makefile) run the kernels in kernels/; the
// rest run TFLM's reference kernels.
//
// On RV64 (CONFIG=rv64 / Spike) every op is bracketed with rdcycle/rdinstret
// through TFLM's MicroProfilerInterface hook, giving per-kernel cycle and
// dynamic instruction counts.

#include <cstdint>
#include <cstring>

#include "kernels/op_resolver.h"
#include "model/dev_net_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "test_data/dev_net_int8_test_data.h"

#ifndef WORKLOAD_ARENA_SIZE
#define WORKLOAD_ARENA_SIZE (256 * 1024)
#endif
#ifndef GARIBALDI_CUSTOM_KERNELS
#define GARIBALDI_CUSTOM_KERNELS ""
#endif

namespace {

#ifdef __riscv
inline uint64_t rv_rdcycle() {
  uint64_t v;
  __asm__ volatile("rdcycle %0" : "=r"(v));
  return v;
}
inline uint64_t rv_rdinstret() {
  uint64_t v;
  __asm__ volatile("rdinstret %0" : "=r"(v));
  return v;
}
#else
inline uint64_t rv_rdcycle() { return 0; }
inline uint64_t rv_rdinstret() { return 0; }
#endif  // __riscv

// One event per op: MicroInterpreterGraph::InvokeSubgraph wraps each
// registration->invoke in a ScopedMicroProfiler.
class OpProfiler : public tflite::MicroProfilerInterface {
 public:
  static constexpr int kMaxEvents = 32;

  uint32_t BeginEvent(const char* tag) override {
    const int i = num_events_ < kMaxEvents ? num_events_++ : kMaxEvents - 1;
    tags_[i] = tag;
    instret_[i] = rv_rdinstret();
    cycles_[i] = rv_rdcycle();
    return i;
  }

  void EndEvent(uint32_t handle) override {
    cycles_[handle] = rv_rdcycle() - cycles_[handle];
    instret_[handle] = rv_rdinstret() - instret_[handle];
  }

  void Clear() { num_events_ = 0; }
  int num_events() const { return num_events_; }
  const char* tag(int i) const { return tags_[i]; }
  uint64_t cycles(int i) const { return cycles_[i]; }
  uint64_t instret(int i) const { return instret_[i]; }

 private:
  int num_events_ = 0;
  const char* tags_[kMaxEvents];
  uint64_t cycles_[kMaxEvents];
  uint64_t instret_[kMaxEvents];
};

alignas(16) uint8_t g_tensor_arena[WORKLOAD_ARENA_SIZE];
OpProfiler g_profiler;

}  // namespace

int main() {
  tflite::InitializeTarget();

  const tflite::Model* model = tflite::GetModel(g_dev_net_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d != supported %d",
                static_cast<int>(model->version()), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // preserve_all_tensors disables activation-buffer reuse so every op's
  // output can be read back after Invoke().
  tflite::MicroMutableOpResolver<kNumberOperators> resolver = get_resolver();
  tflite::MicroInterpreter interpreter(model, resolver, g_tensor_arena,
                                       sizeof(g_tensor_arena),
                                       /*resource_variables=*/nullptr,
                                       &g_profiler, /*preserve_all_tensors=*/true);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return 1;
  }

  TfLiteTensor* input = interpreter.input(0);
  if (input->type != kTfLiteInt8 ||
      input->bytes != static_cast<size_t>(g_dev_net_int8_input_size)) {
    MicroPrintf("Unexpected input tensor: %d B", static_cast<int>(input->bytes));
    return 1;
  }
  std::memcpy(input->data.int8, g_dev_net_int8_input, input->bytes);

  g_profiler.Clear();
  const uint64_t cycle_start = rv_rdcycle();
  const uint64_t instret_start = rv_rdinstret();
  if (interpreter.Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke() failed");
    return 1;
  }
  const uint64_t cycles = rv_rdcycle() - cycle_start;
  const uint64_t instret = rv_rdinstret() - instret_start;

  MicroPrintf("dev_net_int8 | custom kernels: %s",
              GARIBALDI_CUSTOM_KERNELS[0] ? GARIBALDI_CUSTOM_KERNELS : "none");
  if (g_profiler.num_events() != g_dev_net_int8_num_ops) {
    MicroPrintf("profiler saw %d ops, model has %d", g_profiler.num_events(),
                g_dev_net_int8_num_ops);
    return 1;
  }

#ifdef __riscv
  MicroPrintf(" #  op                     cycles      instret  mismatches  max|diff|");
#else
  MicroPrintf(" #  op                mismatches  max|diff|");
#endif
  bool pass = true;
  for (int i = 0; i < g_dev_net_int8_num_ops; ++i) {
    const DevGoldenTensor& golden = g_dev_net_int8_golden[i];
    const TfLiteEvalTensor* t = interpreter.GetTensor(golden.tensor_index);
    if (t == nullptr || t->type != kTfLiteInt8 ||
        tflite::ElementCount(*t->dims) != golden.size) {
      MicroPrintf("op %d (%s): output tensor %d does not match the golden shape",
                  i, golden.op, golden.tensor_index);
      return 1;
    }
    int mismatches = 0;
    int max_abs_diff = 0;
    for (int j = 0; j < golden.size; ++j) {
      int diff = t->data.int8[j] - golden.data[j];
      if (diff < 0) diff = -diff;
      if (diff != 0) ++mismatches;
      if (diff > max_abs_diff) max_abs_diff = diff;
    }
    // Same tolerance as src/software/test: LiteRT and TFLM requantisation
    // can differ by 1 LSB, and that difference propagates to later ops.
    if (max_abs_diff > 1) pass = false;
#ifdef __riscv
    MicroPrintf("%2d  %-16s %12lld %12lld  %5d/%-5d  %d", i, golden.op,
                (long long)g_profiler.cycles(i), (long long)g_profiler.instret(i),
                mismatches, golden.size, max_abs_diff);
#else
    MicroPrintf("%2d  %-16s  %5d/%-5d  %d", i, golden.op, mismatches, golden.size,
                max_abs_diff);
#endif
  }
#ifdef __riscv
  MicroPrintf("invoke cycles:  %lld", (long long)cycles);
  MicroPrintf("invoke instret: %lld", (long long)instret);
#else
  (void)cycles;
  (void)instret;
#endif
  MicroPrintf("arena used: %d / %d bytes",
              static_cast<int>(interpreter.arena_used_bytes()), WORKLOAD_ARENA_SIZE);
  MicroPrintf(pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
