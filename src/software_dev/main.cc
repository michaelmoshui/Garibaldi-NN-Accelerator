// dev_net W8A8 inference on the LiteRT Micro (TFLM) interpreter: the
// kernel-development workload.
//
// Runs one inference on the generated input with every intermediate tensor
// preserved, then checks each op's output against the LiteRT reference-kernel
// output in test_data/, so a kernel bug shows up at the op that introduced it.
// Ops listed in CUSTOM_KERNELS (Makefile) run the kernels in kernels/; the
// rest run TFLM's reference kernels.
//
// On RV64 every op is bracketed with rdcycle/rdinstret through TFLM's
// MicroProfilerInterface hook, giving per-kernel cycle and dynamic
// instruction counts.
//
// Results are always recorded in two C-linkage globals:
//   cvw_results      whole-Invoke() and per-op cycles/instret + check results
//   selfcheck_record CV-Wally testbench self-check record, reported when the
//                    program writes tohost (cvw/crt.S):
//       status 1        "succeeded"
//       status 2        op <index> failed: adr = its output tensor,
//                       sim = max |diff|, signature = allowed max |diff| (1)
//       status 3        profiler saw <actual> ops, model has <expected>
//       status 0x10+k   setup error k (see SetupError)
//       status 0x100+c  trap with mcause c (crt.S also stores index = mepc,
//                       adr = mtval, expected = ra, actual = sp)
// CONFIG=host/rv64 also print a per-op table. CONFIG=cvw (CV-Wally, no UART)
// links cvw/debug_log.cc, which drops MicroPrintf output unformatted.

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

constexpr int kMaxOps = 32;
// Same tolerance as src/software/test: LiteRT and TFLM requantisation can
// differ by 1 LSB, and that difference propagates to later ops.
constexpr int kMaxAbsDiff = 1;

// C linkage so the names appear unmangled in the .objdump.addr/.lab files the
// CV-Wally testbench reads; the section keeps them out of .bss so they are
// part of the program image.
extern "C" {

struct SelfCheckRecord {
  uint64_t status;
  uint64_t index;
  uint64_t adr;
  uint64_t expected;
  uint64_t actual;
  uint64_t reserved[3];
};

struct OpResult {
  uint64_t cycles;
  uint64_t instret;
  uint32_t mismatches;
  uint32_t size;
  uint32_t max_abs_diff;
  uint32_t reserved;
};

struct CvwResults {
  uint64_t invoke_cycles;
  uint64_t invoke_instret;
  uint64_t num_ops;
  uint64_t arena_used;
  OpResult ops[kMaxOps];
};

__attribute__((section(".selfcheck"), aligned(64)))
volatile SelfCheckRecord selfcheck_record = {};
__attribute__((section(".selfcheck")))
volatile CvwResults cvw_results = {};

}  // extern "C"

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

enum SetupError : uint64_t {
  kSchemaVersion = 0x10,
  kAllocateTensors = 0x11,
  kInputTensor = 0x12,
  kInvoke = 0x13,
  kGoldenShape = 0x14,
};

// One event per op: MicroInterpreterGraph::InvokeSubgraph wraps each
// registration->invoke in a ScopedMicroProfiler.
class OpProfiler : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char* /*tag*/) override {
    const int i = num_events_ < kMaxOps ? num_events_++ : kMaxOps - 1;
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
  uint64_t cycles(int i) const { return cycles_[i]; }
  uint64_t instret(int i) const { return instret_[i]; }

 private:
  int num_events_ = 0;
  uint64_t cycles_[kMaxOps];
  uint64_t instret_[kMaxOps];
};

alignas(16) uint8_t g_tensor_arena[WORKLOAD_ARENA_SIZE];
OpProfiler g_profiler;

int Fail(uint64_t status, uint64_t index = 0, uint64_t adr = 0,
         uint64_t expected = 0, uint64_t actual = 0) {
  selfcheck_record.index = index;
  selfcheck_record.adr = adr;
  selfcheck_record.expected = expected;
  selfcheck_record.actual = actual;
  selfcheck_record.status = status;
  MicroPrintf("FAIL (status 0x%x, index %d, expected %d, actual %d)",
              static_cast<int>(status), static_cast<int>(index),
              static_cast<int>(expected), static_cast<int>(actual));
  return 1;
}

}  // namespace

int main() {
  // ── Step 1: InitializeTarget ─────────────────────────────────────────────
  tflite::InitializeTarget();

  // ── Step 2: Load model → check schema version ────────────────────────────
  const tflite::Model* model = tflite::GetModel(g_dev_net_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return Fail(kSchemaVersion, 0, 0, TFLITE_SCHEMA_VERSION, model->version());
  }

  // ── Step 3: Build resolver (±custom kernels) → AllocateTensors ──────────
  // preserve_all_tensors disables activation-buffer reuse so every op's
  // output can be read back after Invoke().
  tflite::MicroMutableOpResolver<kNumberOperators> resolver = get_resolver();
  tflite::MicroInterpreter interpreter(model, resolver, g_tensor_arena,
                                       sizeof(g_tensor_arena),
                                       /*resource_variables=*/nullptr,
                                       &g_profiler, /*preserve_all_tensors=*/true);
  if (interpreter.AllocateTensors() != kTfLiteOk) return Fail(kAllocateTensors);

  // ── Step 4: Fill input tensor ────────────────────────────────────────────
  TfLiteTensor* input = interpreter.input(0);
  if (input->type != kTfLiteInt8 ||
      input->bytes != static_cast<size_t>(g_dev_net_int8_input_size)) {
    return Fail(kInputTensor, 0, 0, g_dev_net_int8_input_size, input->bytes);
  }
  std::memcpy(input->data.int8, g_dev_net_int8_input, input->bytes);

  // ── Step 5: [rdcycle/rdinstret start] → Invoke() → [rdcycle/rdinstret end]
  // OpProfiler.BeginEvent/EndEvent wraps each op inside Invoke().
  g_profiler.Clear();
  const uint64_t cycle_start = rv_rdcycle();
  const uint64_t instret_start = rv_rdinstret();
  if (interpreter.Invoke() != kTfLiteOk) return Fail(kInvoke);
  cvw_results.invoke_cycles = rv_rdcycle() - cycle_start;
  cvw_results.invoke_instret = rv_rdinstret() - instret_start;
  cvw_results.num_ops = g_profiler.num_events();
  cvw_results.arena_used = interpreter.arena_used_bytes();

  MicroPrintf("dev_net_int8 | custom kernels: %s",
              GARIBALDI_CUSTOM_KERNELS[0] ? GARIBALDI_CUSTOM_KERNELS : "none");
  if (g_profiler.num_events() != g_dev_net_int8_num_ops) {
    return Fail(3, 0, 0, g_dev_net_int8_num_ops, g_profiler.num_events());
  }

#ifdef __riscv
  MicroPrintf(" #  op                     cycles      instret  mismatches  max|diff|");
#else
  MicroPrintf(" #  op                mismatches  max|diff|");
#endif
  // ── Step 6: For each op: compare output vs golden, record diff ──────────
  int first_bad = -1;
  for (int i = 0; i < g_dev_net_int8_num_ops; ++i) {
    const DevGoldenTensor& golden = g_dev_net_int8_golden[i];
    const TfLiteEvalTensor* t = interpreter.GetTensor(golden.tensor_index);
    if (t == nullptr || t->type != kTfLiteInt8 ||
        tflite::ElementCount(*t->dims) != golden.size) {
      return Fail(kGoldenShape, i, golden.tensor_index);
    }
    uint32_t mismatches = 0;
    uint32_t max_abs_diff = 0;
    for (int j = 0; j < golden.size; ++j) {
      int diff = t->data.int8[j] - golden.data[j];
      if (diff < 0) diff = -diff;
      if (diff != 0) ++mismatches;
      if (static_cast<uint32_t>(diff) > max_abs_diff) max_abs_diff = diff;
    }
    volatile OpResult& r = cvw_results.ops[i];
    r.cycles = g_profiler.cycles(i);
    r.instret = g_profiler.instret(i);
    r.mismatches = mismatches;
    r.size = golden.size;
    r.max_abs_diff = max_abs_diff;
    if (first_bad < 0 && max_abs_diff > kMaxAbsDiff) first_bad = i;
#ifdef __riscv
    MicroPrintf("%2d  %-16s %12lld %12lld  %5d/%-5d  %d", i, golden.op,
                (long long)g_profiler.cycles(i), (long long)g_profiler.instret(i),
                static_cast<int>(mismatches), golden.size,
                static_cast<int>(max_abs_diff));
#else
    MicroPrintf("%2d  %-16s  %5d/%-5d  %d", i, golden.op,
                static_cast<int>(mismatches), golden.size,
                static_cast<int>(max_abs_diff));
#endif
  }
#ifdef __riscv
  MicroPrintf("invoke cycles:  %lld", (long long)cvw_results.invoke_cycles);
  MicroPrintf("invoke instret: %lld", (long long)cvw_results.invoke_instret);
#endif
  MicroPrintf("arena used: %d / %d bytes",
              static_cast<int>(interpreter.arena_used_bytes()), WORKLOAD_ARENA_SIZE);

  // ── Step 7: if any diff > 1 LSB: Fail(2) → tohost = FAIL ───────────────
  if (first_bad >= 0) {
    const TfLiteEvalTensor* t =
        interpreter.GetTensor(g_dev_net_int8_golden[first_bad].tensor_index);
    return Fail(2, first_bad, reinterpret_cast<uintptr_t>(t->data.int8),
                kMaxAbsDiff, cvw_results.ops[first_bad].max_abs_diff);
  }
  // ── Step 8: else: status=1 → tohost = PASS ──────────────────────────────
  selfcheck_record.status = 1;
  MicroPrintf("PASS");
  return 0;
}
