// Op resolver for dev_net: one entry per operator type in the model.
//
// The Makefile defines GARIBALDI_CUSTOM_<OP> for every op listed in
// CUSTOM_KERNELS; those ops are registered with the garibaldi:: kernel, the
// rest with the TFLM reference kernel (the Add*() default argument).
//
// To add a custom kernel for another op:
//   1. write kernels/<op>.cc defining garibaldi::Register_<OP>() and declare
//      it in kernels/garibaldi_kernels.h
//   2. wrap that op's Add*() call below in #ifdef GARIBALDI_CUSTOM_<OP>
//   3. build with CUSTOM_KERNELS="CONV_2D <OP>"

#pragma once

#include "kernels/garibaldi_kernels.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

constexpr int kNumberOperators = 7;

inline tflite::MicroMutableOpResolver<kNumberOperators> get_resolver() {
  tflite::MicroMutableOpResolver<kNumberOperators> resolver;
  resolver.AddTranspose();
  resolver.AddPad();
#ifdef GARIBALDI_CUSTOM_CONV_2D
  resolver.AddConv2D(garibaldi::Register_CONV_2D());
#else
  resolver.AddConv2D();
#endif
  resolver.AddMaxPool2D();
  resolver.AddAdd();
  resolver.AddAveragePool2D();
  resolver.AddFullyConnected();
  return resolver;
}
