// Custom (Garibaldi) kernel registrations for the software_dev workload.
//
// Each kernel is a TFLMRegistration that kernels/op_resolver.h hands to
// MicroMutableOpResolver in place of the TFLM reference kernel when the op is
// listed in CUSTOM_KERNELS (Makefile). A registration is {init, prepare,
// invoke}; reusing TFLM's own init/prepare (declared in the op's
// tensorflow/lite/micro/kernels/<op>.h) and replacing only invoke keeps the
// quantisation setup identical to the reference kernel.

#pragma once

#include "tensorflow/lite/micro/micro_common.h"

namespace garibaldi {

TFLMRegistration Register_CONV_2D();

}  // namespace garibaldi
