// Garibaldi CONV_2D kernel: int8 activations x int8 per-channel weights.
//
// Init/Prepare are TFLM's ConvInit/ConvPrepare (conv_common.cc): they resolve
// SAME/VALID padding and fold input_scale * filter_scale[c] / output_scale
// into per-channel (multiplier, shift) in OpDataConv. Only Eval is ours.
//
// Layouts: input/output NHWC, filter OHWI (out_ch, k_h, k_w, in_ch).
// Requantisation must stay bit-exact with tflite::MultiplyByQuantizedMultiplier
// for the per-layer golden check in main.cc to pass.

#include <algorithm>
#include <cstdint>

#include "kernels/garibaldi_kernels.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace garibaldi {
namespace {

void ConvInt8(const tflite::ConvParams& params, const int32_t* output_multiplier,
              const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
              const int8_t* input, const tflite::RuntimeShape& filter_shape,
              const int8_t* filter, const int32_t* bias,
              const tflite::RuntimeShape& output_shape, int8_t* output) {
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t act_min = params.quantized_activation_min;
  const int32_t act_max = params.quantized_activation_max;
  const int stride_h = params.stride_height;
  const int stride_w = params.stride_width;
  const int dilation_h = params.dilation_height_factor;
  const int dilation_w = params.dilation_width_factor;
  const int pad_h = params.padding_values.height;
  const int pad_w = params.padding_values.width;

  const int batches = input_shape.Dims(0);
  const int in_h = input_shape.Dims(1);
  const int in_w = input_shape.Dims(2);
  const int in_c = input_shape.Dims(3);
  const int k_h = filter_shape.Dims(1);
  const int k_w = filter_shape.Dims(2);
  const int out_h = output_shape.Dims(1);
  const int out_w = output_shape.Dims(2);
  const int out_c = output_shape.Dims(3);

  for (int b = 0; b < batches; ++b) {
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        int8_t* out_px = output + ((b * out_h + oy) * out_w + ox) * out_c;
        for (int oc = 0; oc < out_c; ++oc) {
          const int8_t* w_oc = filter + oc * k_h * k_w * in_c;
          int32_t acc = 0;
          for (int ky = 0; ky < k_h; ++ky) {
            const int iy = oy * stride_h - pad_h + ky * dilation_h;
            if (iy < 0 || iy >= in_h) continue;
            for (int kx = 0; kx < k_w; ++kx) {
              const int ix = ox * stride_w - pad_w + kx * dilation_w;
              if (ix < 0 || ix >= in_w) continue;
              const int8_t* x = input + ((b * in_h + iy) * in_w + ix) * in_c;
              const int8_t* w = w_oc + (ky * k_w + kx) * in_c;
              for (int ic = 0; ic < in_c; ++ic) {
                acc += w[ic] * (x[ic] + input_offset);
              }
            }
          }
          if (bias != nullptr) acc += bias[oc];
          acc = tflite::MultiplyByQuantizedMultiplier(acc, output_multiplier[oc],
                                                      output_shift[oc]);
          acc += output_offset;
          acc = std::min(std::max(acc, act_min), act_max);
          out_px[oc] = static_cast<int8_t>(acc);
        }
      }
    }
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, tflite::kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, tflite::kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::NumInputs(node) == 3
          ? tflite::micro::GetEvalInput(context, node, tflite::kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, tflite::kConvOutputTensor);

  const auto& params = *static_cast<const TfLiteConvParams*>(node->builtin_data);
  const auto& data = *static_cast<const tflite::OpDataConv*>(node->user_data);

  const tflite::RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
  const tflite::RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
  if (input->type != kTfLiteInt8 || filter->type != kTfLiteInt8 ||
      filter_shape.Dims(3) != input_shape.Dims(3)) {
    MicroPrintf("garibaldi CONV_2D: only ungrouped int8 x int8 is supported");
    return kTfLiteError;
  }

  ConvInt8(tflite::ConvParamsQuantized(params, data),
           data.per_channel_output_multiplier, data.per_channel_output_shift,
           input_shape, tflite::micro::GetTensorData<int8_t>(input), filter_shape,
           tflite::micro::GetTensorData<int8_t>(filter),
           tflite::micro::GetOptionalTensorData<int32_t>(bias),
           tflite::micro::GetTensorShape(output),
           tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(tflite::ConvInit, tflite::ConvPrepare, Eval);
}

}  // namespace garibaldi
