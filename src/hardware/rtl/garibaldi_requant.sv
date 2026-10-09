///////////////////////////////////////////
// garibaldi_requant.sv
//
// Purpose: Requantization datapath matching TFLM's default (double-rounding)
//          MultiplyByQuantizedMultiplier, followed by zero-point add and clamp:
//
//            x  = Acc << max(Shift, 0)                       (32-bit wrap)
//            y  = SaturatingRoundingDoublingHighMul(x, Multiplier)
//            z  = RoundingDivideByPOT(y, max(-Shift, 0))     (round half away from 0)
//            Result = clamp(z + ZeroPoint, ActMin, ActMax)
//
// Combinational reference datapath; operand packing into registers and the
// pipeline integration into CV-Wally are still to be designed.
///////////////////////////////////////////

module garibaldi_requant (
  input  logic [31:0] Acc,         // int32 accumulator
  input  logic [31:0] Multiplier,  // Q31 multiplier, [0, 2^31)
  input  logic [5:0]  Shift,       // signed, [-31, 30]; negative = right shift
  input  logic [31:0] ZeroPoint,   // output zero point (signed)
  input  logic [31:0] ActMin,      // clamp bounds (signed)
  input  logic [31:0] ActMax,
  output logic [31:0] Result
);

  logic        [4:0]  LeftShift, RightShift;
  logic        [31:0] Shifted;
  logic signed [63:0] Prod, Nudged;
  logic        [31:0] Quot, HighMul;
  logic        [31:0] Mask, Remainder, Threshold, Divided, Biased;

  // Split the signed shift into left/right amounts
  always_comb begin
    if ($signed(Shift) > 0) begin
      LeftShift  = Shift[4:0];
      RightShift = '0;
    end else begin
      LeftShift  = '0;
      RightShift = 5'(-$signed(Shift));
    end
  end

  assign Shifted = Acc << LeftShift;

  // SaturatingRoundingDoublingHighMul: (x*m + nudge) / 2^31, truncating toward zero
  assign Prod   = 64'($signed(Shifted)) * 64'($signed(Multiplier));
  assign Nudged = Prod + (Prod[63] ? 64'sd1 - (64'sd1 <<< 30) : (64'sd1 <<< 30));
  assign Quot   = 32'((Nudged + (Nudged[63] ? (64'sd1 <<< 31) - 64'sd1 : 64'sd0)) >>> 31);
  assign HighMul = (Shifted == 32'h8000_0000 && Multiplier == 32'h8000_0000) ?
                   32'h7FFF_FFFF : Quot;

  // RoundingDivideByPOT
  assign Mask      = (32'd1 << RightShift) - 32'd1;
  assign Remainder = HighMul & Mask;
  assign Threshold = (Mask >> 1) + {31'b0, HighMul[31]};
  assign Divided   = 32'($signed(HighMul) >>> RightShift) + {31'b0, (Remainder > Threshold)};

  // Zero point and clamp
  assign Biased = Divided + ZeroPoint;
  always_comb begin
    if      ($signed(Biased) < $signed(ActMin)) Result = ActMin;
    else if ($signed(Biased) > $signed(ActMax)) Result = ActMax;
    else                                        Result = Biased;
  end

endmodule
