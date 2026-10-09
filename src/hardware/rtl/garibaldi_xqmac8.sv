///////////////////////////////////////////
// garibaldi_xqmac8.sv
//
// Purpose: Packed 8 x INT8 multiply-accumulate datapath for the Xqmac8
//          instruction:  Result = Acc + sum_{i=0..7} A.b[i] * W.b[i]
//          (signed x signed, 64-bit wrap-around accumulate).
//
// Combinational reference datapath; pipelining and the decode/regfile
// integration into the CV-Wally IEU are still to be designed.
///////////////////////////////////////////

module garibaldi_xqmac8 (
  input  logic [63:0] Acc,    // accumulator (rd old value)
  input  logic [63:0] A,      // 8 packed INT8 activations (rs1)
  input  logic [63:0] W,      // 8 packed INT8 weights (rs2)
  output logic [63:0] Result
);

  logic signed [15:0] Prod [8];
  logic        [63:0] Sum;

  always_comb begin
    Sum = Acc;
    for (int i = 0; i < 8; i++) begin
      Prod[i] = $signed(A[8*i +: 8]) * $signed(W[8*i +: 8]);
      Sum     = Sum + {{48{Prod[i][15]}}, Prod[i]};
    end
  end

  assign Result = Sum;

endmodule
