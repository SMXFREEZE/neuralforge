//============================================================================
// NeuralForge — MAC (Multiply-Accumulate) Unit
// INT8 inputs, 32-bit accumulator
// The atomic compute primitive used inside the systolic array
//============================================================================

module mac_unit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        clear_acc,      // Clear accumulator for new computation
    input  wire signed [7:0] a,        // INT8 activation input
    input  wire signed [7:0] b,        // INT8 weight input
    output reg  signed [31:0] acc,     // 32-bit accumulated result
    output wire signed [7:0] a_out,    // Pass-through for systolic data flow
    output wire signed [7:0] b_out     // Pass-through for systolic data flow
);

    wire signed [15:0] product;

    // Signed multiplication: 8x8 → 16-bit product
    assign product = a * b;

    // Systolic pass-through (registered for pipeline timing)
    reg signed [7:0] a_reg, b_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_reg <= 8'sd0;
            b_reg <= 8'sd0;
        end else if (en) begin
            a_reg <= a;
            b_reg <= b;
        end
    end

    assign a_out = a_reg;
    assign b_out = b_reg;

    // Accumulate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (clear_acc)
            acc <= 32'sd0;
        else if (en)
            acc <= acc + {{16{product[15]}}, product}; // Sign-extend & accumulate
    end

endmodule
