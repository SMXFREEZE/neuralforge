//============================================================================
// NeuralForge — ReLU Activation Unit
// Zero-latency combinational ReLU using sign-bit masking
// Also supports LeakyReLU (alpha = 1/256 via arithmetic shift)
//============================================================================

module activation (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire [1:0]  mode,           // 00: ReLU, 01: LeakyReLU, 10: passthrough
    input  wire signed [31:0] data_in,
    output reg  signed [31:0] data_out,
    output reg         valid
);

    // Combinational ReLU result
    wire signed [31:0] relu_out;
    wire signed [31:0] leaky_out;

    // ReLU: max(0, x) — just check sign bit
    assign relu_out = data_in[31] ? 32'sd0 : data_in;

    // LeakyReLU: x if x > 0, else x/256 (arithmetic right shift 8)
    assign leaky_out = data_in[31] ? (data_in >>> 8) : data_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 32'sd0;
            valid <= 1'b0;
        end else if (en) begin
            case (mode)
                2'b00:   data_out <= relu_out;      // ReLU
                2'b01:   data_out <= leaky_out;     // LeakyReLU
                2'b10:   data_out <= data_in;       // Passthrough (for last layer)
                default: data_out <= relu_out;
            endcase
            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule
