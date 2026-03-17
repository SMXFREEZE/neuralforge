//============================================================================
// NeuralForge — 2x2 Max Pooling Unit
// Configurable stride, pipelined output
//============================================================================

module pooling (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        start,

    // 2x2 input window (4 values)
    input  wire signed [31:0] p0,
    input  wire signed [31:0] p1,
    input  wire signed [31:0] p2,
    input  wire signed [31:0] p3,

    // Output
    output reg  signed [31:0] pool_out,
    output reg         valid
);

    // Pipeline stage 1: Compare pairs
    reg signed [31:0] max_01, max_23;
    reg                stage1_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_01 <= 32'sd0;
            max_23 <= 32'sd0;
            stage1_valid <= 1'b0;
        end else if (en && start) begin
            max_01 <= (p0 > p1) ? p0 : p1;
            max_23 <= (p2 > p3) ? p2 : p3;
            stage1_valid <= 1'b1;
        end else begin
            stage1_valid <= 1'b0;
        end
    end

    // Pipeline stage 2: Compare winners
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pool_out <= 32'sd0;
            valid <= 1'b0;
        end else if (stage1_valid) begin
            pool_out <= (max_01 > max_23) ? max_01 : max_23;
            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule
