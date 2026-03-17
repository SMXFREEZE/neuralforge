//============================================================================
// NeuralForge — 3x3 Convolution Engine
// Processes 3x3 convolution with configurable stride
// Spatially unrolls the 9 multiply-accumulate operations
//============================================================================

module conv_engine (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        start,           // Start new convolution

    // Input feature map patch (3x3 = 9 pixels)
    input  wire signed [7:0] ifmap [0:8],

    // Convolution kernel weights (3x3 = 9 weights)
    input  wire signed [7:0] kernel [0:8],

    // Bias
    input  wire signed [31:0] bias,

    // Output
    output reg  signed [31:0] conv_out,
    output reg         valid            // Output is valid
);

    // Pipeline stage 1: Multiply all 9 pairs
    reg signed [15:0] products [0:8];
    reg                stage1_valid;

    // Pipeline stage 2: Reduction tree (add pairs)
    reg signed [31:0] sum_01, sum_23, sum_45, sum_67;
    reg signed [31:0] prod_8_ext;
    reg                stage2_valid;

    // Pipeline stage 3: Final reduction + bias
    reg signed [31:0] sum_0123, sum_4567;
    reg                stage3_valid;

    // Stage 1: Parallel multiplication
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 9; i = i + 1)
                products[i] <= 16'sd0;
            stage1_valid <= 1'b0;
        end else if (en && start) begin
            for (i = 0; i < 9; i = i + 1)
                products[i] <= ifmap[i] * kernel[i];
            stage1_valid <= 1'b1;
        end else begin
            stage1_valid <= 1'b0;
        end
    end

    // Stage 2: Pairwise addition (adder tree level 1)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_01 <= 32'sd0;
            sum_23 <= 32'sd0;
            sum_45 <= 32'sd0;
            sum_67 <= 32'sd0;
            prod_8_ext <= 32'sd0;
            stage2_valid <= 1'b0;
        end else if (stage1_valid) begin
            sum_01 <= {{16{products[0][15]}}, products[0]} + {{16{products[1][15]}}, products[1]};
            sum_23 <= {{16{products[2][15]}}, products[2]} + {{16{products[3][15]}}, products[3]};
            sum_45 <= {{16{products[4][15]}}, products[4]} + {{16{products[5][15]}}, products[5]};
            sum_67 <= {{16{products[6][15]}}, products[6]} + {{16{products[7][15]}}, products[7]};
            prod_8_ext <= {{16{products[8][15]}}, products[8]};
            stage2_valid <= 1'b1;
        end else begin
            stage2_valid <= 1'b0;
        end
    end

    // Stage 3: Final reduction + bias addition
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum_0123 <= 32'sd0;
            sum_4567 <= 32'sd0;
            stage3_valid <= 1'b0;
        end else if (stage2_valid) begin
            sum_0123 <= sum_01 + sum_23;
            sum_4567 <= sum_45 + sum_67;
            stage3_valid <= 1'b1;
        end else begin
            stage3_valid <= 1'b0;
        end
    end

    // Stage 4: Output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_out <= 32'sd0;
            valid <= 1'b0;
        end else if (stage3_valid) begin
            conv_out <= sum_0123 + sum_4567 + prod_8_ext + bias;
            valid <= 1'b1;
        end else begin
            valid <= 1'b0;
        end
    end

endmodule
