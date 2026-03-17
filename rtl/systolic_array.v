//============================================================================
// NeuralForge — 4x4 Weight-Stationary Systolic Array
// Performs 4x4 matrix multiplication using pipelined MAC units
// Inspired by Google TPU v1 architecture
//============================================================================

module systolic_array (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        load_weights,    // Load new weights into MAC units
    input  wire        clear_acc,       // Clear all accumulators

    // Weight loading interface (column-major, 4 weights per cycle)
    input  wire [1:0]  weight_col,      // Which column to load (0-3)
    input  wire signed [7:0] w_in [0:3], // 4 weights for the selected column

    // Activation inputs (one per row, fed from left)
    input  wire signed [7:0] a_in [0:3],

    // Result outputs (4x4 accumulated results)
    output wire signed [31:0] result [0:3][0:3],
    output reg         done             // Computation complete flag
);

    // Internal weight storage
    reg signed [7:0] weights [0:3][0:3]; // [row][col]

    // Internal signals between MAC units
    wire signed [7:0] a_pass [0:3][0:4]; // Horizontal activation flow
    wire signed [7:0] b_pass [0:4][0:3]; // Vertical weight flow (unused in weight-stationary)

    // Pipeline counter for tracking computation progress
    reg [3:0] cycle_count;

    // Load weights into local storage
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            integer i, j;
            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1)
                    weights[i][j] <= 8'sd0;
        end else if (load_weights) begin
            integer i;
            for (i = 0; i < 4; i = i + 1)
                weights[i][weight_col] <= w_in[i];
        end
    end

    // Feed activations into left edge
    genvar r;
    generate
        for (r = 0; r < 4; r = r + 1) begin : feed_act
            assign a_pass[r][0] = a_in[r];
        end
    endgenerate

    // Instantiate 4x4 grid of MAC units
    genvar row, col;
    generate
        for (row = 0; row < 4; row = row + 1) begin : mac_row
            for (col = 0; col < 4; col = col + 1) begin : mac_col
                mac_unit u_mac (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .en        (en),
                    .clear_acc (clear_acc),
                    .a         (a_pass[row][col]),
                    .b         (weights[row][col]),   // Weight-stationary: weights don't flow
                    .acc       (result[row][col]),
                    .a_out     (a_pass[row][col+1]),
                    .b_out     ()                     // Not used in weight-stationary mode
                );
            end
        end
    endgenerate

    // Computation progress tracking
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 4'd0;
            done <= 1'b0;
        end else if (clear_acc) begin
            cycle_count <= 4'd0;
            done <= 1'b0;
        end else if (en && !done) begin
            if (cycle_count == 4'd7) // 4 compute + 3 pipeline drain
                done <= 1'b1;
            else
                cycle_count <= cycle_count + 4'd1;
        end
    end

endmodule
