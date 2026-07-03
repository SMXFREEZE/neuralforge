//============================================================================
// NeuralForge — 4x4 Output-Stationary Systolic Array
// Performs 4x4 matrix multiplication (R = A * W) using pipelined MAC units
//
// Dataflow: activations flow left->right, weights flow top->bottom, and each
// PE[r][c] keeps its own accumulator (result stays put — output-stationary).
// Weights are pre-loaded column-by-column into local storage; during compute
// an internal scheduler streams stored column c into the array with the
// systolic skew (W[k][c] enters during cycle k + c), matching activations
// A[r][k] entering row r during cycle k + r. The operands for output element
// (r, c) meet at PE[r][c] on cycle k + r + c, so:
//     result[r][c] = sum_k A[r][k] * W[k][c]
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

    // Internal weight storage: weights[k][c] = W[k][c], k = input-dim index
    reg signed [7:0] weights [0:3][0:3];

    // Internal signals between MAC units
    wire signed [7:0] a_pass [0:3][0:4]; // Horizontal activation flow
    wire signed [7:0] b_pass [0:4][0:3]; // Vertical weight flow

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

    // Feed stored weights into the top edge with the systolic skew:
    // column c receives W[k][c] during cycle k + c (cycle_count tracks the
    // cycle index since clear_acc), zero elsewhere so misaligned operands
    // contribute nothing.
    genvar wc;
    generate
        for (wc = 0; wc < 4; wc = wc + 1) begin : feed_weights
            wire [3:0] k = cycle_count - wc[3:0];
            assign b_pass[0][wc] =
                (cycle_count >= wc && cycle_count <= wc + 3)
                    ? weights[k[1:0]][wc]
                    : 8'sd0;
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
                    .b         (b_pass[row][col]),
                    .acc       (result[row][col]),   // Output-stationary accumulator
                    .a_out     (a_pass[row][col+1]),
                    .b_out     (b_pass[row+1][col])
                );
            end
        end
    endgenerate

    // Computation progress tracking.
    // The last operand pair (k = 3) reaches PE[3][3] during cycle
    // k + r + c = 9 and is accumulated on the following edge, so results
    // are final once cycle_count has passed 10.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cycle_count <= 4'd0;
            done <= 1'b0;
        end else if (clear_acc) begin
            cycle_count <= 4'd0;
            done <= 1'b0;
        end else if (en && !done) begin
            if (cycle_count == 4'd10)
                done <= 1'b1;
            else
                cycle_count <= cycle_count + 4'd1;
        end
    end

endmodule
