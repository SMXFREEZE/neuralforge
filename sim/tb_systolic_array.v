//============================================================================
// NeuralForge — Systolic Array Testbench
// Verifies 4x4 matrix multiplication against known reference values
//============================================================================

`timescale 1ns / 1ps

module tb_systolic_array;

    reg        clk;
    reg        rst_n;
    reg        en;
    reg        load_weights;
    reg        clear_acc;
    reg [1:0]  weight_col;
    reg signed [7:0] w_in [0:3];
    reg signed [7:0] a_in [0:3];
    wire signed [31:0] result [0:3][0:3];
    wire       done;

    // DUT
    systolic_array uut (
        .clk          (clk),
        .rst_n        (rst_n),
        .en           (en),
        .load_weights (load_weights),
        .clear_acc    (clear_acc),
        .weight_col   (weight_col),
        .w_in         (w_in),
        .a_in         (a_in),
        .result       (result),
        .done         (done)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer i, j, t;
    integer pass_count, fail_count;

    // Reference matrices for verification
    // Matrix A (4x4):
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    //  13 14 15 16
    //
    // Weight matrix W (4x4):
    //   1  0  0  1
    //   0  1  0  0
    //   0  0  1  0
    //   1  0  0  1
    //
    // Expected A * W:
    //   5  2  3  5
    //  13  6  7 13
    //  21 10 11 21
    //  29 14 15 29
    reg signed [7:0]  A [0:3][0:3];
    reg signed [31:0] expected [0:3][0:3];

    initial begin
        $display("==========================================================");
        $display("  NeuralForge Systolic Array Testbench");
        $display("==========================================================");

        pass_count = 0;
        fail_count = 0;

        // A[r][k] = r*4 + k + 1  (the matrix documented above)
        for (i = 0; i < 4; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                A[i][j] = i * 4 + j + 1;

        // expected = A * W
        expected[0][0] = 32'sd5;  expected[0][1] = 32'sd2;  expected[0][2] = 32'sd3;  expected[0][3] = 32'sd5;
        expected[1][0] = 32'sd13; expected[1][1] = 32'sd6;  expected[1][2] = 32'sd7;  expected[1][3] = 32'sd13;
        expected[2][0] = 32'sd21; expected[2][1] = 32'sd10; expected[2][2] = 32'sd11; expected[2][3] = 32'sd21;
        expected[3][0] = 32'sd29; expected[3][1] = 32'sd14; expected[3][2] = 32'sd15; expected[3][3] = 32'sd29;

        // Reset
        rst_n = 0;
        en = 0;
        load_weights = 0;
        clear_acc = 0;
        weight_col = 0;
        for (i = 0; i < 4; i = i + 1) begin
            w_in[i] = 0;
            a_in[i] = 0;
        end
        #20;
        rst_n = 1;
        #10;

        //--------------------------------------------------------------
        // Load weights column by column
        //--------------------------------------------------------------
        $display("\n--- Loading weights ---");

        // Column 0: [1, 0, 0, 1]
        load_weights = 1;
        weight_col = 2'd0;
        w_in[0] = 8'sd1; w_in[1] = 8'sd0; w_in[2] = 8'sd0; w_in[3] = 8'sd1;
        @(posedge clk); #1;

        // Column 1: [0, 1, 0, 0]
        weight_col = 2'd1;
        w_in[0] = 8'sd0; w_in[1] = 8'sd1; w_in[2] = 8'sd0; w_in[3] = 8'sd0;
        @(posedge clk); #1;

        // Column 2: [0, 0, 1, 0]
        weight_col = 2'd2;
        w_in[0] = 8'sd0; w_in[1] = 8'sd0; w_in[2] = 8'sd1; w_in[3] = 8'sd0;
        @(posedge clk); #1;

        // Column 3: [1, 0, 0, 1]
        weight_col = 2'd3;
        w_in[0] = 8'sd1; w_in[1] = 8'sd0; w_in[2] = 8'sd0; w_in[3] = 8'sd1;
        @(posedge clk); #1;

        load_weights = 0;
        $display("Weights loaded.");

        //--------------------------------------------------------------
        // Clear accumulators and stream matrix A with the systolic skew
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;

        // Output-stationary schedule: row r receives A[r][k] during cycle
        // k + r (rows enter staggered, zero-padded outside their window).
        en = 1;
        for (t = 0; t < 8; t = t + 1) begin
            for (i = 0; i < 4; i = i + 1) begin
                if (t >= i && t <= i + 3)
                    a_in[i] = A[i][t - i];
                else
                    a_in[i] = 8'sd0;
            end
            @(posedge clk); #1;
        end

        // Drain: keep clocking until the array reports done (bounded)
        for (i = 0; i < 4; i = i + 1)
            a_in[i] = 8'sd0;
        t = 0;
        while (!done && t < 20) begin
            @(posedge clk); #1;
            t = t + 1;
        end
        en = 0;

        //--------------------------------------------------------------
        // Verify results (full 4x4 matrix against A * W)
        //--------------------------------------------------------------
        $display("\n--- Checking results ---");

        if (!done) begin
            fail_count = fail_count + 1;
            $display("[FAIL] done was never asserted");
        end

        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                if (result[i][j] === expected[i][j]) begin
                    pass_count = pass_count + 1;
                    $display("[PASS] R[%0d][%0d] = %0d", i, j, result[i][j]);
                end else begin
                    fail_count = fail_count + 1;
                    $display("[FAIL] R[%0d][%0d] = %0d (expected %0d)", i, j, result[i][j], expected[i][j]);
                end
            end
        end

        // Display full result matrix
        $display("\n--- Full Result Matrix ---");
        for (i = 0; i < 4; i = i + 1) begin
            $display("  [%4d %4d %4d %4d]", result[i][0], result[i][1], result[i][2], result[i][3]);
        end

        //--------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------
        #20;
        $display("\n==========================================================");
        $display("  Results: %0d/%0d passed (%0d failed)", pass_count, pass_count + fail_count, fail_count);
        if (fail_count == 0)
            $display("  STATUS: ALL TESTS PASSED");
        else
            $display("  STATUS: TESTS FAILED");
        $display("==========================================================");
        $finish;
    end

    initial begin
        $dumpfile("systolic_array.vcd");
        $dumpvars(0, tb_systolic_array);
    end

endmodule
