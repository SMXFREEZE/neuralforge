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

    integer i, j;
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

    initial begin
        $display("==========================================================");
        $display("  NeuralForge Systolic Array Testbench");
        $display("==========================================================");

        pass_count = 0;
        fail_count = 0;

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
        // Clear accumulators and compute
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;

        // Feed activation rows (one column of A at a time)
        en = 1;

        // Cycle 0: Column 0 of A
        a_in[0] = 8'sd1; a_in[1] = 8'sd5; a_in[2] = 8'sd9; a_in[3] = 8'sd13;
        @(posedge clk); #1;

        // Cycle 1: Column 1 of A
        a_in[0] = 8'sd2; a_in[1] = 8'sd6; a_in[2] = 8'sd10; a_in[3] = 8'sd14;
        @(posedge clk); #1;

        // Cycle 2: Column 2 of A
        a_in[0] = 8'sd3; a_in[1] = 8'sd7; a_in[2] = 8'sd11; a_in[3] = 8'sd15;
        @(posedge clk); #1;

        // Cycle 3: Column 3 of A
        a_in[0] = 8'sd4; a_in[1] = 8'sd8; a_in[2] = 8'sd12; a_in[3] = 8'sd16;
        @(posedge clk); #1;

        // Wait for pipeline to drain
        a_in[0] = 8'sd0; a_in[1] = 8'sd0; a_in[2] = 8'sd0; a_in[3] = 8'sd0;
        repeat(4) @(posedge clk);
        en = 0;
        #1;

        //--------------------------------------------------------------
        // Verify results
        //--------------------------------------------------------------
        $display("\n--- Checking results ---");

        // Row 0: expected [5, 2, 3, 5]
        if (result[0][0] === 32'sd5)  begin pass_count = pass_count + 1; $display("[PASS] R[0][0] = %0d", result[0][0]); end
        else begin fail_count = fail_count + 1; $display("[FAIL] R[0][0] = %0d (expected 5)", result[0][0]); end

        if (result[0][1] === 32'sd2)  begin pass_count = pass_count + 1; $display("[PASS] R[0][1] = %0d", result[0][1]); end
        else begin fail_count = fail_count + 1; $display("[FAIL] R[0][1] = %0d (expected 2)", result[0][1]); end

        if (result[0][2] === 32'sd3)  begin pass_count = pass_count + 1; $display("[PASS] R[0][2] = %0d", result[0][2]); end
        else begin fail_count = fail_count + 1; $display("[FAIL] R[0][2] = %0d (expected 3)", result[0][2]); end

        if (result[0][3] === 32'sd5)  begin pass_count = pass_count + 1; $display("[PASS] R[0][3] = %0d", result[0][3]); end
        else begin fail_count = fail_count + 1; $display("[FAIL] R[0][3] = %0d (expected 5)", result[0][3]); end

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
