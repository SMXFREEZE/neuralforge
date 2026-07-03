//============================================================================
// NeuralForge — Convolution Engine Testbench
// Verifies 3x3 convolution output against known reference values
//============================================================================

`timescale 1ns / 1ps

module tb_conv_engine;

    reg        clk;
    reg        rst_n;
    reg        en;
    reg        start;
    reg signed [7:0]  ifmap [0:8];
    reg signed [7:0]  kernel [0:8];
    reg signed [31:0] bias;
    wire signed [31:0] conv_out;
    wire       valid;

    // DUT
    conv_engine uut (
        .clk      (clk),
        .rst_n    (rst_n),
        .en       (en),
        .start    (start),
        .ifmap    (ifmap),
        .kernel   (kernel),
        .bias     (bias),
        .conv_out (conv_out),
        .valid    (valid)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    integer i;
    integer pass_count, fail_count;

    // 'valid' is a single-cycle pulse out of the 4-stage pipeline; wait for
    // it with a bounded loop instead of counting a fixed number of cycles.
    // If the pipeline never produces a result, valid stays 0 and the
    // subsequent check fails.
    integer wv;
    task wait_valid;
        begin
            wv = 0;
            @(posedge clk); #1;
            while (!valid && wv < 20) begin
                @(posedge clk); #1;
                wv = wv + 1;
            end
        end
    endtask

    initial begin
        $display("==========================================================");
        $display("  NeuralForge Convolution Engine Testbench");
        $display("==========================================================");

        pass_count = 0;
        fail_count = 0;

        // Reset
        rst_n = 0;
        en = 1;
        start = 0;
        bias = 32'sd0;
        for (i = 0; i < 9; i = i + 1) begin
            ifmap[i] = 8'sd0;
            kernel[i] = 8'sd0;
        end
        #20;
        rst_n = 1;
        #10;

        //--------------------------------------------------------------
        // Test 1: Identity-like kernel (center = 1, rest = 0)
        //--------------------------------------------------------------
        $display("\n--- Test 1: Identity kernel ---");

        // Input patch: [1,2,3,4,5,6,7,8,9]
        ifmap[0] = 8'sd1; ifmap[1] = 8'sd2; ifmap[2] = 8'sd3;
        ifmap[3] = 8'sd4; ifmap[4] = 8'sd5; ifmap[5] = 8'sd6;
        ifmap[6] = 8'sd7; ifmap[7] = 8'sd8; ifmap[8] = 8'sd9;

        // Identity kernel: center only
        kernel[0] = 8'sd0; kernel[1] = 8'sd0; kernel[2] = 8'sd0;
        kernel[3] = 8'sd0; kernel[4] = 8'sd1; kernel[5] = 8'sd0;
        kernel[6] = 8'sd0; kernel[7] = 8'sd0; kernel[8] = 8'sd0;

        bias = 32'sd0;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        wait_valid;

        if (valid && conv_out === 32'sd5) begin
            pass_count = pass_count + 1;
            $display("[PASS] Identity kernel: conv_out = %0d (expected 5)", conv_out);
        end else begin
            fail_count = fail_count + 1;
            $display("[FAIL] Identity kernel: conv_out = %0d, valid = %0b (expected 5)", conv_out, valid);
        end

        //--------------------------------------------------------------
        // Test 2: All-ones kernel (sum of all elements)
        //--------------------------------------------------------------
        $display("\n--- Test 2: All-ones kernel ---");

        for (i = 0; i < 9; i = i + 1)
            kernel[i] = 8'sd1;

        bias = 32'sd0;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        wait_valid;

        // Expected: 1+2+3+4+5+6+7+8+9 = 45
        if (valid && conv_out === 32'sd45) begin
            pass_count = pass_count + 1;
            $display("[PASS] All-ones kernel: conv_out = %0d (expected 45)", conv_out);
        end else begin
            fail_count = fail_count + 1;
            $display("[FAIL] All-ones kernel: conv_out = %0d, valid = %0b (expected 45)", conv_out, valid);
        end

        //--------------------------------------------------------------
        // Test 3: With bias
        //--------------------------------------------------------------
        $display("\n--- Test 3: With bias ---");

        bias = 32'sd100;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        wait_valid;

        // Expected: 45 + 100 = 145
        if (valid && conv_out === 32'sd145) begin
            pass_count = pass_count + 1;
            $display("[PASS] With bias: conv_out = %0d (expected 145)", conv_out);
        end else begin
            fail_count = fail_count + 1;
            $display("[FAIL] With bias: conv_out = %0d, valid = %0b (expected 145)", conv_out, valid);
        end

        //--------------------------------------------------------------
        // Test 4: Negative kernel (edge detection Laplacian)
        //--------------------------------------------------------------
        $display("\n--- Test 4: Laplacian edge detection ---");

        // Laplacian: [0,-1,0,-1,4,-1,0,-1,0]
        kernel[0] = 8'sd0;  kernel[1] = -8'sd1; kernel[2] = 8'sd0;
        kernel[3] = -8'sd1; kernel[4] = 8'sd4;  kernel[5] = -8'sd1;
        kernel[6] = 8'sd0;  kernel[7] = -8'sd1; kernel[8] = 8'sd0;

        bias = 32'sd0;
        start = 1;
        @(posedge clk); #1;
        start = 0;

        wait_valid;

        // Expected: 0*1 + (-1)*2 + 0*3 + (-1)*4 + 4*5 + (-1)*6 + 0*7 + (-1)*8 + 0*9
        //         = 0 - 2 + 0 - 4 + 20 - 6 + 0 - 8 + 0 = 0
        if (valid && conv_out === 32'sd0) begin
            pass_count = pass_count + 1;
            $display("[PASS] Laplacian: conv_out = %0d (expected 0)", conv_out);
        end else begin
            fail_count = fail_count + 1;
            $display("[FAIL] Laplacian: conv_out = %0d, valid = %0b (expected 0)", conv_out, valid);
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
        $dumpfile("conv_engine.vcd");
        $dumpvars(0, tb_conv_engine);
    end

endmodule
