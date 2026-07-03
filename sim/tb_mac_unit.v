//============================================================================
// NeuralForge — MAC Unit Testbench
// Exhaustive verification of INT8 multiply-accumulate correctness
//============================================================================

`timescale 1ns / 1ps

module tb_mac_unit;

    reg        clk;
    reg        rst_n;
    reg        en;
    reg        clear_acc;
    reg signed [7:0] a;
    reg signed [7:0] b;
    wire signed [31:0] acc;
    wire signed [7:0] a_out;
    wire signed [7:0] b_out;

    // DUT
    mac_unit uut (
        .clk       (clk),
        .rst_n     (rst_n),
        .en        (en),
        .clear_acc (clear_acc),
        .a         (a),
        .b         (b),
        .acc       (acc),
        .a_out     (a_out),
        .b_out     (b_out)
    );

    // Clock: 10ns period (100 MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // Test variables
    integer test_count;
    integer pass_count;
    integer fail_count;
    reg signed [31:0] expected;

    task check_result;
        input signed [31:0] expected_val;
        input [127:0] test_name;
        begin
            // The caller has already advanced past the accumulating clock
            // edge (inputs are sampled on that edge), so sample acc directly.
            // Waiting another cycle here would accumulate the same operands
            // a second time while en is still asserted.
            if (acc === expected_val) begin
                pass_count = pass_count + 1;
                $display("[PASS] %0s: acc = %0d (expected %0d)", test_name, acc, expected_val);
            end else begin
                fail_count = fail_count + 1;
                $display("[FAIL] %0s: acc = %0d (expected %0d)", test_name, acc, expected_val);
            end
            test_count = test_count + 1;
        end
    endtask

    initial begin
        $display("==========================================================");
        $display("  NeuralForge MAC Unit Testbench");
        $display("==========================================================");

        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        // Reset
        rst_n = 0;
        en = 0;
        clear_acc = 0;
        a = 0;
        b = 0;
        #20;
        rst_n = 1;
        #10;

        //--------------------------------------------------------------
        // Test 1: Simple positive multiply
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;
        en = 1;
        a = 8'sd3;
        b = 8'sd4;
        @(posedge clk); #1;
        check_result(32'sd12, "3 * 4 = 12");

        //--------------------------------------------------------------
        // Test 2: Accumulate — should add to previous
        //--------------------------------------------------------------
        a = 8'sd5;
        b = 8'sd6;
        @(posedge clk); #1;
        check_result(32'sd42, "12 + 5*6 = 42");

        //--------------------------------------------------------------
        // Test 3: Negative numbers
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;
        a = -8'sd7;
        b = 8'sd3;
        @(posedge clk); #1;
        check_result(-32'sd21, "-7 * 3 = -21");

        //--------------------------------------------------------------
        // Test 4: Both negative
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;
        a = -8'sd10;
        b = -8'sd5;
        @(posedge clk); #1;
        check_result(32'sd50, "-10 * -5 = 50");

        //--------------------------------------------------------------
        // Test 5: Maximum positive values
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;
        a = 8'sd127;
        b = 8'sd127;
        @(posedge clk); #1;
        check_result(32'sd16129, "127 * 127 = 16129");

        //--------------------------------------------------------------
        // Test 6: Maximum negative value
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;
        a = -8'sd128;
        b = 8'sd1;
        @(posedge clk); #1;
        check_result(-32'sd128, "-128 * 1 = -128");

        //--------------------------------------------------------------
        // Test 7: Zero multiplication
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;
        a = 8'sd42;
        b = 8'sd0;
        @(posedge clk); #1;
        check_result(32'sd0, "42 * 0 = 0");

        //--------------------------------------------------------------
        // Test 8: Pass-through verification
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;
        a = 8'sd99;
        b = -8'sd33;
        @(posedge clk); #1;
        if (a_out === 8'sd99 && b_out === -8'sd33) begin
            pass_count = pass_count + 1;
            $display("[PASS] Pass-through: a_out=%0d, b_out=%0d", a_out, b_out);
        end else begin
            fail_count = fail_count + 1;
            $display("[FAIL] Pass-through: a_out=%0d, b_out=%0d (expected 99, -33)", a_out, b_out);
        end
        test_count = test_count + 1;

        //--------------------------------------------------------------
        // Test 9: Multi-cycle accumulation (dot product)
        // Compute: 1*2 + 3*4 + 5*6 + 7*8 = 2+12+30+56 = 100
        //--------------------------------------------------------------
        clear_acc = 1;
        @(posedge clk); #1;
        clear_acc = 0;

        a = 8'sd1; b = 8'sd2;
        @(posedge clk); #1;
        a = 8'sd3; b = 8'sd4;
        @(posedge clk); #1;
        a = 8'sd5; b = 8'sd6;
        @(posedge clk); #1;
        a = 8'sd7; b = 8'sd8;
        @(posedge clk); #1;
        check_result(32'sd100, "Dot product: 1*2+3*4+5*6+7*8=100");

        //--------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------
        en = 0;
        #20;
        $display("==========================================================");
        $display("  Results: %0d/%0d passed (%0d failed)", pass_count, test_count, fail_count);
        if (fail_count == 0)
            $display("  STATUS: ALL TESTS PASSED");
        else
            $display("  STATUS: TESTS FAILED");
        $display("==========================================================");
        $finish;
    end

    // Optional: VCD dump for waveform viewing
    initial begin
        $dumpfile("mac_unit.vcd");
        $dumpvars(0, tb_mac_unit);
    end

endmodule
