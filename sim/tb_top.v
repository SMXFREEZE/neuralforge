//============================================================================
// NeuralForge — Top-Level Integration Testbench
// End-to-end simulation: sends image bytes over simulated UART,
// verifies that classification result is returned
//============================================================================

`timescale 1ns / 1ps

module tb_top;

    // Use a lower clock freq for faster UART simulation
    localparam CLK_FREQ  = 1_000_000;   // 1 MHz (for fast simulation)
    localparam BAUD_RATE = 115_200;
    localparam TICKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam BIT_PERIOD = (1_000_000_000 / CLK_FREQ) * TICKS_PER_BIT; // in ns

    reg  clk;
    reg  rst_n;
    reg  uart_rx;
    wire uart_tx;
    wire [3:0] led_digit;
    wire       led_busy;
    wire       led_done;
    wire       led_error;

    // DUT
    top #(
        .CLK_FREQ  (CLK_FREQ),
        .BAUD_RATE (BAUD_RATE)
    ) uut (
        .clk       (clk),
        .rst_n     (rst_n),
        .uart_rx   (uart_rx),
        .uart_tx   (uart_tx),
        .led_digit (led_digit),
        .led_busy  (led_busy),
        .led_done  (led_done),
        .led_error (led_error)
    );

    // Clock: 1 MHz for simulation
    initial clk = 0;
    always #500 clk = ~clk; // 1us period

    integer i;
    integer byte_count;

    //======================================================================
    // UART byte transmit task (simulates host sending a byte)
    //======================================================================
    task uart_send_byte;
        input [7:0] data;
        integer bit_idx;
        begin
            // Start bit
            uart_rx = 1'b0;
            #(BIT_PERIOD);

            // Data bits (LSB first)
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                uart_rx = data[bit_idx];
                #(BIT_PERIOD);
            end

            // Stop bit
            uart_rx = 1'b1;
            #(BIT_PERIOD);

            // Small inter-byte gap
            #(BIT_PERIOD / 2);
        end
    endtask

    //======================================================================
    // UART byte receive task (captures response from FPGA)
    //======================================================================
    reg [7:0] received_byte;
    reg       received_valid;

    task uart_receive_byte;
        integer bit_idx;
        begin
            received_valid = 0;

            // Wait for start bit (falling edge on uart_tx)
            wait(uart_tx == 1'b0);
            #(BIT_PERIOD / 2); // Sample mid-bit

            // Verify still in start bit
            if (uart_tx != 1'b0) begin
                $display("[WARN] False start bit detected");
                disable uart_receive_byte;
            end

            #(BIT_PERIOD); // Move to first data bit

            // Read 8 data bits
            for (bit_idx = 0; bit_idx < 8; bit_idx = bit_idx + 1) begin
                received_byte[bit_idx] = uart_tx;
                #(BIT_PERIOD);
            end

            // Stop bit (should be high)
            if (uart_tx != 1'b1) begin
                $display("[WARN] Missing stop bit");
            end

            received_valid = 1;
        end
    endtask

    //======================================================================
    // Main test sequence
    //======================================================================
    initial begin
        $display("==========================================================");
        $display("  NeuralForge Top-Level Integration Testbench");
        $display("==========================================================");

        // Initialize
        rst_n = 0;
        uart_rx = 1'b1; // UART idle high
        received_valid = 0;
        #2000;
        rst_n = 1;
        #1000;

        //--------------------------------------------------------------
        // Send a 784-byte MNIST image (simplified: all same value)
        //--------------------------------------------------------------
        $display("\n--- Sending 784-byte test image via UART ---");
        $display("    (All pixels = 0x55 for test)");

        for (i = 0; i < 784; i = i + 1) begin
            uart_send_byte(8'h55);
            if (i % 100 == 0)
                $display("    Sent %0d/784 bytes...", i);
        end

        $display("    Image transmission complete.");

        //--------------------------------------------------------------
        // Wait for processing
        //--------------------------------------------------------------
        $display("\n--- Waiting for inference result ---");

        // Wait for done LED or timeout
        fork
            begin
                wait(led_done == 1'b1);
                $display("    Inference complete! LED indicates done.");
            end
            begin
                #(BIT_PERIOD * 1000);
                $display("[WARN] Timeout waiting for inference result");
            end
        join_any
        disable fork;

        //--------------------------------------------------------------
        // Capture result from UART
        //--------------------------------------------------------------
        $display("\n--- Checking result ---");
        $display("    Predicted digit (LED): %0d", led_digit);
        $display("    led_busy: %0b", led_busy);
        $display("    led_done: %0b", led_done);

        // The placeholder datapath reports pixel[0][7:4]; every pixel sent
        // was 0x55, so the digit must be 5 (and never X)
        if (led_done && led_digit === 4'd5) begin
            $display("\n[PASS] End-to-end pipeline completed successfully");
            $display("       Predicted class: %0d", led_digit);
        end else if (led_done) begin
            $display("\n[FAIL] Completed but digit = %0d (expected 5)", led_digit);
        end else begin
            $display("\n[FAIL] Pipeline did not complete");
        end

        //--------------------------------------------------------------
        // Summary
        //--------------------------------------------------------------
        #5000;
        $display("\n==========================================================");
        $display("  End-to-End Integration Test Complete");
        $display("  Predicted digit: %0d", led_digit);
        $display("==========================================================");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #(BIT_PERIOD * 20000);
        $display("\n[ERROR] Global simulation timeout reached");
        $finish;
    end

    initial begin
        $dumpfile("top.vcd");
        $dumpvars(0, tb_top);
    end

endmodule
