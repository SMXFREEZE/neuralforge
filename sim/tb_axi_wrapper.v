`timescale 1ns / 1ps

module tb_axi_wrapper();

    // Clock and Reset
    reg clk;
    reg rst_n;

    // AXI4-Stream Slave Interface
    reg  [7:0] s_axis_tdata;
    reg        s_axis_tvalid;
    wire       s_axis_tready;
    reg        s_axis_tlast;

    // AXI4-Stream Master Interface
    wire [7:0] m_axis_tdata;
    wire       m_axis_tvalid;
    reg        m_axis_tready;

    // Status
    wire [3:0] led_digit;
    wire       led_busy;
    wire       led_done;

    // Instantiation
    axi_stream_wrapper uut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .led_digit(led_digit),
        .led_busy(led_busy),
        .led_done(led_done)
    );

    // Clock Generation (100 MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    // Test Sequence
    integer i;
    initial begin
        // Initialize Inputs
        rst_n = 0;
        s_axis_tdata = 8'd0;
        s_axis_tvalid = 0;
        s_axis_tlast = 0;
        m_axis_tready = 1; // Always ready to receive by default

        // Wait 100 ns for global reset to finish
        #100;
        rst_n = 1;

        // Wait a bit
        #50;

        $display("--- Starting AXI4-Stream Transfer ---");
        
        // Feed 784 bytes (MNIST image) to S_AXIS
        for (i = 0; i < 784; i = i + 1) begin
            s_axis_tdata = i[7:0]; // Dummy pixel data
            s_axis_tvalid = 1;
            if (i == 783) 
                s_axis_tlast = 1;
            else
                s_axis_tlast = 0;

            // Wait until the slave is ready
            wait(s_axis_tready);
            
            // Random backpressure (Host stalling occasionally)
            if ($random % 5 == 0) begin
                s_axis_tvalid = 0;
                #20;
                s_axis_tvalid = 1;
            end
            
            // Advance clock to push data
            @(posedge clk);
            #1; // Align out of clock edge
        end

        s_axis_tvalid = 0;
        s_axis_tlast = 0;
        $display("--- Transfer Complete. Waiting for Computation ---");

        // Wait for prediction
        wait(m_axis_tvalid);
        $display("Result Output: %d", m_axis_tdata);
        
        // Ack the result
        @(posedge clk);
        #1;

        if (led_done)
            $display("--- Test Passed ---");
        else
            $display("--- Test Failed: led_done not set ---");

        #100;
        $finish;
    end

    // Monitor
    initial begin
        $dumpfile("axi_wrapper.vcd");
        $dumpvars(0, tb_axi_wrapper);
    end

endmodule
