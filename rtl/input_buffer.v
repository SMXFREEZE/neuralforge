//============================================================================
// NeuralForge — Input Feature Map Buffer (Ping-Pong)
// Double-buffered input storage for streaming tile processing
// While one buffer feeds the compute pipeline, the other receives new data
//============================================================================

module input_buffer #(
    parameter DEPTH  = 784,      // 28x28 = 784 pixels for MNIST
    parameter ADDR_W = 10        // log2(DEPTH)
)(
    input  wire              clk,
    input  wire              rst_n,

    // Write interface (from UART/host)
    input  wire              wr_en,
    input  wire [ADDR_W-1:0] wr_addr,
    input  wire signed [7:0] wr_data,

    // Read interface (to compute pipeline)
    input  wire [ADDR_W-1:0] rd_addr,
    output wire signed [7:0] rd_data,

    // Buffer control
    input  wire              swap,          // Swap active/inactive buffer
    output reg               active_buf     // Which buffer is active (0 or 1)
);

    // Two buffers
    reg signed [7:0] buf0 [0:DEPTH-1];
    reg signed [7:0] buf1 [0:DEPTH-1];

    // Initialize
    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            buf0[i] = 8'sd0;
            buf1[i] = 8'sd0;
        end
        active_buf = 1'b0;
    end

    // Swap logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            active_buf <= 1'b0;
        else if (swap)
            active_buf <= ~active_buf;
    end

    // Write goes to INACTIVE buffer
    always @(posedge clk) begin
        if (wr_en) begin
            if (active_buf == 1'b0)
                buf1[wr_addr] <= wr_data;  // active=0, write to buf1
            else
                buf0[wr_addr] <= wr_data;  // active=1, write to buf0
        end
    end

    // Read comes from ACTIVE buffer
    assign rd_data = (active_buf == 1'b0) ? buf0[rd_addr] : buf1[rd_addr];

endmodule
