//============================================================================
// NeuralForge — Weight Buffer (BRAM-based)
// Stores pre-trained INT8 weights for all network layers
// Initializable from .mem hex files for simulation and synthesis
//============================================================================

module weight_buffer #(
    parameter DEPTH     = 2048,     // Number of weight entries
    parameter ADDR_W    = 11,       // log2(DEPTH)
    parameter INIT_FILE = ""        // Optional .mem init file
)(
    input  wire              clk,
    input  wire              rst_n,

    // Port A: Write/load interface
    input  wire              wr_en,
    input  wire [ADDR_W-1:0] wr_addr,
    input  wire signed [7:0] wr_data,

    // Port B: Read interface
    input  wire [ADDR_W-1:0] rd_addr,
    output reg  signed [7:0] rd_data
);

    // Inferred BRAM
    reg signed [7:0] mem [0:DEPTH-1];

    // Optional initialization
    initial begin
        integer i;
        for (i = 0; i < DEPTH; i = i + 1)
            mem[i] = 8'sd0;
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    // Port A: Write
    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= wr_data;
    end

    // Port B: Read (synchronous)
    always @(posedge clk) begin
        if (rst_n)
            rd_data <= mem[rd_addr];
        else
            rd_data <= 8'sd0;
    end

endmodule
