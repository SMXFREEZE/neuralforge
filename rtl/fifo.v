//============================================================================
// NeuralForge — FIFO Buffer
// Synchronous FIFO for pipeline decoupling between stages
// Used between conv→pool and pool→fc transitions
//============================================================================

module fifo #(
    parameter DATA_W = 32,
    parameter DEPTH  = 64,
    parameter ADDR_W = 6       // log2(DEPTH)
)(
    input  wire              clk,
    input  wire              rst_n,

    // Write interface
    input  wire              wr_en,
    input  wire [DATA_W-1:0] wr_data,
    output wire              full,

    // Read interface
    input  wire              rd_en,
    output wire [DATA_W-1:0] rd_data,
    output wire              empty,

    // Status
    output wire [ADDR_W:0]   count
);

    // Memory array
    reg [DATA_W-1:0] mem [0:DEPTH-1];

    // Pointers
    reg [ADDR_W:0] wr_ptr;  // Extra bit for full/empty detection
    reg [ADDR_W:0] rd_ptr;

    // Derived signals
    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_ptr[ADDR_W] != rd_ptr[ADDR_W]) &&
                   (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]);
    assign count = wr_ptr - rd_ptr;
    assign rd_data = mem[rd_ptr[ADDR_W-1:0]];

    // Write logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
        end else if (wr_en && !full) begin
            mem[wr_ptr[ADDR_W-1:0]] <= wr_data;
            wr_ptr <= wr_ptr + 1;
        end
    end

    // Read logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= 0;
        end else if (rd_en && !empty) begin
            rd_ptr <= rd_ptr + 1;
        end
    end

endmodule
