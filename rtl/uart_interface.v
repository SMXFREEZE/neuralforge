//============================================================================
// NeuralForge — UART Interface (8N1)
// 115200 baud UART transceiver for host ↔ FPGA communication
// Handles byte-level framing; higher-level packet protocol in top.v
//============================================================================

module uart_interface #(
    parameter CLK_FREQ  = 100_000_000,  // 100 MHz default
    parameter BAUD_RATE = 115_200
)(
    input  wire       clk,
    input  wire       rst_n,

    // Physical UART pins
    input  wire       rx,
    output wire       tx,

    // RX interface (received bytes)
    output reg  [7:0] rx_data,
    output reg        rx_valid,

    // TX interface (bytes to transmit)
    input  wire [7:0] tx_data,
    input  wire       tx_start,
    output reg        tx_busy,
    output reg        tx_done
);

    // Baud rate tick generation
    localparam TICKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam TICK_W = $clog2(TICKS_PER_BIT);

    //========================================================================
    // UART Receiver
    //========================================================================
    localparam RX_IDLE  = 2'd0,
               RX_START = 2'd1,
               RX_DATA  = 2'd2,
               RX_STOP  = 2'd3;

    reg [1:0]       rx_state;
    reg [TICK_W:0]  rx_tick_count;
    reg [2:0]       rx_bit_idx;
    reg [7:0]       rx_shift;
    reg             rx_sync_0, rx_sync_1; // Metastability guard

    // Double-register the RX input
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_sync_0 <= 1'b1;
            rx_sync_1 <= 1'b1;
        end else begin
            rx_sync_0 <= rx;
            rx_sync_1 <= rx_sync_0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state      <= RX_IDLE;
            rx_tick_count <= 0;
            rx_bit_idx    <= 0;
            rx_shift      <= 8'd0;
            rx_data       <= 8'd0;
            rx_valid      <= 1'b0;
        end else begin
            rx_valid <= 1'b0; // Default: one-cycle pulse

            case (rx_state)
                RX_IDLE: begin
                    if (rx_sync_1 == 1'b0) begin
                        rx_state      <= RX_START;
                        rx_tick_count <= 0;
                    end
                end

                RX_START: begin
                    // Wait half a bit period to sample mid-bit
                    if (rx_tick_count == (TICKS_PER_BIT / 2) - 1) begin
                        if (rx_sync_1 == 1'b0) begin
                            rx_state      <= RX_DATA;
                            rx_tick_count <= 0;
                            rx_bit_idx    <= 0;
                        end else begin
                            rx_state <= RX_IDLE; // False start
                        end
                    end else begin
                        rx_tick_count <= rx_tick_count + 1;
                    end
                end

                RX_DATA: begin
                    if (rx_tick_count == TICKS_PER_BIT - 1) begin
                        rx_tick_count <= 0;
                        rx_shift[rx_bit_idx] <= rx_sync_1;
                        if (rx_bit_idx == 3'd7) begin
                            rx_state <= RX_STOP;
                        end else begin
                            rx_bit_idx <= rx_bit_idx + 1;
                        end
                    end else begin
                        rx_tick_count <= rx_tick_count + 1;
                    end
                end

                RX_STOP: begin
                    if (rx_tick_count == TICKS_PER_BIT - 1) begin
                        rx_data  <= rx_shift;
                        rx_valid <= 1'b1;
                        rx_state <= RX_IDLE;
                    end else begin
                        rx_tick_count <= rx_tick_count + 1;
                    end
                end
            endcase
        end
    end

    //========================================================================
    // UART Transmitter
    //========================================================================
    localparam TX_IDLE  = 2'd0,
               TX_START = 2'd1,
               TX_DATA  = 2'd2,
               TX_STOP  = 2'd3;

    reg [1:0]       tx_state;
    reg [TICK_W:0]  tx_tick_count;
    reg [2:0]       tx_bit_idx;
    reg [7:0]       tx_shift;
    reg             tx_out;

    assign tx = tx_out;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state      <= TX_IDLE;
            tx_tick_count <= 0;
            tx_bit_idx    <= 0;
            tx_shift      <= 8'd0;
            tx_out        <= 1'b1;  // Idle high
            tx_busy       <= 1'b0;
            tx_done       <= 1'b0;
        end else begin
            tx_done <= 1'b0; // One-cycle pulse

            case (tx_state)
                TX_IDLE: begin
                    tx_out <= 1'b1;
                    if (tx_start) begin
                        tx_shift      <= tx_data;
                        tx_state      <= TX_START;
                        tx_tick_count <= 0;
                        tx_busy       <= 1'b1;
                    end
                end

                TX_START: begin
                    tx_out <= 1'b0; // Start bit
                    if (tx_tick_count == TICKS_PER_BIT - 1) begin
                        tx_tick_count <= 0;
                        tx_state      <= TX_DATA;
                        tx_bit_idx    <= 0;
                    end else begin
                        tx_tick_count <= tx_tick_count + 1;
                    end
                end

                TX_DATA: begin
                    tx_out <= tx_shift[tx_bit_idx];
                    if (tx_tick_count == TICKS_PER_BIT - 1) begin
                        tx_tick_count <= 0;
                        if (tx_bit_idx == 3'd7) begin
                            tx_state <= TX_STOP;
                        end else begin
                            tx_bit_idx <= tx_bit_idx + 1;
                        end
                    end else begin
                        tx_tick_count <= tx_tick_count + 1;
                    end
                end

                TX_STOP: begin
                    tx_out <= 1'b1; // Stop bit
                    if (tx_tick_count == TICKS_PER_BIT - 1) begin
                        tx_state <= TX_IDLE;
                        tx_busy  <= 1'b0;
                        tx_done  <= 1'b1;
                    end else begin
                        tx_tick_count <= tx_tick_count + 1;
                    end
                end
            endcase
        end
    end

endmodule
