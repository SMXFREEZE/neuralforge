//============================================================================
// NeuralForge — Register File (Memory-Mapped I/O)
// Provides host-accessible registers for configuration, status, and
// performance counter readout. Compatible with AXI-Lite adaptation.
//============================================================================

module register_file (
    input  wire        clk,
    input  wire        rst_n,

    // Simple bus interface (UART-driven or AXI-Lite adaptable)
    input  wire        bus_wr_en,
    input  wire        bus_rd_en,
    input  wire [7:0]  bus_addr,       // 256 register addresses
    input  wire [31:0] bus_wr_data,
    output reg  [31:0] bus_rd_data,
    output reg         bus_rd_valid,

    // Control outputs (directly drive hardware)
    output reg         ctrl_start_inference,
    output reg         ctrl_soft_reset,
    output reg         ctrl_perf_clear,
    output reg         ctrl_perf_enable,
    output reg  [1:0]  ctrl_activation_mode,
    output reg  [15:0] ctrl_uart_divisor,

    // Status inputs (directly from hardware)
    input  wire        stat_busy,
    input  wire        stat_done,
    input  wire        stat_error,
    input  wire [3:0]  stat_predicted_class,
    input  wire [31:0] stat_inference_count,

    // Performance counter inputs
    input  wire [31:0] perf_total_cycles,
    input  wire [31:0] perf_inference_cycles,
    input  wire [31:0] perf_conv_cycles,
    input  wire [31:0] perf_pool_cycles,
    input  wire [31:0] perf_fc_cycles,
    input  wire [31:0] perf_mac_cycles,
    input  wire [31:0] perf_mem_rd_cycles,
    input  wire [31:0] perf_mem_wr_cycles,
    input  wire [31:0] perf_stall_cycles,
    input  wire [7:0]  perf_mac_util,
    input  wire [7:0]  perf_mem_bw,
    input  wire [7:0]  perf_stall_pct
);

    // Register map
    localparam REG_VERSION       = 8'h00;  // Read-only: version ID
    localparam REG_CONTROL       = 8'h04;  // Write: control bits
    localparam REG_STATUS        = 8'h08;  // Read: status
    localparam REG_RESULT        = 8'h0C;  // Read: predicted class
    localparam REG_INF_COUNT     = 8'h10;  // Read: inference count
    localparam REG_ACT_MODE      = 8'h14;  // R/W: activation mode
    localparam REG_UART_DIV      = 8'h18;  // R/W: UART baud divisor

    // Performance counter registers (read-only)
    localparam REG_PERF_TOTAL    = 8'h40;
    localparam REG_PERF_INFER    = 8'h44;
    localparam REG_PERF_CONV     = 8'h48;
    localparam REG_PERF_POOL     = 8'h4C;
    localparam REG_PERF_FC       = 8'h50;
    localparam REG_PERF_MAC      = 8'h54;
    localparam REG_PERF_MEM_RD   = 8'h58;
    localparam REG_PERF_MEM_WR   = 8'h5C;
    localparam REG_PERF_STALL    = 8'h60;
    localparam REG_PERF_UTIL     = 8'h64;  // Packed utilization metrics

    // Version constant
    localparam VERSION = 32'h4E_46_01_00; // "NF" v1.0

    // Write logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_start_inference <= 1'b0;
            ctrl_soft_reset      <= 1'b0;
            ctrl_perf_clear      <= 1'b0;
            ctrl_perf_enable     <= 1'b1; // Enabled by default
            ctrl_activation_mode <= 2'b00; // ReLU default
            ctrl_uart_divisor    <= 16'd868; // 115200 at 100MHz
        end else begin
            // Auto-clear pulse registers
            ctrl_start_inference <= 1'b0;
            ctrl_soft_reset      <= 1'b0;
            ctrl_perf_clear      <= 1'b0;

            if (bus_wr_en) begin
                case (bus_addr)
                    REG_CONTROL: begin
                        ctrl_start_inference <= bus_wr_data[0];
                        ctrl_soft_reset      <= bus_wr_data[1];
                        ctrl_perf_clear      <= bus_wr_data[2];
                        ctrl_perf_enable     <= bus_wr_data[3];
                    end
                    REG_ACT_MODE:
                        ctrl_activation_mode <= bus_wr_data[1:0];
                    REG_UART_DIV:
                        ctrl_uart_divisor <= bus_wr_data[15:0];
                endcase
            end
        end
    end

    // Read logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bus_rd_data  <= 32'd0;
            bus_rd_valid <= 1'b0;
        end else begin
            bus_rd_valid <= 1'b0;

            if (bus_rd_en) begin
                bus_rd_valid <= 1'b1;

                case (bus_addr)
                    REG_VERSION:    bus_rd_data <= VERSION;
                    REG_STATUS:     bus_rd_data <= {28'd0, stat_error, stat_done, stat_busy, 1'b0};
                    REG_RESULT:     bus_rd_data <= {28'd0, stat_predicted_class};
                    REG_INF_COUNT:  bus_rd_data <= stat_inference_count;
                    REG_ACT_MODE:   bus_rd_data <= {30'd0, ctrl_activation_mode};
                    REG_UART_DIV:   bus_rd_data <= {16'd0, ctrl_uart_divisor};

                    REG_PERF_TOTAL:  bus_rd_data <= perf_total_cycles;
                    REG_PERF_INFER:  bus_rd_data <= perf_inference_cycles;
                    REG_PERF_CONV:   bus_rd_data <= perf_conv_cycles;
                    REG_PERF_POOL:   bus_rd_data <= perf_pool_cycles;
                    REG_PERF_FC:     bus_rd_data <= perf_fc_cycles;
                    REG_PERF_MAC:    bus_rd_data <= perf_mac_cycles;
                    REG_PERF_MEM_RD: bus_rd_data <= perf_mem_rd_cycles;
                    REG_PERF_MEM_WR: bus_rd_data <= perf_mem_wr_cycles;
                    REG_PERF_STALL:  bus_rd_data <= perf_stall_cycles;
                    REG_PERF_UTIL:   bus_rd_data <= {8'd0, perf_stall_pct, perf_mem_bw, perf_mac_util};

                    default: bus_rd_data <= 32'hDEAD_BEEF;
                endcase
            end
        end
    end

endmodule
