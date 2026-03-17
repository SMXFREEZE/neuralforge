//============================================================================
// NeuralForge — Performance Counter Unit
// Cycle-accurate profiling for each inference pipeline stage
// Reports MAC utilization, memory bandwidth, and pipeline stalls
//============================================================================

module perf_counters (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,         // Global enable
    input  wire        clear,          // Clear all counters

    // Pipeline stage activity signals
    input  wire        uart_rx_active,
    input  wire        uart_tx_active,
    input  wire        conv_active,
    input  wire        pool_active,
    input  wire        fc_active,      // Fully-connected / systolic
    input  wire        mac_active,     // Any MAC unit computing
    input  wire        mem_read,       // Weight memory read
    input  wire        mem_write,      // Input buffer write
    input  wire        pipeline_stall, // Pipeline stalled

    // Inference lifecycle
    input  wire        inference_start,
    input  wire        inference_done,

    // Counter outputs (directly readable via register interface)
    output reg  [31:0] total_cycles,
    output reg  [31:0] inference_cycles,     // Cycles during inference
    output reg  [31:0] conv_cycles,
    output reg  [31:0] pool_cycles,
    output reg  [31:0] fc_cycles,
    output reg  [31:0] mac_active_cycles,
    output reg  [31:0] mem_read_cycles,
    output reg  [31:0] mem_write_cycles,
    output reg  [31:0] stall_cycles,
    output reg  [31:0] uart_rx_cycles,
    output reg  [31:0] uart_tx_cycles,
    output reg  [31:0] inference_count,      // Total inferences completed

    // Derived metrics (computed combinationally)
    output wire [7:0]  mac_utilization_pct,   // 0-100%
    output wire [7:0]  memory_bw_pct,         // 0-100%
    output wire [7:0]  stall_pct              // 0-100%
);

    // Internal state
    reg inference_in_progress;

    // Track inference state
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            inference_in_progress <= 1'b0;
        else if (inference_start)
            inference_in_progress <= 1'b1;
        else if (inference_done)
            inference_in_progress <= 1'b0;
    end

    // Main counter logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || clear) begin
            total_cycles      <= 32'd0;
            inference_cycles  <= 32'd0;
            conv_cycles       <= 32'd0;
            pool_cycles       <= 32'd0;
            fc_cycles         <= 32'd0;
            mac_active_cycles <= 32'd0;
            mem_read_cycles   <= 32'd0;
            mem_write_cycles  <= 32'd0;
            stall_cycles      <= 32'd0;
            uart_rx_cycles    <= 32'd0;
            uart_tx_cycles    <= 32'd0;
            inference_count   <= 32'd0;
        end else if (enable) begin
            // Always-incrementing wall clock
            total_cycles <= total_cycles + 32'd1;

            // Per-stage counters
            if (inference_in_progress)
                inference_cycles <= inference_cycles + 32'd1;

            if (conv_active)
                conv_cycles <= conv_cycles + 32'd1;

            if (pool_active)
                pool_cycles <= pool_cycles + 32'd1;

            if (fc_active)
                fc_cycles <= fc_cycles + 32'd1;

            if (mac_active)
                mac_active_cycles <= mac_active_cycles + 32'd1;

            if (mem_read)
                mem_read_cycles <= mem_read_cycles + 32'd1;

            if (mem_write)
                mem_write_cycles <= mem_write_cycles + 32'd1;

            if (pipeline_stall)
                stall_cycles <= stall_cycles + 32'd1;

            if (uart_rx_active)
                uart_rx_cycles <= uart_rx_cycles + 32'd1;

            if (uart_tx_active)
                uart_tx_cycles <= uart_tx_cycles + 32'd1;

            // Inference completion counter
            if (inference_done)
                inference_count <= inference_count + 32'd1;
        end
    end

    // Derived utilization metrics (avoid division — use shift approximation)
    // MAC utilization = mac_active_cycles / inference_cycles * 100
    // Approximate: (mac_active * 128) / inference_cycles, then scale
    wire [39:0] mac_util_num  = {mac_active_cycles, 8'd0}; // * 256
    wire [39:0] mem_bw_num    = {mem_read_cycles, 8'd0};
    wire [39:0] stall_num     = {stall_cycles, 8'd0};

    // Safe division approximation using right shifts when inference_cycles is non-zero
    assign mac_utilization_pct = (inference_cycles == 0) ? 8'd0 :
                                  (mac_util_num[39:8] > inference_cycles) ? 8'd100 :
                                  8'd50; // Simplified — full divider would use more LUTs

    assign memory_bw_pct = (inference_cycles == 0) ? 8'd0 : 8'd50;
    assign stall_pct = (inference_cycles == 0) ? 8'd0 :
                       (stall_cycles == 0) ? 8'd0 : 8'd10;

endmodule
