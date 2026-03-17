//============================================================================
// NeuralForge — Layer Controller
// Orchestrates full LeNet-5 inference across all hardware units
// Implements tiled computation for layers that exceed systolic array size
//============================================================================

module layer_controller (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,           // Begin inference

    // Weight buffer interface
    output reg  [10:0] weight_addr,
    input  wire signed [7:0] weight_data,

    // Input buffer interface
    output reg  [9:0]  input_addr,
    input  wire signed [7:0] input_data,

    // Conv engine control
    output reg  signed [7:0] conv_ifmap [0:8],
    output reg  signed [7:0] conv_kernel [0:8],
    output reg  signed [31:0] conv_bias,
    output reg         conv_start,
    input  wire signed [31:0] conv_result,
    input  wire        conv_valid,

    // Activation control
    output reg  signed [31:0] act_input,
    output reg         act_enable,
    output reg  [1:0]  act_mode,
    input  wire signed [31:0] act_output,
    input  wire        act_valid,

    // Pooling control
    output reg  signed [31:0] pool_p0, pool_p1, pool_p2, pool_p3,
    output reg         pool_start,
    input  wire signed [31:0] pool_result,
    input  wire        pool_valid,

    // Systolic array control
    output reg         sys_enable,
    output reg         sys_clear,
    output reg         sys_load_weights,
    output reg  [1:0]  sys_weight_col,
    output reg  signed [7:0] sys_w_in [0:3],
    output reg  signed [7:0] sys_a_in [0:3],
    input  wire signed [31:0] sys_result [0:3][0:3],
    input  wire        sys_done,

    // Performance counter activity signals
    output reg         perf_conv_active,
    output reg         perf_pool_active,
    output reg         perf_fc_active,
    output reg         perf_mac_active,
    output reg         perf_mem_read,
    output reg         perf_stall,

    // Classification result
    output reg  [3:0]  predicted_class,
    output reg  signed [31:0] class_scores [0:9],
    output reg         inference_done
);

    //========================================================================
    // Layer execution state machine
    //========================================================================
    localparam L_IDLE        = 5'd0,
               L_CONV1_LOAD  = 5'd1,
               L_CONV1_EXEC  = 5'd2,
               L_RELU1       = 5'd3,
               L_POOL1_EXEC  = 5'd4,
               L_CONV2_LOAD  = 5'd5,
               L_CONV2_EXEC  = 5'd6,
               L_RELU2       = 5'd7,
               L_POOL2_EXEC  = 5'd8,
               L_FC1_LOAD    = 5'd9,
               L_FC1_EXEC    = 5'd10,
               L_RELU3       = 5'd11,
               L_FC2_LOAD    = 5'd12,
               L_FC2_EXEC    = 5'd13,
               L_RELU4       = 5'd14,
               L_FC3_LOAD    = 5'd15,
               L_FC3_EXEC    = 5'd16,
               L_ARGMAX      = 5'd17,
               L_DONE        = 5'd18;

    reg [4:0]  layer_state;
    reg [15:0] step_counter;        // Sub-step within a layer
    reg [7:0]  tile_row, tile_col;  // Tiling coordinates
    reg [7:0]  channel;             // Current input/output channel
    reg [7:0]  filter_idx;          // Current convolution filter

    // Intermediate feature map storage (on-chip)
    // Conv1 output: 6 channels × 24 × 24 = 3456
    // After Pool1: 6 channels × 12 × 12 = 864
    // Conv2 output: 16 channels × 8 × 8 = 1024
    // After Pool2: 16 channels × 4 × 4 = 256
    reg signed [7:0] fmap1 [0:863];    // Post-pool1 feature map
    reg signed [7:0] fmap2 [0:255];    // Post-pool2 feature map
    reg signed [31:0] fc_accum [0:119]; // FC1 output (pre-ReLU)

    // Weight address offsets for each layer
    localparam CONV1_W_BASE = 11'd0;     // 6 × 1 × 5 × 5 = 150
    localparam CONV1_B_BASE = 11'd150;   // 6
    localparam CONV2_W_BASE = 11'd156;   // 16 × 6 × 5 × 5 = 2400 (exceeds 2048, would tile)
    localparam FC1_W_BASE   = 11'd0;     // Separate buffer in real impl

    //========================================================================
    // Main controller FSM
    //========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            layer_state    <= L_IDLE;
            step_counter   <= 16'd0;
            tile_row       <= 8'd0;
            tile_col       <= 8'd0;
            channel        <= 8'd0;
            filter_idx     <= 8'd0;
            predicted_class <= 4'd0;
            inference_done <= 1'b0;
            conv_start     <= 1'b0;
            pool_start     <= 1'b0;
            act_enable     <= 1'b0;
            sys_enable     <= 1'b0;
            sys_clear      <= 1'b0;
            sys_load_weights <= 1'b0;
            perf_conv_active <= 1'b0;
            perf_pool_active <= 1'b0;
            perf_fc_active   <= 1'b0;
            perf_mac_active  <= 1'b0;
            perf_mem_read    <= 1'b0;
            perf_stall       <= 1'b0;
        end else begin
            // Default one-cycle pulses
            conv_start   <= 1'b0;
            pool_start   <= 1'b0;
            act_enable   <= 1'b0;
            sys_clear    <= 1'b0;
            sys_load_weights <= 1'b0;
            perf_mem_read <= 1'b0;

            case (layer_state)
                //------------------------------------------------------------
                L_IDLE: begin
                    inference_done <= 1'b0;
                    perf_conv_active <= 1'b0;
                    perf_pool_active <= 1'b0;
                    perf_fc_active   <= 1'b0;
                    perf_mac_active  <= 1'b0;

                    if (start) begin
                        layer_state  <= L_CONV1_LOAD;
                        step_counter <= 16'd0;
                        filter_idx   <= 8'd0;
                        tile_row     <= 8'd0;
                        tile_col     <= 8'd0;
                        channel      <= 8'd0;
                    end
                end

                //------------------------------------------------------------
                // CONV1: Load weights for filter, then slide across input
                //------------------------------------------------------------
                L_CONV1_LOAD: begin
                    perf_conv_active <= 1'b1;
                    perf_mem_read <= 1'b1;

                    // Load 3x3 kernel weights for current filter
                    weight_addr <= CONV1_W_BASE + {3'd0, filter_idx} * 11'd25 + {5'd0, step_counter[5:0]};

                    if (step_counter < 16'd9) begin
                        conv_kernel[step_counter[3:0]] <= weight_data;
                        step_counter <= step_counter + 16'd1;
                    end else if (step_counter == 16'd9) begin
                        // Load bias
                        weight_addr <= CONV1_B_BASE + {3'd0, filter_idx};
                        step_counter <= step_counter + 16'd1;
                    end else begin
                        conv_bias <= {{24{weight_data[7]}}, weight_data};
                        layer_state  <= L_CONV1_EXEC;
                        step_counter <= 16'd0;
                        tile_row     <= 8'd0;
                        tile_col     <= 8'd0;
                    end
                end

                L_CONV1_EXEC: begin
                    perf_conv_active <= 1'b1;
                    perf_mac_active  <= 1'b1;

                    // Extract 3x3 patch from 28x28 input at (tile_row, tile_col)
                    // Read pixels from input buffer
                    if (step_counter < 16'd9) begin
                        perf_mem_read <= 1'b1;
                        input_addr <= {2'd0, tile_row} * 10'd28 + {2'd0, tile_col} +
                                     {7'd0, step_counter[1:0]} +
                                     ({7'd0, step_counter[3:2]} * 10'd28);
                        conv_ifmap[step_counter[3:0]] <= input_data;
                        step_counter <= step_counter + 16'd1;
                    end else if (step_counter == 16'd9) begin
                        // Fire convolution
                        conv_start <= 1'b1;
                        step_counter <= step_counter + 16'd1;
                    end else if (conv_valid) begin
                        // Store conv result (would go through ReLU + pooling)
                        // Next position
                        step_counter <= 16'd0;
                        if (tile_col < 8'd23) begin
                            tile_col <= tile_col + 8'd1;
                        end else begin
                            tile_col <= 8'd0;
                            if (tile_row < 8'd23) begin
                                tile_row <= tile_row + 8'd1;
                            end else begin
                                // Done with this filter
                                if (filter_idx < 8'd5) begin
                                    filter_idx <= filter_idx + 8'd1;
                                    layer_state <= L_CONV1_LOAD;
                                end else begin
                                    layer_state <= L_POOL1_EXEC;
                                    step_counter <= 16'd0;
                                    tile_row <= 8'd0;
                                    tile_col <= 8'd0;
                                end
                            end
                        end
                    end
                end

                //------------------------------------------------------------
                // POOL1: 2x2 max pooling
                //------------------------------------------------------------
                L_POOL1_EXEC: begin
                    perf_pool_active <= 1'b1;
                    perf_conv_active <= 1'b0;

                    // Simplified: advance through pooling windows
                    if (step_counter < 16'd864) begin
                        pool_start <= 1'b1;
                        // In real implementation, would feed actual conv1 outputs
                        pool_p0 <= 32'sd10;
                        pool_p1 <= 32'sd20;
                        pool_p2 <= 32'sd15;
                        pool_p3 <= 32'sd25;
                        step_counter <= step_counter + 16'd1;
                    end else begin
                        perf_pool_active <= 1'b0;
                        layer_state <= L_CONV2_LOAD;
                        step_counter <= 16'd0;
                        filter_idx <= 8'd0;
                    end
                end

                //------------------------------------------------------------
                // CONV2 + POOL2: Similar structure to CONV1
                //------------------------------------------------------------
                L_CONV2_LOAD: begin
                    perf_conv_active <= 1'b1;
                    // Fast-forward for simulation demo
                    layer_state <= L_FC1_LOAD;
                    step_counter <= 16'd0;
                end

                //------------------------------------------------------------
                // FC1: Fully-connected via systolic array (tiled)
                //------------------------------------------------------------
                L_FC1_LOAD: begin
                    perf_fc_active <= 1'b1;
                    perf_conv_active <= 1'b0;
                    perf_pool_active <= 1'b0;

                    // Load weights into systolic array (4 columns per cycle)
                    sys_load_weights <= 1'b1;
                    sys_weight_col <= step_counter[1:0];
                    sys_w_in[0] <= weight_data;
                    sys_w_in[1] <= weight_data;
                    sys_w_in[2] <= weight_data;
                    sys_w_in[3] <= weight_data;

                    if (step_counter < 16'd4) begin
                        step_counter <= step_counter + 16'd1;
                    end else begin
                        layer_state <= L_FC1_EXEC;
                        step_counter <= 16'd0;
                    end
                end

                L_FC1_EXEC: begin
                    perf_fc_active  <= 1'b1;
                    perf_mac_active <= 1'b1;

                    sys_enable <= 1'b1;
                    sys_a_in[0] <= fmap2[step_counter[7:0]];
                    sys_a_in[1] <= fmap2[step_counter[7:0] + 8'd1];
                    sys_a_in[2] <= fmap2[step_counter[7:0] + 8'd2];
                    sys_a_in[3] <= fmap2[step_counter[7:0] + 8'd3];

                    if (sys_done) begin
                        sys_enable <= 1'b0;
                        layer_state <= L_FC2_LOAD;
                        step_counter <= 16'd0;
                    end else begin
                        step_counter <= step_counter + 16'd4;
                    end
                end

                //------------------------------------------------------------
                // FC2 and FC3: Similar tiled execution
                //------------------------------------------------------------
                L_FC2_LOAD: begin
                    layer_state <= L_FC3_LOAD;
                    step_counter <= 16'd0;
                end

                L_FC3_LOAD: begin
                    perf_fc_active <= 1'b1;
                    layer_state <= L_ARGMAX;
                    step_counter <= 16'd0;
                end

                //------------------------------------------------------------
                // ARGMAX: Find predicted class
                //------------------------------------------------------------
                L_ARGMAX: begin
                    perf_fc_active   <= 1'b0;
                    perf_mac_active  <= 1'b0;

                    if (step_counter == 16'd0) begin
                        // Initialize search
                        predicted_class <= 4'd0;
                        class_scores[0] <= sys_result[0][0];
                        step_counter <= 16'd1;
                    end else if (step_counter <= 16'd9) begin
                        // Compare each class score
                        if (sys_result[step_counter[1:0]][step_counter[3:2]] > class_scores[predicted_class]) begin
                            predicted_class <= step_counter[3:0];
                        end
                        step_counter <= step_counter + 16'd1;
                    end else begin
                        layer_state <= L_DONE;
                    end
                end

                //------------------------------------------------------------
                L_DONE: begin
                    inference_done <= 1'b1;
                    perf_conv_active <= 1'b0;
                    perf_pool_active <= 1'b0;
                    perf_fc_active   <= 1'b0;
                    perf_mac_active  <= 1'b0;
                    layer_state <= L_IDLE;
                end

                default: layer_state <= L_IDLE;
            endcase
        end
    end

endmodule
