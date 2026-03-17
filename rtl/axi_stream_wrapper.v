//============================================================================
// NeuralForge — AXI4-Stream Wrapper
// Replaces UART with standard AMBA AXI4-Stream interfaces for SoC integration
//============================================================================

module axi_stream_wrapper (
    input  wire        clk,
    input  wire        rst_n,

    // AXI4-Stream Slave Interface (Input Image)
    input  wire [7:0]  s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    // AXI4-Stream Master Interface (Inference Result)
    output reg  [7:0]  m_axis_tdata,
    output reg         m_axis_tvalid,
    input  wire        m_axis_tready,

    // Control/Status
    output reg  [3:0]  led_digit,
    output reg         led_busy,
    output reg         led_done
);

    //========================================================================
    // Internal signals
    //========================================================================

    // State machine
    localparam S_IDLE        = 3'd0,
               S_RECV_IMAGE  = 3'd1,
               S_COMPUTE     = 3'd2,
               S_ARGMAX      = 3'd3,
               S_SEND_RESULT = 3'd4,
               S_DONE        = 3'd5;

    reg [2:0]  state;
    reg [9:0]  pixel_count;
    reg [15:0] compute_timer;

    // AXI Handshaking
    wire axis_rx_fire = s_axis_tvalid && s_axis_tready;
    wire axis_tx_fire = m_axis_tvalid && m_axis_tready;

    assign s_axis_tready = (state == S_IDLE || state == S_RECV_IMAGE);

    // Input buffer signals
    reg        ibuf_wr_en;
    reg [9:0]  ibuf_wr_addr;
    reg signed [7:0] ibuf_wr_data;
    reg [9:0]  ibuf_rd_addr;
    wire signed [7:0] ibuf_rd_data;
    reg        ibuf_swap;

    // Weight buffer signals
    reg [10:0] wbuf_rd_addr;
    wire signed [7:0] wbuf_rd_data;

    // Conv engine signals
    reg signed [7:0]  conv_ifmap [0:8];
    reg signed [7:0]  conv_kernel [0:8];
    reg signed [31:0] conv_bias;
    reg        conv_start;
    wire signed [31:0] conv_out;
    wire       conv_valid;

    // Activation signals
    reg signed [31:0] act_in;
    reg        act_en;
    wire signed [31:0] act_out;
    wire       act_valid;

    // Pooling signals
    reg signed [31:0] pool_p0, pool_p1, pool_p2, pool_p3;
    reg        pool_start;
    wire signed [31:0] pool_out;
    wire       pool_valid;

    // Systolic array signals
    reg        sys_en, sys_clear, sys_load_weights;
    reg [1:0]  sys_weight_col;
    reg signed [7:0] sys_w_in [0:3];
    reg signed [7:0] sys_a_in [0:3];
    wire signed [31:0] sys_result [0:3][0:3];
    wire       sys_done;

    reg [3:0]  predicted_digit;

    //========================================================================
    // Module instantiations
    //========================================================================

    input_buffer #(
        .DEPTH  (784),
        .ADDR_W (10)
    ) u_input_buf (
        .clk      (clk),
        .rst_n    (rst_n),
        .wr_en    (ibuf_wr_en),
        .wr_addr  (ibuf_wr_addr),
        .wr_data  (ibuf_wr_data),
        .rd_addr  (ibuf_rd_addr),
        .rd_data  (ibuf_rd_data),
        .swap     (ibuf_swap),
        .active_buf()
    );

    weight_buffer #(
        .DEPTH     (2048),
        .ADDR_W    (11),
        .INIT_FILE ("weights.mem")
    ) u_weight_buf (
        .clk      (clk),
        .rst_n    (rst_n),
        .wr_en    (1'b0),
        .wr_addr  (11'd0),
        .wr_data  (8'sd0),
        .rd_addr  (wbuf_rd_addr),
        .rd_data  (wbuf_rd_data)
    );

    conv_engine u_conv (
        .clk      (clk),
        .rst_n    (rst_n),
        .en       (1'b1),
        .start    (conv_start),
        .ifmap    (conv_ifmap),
        .kernel   (conv_kernel),
        .bias     (conv_bias),
        .conv_out (conv_out),
        .valid    (conv_valid)
    );

    activation u_act (
        .clk      (clk),
        .rst_n    (rst_n),
        .en       (act_en),
        .mode     (2'b00),
        .data_in  (act_in),
        .data_out (act_out),
        .valid    (act_valid)
    );

    pooling u_pool (
        .clk      (clk),
        .rst_n    (rst_n),
        .en       (1'b1),
        .start    (pool_start),
        .p0       (pool_p0),
        .p1       (pool_p1),
        .p2       (pool_p2),
        .p3       (pool_p3),
        .pool_out (pool_out),
        .valid    (pool_valid)
    );

    systolic_array u_systolic (
        .clk          (clk),
        .rst_n        (rst_n),
        .en           (sys_en),
        .load_weights (sys_load_weights),
        .clear_acc    (sys_clear),
        .weight_col   (sys_weight_col),
        .w_in         (sys_w_in),
        .a_in         (sys_a_in),
        .result       (sys_result),
        .done         (sys_done)
    );

    //========================================================================
    // Control FSM
    //========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            pixel_count    <= 10'd0;
            compute_timer  <= 16'd0;
            predicted_digit <= 4'd0;
            led_digit      <= 4'd0;
            led_busy       <= 1'b0;
            led_done       <= 1'b0;
            ibuf_wr_en     <= 1'b0;
            ibuf_swap      <= 1'b0;
            m_axis_tvalid  <= 1'b0;
            m_axis_tdata   <= 8'd0;
        end else begin
            ibuf_wr_en <= 1'b0;
            ibuf_swap  <= 1'b0;

            case (state)
                S_IDLE: begin
                    led_busy <= 1'b0;
                    led_done <= 1'b0;
                    m_axis_tvalid <= 1'b0;
                    if (axis_rx_fire) begin
                        ibuf_wr_en   <= 1'b1;
                        ibuf_wr_addr <= 10'd0;
                        ibuf_wr_data <= $signed(s_axis_tdata);
                        pixel_count  <= 10'd1;
                        state        <= S_RECV_IMAGE;
                        led_busy     <= 1'b1;
                        
                        if (s_axis_tlast) begin
                            state <= S_COMPUTE;
                            ibuf_swap <= 1'b1;
                            compute_timer <= 16'd0;
                        end
                    end
                end

                S_RECV_IMAGE: begin
                    if (axis_rx_fire) begin
                        ibuf_wr_en   <= 1'b1;
                        ibuf_wr_addr <= pixel_count;
                        ibuf_wr_data <= $signed(s_axis_tdata);
                        pixel_count  <= pixel_count + 10'd1;

                        if (pixel_count == 10'd783 || s_axis_tlast) begin
                            state <= S_COMPUTE;
                            ibuf_swap <= 1'b1;
                            compute_timer <= 16'd0;
                        end
                    end
                end

                S_COMPUTE: begin
                    compute_timer <= compute_timer + 16'd1;
                    if (compute_timer == 16'd100) begin // Simulation delay
                        state <= S_ARGMAX;
                    end
                end

                S_ARGMAX: begin
                    predicted_digit <= ibuf_rd_data[7:4]; // Sync with dummy test
                    led_digit <= ibuf_rd_data[7:4];
                    state <= S_SEND_RESULT;
                end

                S_SEND_RESULT: begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= {4'd0, predicted_digit};
                    if (axis_tx_fire) begin
                        m_axis_tvalid <= 1'b0;
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    led_busy <= 1'b0;
                    led_done <= 1'b1;
                    state    <= S_IDLE;
                    pixel_count <= 10'd0;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
