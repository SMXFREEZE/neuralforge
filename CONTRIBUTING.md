# Contributing to NeuralForge

First off, thank you for considering contributing to NeuralForge! This project aims to be a top-tier open-source educational resource and portfolio piece demonstratiing full-stack hardware/software integration for Machine Learning on FPGAs.

## Getting Started

1. **Fork the repository** to your own GitHub account.
2. **Clone the project** to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/neuralforge.git
   cd neuralforge
   ```
3. **Install dependencies**:
   ```bash
   pip install -r sw/requirements.txt
   ```

## Development Workflow

### Python Software Pipeline (`sw/`)
- Ensure any modifications to the `FPGASimulator` remain cycle-accurate and mathematically identical to the Verilog hardware (INT8 inputs, INT32 accumulators).
- Performance optimizations to the simulator (e.g., NumPy vectorization) are welcome as long as they do not change the underlying mathematical operations.
- Run the accuracy testing suite (`python sw/dashboard_server.py` and use the UI batch tester) to verify your changes haven't degraded model accuracy.

### Verilog Hardware (`rtl/`)
- All new hardware modules must have corresponding testbenches in `sim/`.
- Verilog code should be strictly synthesizable (no initial blocks outside of testbenches, proper reset logic).
- Follow the existing pipeline timing: 1 cycle per MAC array step, with data flowing via Valid/Ready handshakes.

### Web Dashboard (`dashboard/`)
- The dashboard is built with vanilla JavaScript, HTML, and CSS to keep dependencies zero. Please do not introduce large frameworks (React/Vue/etc.) unless absolutely necessary.
- Maintain the premium dark-mode aesthetic. Use CSS variables defined in `style.css` for colors and spacing.

## Submitting Pull Requests

1. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name`
2. Make your changes and commit with clear, descriptive messages.
3. Push your branch to your fork: `git push origin feature/your-feature-name`
4. Open a Pull Request against the `main` branch of this repository.

Please include screenshots or performance metrics if your PR changes the UI or improves simulation/hardware throughput.

## License

By contributing to NeuralForge, you agree that your contributions will be licensed under the project's MIT License.
