# Contributing to NeuralForge

First off, thank you for considering contributing to NeuralForge! This project aims to be a top-tier open-source educational resource and portfolio piece demonstrating full-stack hardware/software integration for Machine Learning on FPGAs.

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
- Ensure any modifications to the `FPGASimulator` keep the arithmetic rules of the hardware blocks (INT8 inputs, INT32 accumulators) and keep the cycle model deterministic.
- Performance optimizations to the simulator (e.g., NumPy vectorization) are welcome as long as they do not change the underlying mathematical operations.
- Run the test suite (`python -m pytest tests/`) to verify your changes haven't degraded model accuracy or broken the cycle model.

### Verilog Hardware (`rtl/`)
- All new hardware modules must have corresponding testbenches in `sim/`.
- Verilog code should be strictly synthesizable (no initial blocks outside of testbenches, proper reset logic).
- Follow the existing pipeline timing: 1 cycle per MAC array step, with data flowing via Valid/Ready handshakes.

### Web Dashboard (`app/` + `components/`)
- The dashboard is a Next.js 16 + React 19 app with canvas-drawn charts; please do not add chart libraries or heavy dependencies unless absolutely necessary.
- Maintain the premium dark-mode aesthetic. Use the CSS variables defined in `app/globals.css` for colors and spacing.
- `npm run lint` and `npm run build` must pass (CI enforces both).
- The original vanilla-JS dashboard lives in `legacy/` and is frozen except for bug fixes.

## Submitting Pull Requests

1. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name`
2. Make your changes and commit with clear, descriptive messages.
3. Push your branch to your fork: `git push origin feature/your-feature-name`
4. Open a Pull Request against the `main` branch of this repository.

Please include screenshots or performance metrics if your PR changes the UI or improves simulation/hardware throughput.

## License

By contributing to NeuralForge, you agree that your contributions will be licensed under the project's MIT License.
