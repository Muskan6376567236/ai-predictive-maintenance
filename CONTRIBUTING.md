# Contributing to AI Predictive Maintenance System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-predictive-maintenance.git
cd ai-predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## Code Standards

- Follow PEP 8 style guide
- Write docstrings for all functions and classes
- Add type hints where applicable
- Ensure all tests pass before submitting

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Add tests for any new functionality
3. Ensure all CI checks pass
4. Request review from maintainers

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
