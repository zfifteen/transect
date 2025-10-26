# Contributing to TRANSEC

Thank you for your interest in contributing to TRANSEC! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/transec.git
   cd transec
   ```
3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Requirements

- Python 3.8 or higher
- cryptography >= 42.0.0
- pytest >= 7.0.0 (for testing)
- mpmath >= 1.3.0 (optional, for prime optimization)

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=transec --cov-report=html

# Run specific test file
python tests/test_transec.py
python tests/test_advanced.py
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Keep lines under 100 characters when practical

### Testing

- Write unit tests for all new features
- Ensure all tests pass before submitting PR
- Aim for >90% code coverage
- Test edge cases and error conditions

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/adaptive-slot-duration`
- `bugfix/replay-protection-cache`
- `docs/quickstart-guide`

### Commit Messages

Write clear commit messages:
```
Add adaptive slot duration feature

- Implement ChaCha20-based PRNG for jitter
- Add AdaptiveTransecCipher class
- Include tests for deterministic variation
- Update documentation

Resolves #42
```

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make your changes** with clear commits

3. **Run tests** to ensure nothing breaks:
   ```bash
   python -m pytest tests/
   ```

4. **Update documentation** if needed

5. **Push to your fork**:
   ```bash
   git push origin feature/my-feature
   ```

6. **Open a Pull Request** on GitHub

### Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include test results
- Update CHANGELOG.md if applicable
- Ensure CI passes

## Types of Contributions

### Bug Reports

When filing a bug report, include:
- Python version and OS
- TRANSEC version
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Error messages and tracebacks

### Feature Requests

When suggesting a feature:
- Describe the use case
- Explain why it's valuable
- Provide examples if possible
- Consider implementation complexity

### Code Contributions

Priority areas:
- Additional variance reduction modes
- Hardware acceleration integration
- Post-quantum key derivation
- LoRaWAN/mesh network bindings
- Performance optimizations
- Documentation improvements

## Security

### Reporting Security Issues

**Do not** open public GitHub issues for security vulnerabilities.

Instead, email: security@example.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

### Security Review

All cryptographic changes require:
- Clear justification
- Reference to standards/papers
- Security analysis
- Review by project maintainers

## Documentation

### Types of Documentation

- **README.md**: Overview and quick start
- **QUICKSTART.md**: Getting started guide
- **docs/**: Detailed specifications
- **Docstrings**: In-code API documentation
- **Examples**: Working code samples

### Documentation Style

- Use clear, concise language
- Include code examples
- Provide context and rationale
- Link to relevant standards/RFCs

## Code Review Process

1. Maintainer reviews code and tests
2. Feedback provided via PR comments
3. Author addresses feedback
4. Maintainer approves and merges

Review criteria:
- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Security considerations

## Release Process

1. Update version in `setup.py` and `__init__.py`
2. Update CHANGELOG.md
3. Tag release: `git tag v0.2.0`
4. Build and publish to PyPI
5. Create GitHub release

## Community

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

### Getting Help

- GitHub Issues: For bugs and features
- GitHub Discussions: For questions and ideas
- Email: For sensitive topics

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Contributors will be listed in:
- README.md acknowledgments section
- Release notes
- Git commit history

Thank you for contributing to TRANSEC! 🚀
