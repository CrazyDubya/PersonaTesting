# Changelog

All notable changes to the Persona Evaluation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-12-17

### Added
- Comprehensive `.gitignore` file to exclude Python cache and build artifacts
- Development dependencies file (`requirements-dev.txt`) with linting and testing tools
- Better error messages throughout the codebase for easier debugging
- Input validation in configuration loading to catch issues early
- Edge case handling for empty responses and invalid data
- Enhanced README with troubleshooting, development, and contribution sections
- CHANGELOG to track project improvements

### Changed
- Improved error handling in `model_api.py` with detailed error messages
- Enhanced CLI error messages with helpful examples
- Optimized performance in `scoring.py` using generator expressions
- Updated `pyproject.toml` with correct repository URLs and Python 3.8+ support
- Better I/O error handling with detailed messages in `io_utils.py`
- More robust question loading with validation and detailed warnings

### Fixed
- Configuration validation now catches missing required fields
- Token count estimation handles empty strings correctly
- Confidence interval computation clamps accuracy to valid range [0, 1]
- File I/O operations now handle missing directories automatically

### Security
- No known security vulnerabilities

## [0.1.0] - 2025-12-09

### Added
- Initial release of persona evaluation framework
- Support for OpenAI, Anthropic, and OpenRouter providers
- Six experimental conditions (baseline, shallow/deep personas, etc.)
- MCQ and open-ended question support with judge-based scoring
- Comprehensive metrics: accuracy, refusal rate, robust correctness thresholds
- Sample GPQA and MMLU-Pro data files
- CLI with full experiment and quick test modes
- Retry logic with exponential backoff for API calls
- JSONL-based data format for questions and results

[0.1.1]: https://github.com/CrazyDubya/PersonaTesting/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/CrazyDubya/PersonaTesting/releases/tag/v0.1.0
