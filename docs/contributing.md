# AI Trading Agent Contribution Guide

Thank you for your interest in contributing to the AI Trading Agent project! We welcome contributions to improve the agent's capabilities, fix bugs, and enhance documentation.

## Getting Started

1.  **Fork the Repository**: Click the "Fork" button on the top right of the GitHub repository page.
2.  **Clone Your Fork**: Clone your forked repository to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/AI-Trading-Agent.git
    cd AI-Trading-Agent
    ```
3.  **Set Upstream Remote**: Add the original repository as the upstream remote:
    ```bash
    git remote add upstream https://github.com/Efficientlyy/AI-Trading-Agent.git
    ```
4.  **Create a Virtual Environment**: Follow the setup steps in the main `README.md` to create and activate a virtual environment and install dependencies.
5.  **Create a Branch**: Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name  # Or fix/your-bug-fix-name
    ```

## Development Workflow

1.  **Code**: Implement your changes. Adhere to the existing code style (we aim for PEP 8 compliance and use tools like Black and Flake8 - configuration to be added).
2.  **Test**: 
    *   Add new unit tests for any new functionality.
    *   Ensure all existing tests pass by running `pytest`.
    *   Add integration tests if applicable.
3.  **Document**: Update any relevant documentation (`README.md`, `docs/`, docstrings) to reflect your changes.
4.  **Commit**: Make clear, concise commit messages.
5.  **Rebase**: Keep your branch updated with the main branch:
    ```bash
    git fetch upstream
    git rebase upstream/main  # Or the appropriate base branch
    ```
6.  **Push**: Push your changes to your fork:
    ```bash
    git push origin feature/your-feature-name
    ```
7.  **Pull Request**: Open a pull request (PR) from your fork's branch to the `main` branch of the upstream repository.
    *   Provide a clear description of the changes in the PR.
    *   Link any relevant issues.

## Code Style

*   Follow PEP 8 guidelines.
*   Use type hints (`typing` module).
*   Write clear and concise docstrings (Google style preferred).
*   (We will add automatic formatters like Black and linters like Flake8 soon).

## Testing

*   All new features must include corresponding unit tests.
*   All tests must pass before a PR can be merged.
*   Aim for good test coverage.

## Reporting Bugs

If you find a bug, please open an issue on the GitHub repository. Include:
*   A clear description of the bug.
*   Steps to reproduce the bug.
*   Expected behavior.
*   Actual behavior.
*   Your environment details (OS, Python version).

Thank you for contributing!
