---
name: python-documentation-writer
description: Write and review Python documentation using Sphinx/rST style docstrings for mkdocstrings. Use when writing docstrings, documenting classes, methods, or reviewing documentation for proper formatting.
---

# Python Documentation Writer

Guidelines for writing Python documentation that renders correctly with mkdocstrings (Sphinx style).

## Workflow

1. **Run the linter** on target files to detect formatting issues
2. **Fix** reported issues using the reference below
3. **Re-run the linter** until no issues remain
4. **Manual review**: check for clarity, missing explanations, ambiguity

```bash
uv run python .github/skills/python-documentation-writer/check_docstrings.py path/to/file.py
```

The linter detects: Google/NumPy style sections, missing blank lines before bullet lists, indented parameter blocks, redundant `:vartype:` tags.

## Quick Reference

| Element        | Syntax                                             |
| -------------- | -------------------------------------------------- |
| Parameter      | `:param name: Description.`                        |
| Return         | `:return: Description.`                            |
| Exception      | `:raises TypeError: Description.`                  |
| Instance var   | `:ivar name: Description.` (in class docstring)    |
| Attribute type | `self.name: Type = value` in `__init__` (required) |

## Sphinx/rST Style

Use `:param:`, `:return:`, `:raises:`. Do NOT use Google (`Returns:`) or NumPy style.

```python
def integrate(self, ode_system: ODESystem, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate the ODE system from initial conditions.

    :param ode_system: The ODE system to integrate.
    :param y0: Initial conditions tensor of shape (batch, state_dim).
    :return: Tuple of (time points, solution trajectories).
    :raises ValueError: If y0 has incorrect shape.
    """
```

## Class Attributes

Use `:ivar:` in class docstring + explicit type annotations in `__init__`:

```python
class Solution:
    """
    Represents the result of integrating an ODE system.

    :ivar initial_condition: Initial conditions used for integration.
    :ivar time: Time points of the integration.
    :ivar y: Solution trajectories of shape (N, B, S).
    """

    def __init__(self, initial_condition: torch.Tensor, time: torch.Tensor, y: torch.Tensor):
        self.initial_condition: torch.Tensor = initial_condition
        self.time: torch.Tensor = time
        self.y: torch.Tensor = y
```

**Critical**: Always annotate types explicitly (`self.x: Type = x`). Type inference does not work for docs.

## Markdown in Docstrings

**Bullet lists**: Always add a blank line before:

```python
"""
The errors are based on Bernoulli statistics:

- e_abs = sqrt(S_B(A) * (1 - S_B(A)) / N)
- e_rel = 1 / sqrt(N * S_B(A))
"""
```

**Code examples**: Use fenced blocks with language specifier. Do NOT use `>>>` or `::`.

````python
"""
Example:

```python
estimator = BasinStabilityEstimator(ode_system, sampler)
```
"""
````

## Manual Review Checklist

After the linter passes, verify:

1. Descriptions are clear and unambiguous
2. All parameters have meaningful explanations
3. Return values describe what is returned, not just the type
4. Edge cases and exceptions are documented
5. Complex logic has explanatory context
