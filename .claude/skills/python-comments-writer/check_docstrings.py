#!/usr/bin/env python3
"""Check Python docstrings for common formatting issues.

This script detects:
1. Google-style sections (Returns:, Raises:, Args:, Attributes:) that should use Sphinx style
2. Bullet lists without a blank line before them
3. NumPy-style sections (Parameters, Returns with dashes underneath)
4. Redundant :vartype: tags when type hints are present in __init__

Usage:
    python check_docstrings.py <file_or_directory> [<file_or_directory> ...]
    python check_docstrings.py src/pybasin/
    python check_docstrings.py src/pybasin/ --exclude ts_torch/calculators
    python check_docstrings.py src/pybasin/basin_stability_estimator.py
    python check_docstrings.py file1.py file2.py file3.py
"""

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


class DocstringIssue:
    def __init__(self, file: str, line: int, issue_type: str, message: str, suggestion: str):
        self.file = file
        self.line = line
        self.issue_type = issue_type
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: [{self.issue_type}] {self.message}\n  Suggestion: {self.suggestion}"


def check_google_style_sections(docstring: str, base_line: int, file: str) -> list[DocstringIssue]:
    """Detect Google-style section headers that should be Sphinx style."""
    issues: list[DocstringIssue] = []

    google_sections = {
        r"^\s*Args:\s*$": (":param name:", "Use ':param name: description' for each parameter"),
        r"^\s*Arguments:\s*$": (
            ":param name:",
            "Use ':param name: description' for each parameter",
        ),
        r"^\s*Parameters:\s*$": (
            ":param name:",
            "Use ':param name: description' for each parameter",
        ),
        r"^\s*Returns:\s*$": (":return:", "Use ':return: description' on a single line"),
        r"^\s*Return:\s*$": (":return:", "Use ':return: description' on a single line"),
        r"^\s*Raises:\s*$": (":raises ExceptionType:", "Use ':raises ExceptionType: description'"),
        r"^\s*Attributes:\s*$": (
            ":ivar name:",
            "Use ':ivar name: description' and ':vartype name: type'",
        ),
        r"^\s*Yields:\s*$": (":yields:", "Use ':yields: description'"),
        r"^\s*Examples:\s*$": ("Example::", "Use 'Example::' followed by indented code block"),
    }

    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        for pattern, (sphinx_equiv, suggestion) in google_sections.items():
            if re.match(pattern, line):
                issues.append(
                    DocstringIssue(
                        file=file,
                        line=base_line + i,
                        issue_type="google-style",
                        message=f"Google-style section '{line.strip()}' detected",
                        suggestion=f"{suggestion} (Sphinx: {sphinx_equiv})",
                    )
                )

    return issues


def check_numpy_style_sections(docstring: str, base_line: int, file: str) -> list[DocstringIssue]:
    """Detect NumPy-style section headers (with underlines)."""
    issues: list[DocstringIssue] = []

    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            if re.match(r"^\s*-+\s*$", next_line) and line.strip():
                section_name = line.strip()
                if section_name in [
                    "Parameters",
                    "Returns",
                    "Raises",
                    "Attributes",
                    "Examples",
                    "Yields",
                ]:
                    issues.append(
                        DocstringIssue(
                            file=file,
                            line=base_line + i,
                            issue_type="numpy-style",
                            message=f"NumPy-style section '{section_name}' with underline detected",
                            suggestion="Use Sphinx-style :param:, :return:, :raises: instead",
                        )
                    )

    return issues


def check_bullet_list_spacing(docstring: str, base_line: int, file: str) -> list[DocstringIssue]:
    """Detect bullet lists without a blank line before them."""
    issues: list[DocstringIssue] = []

    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^\s*[-*+]\s+\S", line) and i > 0:
            prev_line = lines[i - 1]
            if (
                prev_line.strip()
                and not re.match(r"^\s*[-*+]\s+", prev_line)
                and not prev_line.strip().endswith(":")
            ):
                issues.append(
                    DocstringIssue(
                        file=file,
                        line=base_line + i,
                        issue_type="bullet-spacing",
                        message="Bullet list item without blank line before it",
                        suggestion="Add a blank line before the bullet list for proper Markdown rendering",
                    )
                )

    return issues


def check_doctest_examples(docstring: str, base_line: int, file: str) -> list[DocstringIssue]:
    """Detect example patterns that don't render well in mkdocs."""
    issues: list[DocstringIssue] = []

    lines = docstring.split("\n")
    reported_doctest = False
    reported_example_colon = False

    for i, line in enumerate(lines):
        if re.match(r"^\s*>>>\s*", line) and not reported_doctest:
            issues.append(
                DocstringIssue(
                    file=file,
                    line=base_line + i,
                    issue_type="doctest-example",
                    message="Doctest-style example (>>>) detected - renders poorly in mkdocs",
                    suggestion="Use markdown code block: ```python ... ``` for syntax highlighting",
                )
            )
            reported_doctest = True

        if (
            re.match(r"^\s*Example(\s+usage)?::\s*$", line, re.IGNORECASE)
            and not reported_example_colon
        ):
            issues.append(
                DocstringIssue(
                    file=file,
                    line=base_line + i,
                    issue_type="example-double-colon",
                    message="Example with double colon (::) detected - renders poorly in mkdocs",
                    suggestion="Use markdown code block: ```python ... ``` for syntax highlighting",
                )
            )
            reported_example_colon = True

    return issues


def check_content_after_field_lists(
    docstring: str, base_line: int, file: str
) -> list[DocstringIssue]:
    """Detect narrative content (examples, notes, text) after Sphinx field lists.

    In Sphinx/rST, field lists (:param:, :ivar:, :return:, etc.) must come at the END
    of the docstring. Any examples, notes, or other content should come before them.
    """
    issues: list[DocstringIssue] = []

    lines = docstring.split("\n")
    first_field_line = None
    field_pattern = re.compile(
        r"^\s*:(param|ivar|vartype|type|return|returns|rtype|raises|raise|yields|yield)(\s+\w+)?:"
    )

    # Find first field list directive
    for i, line in enumerate(lines):
        if field_pattern.match(line):
            first_field_line = i
            break

    # If we found field lists, check for content after them
    if first_field_line is not None:
        for i in range(first_field_line + 1, len(lines)):
            line = lines[i]

            # Skip empty lines and continuation of field descriptions (indented lines right after fields)
            if not line.strip():
                continue

            # If it's another field directive, that's fine
            if field_pattern.match(line):
                continue

            # Check if it's a continuation (indented relative to field)
            # Field continuations are typically indented
            if i > 0 and field_pattern.match(lines[i - 1]):
                # Line right after a field directive is likely its description
                continue

            # If the line is indented and the previous non-empty line was a field or continuation, skip
            prev_non_empty_idx = i - 1
            while prev_non_empty_idx >= first_field_line and not lines[prev_non_empty_idx].strip():
                prev_non_empty_idx -= 1

            if prev_non_empty_idx >= first_field_line and (
                field_pattern.match(lines[prev_non_empty_idx])
                or (len(line) - len(line.lstrip()) > 0 and prev_non_empty_idx >= first_field_line)
            ):
                # This could be a continuation, but let's check if it looks like a new section
                if any(
                    keyword in line.lower()
                    for keyword in ["example", "note", "warning", "```", ".. code"]
                ):
                    issues.append(
                        DocstringIssue(
                            file=file,
                            line=base_line + i,
                            issue_type="content-after-fields",
                            message=f"Content found after field list: '{line.strip()[:50]}...'",
                            suggestion="Move examples, notes, and narrative content BEFORE :param/:ivar/:return: field lists",
                        )
                    )
                    break
                continue

            # If we're here and the line is not empty and not indented much, it's likely new content
            if line.strip() and not line.startswith("    "):
                issues.append(
                    DocstringIssue(
                        file=file,
                        line=base_line + i,
                        issue_type="content-after-fields",
                        message=f"Content found after field list: '{line.strip()[:50]}...'",
                        suggestion="Move examples, notes, and narrative content BEFORE :param/:ivar/:return: field lists",
                    )
                )
                break

    return issues


def check_indented_params(docstring: str, base_line: int, file: str) -> list[DocstringIssue]:
    """Detect indented parameter descriptions (Google/NumPy style)."""
    issues: list[DocstringIssue] = []

    lines = docstring.split("\n")
    in_section = False
    section_indent = 0

    for i, line in enumerate(lines):
        if re.match(r"^\s*(Args|Arguments|Parameters|Returns|Raises|Attributes):\s*$", line):
            in_section = True
            section_indent = len(line) - len(line.lstrip())
            continue

        if in_section:
            if line.strip() == "":
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= section_indent and line.strip():
                in_section = False
            elif re.match(r"^\s+\w+.*:", line) and current_indent > section_indent:
                param_match = re.match(r"^\s+(\w+)\s*(\([^)]+\))?:\s*(.*)$", line)
                if param_match:
                    param_name = param_match.group(1)
                    issues.append(
                        DocstringIssue(
                            file=file,
                            line=base_line + i,
                            issue_type="indented-param",
                            message=f"Indented parameter '{param_name}' in Google/NumPy style",
                            suggestion=f"Use ':param {param_name}: description' instead",
                        )
                    )

    return issues


def extract_docstrings(file_path: Path) -> list[tuple[str, int]]:
    """Extract all docstrings from a Python file with their line numbers."""
    docstrings: list[tuple[str, int]] = []

    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            docstring = ast.get_docstring(node)
            if docstring:
                line_no = node.lineno if hasattr(node, "lineno") else 1
                if isinstance(node, ast.Module):
                    line_no = 1
                docstrings.append((docstring, line_no))

    return docstrings


def get_typed_init_attributes(class_node: ast.ClassDef) -> set[str]:
    """Extract attribute names that have type annotations in __init__.

    Looks for patterns like: self.name: Type = value

    :param class_node: The AST node for the class.
    :return: Set of attribute names with type annotations.
    """
    typed_attrs: set[str] = set()

    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for stmt in ast.walk(node):
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Attribute)
                    and isinstance(stmt.target.value, ast.Name)
                    and stmt.target.value.id == "self"
                ):
                    typed_attrs.add(stmt.target.attr)
            break

    return typed_attrs


def get_class_level_annotations(class_node: ast.ClassDef) -> set[str]:
    """Extract attribute names that have class-level type annotations.

    Looks for patterns like: attr: Type (typically in Protocol classes or dataclasses)

    :param class_node: The AST node for the class.
    :return: Set of attribute names with class-level type annotations.
    """
    typed_attrs: set[str] = set()

    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            typed_attrs.add(node.target.id)

    return typed_attrs


def check_redundant_vartype(class_node: ast.ClassDef, file: str) -> list[DocstringIssue]:
    """Detect redundant :vartype: tags when type hints are present.

    Checks both __init__ annotations (self.attr: Type) and class-level annotations (attr: Type).

    :param class_node: The AST node for the class.
    :param file: The file path for error reporting.
    :return: List of docstring issues found.
    """
    issues: list[DocstringIssue] = []

    docstring = ast.get_docstring(class_node)
    if not docstring:
        return issues

    # Check both __init__ attributes and class-level annotations (e.g., Protocol classes)
    typed_attrs = get_typed_init_attributes(class_node)
    class_level_attrs = get_class_level_annotations(class_node)
    all_typed_attrs = typed_attrs | class_level_attrs

    if not all_typed_attrs:
        return issues

    vartype_pattern = re.compile(r":vartype\s+(\w+):")
    lines = docstring.split("\n")
    base_line = class_node.lineno

    for i, line in enumerate(lines):
        match = vartype_pattern.search(line)
        if match:
            attr_name = match.group(1)
            if attr_name in all_typed_attrs:
                if attr_name in class_level_attrs:
                    suggestion = f"Remove ':vartype {attr_name}: ...' line; type is already declared as class attribute '{attr_name}: Type'"
                else:
                    suggestion = f"Remove ':vartype {attr_name}: ...' line; type is inferred from 'self.{attr_name}: Type = ...' in __init__"

                issues.append(
                    DocstringIssue(
                        file=file,
                        line=base_line + i,
                        issue_type="redundant-vartype",
                        message=f"Redundant ':vartype {attr_name}:' - type hint already present",
                        suggestion=suggestion,
                    )
                )

    return issues


def extract_class_nodes(file_path: Path) -> list[tuple[ast.ClassDef, str]]:
    """Extract all class nodes from a Python file.

    :param file_path: Path to the Python file.
    :return: List of (class_node, file_path_str) tuples.
    """
    classes: list[tuple[ast.ClassDef, str]] = []

    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append((node, str(file_path)))

    return classes


def check_file(file_path: Path) -> list[DocstringIssue]:
    """Check a single Python file for docstring issues."""
    all_issues: list[DocstringIssue] = []
    file_str = str(file_path)

    for docstring, line_no in extract_docstrings(file_path):
        all_issues.extend(check_google_style_sections(docstring, line_no, file_str))
        all_issues.extend(check_numpy_style_sections(docstring, line_no, file_str))
        all_issues.extend(check_bullet_list_spacing(docstring, line_no, file_str))
        all_issues.extend(check_indented_params(docstring, line_no, file_str))
        all_issues.extend(check_doctest_examples(docstring, line_no, file_str))
        all_issues.extend(check_content_after_field_lists(docstring, line_no, file_str))

    for class_node, file_str in extract_class_nodes(file_path):
        all_issues.extend(check_redundant_vartype(class_node, file_str))

    return all_issues


DEFAULT_EXCLUDES = [
    "ts_torch",
    "ts_torch/calculators",
]


def check_directory(dir_path: Path, excludes: list[str] | None = None) -> list[DocstringIssue]:
    """Check all Python files in a directory recursively."""
    all_issues: list[DocstringIssue] = []
    exclude_patterns = excludes if excludes is not None else DEFAULT_EXCLUDES

    for py_file in dir_path.rglob("*.py"):
        file_str = str(py_file)
        if "__pycache__" in file_str:
            continue
        if any(pattern in file_str for pattern in exclude_patterns):
            continue
        all_issues.extend(check_file(py_file))

    return all_issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check Python docstrings for formatting issues (Google/NumPy style vs Sphinx)"
    )
    parser.add_argument("paths", nargs="+", help="File(s) or directory(ies) to check")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Patterns to exclude (can be specified multiple times). Defaults to: ts_torch/calculators",
    )
    parser.add_argument(
        "--no-default-excludes", action="store_true", help="Don't use default exclude patterns"
    )
    args = parser.parse_args()

    # Build exclude list
    if args.no_default_excludes:
        excludes = args.exclude or []
    else:
        excludes = (args.exclude or []) + DEFAULT_EXCLUDES

    issues: list[DocstringIssue] = []
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Error: {path} does not exist", file=sys.stderr)
            return 1

        if path.is_file():
            issues.extend(check_file(path))
        else:
            issues.extend(check_directory(path, excludes))

    if args.json:
        output = [
            {
                "file": issue.file,
                "line": issue.line,
                "type": issue.issue_type,
                "message": issue.message,
                "suggestion": issue.suggestion,
            }
            for issue in issues
        ]
        print(json.dumps(output, indent=2))
    else:
        if issues:
            # Group issues by file
            issues_by_file: dict[str, list[DocstringIssue]] = defaultdict(list)
            for issue in issues:
                issues_by_file[issue.file].append(issue)

            # Count by type
            type_counts: dict[str, int] = defaultdict(int)
            for issue in issues:
                type_counts[issue.issue_type] += 1

            # Print grouped by file
            for file_path in sorted(issues_by_file.keys()):
                file_issues = sorted(issues_by_file[file_path], key=lambda x: x.line)
                print(f"\n{'=' * 80}")
                print(f"FILE: {file_path} ({len(file_issues)} issue(s))")
                print("=" * 80)
                for issue in file_issues:
                    print(f"  Line {issue.line}: [{issue.issue_type}] {issue.message}")
                    print(f"    → {issue.suggestion}")

            # Print summary
            print(f"\n{'=' * 80}")
            print("SUMMARY")
            print("=" * 80)
            print(f"Total issues: {len(issues)}")
            print(f"Files affected: {len(issues_by_file)}")
            print("\nIssues by type:")
            for issue_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"  {issue_type}: {count}")
        else:
            print("✓ No docstring issues found.")

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
