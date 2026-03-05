---
name: pybasin_documentation_writer
description: Use whenever the user asks to write, update, or review documentation pages (user guides or API reference) for the pybasin project.
---

# Documentation Writer for pybasin User Guides and API Reference

## Task

Write or update documentation pages for the pybasin project. There are two types of doc pages:

1. **User Guide pages** (`docs/user-guide/`) — conceptual, tutorial-style docs that explain _how and why_ to use a component.
2. **API Reference pages** (`docs/api/`) — auto-generated from source code using mkdocstrings directives.

All prose sections must follow the `human-writer` skill guidelines (see `.github/skills/human-writer/SKILL.md`). Read that skill file before writing any documentation prose, and apply its rules on sentence variation, natural connectives, and avoiding formulaic patterns.

## Documentation Stack

- **MkDocs Material** theme with `mkdocstrings` (Sphinx-style docstrings, `paths: [src]`)
- Navigation defined in `mkdocs.yml` under `nav:`
- Admonitions via `!!! type "Title"` (indent body by 4 spaces)
- Code blocks with `python` syntax highlighting

## User Guide Page Structure

Follow this structure (derived from existing pages like `feature-extractors.md`, `samplers.md`, `solvers.md`):

1. **Title** (`# Component Name`) — one `#` heading.
2. **Opening paragraph** — 2–3 sentences explaining what the component does and its role in the pipeline.
3. **Available implementations table** — comparison table with columns like Class, key traits, and "Best for" recommendation.
4. **Admonition warnings/notes** — for experimental status, gotchas, or important caveats. Keep these concise (1–2 sentences). Do not over-explain implementation details (e.g., don't say "due to numerical precision, FFT implementations, and edge-case handling" — just say "results are close but not identical").
5. **Common concepts** — shared parameters, configuration patterns, or type aliases that apply across implementations (e.g., `FCParameters`, `time_span`).
6. **Per-implementation sections** (`## ClassName`) separated by `---` dividers, each containing:
   - Brief description (1 sentence).
   - Code example showing common usage patterns (minimal, comprehensive, custom).
   - Per-implementation-specific features (e.g., per-state config).
   - Constructor parameters table with columns: Parameter, Type, Default, Description.
   - Admonition notes for gotchas specific to that implementation.
7. **Cross-cutting concerns** — sections covering behavior that applies across implementations (e.g., normalization, imputation, naming conventions).
8. **Standalone usage example** — if the component can be used outside `BasinStabilityEstimator`.
9. **Cross-reference to related guides** — link to custom/advanced guides if they exist.

### Writing Style

- Tables for structured comparisons (implementations, parameters, features).
- Code examples are practical and copy-pasteable — show real imports and constructor calls.
- Use `backtick` formatting for all code symbols (classes, methods, parameters, values).
- Admonitions: `!!! warning`, `!!! tip`, `!!! note`, `!!! info` — always indent the body by 4 spaces.
- Em dashes (`--`) not en dashes.
- Keep prose concise. No filler. Each sentence should add information.
- When referencing API details, link to the API reference page rather than duplicating signatures. Example: `For full function signatures, see the [API reference](../api/page.md).`
- Follow the `human-writer` skill: vary sentence length and openings, avoid formulaic repetition, prefer concrete language over abstract.

## API Reference Page Structure

Two patterns:

### Class-based API pages (e.g., `feature-extractors.md`, `samplers.md`)

Use mkdocstrings `:::` directives pointing to the fully qualified module path of each class, separated by `---`:

```markdown
# Page Title

::: pybasin.module.ClassName

---

::: pybasin.module.AnotherClass
```

### Function-based API pages (e.g., `torch-feature-calculators.md`)

When documenting a module full of standalone functions:

1. **Title** (`# Page Title`).
2. **Experimental/status admonition** if applicable.
3. **Brief convention description** (input/output shapes, common patterns).
4. **Sections by category** (`## Category Name`) with mkdocstrings `:::` directives per module:

```markdown
## Category Name

::: pybasin.module.submodule
options:
show_root_heading: false
heading_level: 3
```

### Submenu Pattern for Large API Sections

When an API section would be too large as a single page, split it into a submenu in `mkdocs.yml`:

```yaml
# Before (single page):
- Feature Extraction: api/feature-extractors.md

# After (submenu):
- Feature Extraction:
    - Feature Extractors: api/feature-extractors.md
    - Torch Feature Calculators: api/torch-feature-calculators.md
```

Then add a cross-reference from the user guide to the new sub-page.

## Navigation Updates

After creating any new page, add it to the `nav:` section in `mkdocs.yml`. Verify the build succeeds with `uv run mkdocs build`.

## Rules

- Do NOT create files unless the user explicitly asks.
- Do NOT modify `pyrightconfig.json`.
- Always verify the docs build after changes.
- Keep admonition text concise — no redundant technical details.
- User guide pages explain concepts and show usage. API pages auto-generate from source. Don't duplicate API signatures in user guides.
- Always read and apply the `human-writer` skill before writing prose.
