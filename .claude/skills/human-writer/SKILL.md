---
name: human-writer
description: Write and revise documentation (Markdown, prose sections) so that the text reads naturally and avoids patterns commonly flagged as AI-generated. Use when creating or editing .md files, case study pages, README sections, or any prose-heavy documentation.
---

# Human Writer

Guidelines for producing documentation prose that reads naturally, varies in rhythm, and avoids the formulaic patterns that AI-detection tools flag.

## Core Principles

1. **Vary sentence length deliberately.** Follow a long, multi-clause sentence with a short one. Then try a medium-length sentence. Monotonous uniformity is the single biggest trigger for AI detectors.
2. **Vary sentence openings.** Never start three consecutive sentences the same way. Mix subject-first, prepositional, participial, inverted, and adverbial openings.
3. **Prefer concrete over abstract.** Say "the solver diverges after 200 steps" instead of "the solver exhibits divergent behavior."
4. **Break parallel structure occasionally.** Lists and bullet points are fine, but surrounding prose should not read like a list in disguise.
5. **Use natural connectives sparingly.** Words like "moreover," "furthermore," and "additionally" are fine once or twice, but stacking them signals generated text. Prefer varied transitions: "Beyond that," "On the other hand," "Worth noting is that," or simply starting a new thought without an explicit connective.

## Sentence Structure Rules

### Openings

Rotate among these patterns -- no pattern should appear more than twice in a row:

| Pattern                  | Example                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------ |
| Subject-first            | "The integration tolerances control numerical accuracy."                             |
| Prepositional            | "Under these parameter values, the system exhibits two stable limit cycles."         |
| Participial              | "Comparing the two implementations, we find close agreement."                        |
| Inverted / existential   | "Documented here are the five case studies used for validation."                     |
| Adverbial                | "Concretely, each test proceeds as follows."                                         |
| Subordinate-clause-first | "Because the friction model is non-smooth, standard tolerances may be insufficient." |

### Length Variation

Aim for a mix across any given paragraph. A rough target:

- 1-2 short sentences (under 12 words)
- 1-2 medium sentences (12-25 words)
- 1 long sentence (25+ words) that carries more detail

### Avoiding Formulaic Patterns

Do **not** use these patterns repeatedly:

- "This section documents..." / "This section describes..." (use at most once per page)
- "The X measures Y" three times in a row -- rephrase the second and third
- Starting every bullet list with the same part of speech
- "In this study, we..." as an opener for multiple paragraphs
- Uniform "X. Y. Z." triplet rhythm throughout a section

## Word Choice

- **Avoid overused hedging clusters**: "It is important to note that," "It should be noted that," "It is worth mentioning that." Pick one or rephrase entirely.
- **Limit adverb stacking**: "significantly," "particularly," "especially" -- one per paragraph is enough.
- **Use specific verbs**: "diverges," "oscillates," "saturates" over generic "exhibits," "demonstrates," "utilizes."
- **Vary synonyms across sections**: if you wrote "method" in one paragraph, try "approach," "procedure," or "strategy" in the next -- but stay within scientific register.

## Structural Guidelines

### Paragraphs Over Lists (When Appropriate)

If a section has three or fewer points, write them as a paragraph rather than a bulleted list. Lists are appropriate for four or more items or for reference material (metrics, commands, file paths).

**Instead of:**

```markdown
The case studies serve multiple purposes:

- Validation: Verify correctness
- Examples: Demonstrate usage
- Benchmarking: Compare performance
```

**Write:**

```markdown
These case studies serve primarily as a correctness check against the MATLAB reference. They also double as usage examples and provide data for performance comparisons.
```

### Section Transitions

Between major sections, include at least one sentence that bridges the preceding topic to the next. Do not rely on headings alone to carry transitions.

### Bullet List Variation

When bullet lists are appropriate, vary the internal structure:

- Some items can be sentence fragments
- Others should be complete sentences with a period at the end.
- Mixing lengths within a list is acceptable and desirable

## Character Encoding -- ASCII Only

AI-detection tools flag invisible and non-ASCII characters as a strong signal of generated text. Every character in the output must be plain ASCII.

**Forbidden characters (and their ASCII replacements):**

| Forbidden | Name | Use instead |
|-----------|------|-------------|
| `–` (U+2013) | en-dash | `-` or `--` |
| `—` (U+2014) | em-dash | `--` |
| `'` `'` (U+2018, U+2019) | curly single quotes | `'` |
| `"` `"` (U+201C, U+201D) | curly double quotes | `"` |
| `…` (U+2026) | ellipsis | `...` |
| `−` (U+2212) | minus sign | `-` |
| zero-width spaces, soft hyphens, BOM markers | invisible characters | remove entirely |

**Rules:**

- Never insert any Unicode punctuation. Use only ASCII hyphens, quotes, and periods.
- If source material contains Unicode characters, replace them when incorporating text.
- Exception: proper names with accented characters (e.g., "Rossler" can be written as "Rossler" or kept as the original if the font supports it) and LaTeX math blocks where Unicode is rendered by KaTeX/MathJax.

## Scientific Register

This skill applies to scientific and technical documentation. Keep the tone precise and professional.

**Do not:**

- Use colloquial language ("this one is trickier than it sounds")
- Add rhetorical questions ("But what does this really mean?")
- Use first-person singular ("I implemented...") -- prefer "we" or passive constructions
- Use exclamation marks
- Use filler phrases ("In order to," "Due to the fact that") -- prefer direct phrasing ("To," "Because")
- Use unicode characters (en-dashes, em-dashes, curly quotes, etc.) -- stick to ASCII: hyphens (`-`), double hyphens (`--`), and straight quotes (`"`, `'`)

**Do:**

- Use the technical term when one exists
- Cite references in a consistent style
- Write in present tense for descriptions of the system, past tense for describing what was done during a specific experiment

## Checklist Before Submitting

1. Read the text aloud (or mentally). Does any stretch of three or more sentences feel monotonous?
2. Count sentence openings in each paragraph -- are at least three different patterns used?
3. Check for repeated transition words. Is "additionally" or "furthermore" used more than once on the page?
4. Verify sentence length variation: does every paragraph contain at least one short and one long sentence?
5. Confirm no section starts with "This section..." more than once across the entire document.
6. Ensure lists with three or fewer items are written as prose instead.
7. Scan for non-ASCII characters (en-dashes, curly quotes, ellipsis, invisible chars) and replace with ASCII equivalents.
