---
name: clean-code
description: "Apply clean-code heuristics when reviewing or writing code. Use when: spotting duplicate branches, fixing API contracts that push work onto callers, simplifying conditionals, or identifying structural smells that make code hard to extend."
---

# Clean Code Heuristics

## Unify Diverging Paths

Two branches that do near-identical work signal a missing abstraction.

**Smell:** Separate `if/else` blocks with duplicated logic differing only in one argument.

**Fix:** Compute the differing value before the common operation, then write the operation once.

```python
# Bad -- two almost-identical branches
if params is not None:
    result = solve(y0, params)
else:
    result = solve(y0, default_p)

# Good -- unified
effective = params if params is not None else default_p
result = solve(y0, effective)
```

**For collections:** use a ternary or `or` to pick between two arrays/lists, then process once.

---

## Own Your Expansion

If a function accepts `(A, B)` and internally needs to work over all combinations of A and B, the expansion belongs inside the function -- not in every caller.

**Smell:** Callers flatten/repeat inputs before passing them in. The function's docstring says "pre-expanded."

**Fix:** Accept the natural shapes `(n_a, ...)` and `(n_b, ...)`, expand internally, return the full result.

```python
# Bad -- caller does the work
y0_flat = y0.repeat_interleave(P, dim=0)
params_flat = params.repeat(B, 1)
y = solver.integrate(ode, y0_flat, params_flat)

# Good -- solver owns the expansion
y = solver.integrate(ode, y0, params)  # (B, dims) x (P, n_params) -> (B*P, dims)
```

**Check:** If every call site has the same reshape/repeat before calling, the function has the wrong contract.

---

## Treat `broadcast_to` / `repeat` as a Smell Signal

`broadcast_to` or `repeat` in caller code often means the API forces the shape the function actually needs. Pull the broadcast inside.

---

## Push Defaults Inward

A default value that every caller must construct independently belongs inside the function.

```python
# Bad -- every caller repeats this
p = ode.params_to_array()[None, :]  # (1, n_params)
y = solver.integrate(ode, y0, p)

# Good -- solver handles the None case
y = solver.integrate(ode, y0)  # params=None -> use default internally
```

---

## Skip Unnecessary Parameters

If a value is constant across all calls and has no reason to vary, hardcode it. Parameters are a contract -- every one adds cognitive load.

```python
# Bad -- alpha never changes
def compute(graph, alpha=0.1232):
    ...

# Good -- constant lives inside
def compute(graph):
    alpha = 0.1232
    ...
```

---

## Set Defaults Before the Branch

When a variable has a common default and only sometimes needs overriding, declare it before the `if`. The `else` branch disappears and all variables are always bound.

```python
# Bad -- variables only exist inside branches; easy to miss one
if condition:
    x = a
else:
    x = default_a

# Good -- defaults unconditional, if only overrides what changes
x = default_a
if condition:
    x = a
```

**Check:** If the `else` branch only sets defaults, the defaults belong above the `if`.

---

## Use Self-Descriptive Names

A name should say what the value represents, not how it was computed or its role in a formula.

```python
# Bad -- abbreviation requires context to decode
n_p = len(run_configs)

# Good -- readable at the use site
n_configs = len(run_configs)
```

Avoid: single letters (`n`, `p`, `B`), Hungarian notation (`n_p`, `i_idx`), and acronyms that are not universally known in the domain.

---

## Review Checklist

When reviewing a diff or writing code, check:

1. Are there two branches doing near-identical work? Unify them.
2. Do callers reshape/flatten inputs before every call? Move expansion inside.
3. Does `broadcast_to` / `repeat` appear in caller code? API contract is wrong.
4. Are defaults reconstructed at every call site? Push them into the function.
5. Are there parameters that never actually vary? Hardcode them.
6. Does an `if/else` only set defaults in the `else`? Move defaults above the `if`, drop the `else`.
7. Do variable names require context to decode? Rename to be self-descriptive.
