# Repository conventions

## Prime directive: API consistency is not negotiable

This project builds APIs meant for wide use (currently: the tstorch `Feature`/
`ParameterlessFeature` calculator API, used by every consumer of `TorchFeatureExtractor` and
every direct caller of a feature class). The primary job on any API surface is not "does this
task's code work" - it's "does the same member mean the same kind of thing for every variant of
this API, regardless of which internal class/branch/path produced it." An oversight where that
isn't true must never reach the user - catch it while designing or reviewing the change, not
after it ships.

A property or method that returns a structurally different *kind* of value depending on an
invisible implementation detail (which base class was inherited, which internal branch fired) is
a defect in the API, not a documentation gap and not a style nit. Concretely:

- Before shipping any shared interface (a protocol, a base class, a property every variant
  implements), check every variant against the identical contract: would a caller who does not
  know the internal class hierarchy be surprised by what they get back from any one of them? If
  yes, the contract itself is wrong.
- If explaining a rule - in a comment, docstring, or skill file - requires an "except when X, in
  which case Y behaves differently" carve-out for what should be one concept, that is a signal to
  fix X in the code, not to write clearer prose around it. Prose is never a substitute for a
  consistent contract.
- Never rationalize an inconsistency as intentional in order to finish the current task faster.
  Stop, name it explicitly, and either fix it or ask before proceeding - do not paper over it and
  move on.

## Comments

Never write comments or docstrings that narrate a change: what a piece of code used to
do, that something is deprecated, that a name is destined for deletion, or how an old
shape differed from the new one. When removing or replacing code, delete it outright and
say nothing about its history in the source. Never write a deprecation comment/notice
unless the user explicitly asks for one. A comment only earns its place by explaining a
non-obvious WHY a future reader needs to safely change the code going forward - a real
invariant, a subtle bug workaround, a constraint imposed from outside the file - never a
changelog of what was tried, removed, or is pending removal.

Never reference internal skills (e.g. `.claude/skills/...`, "see the X skill") from source
code comments or docstrings. Skills are a workflow tool for the assistant, not part of the
codebase's own documentation - a reader of the code has no access to them and shouldn't
need to.

## Maintaining skills

Every skill has a "## Skill structure" section (or equivalently named) in its SKILL.md that
maps each file in the skill to the one kind of content it owns. Before editing any file in a
skill - adding a rule, fixing a rule, adding a note, anything - read that section first, every
time, even for a one-line change. This is not optional and not a one-time read.

1. Read "## Skill structure" (or the skill's equivalent index section) before touching any
   file.
2. From it, identify the single file that owns the category of information you're adding
   (a design rule, a process step, a performance finding, a per-feature history entry,
   whatever categories that skill defines).
3. Write the information there, once. Every other file that touches the same topic gets at
   most a bare pointer already established by that file's own structure (a name, a section
   title) - never a restatement of the rule, its rationale, or its exceptions, and never a
   second copy "just in case." If a file already has its own pointer to the owning section,
   leave it as a pointer; don't expand it into a summary.
4. If "Skill structure" doesn't make it clear which file owns the new information, stop and
   ask before writing anything. Do not guess, and do not write it in two places to be safe -
   duplicated information is the specific failure mode this process exists to prevent.
5. When a rule changes, edit it in its single owning location only. Touch a pointer elsewhere
   only if the pointer itself is now wrong (e.g. a renamed section).
