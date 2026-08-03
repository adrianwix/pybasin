# Reproducibility

Documentation builds do not execute numerical studies. Expensive runs are performed deliberately, their structured outputs are committed, and the site renders those outputs later. This makes a routine deployment fast while preserving the evidence behind each table and plot.

A published study should record:

- the exact command and entry point;
- solver, tolerance, precision, batch, and device settings;
- software and hardware metadata;
- a structured output file;
- the interpretation of the result, including failed or incomplete runs.

Case-study artifacts are refreshed by the integration suite. Solver design studies are self-contained, executed notebooks under [Design Studies](design-studies/index.md). Each notebook owns the experiment implementation and reads a committed JSON artifact by default; changing its execution flag reruns the expensive measurement.
