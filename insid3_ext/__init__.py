"""Research extensions to INSID3 (CVPR 2026).

This package *extends* the official INSID3 implementation that is vendored,
unmodified, at the repository root (``models/``, ``utils/``, ``datasets/``).
Nothing in the official code is edited; every new behaviour lives here.

Modules
-------
config       Typed experiment configuration, YAML loading and validation.
debiasing    Task 1: adaptive positional debiasing (hard projection is a special case).
calibration  Task 1: measurement of positional components across images and tokens.
prototypes   Task 2: multi-prototype reference representation.
matching     Task 2: prototype aggregation and region (many-to-many) matching.
model        ``INSID3Ext``: subclass of the official ``INSID3`` wiring both tasks in.
builder      Encoder / model construction with configurable (env-var) paths.
runtime      Device checks, seeding, timing and memory measurement.
experiment   Resumable evaluation loops with per-episode logging.
"""

__version__ = "0.1.0"
