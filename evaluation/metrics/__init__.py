"""Per-axis metric implementations.

Each module holds the pure callables for one metric axis. Only the parameter axis
exists today; magnitude / timbre / loudness / pitch / perceptual modules land in
later build-order slices (see ``docs/DECISIONS.md`` and issue #8).
"""
