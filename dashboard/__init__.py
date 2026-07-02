"""Local Streamlit control-panel dashboard for the sound-matching framework.

A private accelerator over the existing terminal scripts: each page builds the
exact ``python scripts/... --flags`` command, subprocesses it, streams the output
live, and reads the resulting files (run_summary.json / per_sample.csv / ...) for
display. The dashboard never imports the pipeline library (DatasetBuilder,
Evaluator); it only drives the scripts, so it can never drift from CLI behaviour.

Run with::

    streamlit run dashboard/app.py
"""
