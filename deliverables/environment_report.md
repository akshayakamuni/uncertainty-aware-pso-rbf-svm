# Analysis Environment Report

Captured from the Python interpreter used to run the saved-results analysis
and its tests. The dependency list in the repository's `requirements.txt` is
unpinned; the versions below are observations of this environment, not
reproducible lockfile entries.

| Component | Observed version / availability |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.2.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.10.1 |
| seaborn | 0.13.2 |
| pyswarms | 1.3.0 |
| python-pptx | Not installed |
| pywin32 / `win32com` | Available |

The final analysis uses the installed pandas, NumPy, and matplotlib. It does
not require `tabulate` or seaborn. PowerPoint automation is available through
`win32com`; the presentation is produced as a separate final copy so the
working draft remains unchanged.

The run environment and dependency availability do not verify dataset
provenance, licensing, or long-term reproducibility. No package versions
were added to or invented for `requirements.txt`.
