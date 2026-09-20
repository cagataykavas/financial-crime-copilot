# Review-capacity evaluation

Financial-crime alert models operate behind a finite analyst queue. Aggregate AUC can look
healthy while the alerts actually reachable by analysts contain few confirmed cases. The
`capacity_evaluation` module evaluates the ranked queue at an explicit review budget.

It reports precision and recall at capacity, base rate, lift over random selection, workload
reduction, missed positives, and the exact reviewed alert IDs. A configurable release gate can
require minimum precision, recall, and lift. Sparse outcome evidence, duplicate IDs, non-finite
scores, and invalid policies fail closed.

## Interpretation boundaries

- Labels must be mature, consistently adjudicated outcomes; unresolved alerts must not be
  silently treated as negatives.
- This is a ranking and operational-capacity audit, not a causal estimate of prevented loss.
- One capacity point cannot show behavior across all staffing levels. Production monitoring
  should evaluate several policy-relevant budgets and time windows.
- Ties are resolved by alert ID for reproducibility, not business priority.
- Minimum counts do not replace confidence intervals, subgroup analysis, or temporal validation.
