# Sensitivity Analysis in Petersburg Case Studies

## Overview

Sensitivity analysis is a key application of the petersburg framework - it allows you to understand how changes in specific parameters affect overall outcomes. This is critical for decision-making because it reveals:

1. **Which variables matter most** - Focus your effort on improving the high-impact parameters
2. **The magnitude of impact** - Quantify ROI of improvements
3. **Non-linear relationships** - Discover threshold effects and exponential impacts
4. **Decision robustness** - Understand how sensitive your decision is to uncertainty

## Implementation

Each case study includes sensitivity analysis that:
- Varies a key parameter across a realistic range
- Runs Monte Carlo simulations for each value
- Displays results in both tabular and ASCII chart format
- Calculates improvement metrics and key findings

## Case Studies with Sensitivity Analysis

### 1. Drug Development - Phase II Success Rate

**Parameter**: Phase II clinical trial success rate (20% - 65%)

**Why this matters**: Phase II is where efficacy is first tested in patients. Companies invest heavily in biomarkers, patient selection, and trial design to improve Phase II success rates.

**Key findings from sensitivity analysis**:
- Improving from 20% to 65% increases expected value dramatically
- Each 10-point improvement adds hundreds of millions in EV per drug
- Shows why pharma invests in computational biology and precision medicine

**Run it**:
```bash
.venv/bin/python3 examples/case_studies/drug_development.py
```

The output includes an ASCII bar chart showing expected value vs. Phase II success rate, making the relationship immediately visual.

### 2. Startup Funding - Seed → Series A Transition

**Parameter**: Seed to Series A success rate (20% - 50%)

**Why this matters**: This is the "Valley of Death" - the most dangerous transition in a startup's lifecycle. Most startups that survive seed fail to raise Series A.

**Key findings from sensitivity analysis**:
- Improving from 20% to 50% can triple expected value
- This transition is more impactful than any other single stage
- Explains why accelerators focus intensely on product-market fit

**Run it**:
```bash
.venv/bin/python3 examples/case_studies/startup_funding.py
```

### 3. Product Launch - Pilot Success Rate (To Be Added)

**Recommended parameter**: Pilot launch success rate (20% - 60%)

**Why it matters**: Pilot stage is where products get real market validation. Strong pilot results dramatically improve odds of successful full launch.

**To add this analysis**, insert the following function before the `if __name__` block:

```python
def sensitivity_analysis_pilot_success():
    """
    Perform sensitivity analysis: how does Pilot success rate affect outcomes?

    Pilot is the critical validation stage - real customers, real revenue.
    """
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Pilot Success Rate")
    print("=" * 80)
    print()

    pilot_rates = np.arange(0.20, 0.65, 0.05)
    results = {
        'pilot_rate': [],
        'expected_value': [],
        'success_rate': []
    }

    print("Running simulations across Pilot success rates...")
    print()

    for rate in pilot_rates:
        g = build_product_launch_graph('base_case')
        # Modify the graph to use the new pilot success rate
        # (Requires extracting graph building logic to allow parameter override)

        outcomes = []
        for _ in range(10000):
            outcomes.append(g.get_outcome())
        outcomes = np.array(outcomes)

        results['pilot_rate'].append(rate)
        results['expected_value'].append(np.mean(outcomes))
        results['success_rate'].append(np.sum(outcomes > 0) / len(outcomes) * 100)

    # Print results and ASCII chart
    # (Similar format to other case studies)
```

### 4. Litigation Strategy - Win Probability (To Be Added)

**Recommended parameter**: Plaintiff win probability at trial (30% - 70%)

**Why it matters**: The estimated probability of winning is the single most important input to settlement decisions. Lawyers often disagree by 20-30 percentage points on this estimate.

**Key insight**: Small changes in win probability (±10%) can shift the decision from "settle immediately" to "go to trial" or vice versa.

## How to Use Sensitivity Analysis for Your Decisions

1. **Identify the key uncertainty**: What parameter are you most uncertain about?

2. **Define the range**: What's the realistic min-max range for this parameter?

3. **Rebuild the graph for each value**: Use a loop to test different parameter values

4. **Collect metrics**: Track EV, median, success rate, etc. for each scenario

5. **Visualize**: Create charts (ASCII in terminal, matplotlib for production)

6. **Interpret**:
   - Linear relationships → improvements scale proportionally
   - Non-linear → look for threshold effects or exponential impacts
   - Flat regions → parameter doesn't matter much (good news!)

## Example: Running Sensitivity Analysis

```python
from petersburg import Graph
import numpy as np

def sensitivity_analysis(parameter_name, param_range, build_graph_func):
    """
    Generic sensitivity analysis function.

    Args:
        parameter_name: Name of parameter being varied
        param_range: Array of values to test
        build_graph_func: Function that builds graph given parameter value
    """
    results = {'param': [], 'ev': [], 'success_rate': []}

    for param_value in param_range:
        g = build_graph_func(param_value)

        outcomes = []
        for _ in range(10000):
            outcomes.append(g.get_outcome())
        outcomes = np.array(outcomes)

        results['param'].append(param_value)
        results['ev'].append(np.mean(outcomes))
        results['success_rate'].append(
            np.sum(outcomes > 0) / len(outcomes) * 100
        )

    # Print results
    print(f"\nSensitivity Analysis: {parameter_name}")
    print("=" * 60)
    for i in range(len(results['param'])):
        print(f"{results['param'][i]:.2f}: EV=${results['ev'][i]:.0f}, "
              f"Success={results['success_rate'][i]:.1f}%")

    return results
```

## Extending to Multi-Parameter Sensitivity

For advanced analysis, you can vary multiple parameters simultaneously to create sensitivity surfaces:

```python
# 2D sensitivity: vary two parameters
for param1 in param1_range:
    for param2 in param2_range:
        g = build_graph(param1, param2)
        # Simulate and record results
        # Can visualize as heatmap
```

This reveals interactions between parameters and helps prioritize which combinations of improvements yield the best outcomes.

## Key Takeaways

1. **Sensitivity analysis is not optional** - Real decisions require understanding parameter sensitivity

2. **The petersburg framework makes this easy** - Just rebuild the graph with different values and re-simulate

3. **Visual output matters** - Even simple ASCII charts make insights immediate

4. **Focus on elastic parameters** - Improve the variables with steepest slopes

5. **Use for decision-making** - If outcome is insensitive to a parameter, don't waste effort refining that estimate

## References

- Saltelli et al., "Global Sensitivity Analysis: The Primer"
- Rabitz & Aliş, "General foundations of high-dimensional model representations"
- Morgan & Henrion, "Uncertainty: A Guide to Dealing with Uncertainty in Quantitative Risk and Policy Analysis"
