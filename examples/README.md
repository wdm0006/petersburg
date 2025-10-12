# Petersburg Framework Examples

This directory contains examples demonstrating the Petersburg framework's capabilities for modeling sequential decisions under uncertainty.

## Directory Structure

- **[case_studies/](case_studies/)** - Comprehensive real-world applications with detailed analysis
- **Basic examples** - Classic paradoxes and simple demonstrations (root level)
- **[analysis/](analysis/)** - Analysis utilities and tools
- **[estimation/](estimation/)** - Estimation method examples

## Quick Start Examples

These root-level scripts demonstrate core framework concepts through classic decision problems and paradoxes:

### [stpetersburg.py](stpetersburg.py)
The classic St. Petersburg Paradox that inspired this framework.

**Problem**: A casino offers a coin-flip game where you win $2 if it lands heads on the first flip, $4 if it lands heads on the second flip, $8 on the third, and so on (doubling each time). How much should you pay to play?

**Key concept**: Demonstrates infinite expected value vs finite willingness to pay. Shows the difference between expected value calculations and practical decision-making.

**Run it**:
```bash
.venv/bin/python3 examples/stpetersburg.py
```

### [stpetersburg_w_bankroll.py](stpetersburg_w_bankroll.py)
Extension of the St. Petersburg Paradox with finite bankroll constraints.

**Key concept**: How realistic constraints (casino has finite bankroll) change the expected value from infinite to finite.

### [two_envelope_problem.py](two_envelope_problem.py)
The famous Two Envelope Paradox.

**Problem**: You're given two envelopes, one containing twice as much money as the other. You pick one, open it, and find $X. Should you switch?

**Key concept**: Demonstrates how naive expected value reasoning can lead to paradoxes. The "always switch" strategy seems to have positive EV, but that can't be right for both players.

**Run it**:
```bash
.venv/bin/python3 examples/two_envelope_problem.py
```

### [necktie_paradox.py](necktie_paradox.py)
A variant of the two envelope problem with ties.

**Problem**: Two people compare ties, each thinking theirs is cheaper and preferring to trade. How can both think trading has positive expected value?

**Key concept**: Shows how asymmetric information and subjective probabilities affect decision-making.

**Run it**:
```bash
.venv/bin/python3 examples/necktie_paradox.py
```

### [automatic_sensitivity_demo.py](automatic_sensitivity_demo.py)
Demonstrates the framework's automatic sensitivity analysis feature.

**Key concept**: Shows how to automatically identify which parameters have the most impact on expected value without manually testing each one.

**Run it**:
```bash
.venv/bin/python3 examples/automatic_sensitivity_demo.py
```

### [decision_vs_research.py](decision_vs_research.py)
Compares the value of making a decision now vs doing more research first.

**Key concept**: Demonstrates that research has value, but only if it changes your decision. Shows how to calculate the value of information.

**Run it**:
```bash
.venv/bin/python3 examples/decision_vs_research.py
```

### [outsourcing.py](outsourcing.py)
Models an outsourcing decision with uncertain outcomes.

**Key concept**: Simple business decision with quality uncertainty. Should you outsource or keep work in-house?

**Run it**:
```bash
.venv/bin/python3 examples/outsourcing.py
```

### [costwise_gradient.py](costwise_gradient.py)
Demonstrates cost-based sensitivity analysis.

**Key concept**: Shows how to find the "cost gradient" - which costs matter most to the final expected value.

**Run it**:
```bash
.venv/bin/python3 examples/costwise_gradient.py
```

### [print.py](print.py)
Demonstrates various output formats and visualization options.

**Key concept**: Shows how to export graphs to different formats (Mermaid, DOT) and print sensitivity reports.

**Run it**:
```bash
.venv/bin/python3 examples/print.py
```

## Case Studies

For comprehensive, real-world applications with detailed documentation and analysis, see the **[case_studies/](case_studies/)** directory.

Each case study includes:
- **README.md**: Blog-post style documentation (3,000-4,000 words)
- **analyze.py**: Python implementation with multiple analysis functions (800-1,100 lines)
- Monte Carlo simulation with 100K+ trials
- Sensitivity analysis
- Strategic insights and business implications

### Available Case Studies:

1. **[Drug Development](case_studies/drug_development/)** - Pharmaceutical R&D pipeline modeling
   - Why companies invest $2.6B when 95% of drugs fail
   - Portfolio strategy and power law returns
   - Phase II "Valley of Death" analysis

2. **[Startup Funding](case_studies/startup_funding/)** - Venture capital investment analysis
   - How VC funds achieve 3x returns with 85% failure rate
   - The "Series A Crunch" bottleneck
   - Portfolio sizing (10-100 companies)

3. **[Product Launch](case_studies/product_launch/)** - Consumer product launch strategy
   - Pilot validation creates $5M+ option value
   - Value of killing bad products early
   - With vs without pilot comparison

4. **[Litigation Strategy](case_studies/litigation_strategy/)** - Legal settlement decisions
   - Why 90-97% of cases settle before trial
   - Settlement zone calculation
   - Trial vs settlement economics

**Run all case studies**:
```bash
make case-studies
```

## Running Examples

### All examples at once:
```bash
make examples         # Run basic examples
make case-studies     # Run comprehensive case studies
```

### Individual examples:
```bash
.venv/bin/python3 examples/stpetersburg.py
.venv/bin/python3 examples/case_studies/drug_development/analyze.py
```

## Learning Path

**If you're new to Petersburg:**

1. Start with **stpetersburg.py** to understand the motivating paradox
2. Try **two_envelope_problem.py** and **necktie_paradox.py** for classic examples
3. Run **automatic_sensitivity_demo.py** to see key framework features
4. Pick a case study that matches your domain:
   - Pharma/biotech → [Drug Development](case_studies/drug_development/)
   - Startups/VC → [Startup Funding](case_studies/startup_funding/)
   - Product management → [Product Launch](case_studies/product_launch/)
   - Legal → [Litigation Strategy](case_studies/litigation_strategy/)

**If you want to solve your own problem:**

1. Read a relevant case study README to see the problem structure
2. Examine the corresponding analyze.py to see the implementation
3. Copy the structure and adapt to your specific problem
4. Use the framework features:
   - `g.simulate()` - Monte Carlo simulation
   - `g.print_sensitivity_report()` - Automatic sensitivity analysis
   - `g.get_probability_of_success()` - Success rate estimation
   - `g.mermaid()` - Visualization

## Framework Features Demonstrated

| Feature | Example Files |
|---------|--------------|
| Basic graph construction | All files |
| Monte Carlo simulation | All case studies, stpetersburg.py |
| Sensitivity analysis | automatic_sensitivity_demo.py, all case studies |
| Inversion analysis | drug_development/, startup_funding/ |
| Portfolio analysis | drug_development/, startup_funding/ |
| Visualization (Mermaid) | print.py, all case studies |
| Cost gradients | costwise_gradient.py |
| Value of information | decision_vs_research.py |
| Power law outcomes | drug_development/, startup_funding/ |
| Option value | product_launch/ |
| Settlement zones | litigation_strategy/ |

## Contributing Examples

Have a great example? Consider contributing!

**Good examples:**
- Demonstrate a specific framework feature
- Solve a real problem people face
- Include clear comments explaining the logic
- Show both the graph construction and analysis

**Great examples (case studies):**
- Include comprehensive documentation
- Use real-world research-backed data
- Show multiple analysis types
- Provide strategic insights
- Include visualizations

## Questions?

- For framework documentation, see [../README.md](../README.md)
- For technical architecture, see [../CLAUDE.md](../CLAUDE.md)
- For case study methodology, see [case_studies/README.md](case_studies/README.md)
