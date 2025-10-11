# Petersburg Framework: Real-World Case Studies

This directory contains in-depth case studies demonstrating how the petersburg framework can be applied to complex, real-world decision problems. Each case study includes both a markdown explanation and a Python implementation using the petersburg framework.

## Overview

The petersburg framework is designed to help with complex decisions involving:
- **Sequential stages** with dependencies
- **Uncertain outcomes** at each decision point
- **Escalating costs** as you progress
- **Multiple terminal states** ranging from failure to success
- **Path dependencies** where early choices affect later probabilities

These case studies demonstrate the framework's power for:
1. **Simulation**: Understanding the distribution of possible outcomes
2. **Optimization**: Finding the best decision at each stage
3. **Inversion**: Working backward from desired outcomes to required inputs
4. **Sensitivity analysis**: Identifying which uncertainties matter most

## Case Studies

### 1. Drug Development ([drug_development.md](drug_development.md) | [drug_development.py](drug_development.py))

**Domain**: Pharmaceutical R&D

**Problem**: Should a pharmaceutical company invest $2B+ to develop a new drug through clinical trials?

**Key Features**:
- 5 sequential phases (Pre-clinical → Phase I → Phase II → Phase III → FDA Review)
- Only 5-14% of drugs that enter trials receive approval
- Costs escalate from $150M (pre-clinical) to $225M (Phase III)
- Requires portfolio strategy: need 10-20 drugs in pipeline to expect 1 blockbuster

**Key Insights**:
- Individual drugs have negative expected value, but portfolios are profitable
- Phase II success rate is a critical leverage point
- Early-stage improvements compound exponentially
- Demonstrates power of inversion: "What would have to be true for this $2B investment to pay off?"

**Run the analysis**:
```bash
python drug_development.py
```

### 2. Startup Funding ([startup_funding.md](startup_funding.md) | [startup_funding.py](startup_funding.py))

**Domain**: Venture capital and entrepreneurship

**Problem**: Should a startup raise another round of funding? Should a VC invest?

**Key Features**:
- 5 funding stages (Pre-seed → Seed → Series A → Series B → Series C → Exit)
- Overall success rate: ~1-2% reach unicorn exit
- Power law dynamics: top 10% of exits contribute 70%+ of returns
- The "Valley of Death" between seed and Series A has lowest success rate (~30%)

**Key Insights**:
- VC portfolio strategy essential: need 20-30 investments to capture winners
- Each funding round is both validation and commitment to harder milestones
- Small improvements at each stage compound to huge impacts
- Inversion helps: work backward from $1B exit to determine required milestones

**Run the analysis**:
```bash
python startup_funding.py
```

### 3. Product Launch ([product_launch.md](product_launch.md) | [product_launch.py](product_launch.py))

**Domain**: Product management and innovation

**Problem**: Should we launch this new product? Should we kill it after pilot testing?

**Key Features**:
- 5 stages (Concept → Prototype → Pilot → Regional → Full Launch)
- 40-90% of new products fail (industry-dependent)
- Pilot stage is critical decision point (40% proceed to full launch)
- Costs escalate from $300K (concept) to $25M+ (full launch)

**Key Insights**:
- Strong early validation improves EV by 250%+
- Pilot stage is the critical filter - weak signals should trigger STOP decision
- Sunk cost fallacy is deadly: having spent $3.8M doesn't justify risking $33M more
- Portfolio approach: 60-70% should be killed before pilot (this is good!)

**Run the analysis**:
```bash
python product_launch.py
```

### 4. Litigation Strategy ([litigation_strategy.md](litigation_strategy.md) | [litigation_strategy.py](litigation_strategy.py))

**Domain**: Legal decision-making

**Problem**: Should we settle this lawsuit or proceed to trial?

**Key Features**:
- 6 stages (Investigation → Filing → Discovery → Pre-trial → Trial → Appeal)
- 70-80% of cases settle before trial
- Costs range from $50K (investigation) to $1.5M+ (trial)
- Both plaintiff and defendant face settlement vs. trial decisions

**Key Insights**:
- Settlement range exists when defendant's expected loss > plaintiff's minimum
- Discovery is expensive ($500K-$2M) creating pressure to settle
- Risk aversion strongly favors settlement over trial uncertainty
- Case strength assessment is crucial: weak cases should settle or drop early
- Information from discovery can justify higher settlements

**Run the analysis**:
```bash
python litigation_strategy.py
```

## Common Themes

Across all these case studies, several patterns emerge:

### 1. Sequential Decision-Making
All involve multiple stages where you can stop, pivot, or continue. The decision at each stage depends on:
- Results from prior stages
- Updated probability estimates
- Remaining costs to completion
- Expected value of continuing vs. stopping

### 2. Compounding Probabilities
Success requires passing ALL stages. This creates exponential decay:
- 70% × 60% × 50% × 40% = 8.4% overall success
- Improving each stage by 10 points → 80% × 70% × 60% × 50% = 16.8% (+100% improvement!)

### 3. Portfolio Strategy
Because individual attempts often have negative expected value, success requires:
- Multiple parallel bets
- Asymmetric payoffs (losses capped, gains unbounded)
- Disciplined killing of weak candidates
- Doubling down on winners

### 4. The Power of Inversion
Working backward from desired outcome reveals required conditions:
- "What would have to be true for this to work?"
- Forces explicit articulation of assumptions
- Identifies critical uncertainties
- Often reveals that the decision is NOT justified

### 5. Stage Gates Matter
The highest-value decisions are often STOP decisions:
- Kill early and often (fail fast, fail cheap)
- Don't throw good money after bad (avoid sunk cost fallacy)
- Weak signals get weaker, strong signals get stronger
- The pilot/prototype stage is typically the critical filter

### 6. Information Has Value
Each stage purchases information:
- Discovery in litigation reveals evidence
- Prototype testing reveals product issues
- Phase II trials reveal efficacy signals
- Sometimes worth investing just to learn, even if you expect to stop

## Using These Examples

### For Learning:
1. Read the markdown file to understand the domain and decision structure
2. Run the Python simulation to see quantitative results
3. Modify parameters to explore sensitivity
4. Try different scenarios (optimistic vs. pessimistic)

### For Your Own Problems:
1. Map your decision to the framework:
   - What are the sequential stages?
   - What are the costs at each stage?
   - What are the success probabilities?
   - What are the terminal outcomes?

2. Build a petersburg graph:
   - Start with a simple version
   - Add complexity incrementally
   - Validate against historical data if available

3. Use for decision support:
   - Simulate outcomes (Monte Carlo)
   - Compare scenarios
   - Perform sensitivity analysis
   - Apply inversion thinking

## Further Reading

Each case study includes references to domain-specific resources. For the petersburg framework itself, see:

- [README.md](../../README.md) - Framework overview
- [CLAUDE.md](../../CLAUDE.md) - Technical architecture
- [examples/](../) - Simpler examples and paradoxes

## Contributing

Have a great case study idea? Consider contributing! Ideal case studies:
- Involve sequential decisions under uncertainty
- Have real-world business or personal relevance
- Demonstrate unique aspects of the framework
- Include both explanation and working code

## Questions?

These case studies demonstrate the "why" and "how" of using petersburg for complex decisions. They show that the framework isn't just about simulation - it's about structured thinking for high-stakes, uncertain decisions.
