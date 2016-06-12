petersburg
==========

version number: 0.0.1
author: Will McGinnis

Overview
========

A simple framework for complex decisions.

Simulating and Predicting uncertain decisions
---------------------------------------------

Petersburg is a framework based on the decision theoretic concept of an individual agent approaching a network
of discrete decisions or probabilistic options. We model these networks as directed acyclic graphs that have a 
 few extra concepts:
 
 * Node Payoff: a potential reward for reaching some point
 * Edge Cost: a cost of taking a certain path
 * Edge Weight: a term related to the conditional likelihood of that edge being traversed (either a static number of classification model)

Using petersburg, you can:

 * Build graphs for a problem and understand
    * The most likely outcomes
    * The worst/best case scenarios
    * The choices or events which impact the outcome the most
    * The highest or most optimal costs for an edge
 * Build graphs for a problem and predict
    * Most likely outcome given some feature data with high accuracy even if intermediate decisions aren't well defined
    * Distributions of outcomes
    * Build static graphs from classifier based models and a sample of new data to perform the above analyses 
    
All in all, it's pretty neat.

Installation / Usage
====================

To install use pip:

    $ pip install petersburg


Or clone the repo:

    $ git clone https://github.com/wdm0006/petersburg.git
    $ python setup.py install
    
Contributing
============

TBD

Example Static Graph
====================

Here is a simple example of simulating the St. Petersburg Paradox game, with some slight variations. In this case the 
entrance fee is $10, and the game only has a maximum of 10,000 flips and is played 10,000,000 times.

    from petersburg import Graph
    
    if __name__ == '__main__':
        g = Graph()
    
        # st petersburg paradox w/ $10 entrance fee and only 10000 possible flips
        entrance_fee = 10
        gd = {1: {'payoff': 0, 'after': []}, 2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': entrance_fee}]}}
        nn = 3
        for idx in range(10000):
            node_id = 2 * (idx + 1)
            payoff = 2 ** (idx + 1)
            gd[nn] = {'payoff': payoff, 'after': [{'node_id': node_id, 'cost': 0, 'weight': 1}]}
            nn += 1
            gd[nn] = {'payoff': 0, 'after': [{'node_id': node_id, 'cost': 0, 'weight': 1}]}
            nn += 1
        g.from_dict(gd)
    
        outcomes = []
        for _ in range(10000000):
            outcomes.append(g.get_outcome())
    
        print('\n\nSimulated Output')
        print(sum(outcomes))

Via simulation, the outcome of this is a profit of: $197,592,288.  This will, of course, vary depending on the run, but
will approach infinity as the number of games goes to infinity, regardless of cost-to-play.

Example Prediction
==================

There are two prediction objects, both of which are scikit-learn style classes. 

 * MixedModeEstimator
 * FrequencyEstimator
 
Both have full working examples in the examples/estimation/* directory.