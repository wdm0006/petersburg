petersburg
==========

version number: 0.0.1
author: Will McGinnis

Overview
========

A simple framework for analyzing decision processes as directed graphs. 

Simulating uncertain decisions
------------------------------

Inspired by the [St. Petersburg paradox](https://en.wikipedia.org/wiki/St._Petersburg_paradox),
where the expected value of a simple game is infinity, unless reasonable contraints are set (i.e. the bank doesn't have
and infinite bankroll, you don't have infinite time to play, etc.). The idea is that we can represent decisions and the
decisions they lead to as directed acyclic graphs, wherein the nodes have payoffs, edges have costs, and at each node, 
the edge to progress on to is selected at random (with weights). 
 
With such a graph assembled, we can simulate the different first choices to see what the expected (simulated) outcome of 
each is.

Future goals
------------

In the future the plan is to add things like extremely rare and extreme events at each node (i.e. unknown unknowns, or 
black swans), more complex switching logic at nodes than weighted random, more complex cost and payoff models (utility),
and better methods for building complex graphs.

The first outcome will be more detailed metrics on the simulated outcomes than just mean (simulated expected value). Of 
course a positive expected value with extremely high risk is different than the same expected value without. We'd love 
to capture that concisely. 

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

Example
=======

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

