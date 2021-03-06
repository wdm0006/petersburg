"""
A variation of the st petersburg game where the bank has a limited bankroll, and therefore cannot payout an infinite
amount of money on any given hand.

"""

from petersburg import Graph

__author__ = 'willmcginnis'

if __name__ == '__main__':
    for bankroll in [100, 10e6, 10e9, 79200000000]:
        g = Graph()

        entrance_fee = 0
        gd = {1: {'payoff': 0, 'after': []}, 2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': entrance_fee}]}}
        nn = 3
        idx = 0
        payoff = 2 ** (idx + 1)
        while payoff <= bankroll:
            node_id = 2 * (idx + 1)
            gd[nn] = {'payoff': payoff, 'after': [{'node_id': node_id, 'cost': 0, 'weight': 1}]}
            nn += 1
            gd[nn] = {'payoff': 0, 'after': [{'node_id': node_id, 'cost': 0, 'weight': 1}]}
            nn += 1
            idx += 1
            payoff = 2 ** (idx + 1)

        g.from_dict(gd)

        outcomes = []
        for _ in range(10000000):
            outcomes.append(g.get_outcome())

        print('\n\nExpected value of game with a $%d bankroll' % (bankroll, ))
        print(float(sum(outcomes)) / len(outcomes))