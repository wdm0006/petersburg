from petersburg import Graph

__author__ = 'willmcginnis'

if __name__ == '__main__':
    g = Graph()

    # st petersburg paradox w/ $10 entrance fee and only 1000 possible flips
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