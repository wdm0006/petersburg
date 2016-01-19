from petersburg import Graph

__author__ = 'willmcginnis'

if __name__ == '__main__':
    g = Graph()

    # st petersburg paradox w/ $10 entrance fee, a 1000 dollar bankroll and only 100 possible flips
    entrance_fee = 10
    gd = {1: {'payoff': 0, 'after': []}, 2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': entrance_fee}]}}
    nn = 3
    for idx in range(100):
        node_id = 2 * (idx + 1)
        payoff = 2 ** (idx + 1)
        gd[nn] = {'payoff': payoff, 'after': [{'node_id': node_id, 'cost': 0, 'weight': 1}]}
        nn += 1
        gd[nn] = {'payoff': 0, 'after': [{'node_id': node_id, 'cost': 0, 'weight': 1}]}
        nn += 1
    g.from_dict(gd)

    outcomes = []
    for _ in range(1000):
        outcome = g.get_outcome(iters=1000, ruin=True, starting_bank=1000)
        outcomes.append(outcome)

    print(float(sum(outcomes))/len(outcomes))
    print(min(outcomes))
    print(max(outcomes))