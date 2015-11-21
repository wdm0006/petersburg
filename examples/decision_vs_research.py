from petersburg import Graph

__author__ = 'willmcginnis'

if __name__ == '__main__':
    g = Graph()

    # decide or research then decide (w/o switching)
    g.from_dict({
          1: {'payoff': 0, 'after': []},
          2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10, 'weight': 2}]},
          3: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10}]},
          4: {'payoff': 0, 'after': [{'node_id': 1, 'cost': 10, 'weight': 1.5}]},
          5: {'payoff': 0, 'after': [{'node_id': 2, 'cost': 5}, {'node_id': 3, 'cost': 10}]},
          6: {'payoff': 0, 'after': [{'node_id': 2, 'cost': 5}, {'node_id': 4, 'cost': 10}]},
          7: {'payoff': 25, 'after': [{'node_id': 5, 'cost': 0}]},
          8: {'payoff': 15, 'after': [{'node_id': 5, 'cost': 0}]},
          9: {'payoff': 25, 'after': [{'node_id': 6, 'cost': 0}]},
          10: {'payoff': 15, 'after': [{'node_id': 6, 'cost': 0}]},
     })

    outcomes = []
    for _ in range(100000):
        outcomes.append(g.get_outcome())

    print('\n\nSimulated Output With Random Start')
    print(float(sum(outcomes))/len(outcomes))

    print('\n\nSimulated Profit of Each Starting Move')
    print(g.get_options(iters=10))