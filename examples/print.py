from petersburg import Graph
import os

__author__ = 'willmcginnis'

if __name__ == '__main__':
    data = []
    c_switch = 2
    in_house = 1
    third_party = in_house * 0.5
    switching = in_house * c_switch
    weight = 0.67
    g = Graph()
    g.from_dict({
          1: {'payoff': 0, 'after': []},
          2: {'payoff': 0, 'after': [{'node_id': 1, 'cost': in_house}]},
          3: {'payoff': 0, 'after': [{'node_id': 2, 'cost': in_house}]},
          4: {'payoff': 0, 'after': [{'node_id': 1, 'cost': third_party}]},
          5: {'payoff': 0, 'after': [{'node_id': 4, 'cost': third_party}]},
          6: {'payoff': 0, 'after': [{'node_id': 4, 'cost': switching, 'weight': weight}]},
          7: {'payoff': 0, 'after': [{'node_id': 6, 'cost': in_house}, {'node_id': 5, 'cost': 2 * switching, 'weight': weight}]},
          8: {'payoff': 0, 'after': [{'node_id': 3, 'cost': in_house}]},
          9: {'payoff': 0, 'after': [{'node_id': 5, 'cost': third_party}]},
          10: {'payoff': 0, 'after': [{'node_id': 7, 'cost': in_house}, {'node_id': 5, 'cost': 3 * switching, 'weight': weight}]},
          11: {'payoff': 0, 'after': [{'node_id': 8, 'cost': 0}]},
          12: {'payoff': 0, 'after': [{'node_id': 9, 'cost': 0}]},
          13: {'payoff': 0, 'after': [{'node_id': 10, 'cost': 0}]},
    })

    g.plot(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img', 'print.png'))