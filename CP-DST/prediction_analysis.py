import json
import matplotlib.pyplot as plt

prediction_true = json.load(open("cls-graph2p-correct-preds.json"))
prediction_false = json.load(open("cls-graph2p-errors.json"))

def extract_scores_true(data):
    graph_prob = []
    value_prob = []
    for dial in data:
        for turn in dial["pred"]:
            for slot, slot_pred in turn.items():
                max_graph_score = max(slot_pred["graph_scores"])
                if max_graph_score > 0:
                    graph_prob.append(max_graph_score)
                value_prob.append(slot_pred["value_prob"])

    return graph_prob, value_prob


def extract_scores_false(data):
    graph_prob = []
    value_prob = []
    for dial in data:
        for turn in dial["errors"]:
            for slot, slot_pred in turn["wrong_slots"].items():
                max_graph_score = max(slot_pred["graph_scores"])
                if max_graph_score > 0:
                    graph_prob.append(max_graph_score)
                value_prob.append(slot_pred["value_prob"])

    return graph_prob, value_prob


graph_prob_a, value_prob_a = extract_scores_true(prediction_true)
graph_prob_b, value_prob_b = extract_scores_false(prediction_false)

graph_prob = graph_prob_a + graph_prob_b
value_prob = value_prob_a + value_prob_b


plt.hist(graph_prob, bins=20)
# plt.hist(value_prob, bins=20)
plt.show()
