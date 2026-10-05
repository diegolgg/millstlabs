"""Controlled cue test. Synthetic observations, not a held-out survival result."""
import copy

import numpy as np

from .observations import Heuristic


def cue_pair(example):
    base = copy.deepcopy(example)
    base["self"][0] = 30
    base["self"][2:4] = [6, 6]
    base["self"][6] = 10
    base["self"][7:] = [0, 0, 20]
    base["local"][:] = 0
    base["threats"][:] = -1
    base["companions"][:] = -1
    base["stations"][:] = [[2, 6, -1, -1], [10, 6, -1, -1], [6, 16, 0, 10]]
    base["action_mask"] = np.array([1, 1, 1, 1, 0, 1, 1], np.int8)
    base["note_candidate"] = []
    base["note_opportunity"] = False
    pair = []
    for direction, stocks in [("west", (30, 0)), ("east", (0, 30))]:
        obs = copy.deepcopy(base)
        obs["corpus"][:] = -1
        for row, x, stock in zip(obs["corpus"], [2, 10], stocks):
            row[:] = [1, x, 6, stock, 1, 1, 1]
        evidence = f"food (2,6) stock {stocks[0]} at tick 9; food (10,6) stock {stocks[1]} at tick 9"
        obs["notes"] = [{"id": 0, "author": "peer", "age": 1,
                         "text": f"The {direction} station has food; the other station is empty.", "evidence": evidence}]
        pair.append(obs)
    return pair


def probe_notes(bank, policies, example):
    if len(example["stations"]) != 3 or len(example["corpus"]) < 2:
        return {"skipped": "Cue probe requires three stations and at least two corpus slots"}
    pair = cue_pair(example)
    probabilities = np.mean([bank.probabilities(i, [[o] for o in pair]).numpy() for i in policies], axis=0)
    return {"synthetic": True, "local_sensors_identical": True,
            "teacher_actions": [Heuristic(0).act(o) for o in pair],
            "west_food_action_probabilities": probabilities[0].tolist(),
            "east_food_action_probabilities": probabilities[1].tolist(),
            "max_action_probability_change": float(np.abs(probabilities[0]-probabilities[1]).max()),
            "correct_direction_probability": float((probabilities[0, 3]+probabilities[1, 2])/2),
            "both_argmax_directions_correct": bool(probabilities[0].argmax() == 3 and probabilities[1].argmax() == 2)}
