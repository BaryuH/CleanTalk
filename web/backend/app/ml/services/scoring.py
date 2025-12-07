LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

weights = {
    "toxic": 1,
    "obscene": 1,
    "insult": 1,
    "severe_toxic": 3,
    "identity_hate": 3,
    "threat": 4,
}

SAFE_MAX = 2
WARNING_MAX = 4


def compute_points(pred_vector):
    total = 0
    for label, value in zip(LABELS, pred_vector):
        if int(value) == 1:
            total += weights[label]
    return total


def classify(points, pred_vector):
    toxic, severe_toxic, obscene, threat, insult, identity_hate = pred_vector

    light_count = int(toxic) + int(obscene) + int(insult)
    heavy_count = int(severe_toxic) + int(identity_hate) + int(threat)
    total_on = light_count + heavy_count

    if total_on == 3:
        if heavy_count == 1 and light_count == 2:
            return "warning"
        if heavy_count == 2 and light_count == 1:
            return "ban"

    if int(threat) == 1:
        return "ban"

    if points <= SAFE_MAX:
        return "safe"

    if points <= WARNING_MAX:
        return "warning"

    return "ban"
