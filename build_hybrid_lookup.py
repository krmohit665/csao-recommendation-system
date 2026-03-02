import pickle
from collections import defaultdict

print("Loading conditional lookup...")
with open("data/conditional_lookup.pkl", "rb") as f:
    conditional_lookup = pickle.load(f)

print("Loading lift lookup...")
with open("data/lift_lookup.pkl", "rb") as f:
    lift_lookup = pickle.load(f)

print("Building hybrid lookup...")

hybrid_lookup = defaultdict(list)

for item in conditional_lookup:

    cond_dict = dict(conditional_lookup.get(item, []))
    lift_dict = dict(lift_lookup.get(item, []))

    all_related = set(cond_dict.keys()).union(set(lift_dict.keys()))

    for related in all_related:
        cond_score = cond_dict.get(related, 0)
        lift_score = lift_dict.get(related, 0)

        # weighted combination (tunable)
        hybrid_score = 0.7 * cond_score + 0.3 * lift_score

        hybrid_lookup[item].append((related, hybrid_score))

print("Saving hybrid lookup...")
with open("data/hybrid_lookup.pkl", "wb") as f:
    pickle.dump(dict(hybrid_lookup), f)

print("Done.")