from sklearn.metrics.pairwise import cosine_similarity
def overlapping_coefficient(set1, set2):
    intersection = set1 & set2
    absolute_sim = len(intersection)
    min_cardinality = min(len(set1), len(set2))
    if min_cardinality == 0:
        sim = 0
    else:
        sim = absolute_sim / min_cardinality
    return intersection, absolute_sim, sim


# def overlapping(set1, set2):
#     intersection = set1 & set2
#     sim = len(intersection)
#     return intersection, sim


def my_cosine_similarity(v1, v2):
    return cosine_similarity(v1, v2)[0][0]