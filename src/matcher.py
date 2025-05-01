from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def match_new_to_base(new_filenames, new_vectors, base_vectors):
    results = []
    for filename, new_vec in zip(new_filenames, new_vectors):
        sims = cosine_similarity([new_vec], base_vectors)[0]
        mean_sim = np.round(np.mean(sims) * 100, 2)
        results.append((filename, mean_sim))
    return results