import numpy as np


class NearestNeighbor:
    """Class supporting finding neareast embeddings of a query embeddings.

    Attrubutes:
        item_embeddings: a matrix of shape [N, k], such that row i is the embedding of
            item i.
        measure: One of ("cosine", "dot", "l2") specifying the similarity measure to be used
    """

    def __init__(self, item_embeddings, measure="cosine"):
        assert measure in ("dot", "cosine", "l2")
        self.measure = measure
        self.item_embeddings = item_embeddings
        if self.measure == "cosine":
            # nomalize embeding
            self.item_embeddings = item_embeddings / np.linalg.norm(
                item_embeddings, axis=1, keepdims=True
            )
        elif self.measure == "l2":
            self.squared_item_embedding = (item_embeddings ** 2).sum(axis=1)

    def find_nearest_neighbors(self, query_embedding, k=10):
        """Returns indices of k nearest neighbors"""
        # Denote q as query_emebdding vector, V as item_embeddings matrix.
        dot_products = query_embedding.dot(self.item_embeddings.T)
        if self.measure in ("dot", "cosine"):
            scores = dot_products
        elif self.measure == "l2":
            # ignore squared_query_embedding since it's the same for all item_embeddings
            scores = -(self.squared_item_embedding - 2 * dot_products)

        return (-scores).argsort()[:k]


def test_nearest_neighbors():
    query = np.array([1, 2])
    items = np.array(
        [
            [1, 2.2],  # neareast in l2
            [10, 21],  # neareast in dot product similarity
            [2, 4],  # nearest in cosine similarity
        ]
    )

    assert NearestNeighbor(items, "l2").find_nearest_neighbors(query, 1)[0] == 0
    assert NearestNeighbor(items, "dot").find_nearest_neighbors(query, 1)[0] == 1
    assert NearestNeighbor(items, "cosine").find_nearest_neighbors(query, 1)[0] == 2
    print("All tests passed")


test_nearest_neighbors()
