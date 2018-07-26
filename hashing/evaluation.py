from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

class Database(object):

    def __init__(self, database_vectors, targets, metric='hamming', n_limit=0):
        """
        Creates a database object that contains the database_vectors along with their labels
        :param database_vectors:
        :param targets:
        :param metric: metric used for retrieval
        :param n_limit: limit retrieval to the top n_limit results (this can severely impact the metrics, use with care)
        """

        if n_limit == 0:
            n_neighbors = len(targets)
        else:
            n_neighbors = n_limit

        self.nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric=metric)
        self.nn.fit(database_vectors)

        self.targets = np.cast[np.int](targets)

        bins = np.bincount(self.targets)
        idx = np.nonzero(bins)[0]
        self.instances_per_target = dict(zip(idx, bins[idx]))
        self.number_of_instances = float(len(targets))

        self.recall_levels = np.asarray([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    def get_binary_relevances(self, queries, targets, hamming_distance=2.0):
        """
        Executes the queries and returns the binary relevance vectors (one vector for each query)
        :param queries: the queries
        :param targets: the label of each query
        :return:
        """

        distances, indices = self.nn.kneighbors(queries)

        relevant_vectors = np.zeros_like(indices)
        hamming_2_prec = np.zeros((indices.shape[0], ))
        for i in range(targets.shape[0]):
            relevant_vectors[i, :] = self.targets[indices[i, :]] == targets[i]

            # Hamming radius according to sk-learn hamming distance definition
            idx = (distances[i, :] <= (hamming_distance/(queries.shape[1])))
            N = np.sum(idx)
            if N > 0:
                hamming_2_prec[i] = np.sum(self.targets[indices[i, idx]] == targets[i]) / N

        return relevant_vectors, hamming_2_prec

    def get_metrics(self, relevant_vectors, targets):
        """
        Evaluates the retrieval performance
        :param relevant_vectors: the relevant vectors for each query
        :param targets: labels of the queries
        :return:
        """

        # Calculate precisions per query
        precision = np.cumsum(relevant_vectors, axis=1) / np.arange(1, relevant_vectors.shape[1] + 1)

        # Calculate recall per query
        instances_per_query = np.zeros((targets.shape[0], 1))
        for i in range(targets.shape[0]):
            instances_per_query[i] = self.instances_per_target[targets[i]]
        recall = np.cumsum(relevant_vectors, axis=1) / instances_per_query

        # Calculate precision @ 11 recall point
        precision_at_recall_levels = np.zeros((targets.shape[0], self.recall_levels.shape[0]))
        for i in range(len(self.recall_levels)):
            idx = np.argmin(np.abs(recall - self.recall_levels[i]), axis=1)
            precision_at_recall_levels[:, i] = precision[np.arange(targets.shape[0]), idx]

        # Calculate the means values of the metrics
        ap = np.mean(precision_at_recall_levels, axis=1)

        return precision, recall, np.float32(precision_at_recall_levels), ap

    def evaluate(self, queries, targets, batch_size=1024):
        """
        Evaluates the performance of the database using the following metrics: interpolated map, interpolated precision,
        precision-recall curve, precision withing hamming radius 2
        It uses batches to reduce the memory required for the evaluation
        :param queries: the queries
        :param targets: the labels
        :return: the evaluated metrics
        """

        n_batches = int(len(queries) / batch_size)

        precision, recall, precision_at_recall_levels, ap = [], [], [], []
        hamming_2_prec = []
        for i in tqdm(range(n_batches)):
            relevant_vectors, c_hamming_2_prec = self.get_binary_relevances(queries[i * batch_size:(i + 1) * batch_size],
                                                          targets[i * batch_size:(i + 1) * batch_size])
            c_precision, c_recall, c_precision_at_recall_levels, c_ap = \
                self.get_metrics(relevant_vectors, targets[i * batch_size:(i + 1) * batch_size])
            precision.append(np.sum(c_precision, axis=0))
            recall.append(np.sum(c_recall, axis=0))
            precision_at_recall_levels.append(np.sum(c_precision_at_recall_levels, axis=0))
            ap.append(c_ap)
            hamming_2_prec.append(c_hamming_2_prec)

        # Final batch (if exists)
        if n_batches * batch_size < len(queries):
            relevant_vectors, c_hamming_2_prec = self.get_binary_relevances(queries[n_batches * batch_size:],
                                                          targets[n_batches * batch_size:])
            c_precision, c_recall, c_precision_at_recall_levels, c_ap = \
                self.get_metrics(relevant_vectors, targets[n_batches * batch_size:])
            precision.append(np.sum(c_precision, axis=0))
            recall.append(np.sum(c_recall, axis=0))
            precision_at_recall_levels.append(np.sum(c_precision_at_recall_levels, axis=0))
            ap.append(c_ap)
            hamming_2_prec.append(c_hamming_2_prec)


        ap = np.float64(np.concatenate(ap))
        N = np.float64(len(targets))

        precision_at_recall_levels = np.sum(np.float64(precision_at_recall_levels), axis=0) / N
        precision = np.sum(np.float64(precision), axis=0) / N
        recall = np.sum(np.float64(recall), axis=0) / N
        m_ap = np.mean(ap)
        hamming_2_prec = np.concatenate(hamming_2_prec)
        hamming_2_prec = np.mean(hamming_2_prec)

        return m_ap, precision, recall, precision_at_recall_levels, ap, hamming_2_prec


def evaluate_database(train_data, train_labels, test_data, test_labels, metric='hamming', n_limit=0, batch_size=1000):

    database = Database(train_data, train_labels, metric=metric, n_limit=n_limit)
    m_ap, precision, recall, precision_at_recall_levels, ap, hamming_2_prec = database.evaluate(test_data,
                                                                                        test_labels,
                                                                                        batch_size=batch_size)
    return m_ap, precision, recall, precision_at_recall_levels, ap, hamming_2_prec

