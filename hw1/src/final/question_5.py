import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

# Set random seed for reproducibility
np.random.seed(42)


class FeatureSet:
    """Represents a set of features conditioned on a given category"""

    def __init__(self, features):
        """Initializes the feature set using the specified array of feature
        vectors.

        :param features: An array of feature vectors, where each row corresponds
        to a different sample and each column corresponds to a different
        feature.
        """
        self._features = features
        self._mean = np.mean(self._features, axis=0)
        self._cov = np.cov(self._features, rowvar=False)
        self._name = ""

    @property
    def name(self):
        """Returns the name of the FeatureSet."""
        return self._name

    @property
    def features(self):
        """Returns the array of feature vectors."""
        return self._features

    @property
    def mean(self):
        """Returns the mean of the feature vectors."""
        return self._mean

    @property
    def cov(self):
        """Returns the covariance matrix of the feature vectors."""
        return self._cov

    @property
    def x1(self):
        """Returns the first feature vector (x1) from the feature set."""
        return self._features[:, 0]

    @property
    def x2(self):
        """Returns the second feature vector (x2) from the feature set."""
        return self._features[:, 1]

    def __add__(self, other):
        """Concatenates two FeatureSet objects by combining their feature
        vectors.

        :param other: Another FeatureSet object to concatenate with.
        :return: A new FeatureSet object containing the combined feature
        vectors.
        """
        combined = np.concatenate((self._features, other._features))
        return FeatureSet(combined)

    def __len__(self):
        """Returns the number of feature vectors in the FeatureSet."""
        return len(self._features)

    def __getitem__(self, index):
        """Returns the feature vector at the specified index.

        :param index: The index of the feature vector to retrieve.
        :return: The feature vector at the specified index.
        """
        return self._features[index]

    def set_name(self, name):
        """Sets the name of the FeatureSet.

        :param name: A string representing the name of the FeatureSet.
        """
        self._name = name
        return self

    def to_dist(self):
        """Converts the feature set into a multivariate Gaussian distribution
        using the mean and covariance of the feature vectors.

        :return: A multivariate Gaussian distribution representing the
        feature set.
        """
        return multivariate_normal(self._mean, self._cov)

    @staticmethod
    def from_dist(dist, num_samples):
        """Factory method to create a FeatureSet from a given distribution.
        This is done by sampling a category conditioned distribution for a
        specified number of samples.

        :param dist: A multivariate gaussian that typically represents the
        category conditioned distribution of the feature vectors.
        :param num_samples: The number of samples to extract.
        :return: A FeatureSet object containing the extracted features.
        """
        features = dist.rvs(size=num_samples)
        return FeatureSet(features)


class Metrics:
    """Represents evaluation metrics for a classifier against some data set."""

    def __init__(self, tau, tpr, fpr):
        """Initialize the metrics with the specified true positive rate (TPR)
        and false positive rate (FPR).
        :param tau: The threshold value used for classification.
        :param tpr: True positive rate (TPR) of the classifier.
        :param fpr: False positive rate (FPR) of the classifier.
        """
        self._tau = tau
        self._tpr = tpr
        self._fpr = fpr

    @property
    def tau(self):
        """Returns the threshold value (tau) used for classification."""
        return self._tau

    @property
    def tpr(self):
        """Returns the true positive rate (TPR) computed during evaluation."""
        return self._tpr

    @property
    def fpr(self):
        """Returns the false positive rate (FPR) computed during evaluation."""
        return self._fpr


class GeneralGaussianDiscriminant:
    """Represents a Bayesian discriminant function for a single category."""

    def __init__(self, prior, dist):
        """Instantiate the discriminant for a given cateogory using the
        specified prior and feature set.
        :param prior: Prior probability of the category.
        :param dist: A multivariate Gaussian distribution representing the
        category conditioned distribution of the feature vectors.
        """
        self._dist = dist
        self._prior = prior

    def calculate_score(self, x):
        """Compute the discriminant score for a given input x.
        :param x: An array of feature vectors for which to compute the
        discriminant score.
        :return: The discriminant score for the input feature vector.
        """
        return self._dist.logpdf(x) + np.log(self._prior)

    @staticmethod
    def from_mean_cov(prior, mean, cov):
        """Factory method to create a GeneralGaussianDiscriminant from a given
        mean and covariance matrix.
        :param prior: Prior probability of the category.
        :param mean: Mean vector of the category conditioned distribution.
        :param cov: Covariance matrix of the category conditioned distribution.
        :return: A GeneralGaussianDiscriminant object representing the category.
        """
        dist = multivariate_normal(mean, cov)
        return GeneralGaussianDiscriminant(prior, dist)

    @staticmethod
    def from_mean(prior, mean):
        """Factory method to create a GeneralGaussianDiscriminant from a given
        mean vector, assuming an identity covariance matrix.
        :param prior: Prior probability of the category.
        :param mean: Mean vector of the category conditioned distribution.
        :return: A GeneralGaussianDiscriminant object representing the category.
        """
        cov = np.eye(len(mean))
        return GeneralGaussianDiscriminant.from_mean_cov(prior, mean, cov)


class IdentityCovarianceDiscriminant:
    """Represents a Bayesian discriminant function for a single category with
    an identity covariance matrix."""

    def __init__(self, prior, mean):
        """Instantiate the discriminant for a given category using the
        specified prior and mean vector.
        :param prior: Prior probability of the category.
        :param mean: Mean vector of the category conditioned distribution.
        """
        self._prior = prior
        self._mean = mean

    def calculate_score(self, x):
        """Compute the discriminant score for a given input x using  an
        expression that is simplified for the case of an identity covariance
        matrix.
        :param x: An array of feature vectors for which to compute the
        discriminant score.
        :return: The discriminant score for the input feature vector.
        """

        diff = x - self._mean
        square_distance = np.sum(diff**2, axis=-1)
        return -0.5 * square_distance + np.log(self._prior)


class MahalanobisCovarianceDiscriminant:
    """Represents a Bayesian discriminant function for a single category with
    a Mahalanobis distance-based covariance matrix."""

    def __init__(self, prior, mean, cov):
        """Instantiate the discriminant for a given category using the
        specified prior, mean vector, and covariance matrix.
        :param prior: Prior probability of the category.
        :param mean: Mean vector of the category conditioned distribution.
        :param cov: Covariance matrix of the category conditioned distribution.
        """
        self._prior = prior
        self._mean = mean
        self._cov_inv = np.linalg.inv(cov)

    def calculate_score(self, x):
        """Compute the discriminant score for a given input x using an
        expression that is simplified for the case of a Mahalanobis distance-
        based covariance matrix.
        :param x: An array of feature vectors for which to compute the
        discriminant score.
        :return: The discriminant score for the input feature vector.
        """
        diff = x - self._mean
        mahalanobis_distance = np.sum(diff @ self._cov_inv * diff, axis=-1)
        return -0.5 * mahalanobis_distance + np.log(self._prior)


class Classifier:
    """Abstract base class that represents a classifier"""

    def __init__(self, name):
        """Initialize the classifier with a name.
        :param name: A string representing the name of the classifier.
        """
        self._name = name
        self._color = "gray"  # Default color for plotting

    @property
    def name(self):
        """Returns the name of the classifier."""
        return self._name

    @property
    def color(self):
        """Returns the color associated with the classifier for plotting."""
        return self._color

    def set_color(self, color):
        """Sets the color associated with the classifier for plotting.
        :param color: A string representing the color to set.
        """
        self._color = color
        return self

    def classify(self, feature_set, threshold):
        """Classify the input feature vector x using the naive approach.
        :param x: A set of feature vectors to classify.
        :param threshold: The threshold used to classify samples as positive or
        negative.
        :return: An array of predicted class labels for the input feature
        vectors.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def calculate_tau_range(self, x):
        """Calculate the range of threshold values (tau) for the classifier.
        :param x: An array of feature vectors for which to calculate the tau
        range.
        :return: A tuple (tau_min, tau_max) representing the minimum and
        maximum threshold values.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate(self, threshold, feature_set, labels):
        """Generate evaluation metrics for the classifier using the specified
        thresholod, feature set and true class labels.

        :param threshold: The threshold used to classify samples as positive or
        negative.
        :param feature_set: A featureset object containing the feature vectors
        to classify.
        :param labels: An array of true class labels corresponding to the
        feature vectors.
        """

        pred = self.classify(feature_set, threshold)

        # We're assuming the vectors are 2D.
        pred_actual_pairs = [(int(x), int(y)) for (x, y) in zip(pred, labels)]
        pred_result_pairs = [
            (pred_label, pred_label * true_label)
            for (pred_label, true_label) in pred_actual_pairs
        ]

        true_pos = 1.0 * pred_result_pairs.count((1, 1))
        false_pos = 1.0 * pred_result_pairs.count((1, -1))
        true_neg = 1.0 * pred_result_pairs.count((-1, 1))
        false_neg = 1.0 * pred_result_pairs.count((-1, -1))
        total_pos = true_pos + false_neg
        total_neg = true_neg + false_pos

        tpr = 1.0 * true_pos / total_pos if total_pos > 0 else 0.0
        fpr = 1.0 * false_pos / total_neg if total_neg > 0 else 0.0
        return Metrics(threshold, tpr, fpr)


class NaiveClassifier(Classifier):
    """A naive classifier that randomly assigns class labels based on a
    specified probability threshold."""

    def __init__(self):
        """Initialize the NaiveClassifier with a name."""
        super().__init__("Naive Classifier")

    def classify(self, x, threshold):
        return -np.sign(np.random.rand(len(x)) - threshold)

    def calculate_tau_range(self, x):
        return 0.0, 1.0


class BayesianClassifier(Classifier):
    """A Bayesian classifier that uses discriminant functions to classify
    samples based on their feature vectors."""

    def __init__(self, name, g_pos, g_neg):
        """Initialize the BayesianClassifier with positive and negative
        discriminant functions.
        :param name: A string representing the name of the classifier.
        :param g_pos: A GeneralGaussianDiscriminant object representing the
        positive class.
        :param g_neg: A GeneralGaussianDiscriminant object representing the
        negative class.
        """
        super().__init__(name)
        self._g_pos = g_pos
        self._g_neg = g_neg

    def classify(self, feature_set, threshold):
        pred = []
        for feature in feature_set.features:
            score = self._g_pos.calculate_score(
                feature
            ) - self._g_neg.calculate_score(feature)
            pred.append(np.sign(score - threshold))
        return np.array(pred)

    def calculate_tau_range(self, feature_set):
        """Calculate the range of threshold values (tau) for the Bayesian
        classifier based on the input feature vectors.
        :param feature_set: A featureset object containing the feature vectors
        for which to calculate the tau
        :return: A tuple (tau_min, tau_max) representing the minimum and
        maximum threshold values.
        """
        scores = [
            self._g_pos.calculate_score(feature)
            - self._g_neg.calculate_score(feature)
            for feature in feature_set.features
        ]
        return min(scores), max(scores)


def plot_contours(
    features_pos, features_neg, likelihood_pos, likelihood_neg, file_name=None
):
    """Plot the contours of the positive and negative class distributions
    along with the samples from each class.
    :param features_pos: A FeatureSet object containing the positive class
    samples.
    :param features_neg: A FeatureSet object containing the negative class
    samples.
    :param likelihood_pos: A multivariate Gaussian distribution representing
    the positive class conditioned distribution.
    :param likelihood_neg: A multivariate Gaussian distribution representing
    the negative class conditioned distribution.
    :param file_name: Optional. If provided, the contour plot will be saved to
    the specified file name.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x1, x2 = np.mgrid[-1:8:0.01, -1:6:0.01]
    positions = np.dstack((x1, x2))

    for features, dist, color in zip(
        [features_pos, features_neg],
        [likelihood_pos, likelihood_neg],
        ["blue", "orange"],
    ):
        ax.scatter(
            features.x1, features.x2, label=features.name, alpha=0.7, c=color
        )
        ax.contour(x1, x2, dist.pdf(positions), levels=5, colors=color)

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    ax.legend()
    if file_name:
        plt.savefig(f"outputs/{file_name}", format="pdf", bbox_inches="tight")
    else:
        plt.show()


def plot_roc_curves(
    classifiers, features_combined, true_labels, file_name=None
):
    """Plot ROC curves for multiple classifiers based on their evaluation
    metrics.
    :param classifiers: A list of Classifier objects to evaluate and plot.
    :param features_combined: A FeatureSet object containing the combined
    feature vectors for evaluation.
    :param true_labels: An array of true class labels corresponding to the
    feature vectors.
    :param file_name: Optional. If provided, the ROC curve plot will be saved
    to the specified file name.
    """
    fig = plt.figure()
    sub_plot = fig.add_subplot(111)
    plt.title(f"ROC Curve - Multiple Classifiers")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid()

    for classifier in classifiers:
        print(f"Evaluating classifier: {classifier.name}")
        tau_min, tau_max = classifier.calculate_tau_range(features_combined)
        fpr_values, tpr_values = [], []
        for tau in np.linspace(tau_min, tau_max, 11):
            metrics = classifier.evaluate(tau, features_combined, true_labels)
            fpr_values.append(metrics.fpr)
            tpr_values.append(metrics.tpr)

        sub_plot.plot(
            fpr_values,
            tpr_values,
            linestyle="-",
            label=classifier.name,
            color=classifier.color,
        )

    plt.legend(loc="lower right")
    if file_name:
        plt.savefig(f"outputs/{file_name}", format="pdf", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    PRIOR_POS = 0.5
    NUM_SAMPLES = 1000
    POS_MEAN = [1, 3]
    POS_COV = [[1, 0.8], [0.8, 1]]
    NEG_MEAN = [4, 4]
    NEG_COV = [[0.99, 0.81], [0.81, 1.01]]
    SHARED_COV = [[1.0, -0.8], [-0.8, 1.0]]

    likelihood_pos = multivariate_normal(POS_MEAN, POS_COV)
    likelihood_neg = multivariate_normal(NEG_MEAN, NEG_COV)

    num_samples_pos = np.random.binomial(NUM_SAMPLES, PRIOR_POS)
    num_samples_neg = NUM_SAMPLES - num_samples_pos

    estimated_prior_pos = num_samples_pos / NUM_SAMPLES
    estimated_prior_neg = num_samples_neg / NUM_SAMPLES

    # Positive samples drawn from distribution conditioned on positive class
    features_pos = FeatureSet.from_dist(
        likelihood_pos, num_samples_pos
    ).set_name("Sea Bass (+)")

    # Negative samples drawn from distribution conditioned on negative class
    features_neg = FeatureSet.from_dist(
        likelihood_neg, num_samples_neg
    ).set_name("Salmon (-)")

    # Combine positive and negative samples into a single feature set and create
    # true labels
    features_combined = features_pos + features_neg
    true_labels = np.concatenate(
        (np.ones(num_samples_pos), -np.ones(num_samples_neg))
    )

    plot_contours(features_pos, features_neg, likelihood_pos, likelihood_neg)

    classifier_list_1 = [
        NaiveClassifier().set_color("orange"),
        BayesianClassifier(
            "General Gaussian - Identity Covariance",
            GeneralGaussianDiscriminant.from_mean(
                estimated_prior_pos, features_pos.mean
            ),
            GeneralGaussianDiscriminant.from_mean(
                estimated_prior_neg, features_neg.mean
            ),
        ).set_color("blue"),
        BayesianClassifier(
            "General Gaussian - Shared Covariance",
            GeneralGaussianDiscriminant.from_mean_cov(
                estimated_prior_pos, features_pos.mean, SHARED_COV
            ),
            GeneralGaussianDiscriminant.from_mean_cov(
                estimated_prior_neg, features_neg.mean, SHARED_COV
            ),
        ).set_color("green"),
    ]
    plot_roc_curves(
        classifier_list_1,
        features_combined,
        true_labels,
        "roc_curves_general.pdf",
    )

    classifier_list_2 = [
        NaiveClassifier().set_color("orange"),
        BayesianClassifier(
            "Simplified Gaussian - Identity Covariance",
            IdentityCovarianceDiscriminant(
                estimated_prior_pos, features_pos.mean
            ),
            IdentityCovarianceDiscriminant(
                estimated_prior_neg, features_neg.mean
            ),
        ).set_color("purple"),
        BayesianClassifier(
            "Simplified Gaussian - Mahalanobis Covariance",
            MahalanobisCovarianceDiscriminant(
                estimated_prior_pos, features_pos.mean, SHARED_COV
            ),
            MahalanobisCovarianceDiscriminant(
                estimated_prior_neg, features_neg.mean, SHARED_COV
            ),
        ).set_color("brown"),
    ]

    plot_roc_curves(
        classifier_list_2,
        features_combined,
        true_labels,
        "roc_curves_simplified.pdf",
    )
