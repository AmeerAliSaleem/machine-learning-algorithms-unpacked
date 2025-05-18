import numpy as np
from manim import *
from scipy.stats import multivariate_normal

class GMM:
    """
    A class to implement a __single__ training step of the GMM algorithm.

    Code for this class is taken from the following source: https://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
    """

    def __init__(self, k=3):
        self.k = k


    def initialise(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.pi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full(shape=self.shape, fill_value=1/self.k)

        random_row_indices = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [X[row_index,:] for row_index in random_row_indices]
        self.sigma = [np.cov(X.T) for _ in range(self.k)]
    

    def e_step(self, X):
        """
        Conducts the E-step of the Expectation-Maximisation algorithm.
        """

        self.weights = self.predict_proba(X)
        self.pi = self.weights.mean(axis=0)


    def m_step(self, X):
        """
        Conducts the M-step of the Expectation-Maximisation algorithm.
        """
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X*weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, aweights=(weight/total_weight).flatten(), bias=True)

    def fit(self, X):
        """
        Conducts a single training step of Expectation-Maximisation for the Gaussian Mixture Model.
        """

        self.e_step(X)
        self.m_step(X)

    def predict_proba(self, X):
        """
        Compute the w_{i,j} values and store them in a matrix.
        """

        likelihood = np.zeros((self.n, self.k))
        for i in range(self.k):
            dist = multivariate_normal(self.mu[i], self.sigma[i])
            likelihood[:, i] = dist.pdf(X)
        
        numerator = likelihood * self.pi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def predict(self, X_new):
        """
        Takes a new data point and returns the predicted class.
        """

        weights = self.predict_proba(X_new)
        return np.argmax(weights, axis=1)


class gmm_animation_test(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[-3, 3, 1],
            axis_config={'tip_shape': StealthTip}
            )
        
        gaussians = [
            (np.array([0, 0]), np.array([[1, 0], [0, 1]]), RED),
            (np.array([2, 2]), np.array([[1, 0.8], [0.8, 1]]), GREEN),
            (np.array([-2, -2]), np.array([[1.5, 0], [0, 0.5]]), BLUE),
        ]

        # Grid resolution: lower values give more detail but take longer to render
        res = 0.25

        x_vals = np.arange(-5, 5, res)
        y_vals = np.arange(-5, 5, res)

        self.add(ax)

        for (mean, cov, color) in gaussians:
            rv = multivariate_normal(mean, cov)
            for x in x_vals:
                for y in y_vals:
                    pos = np.array([x, y])
                    pdf_val = rv.pdf(pos)
                    # Only draw squares for points with a non-negligible probability density
                    if pdf_val > 1e-3:
                        alpha_scale = 5
                        alpha = min(pdf_val*alpha_scale, 1.0)
                        square = Square(side_length=res).set_fill(color=color, opacity=alpha).set_stroke(width=0)
                        square.move_to(ax.c2p(x, y))
                        self.add(square)

def generate_data(means, covs, n_points):
    """
    Some code from GPT to generate data from three 2D Multivariate Gaussians.
    The aim is to see whether the GMM can recover the Gaussians' parameters.
    """
    np.random.seed(8)

    # Generate points
    points = []
    labels = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        cluster = np.random.multivariate_normal(mean, cov, size=n_points)
        points.append(cluster)
        labels.append(np.full(n_points, i))

    # Concatenate everything
    X = np.vstack(points)
    y = np.concatenate(labels)

    return X, y


class gmm_animation(Scene):
    def construct(self):
        np.random.seed(8)

        ax = Axes(
            x_range=[-6, 6, 1],
            y_range=[-3, 3, 1],
            axis_config={'tip_shape': StealthTip}
            )

        # Constants
        RES = 0.1 # Grid resolution: lower values give more detail but take longer to render
        COLOURS = [RED, GREEN, BLUE]
        NUM_ITER = 50

        x_vals = np.arange(-5, 5, RES)
        y_vals = np.arange(-5, 5, RES)

        # Initialise grid of Manim Squares
        squares = [
            Square(side_length=RES).set_fill(color=BLACK).set_stroke(width=0).move_to(ax.c2p(x,y))
            for x in x_vals for y in y_vals
        ]

        # Generate data coordinates
        X, y = generate_data(
            n_points=100,
            means = [
                np.array([-2, 2]),
                np.array([0, 0]),
                np.array([2, -2]),
            ],

            covs = [
                np.array([[1, 0], [0, 0.25]]),
                np.array([[1, 0.8], [0.8, 1]]),
                np.array([[1.5, 0.5], [0.5, 0.25]]),
            ]
        )

        # Create corresponding dots
        dots = [
            Dot(ax.c2p(point[0], point[1], 0)).set_opacity(0.5) for point in X
        ]

        text_iter = Text("Iteration: ", color=TEAL_B, font_size=30).shift(RIGHT*4, UP*3)
        text_iter_number = [MathTex(f'{i}', color=TEAL_B).next_to(text_iter, RIGHT) for i in range(NUM_ITER+1)]

        gmm = GMM(k=3)
        gmm.initialise(X)

        self.add(ax, *squares, *dots, text_iter, text_iter_number[0])

        for i in range(NUM_ITER):
            gmm.fit(X)

            mu_vectors = gmm.mu
            sigma_matrices = gmm.sigma
            # pi = gmm.pi

            print(f"Iteration {i+1}:")
            print(f"Mu vectors: {mu_vectors}")
            print(f"Sigma matrices: {sigma_matrices}")

            square_colour_updates = []
            for square in squares:
                pdf_val = 0
                for j, (mean, cov) in enumerate(zip(mu_vectors, sigma_matrices)):
                    # Calculate probability density function value for the currrent GMM
                    current_pdf_val = multivariate_normal(mean, cov).pdf(square.get_center()[:2])

                    if current_pdf_val > max(1e-3, pdf_val):  # filter very low values
                        pdf_val = current_pdf_val
                        ALPHA_SCALE = 10
                        alpha = min(pdf_val*ALPHA_SCALE, 1.0)
                        square_colour_updates.append(square.animate.set_fill(color=COLOURS[j], opacity=alpha))

            self.play(ReplacementTransform(text_iter_number[i], text_iter_number[i+1]), *square_colour_updates, 
                      run_time=0.25)
    
    
        self.wait(2)

        