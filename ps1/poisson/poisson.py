import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Load validation set
    x_valid, y_valid = util.load_dataset(eval_path, label_col='y', add_intercept=True)

    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    np.savetxt(save_path, clf.predict(x_valid))

    # Create scatter plot
    plt.scatter(y_valid, clf.predict(x_valid))
    plt.savefig('poisson_pred.jpg')

    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***

        # Get dimensions of x (m = nb of rows, n = nb of columns)
        m, n = x.shape

        # Use zero vector if initial guess for theta is None
        if self.theta is None:
            self.theta = np.zeros(n)

        # Run full batch gradient ascent algorithm

        i = 0

        for i in range(self.max_iter):

            grad = np.expand_dims(y - np.exp(x.dot(self.theta)), 1) * x
            self.theta = self.theta + self.step_size * np.sum(grad, axis=0)

            if i > 0 and np.sum(np.abs(self.theta - self.prev_theta)) <= self.eps:
                break

            self.prev_theta = np.copy(self.theta)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***

        return np.exp(x.dot(self.theta))

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
