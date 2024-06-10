import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***

    # Load validation set
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    # Train logistic regression classifier using Newton's Method
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot validation data and decision boundary
    util.plot(x_valid, y_valid, clf.theta, save_path[:-3] + 'jpg', correction=1.0)

    # Save predicted probabilities on the validation set
    np.savetxt(save_path, clf.predict(x_valid))

    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

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

        # Run Newton's Method algorithm

        # Initialize i
        i = 0

        while i < self.max_iter:

            # Calculate predicted probabilities using the sigmoid function
            h = 1/(1+np.exp(-x.dot(self.theta)))

            # Compute the gradient
            gradient = (1/m) * x.T.dot(h-y)

            # Compute the Hessian
            hessian = (1/m) * x.T.dot(np.diag(h * (1-h))).dot(x)

            # Update theta
            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size * np.linalg.inv(hessian).dot(gradient)

            # Calculate the loss
            loss = -np.mean(y*np.log(h+self.eps) + (1-y) * np.log(1-h+self.eps))

            # Print loss at each step if verbose active
            if self.verbose:
                print("Loss = ", loss)

            # Break condition
            if np.sum(np.abs(prev_theta - self.theta)) < self.eps:
                break

            # Increment iteration counter
            i+=1

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***

        return 1/(1+np.exp(-(x.dot(self.theta))))

        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')
