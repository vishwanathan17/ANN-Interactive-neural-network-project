"""
Explore the World of Neural Networks - Python version.

A fully connected Artificial Neural Network (ANN) built from scratch with NumPy.
Features: configurable hidden layers / neurons / activation, forward pass,
backpropagation, gradient descent, and result plots (saved as PNG files).

Usage:
    python ann.py                              # defaults: XOR, 3 hidden layers x 5 neurons
    python ann.py --dataset spiral --layers 3 --neurons 8 --activation tanh --epochs 3000
    python ann.py --help
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless backend: works on GitHub Actions / servers
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Activation functions (each returns f(z) and derivative in terms of a=f(z))
# ----------------------------------------------------------------------
ACTIVATIONS = {
    "tanh": (np.tanh, lambda a: 1.0 - a ** 2),
    "relu": (lambda z: np.maximum(0.0, z), lambda a: (a > 0).astype(float)),
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a: a * (1.0 - a)),
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ----------------------------------------------------------------------
# Neural network
# ----------------------------------------------------------------------
class NeuralNetwork:
    def __init__(self, sizes, activation="tanh", seed=None):
        """sizes example: [2, 5, 5, 5, 1] = input, three hidden layers, output."""
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.act, self.act_deriv = ACTIVATIONS[activation]
        self.weights, self.biases = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))  # Xavier/Glorot initialisation
            self.weights.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))

    def forward(self, X):
        """Forward pass. Returns list of activations for every layer."""
        activations = [X]
        last = len(self.weights) - 1
        for l, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            activations.append(sigmoid(z) if l == last else self.act(z))
        return activations

    def predict_proba(self, X):
        return self.forward(X)[-1]

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    @staticmethod
    def loss(y_true, y_prob):
        """Binary cross-entropy."""
        eps = 1e-9
        return float(-np.mean(y_true * np.log(y_prob + eps)
                              + (1 - y_true) * np.log(1 - y_prob + eps)))

    def train_step(self, X, y, lr):
        """One full-batch gradient descent step using backpropagation."""
        n = X.shape[0]
        acts = self.forward(X)
        delta = acts[-1] - y  # sigmoid + cross-entropy -> output error
        for l in reversed(range(len(self.weights))):
            grad_W = acts[l].T @ delta / n
            grad_b = delta.mean(axis=0, keepdims=True)
            if l > 0:  # propagate the error backwards (before updating W)
                delta = (delta @ self.weights[l].T) * self.act_deriv(acts[l])
            self.weights[l] -= lr * grad_W
            self.biases[l] -= lr * grad_b
        return self.loss(y, acts[-1])

    def fit(self, X, y, epochs=1000, lr=0.1, verbose=True):
        history = []
        for epoch in range(1, epochs + 1):
            history.append(self.train_step(X, y, lr))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                acc = accuracy(y, self.predict(X))
                print(f"Epoch {epoch:5d}/{epochs}  loss={history[-1]:.4f}  accuracy={acc:6.2%}")
        return history


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# ----------------------------------------------------------------------
# Datasets (2 input features, binary label)
# ----------------------------------------------------------------------
def make_dataset(name, n=300, seed=0):
    rng = np.random.default_rng(seed)
    if name == "xor":
        X = rng.uniform(-1, 1, (n, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(float)
    elif name == "circle":
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X ** 2).sum(axis=1) < 0.45).astype(float)
    elif name == "spiral":
        half = n // 2
        t = np.linspace(0, 3.2 * np.pi, half)
        r = np.linspace(0.05, 0.9, half)
        X0 = np.c_[r * np.cos(t), r * np.sin(t)]
        X1 = np.c_[r * np.cos(t + np.pi), r * np.sin(t + np.pi)]
        X = np.vstack([X0, X1]) + rng.normal(0, 0.03, (2 * half, 2))
        y = np.r_[np.zeros(half), np.ones(half)]
    elif name == "clusters":
        half = n // 2
        X = np.vstack([rng.normal(-0.45, 0.22, (half, 2)), rng.normal(0.45, 0.22, (half, 2))])
        y = np.r_[np.zeros(half), np.ones(half)]
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y.reshape(-1, 1)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_results(net, X, y, history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Loss curve
    plt.figure(figsize=(6, 4))
    plt.plot(history, color="#8b5cf6")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # 2. Decision boundary
    xx, yy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = net.predict_proba(grid).reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="coolwarm", edgecolors="k", s=20)
    plt.title("Decision boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(label="P(class = 1)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "decision_boundary.png"), dpi=150)
    plt.close()

    # 3. Network diagram (edge colour = weight sign, thickness = magnitude)
    sizes = net.sizes
    fig, ax = plt.subplots(figsize=(9, 5))
    pos = [[(l, (i - (n - 1) / 2) ) for i in range(n)] for l, n in enumerate(sizes)]
    for l, W in enumerate(net.weights):
        wmax = np.abs(W).max() or 1.0
        for i, (x0, y0) in enumerate(pos[l]):
            for j, (x1, y1) in enumerate(pos[l + 1]):
                w = W[i, j]
                ax.plot([x0, x1], [y0, y1], color="#2563eb" if w >= 0 else "#db2777",
                        linewidth=0.3 + 3 * abs(w) / wmax, alpha=0.6, zorder=1)
    colors = ["#4cc9f0"] + ["#8b5cf6"] * (len(sizes) - 2) + ["#f472b6"]
    for l, layer in enumerate(pos):
        xs, ys = zip(*layer)
        ax.scatter(xs, ys, s=220, color=colors[l], zorder=2, edgecolors="k")
        name = "Input" if l == 0 else "Output" if l == len(sizes) - 1 else f"Hidden {l}"
        ax.text(l, -max(sizes) / 2 - 0.4, f"{name} ({sizes[l]})", ha="center")
    ax.set_title("Trained network (blue = positive weight, pink = negative)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "network_diagram.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="ANN from scratch with NumPy")
    p.add_argument("--dataset", default="xor", choices=["xor", "circle", "spiral", "clusters"])
    p.add_argument("--layers", type=int, default=3, help="number of hidden layers")
    p.add_argument("--neurons", type=int, default=5, help="neurons per hidden layer")
    p.add_argument("--activation", default="tanh", choices=list(ACTIVATIONS))
    p.add_argument("--lr", type=float, default=0.1, help="learning rate")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs", help="folder for result images")
    args = p.parse_args()

    X, y = make_dataset(args.dataset, seed=args.seed)
    sizes = [2] + [args.neurons] * args.layers + [1]
    print(f"Dataset: {args.dataset} | Architecture: {sizes} | Activation: {args.activation}")

    net = NeuralNetwork(sizes, args.activation, seed=args.seed)
    history = net.fit(X, y, epochs=args.epochs, lr=args.lr)

    acc = accuracy(y, net.predict(X))
    print(f"\nTraining complete. Final loss={history[-1]:.4f}, accuracy={acc:.2%}")
    plot_results(net, X, y, history, args.out)
    print(f"Plots saved in '{args.out}/' (loss_curve.png, decision_boundary.png, network_diagram.png)")


if __name__ == "__main__":
    main()
