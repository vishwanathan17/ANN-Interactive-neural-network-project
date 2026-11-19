# Neural Network from Scratch (Python)

A fully connected Artificial Neural Network (ANN) built from scratch with NumPy.
It trains with backpropagation and saves a loss curve, a decision boundary and a
network diagram as images.

## Run

```bash
pip install -r requirements.txt
python ann.py
python ann.py --dataset spiral --layers 3 --neurons 8 --activation tanh --epochs 3000
```

Options: `--dataset` (xor, circle, spiral, clusters), `--layers`, `--neurons`,
`--activation` (tanh, relu, sigmoid), `--lr`, `--epochs`, `--seed`, `--out`.

## Output

Results are written to the `outputs/` folder:
`loss_curve.png`, `decision_boundary.png`, `network_diagram.png`.

## How it works

| Step | Formula |
|------|---------|
| Forward pass | `a = f(W·x + b)` (output layer uses sigmoid) |
| Loss | Binary cross-entropy |
| Backpropagation | Chain rule; output error = `ŷ − y` |
| Update | `w ← w − η · ∂L/∂w` |

## Files

```
ann.py            the neural network and training script
requirements.txt  dependencies (numpy, matplotlib)
```
