"""
Configuration and hyperparameters for the DBN MNIST project.

All tunable parameters are centralized here so you can experiment
without touching the core logic.
"""

# ── Reproducibility ──────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data ─────────────────────────────────────────────────────────
VALIDATION_SPLIT = 0.10          # 10 % of training set for validation
NORMALIZE = True                 # Scale pixel values to [0, 1]
FLATTEN = True                   # Reshape 28×28 → 784

# ── DBN Architecture ────────────────────────────────────────────
HIDDEN_LAYERS = [128, 128]       # Two RBM layers, 128 units each (much faster)
LEARNING_RATE_RBM = 0.05         # Learning rate for CD training
LEARNING_RATE_BP = 0.1           # Learning rate for backprop fine-tune
N_EPOCHS_RBM = 3                 # Training epochs per RBM layer (reduced)
N_ITER_BACKPROP = 15             # Backpropagation iterations (reduced)
BATCH_SIZE = 128                 # Mini-batch size (increased for speed)
ACTIVATION = "relu"              # Hidden-unit activation
DROPOUT = 0.2                    # Dropout probability (fine-tune phase)
VERBOSE = True                   # Print training progress

# ── Paths ────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
MODEL_FILENAME = "dbn_mnist_model.pkl"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"
SAMPLE_PREDICTIONS_FILE = "sample_predictions.png"
PER_CLASS_ACCURACY_FILE = "per_class_accuracy.png"
PRETRAINING_COMPARISON_FILE = "pretraining_comparison.png"

# ── Visualization ────────────────────────────────────────────────
FIG_DPI = 150                    # Dots per inch for saved figures
NUM_SAMPLE_PREDICTIONS = 10      # Number of random test samples to show
