"""
DBN model creation, training, prediction, and persistence.
"""

import os
import pickle

from .config import (
    HIDDEN_LAYERS,
    LEARNING_RATE_RBM,
    LEARNING_RATE_BP,
    N_EPOCHS_RBM,
    N_ITER_BACKPROP,
    BATCH_SIZE,
    ACTIVATION,
    DROPOUT,
    VERBOSE,
    OUTPUT_DIR,
    MODEL_FILENAME,
)


def _get_classifier_class():
    """
    Import the SupervisedDBNClassification from the deep-belief-network
    library, trying the TensorFlow backend first and falling back to NumPy.

    Returns
    -------
    class
        The SupervisedDBNClassification class.
    str
        The backend name ("tensorflow" or "numpy").
    """
    try:
        from dbn.tensorflow import SupervisedDBNClassification
        return SupervisedDBNClassification, "tensorflow"
    except (ImportError, AttributeError):
        pass
    try:
        from dbn.models import SupervisedDBNClassification
        return SupervisedDBNClassification, "numpy"
    except ImportError:
        raise ImportError(
            "deep-belief-network library not found.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/albertbup/deep-belief-network.git"
        )


def build_model():
    """
    Create a SupervisedDBNClassification instance with the hyperparameters
    defined in config.py.

    Returns
    -------
    classifier : SupervisedDBNClassification
        Untrained DBN model.
    backend : str
        Name of the backend ("tensorflow" or "numpy").
    """
    Classifier, backend = _get_classifier_class()

    classifier = Classifier(
        hidden_layers_structure=HIDDEN_LAYERS,
        learning_rate_rbm=LEARNING_RATE_RBM,
        learning_rate=LEARNING_RATE_BP,
        n_epochs_rbm=N_EPOCHS_RBM,
        n_iter_backprop=N_ITER_BACKPROP,
        batch_size=BATCH_SIZE,
        activation_function=ACTIVATION,
        dropout_p=DROPOUT,
        verbose=VERBOSE,
    )

    print(f"[model] DBN initialised — backend: {backend}")
    print(f"[model]   hidden_layers  = {HIDDEN_LAYERS}")
    print(f"[model]   lr_rbm         = {LEARNING_RATE_RBM}")
    print(f"[model]   lr_backprop    = {LEARNING_RATE_BP}")
    print(f"[model]   epochs_rbm     = {N_EPOCHS_RBM}")
    print(f"[model]   iter_backprop  = {N_ITER_BACKPROP}")
    print(f"[model]   batch_size     = {BATCH_SIZE}")
    print(f"[model]   activation     = {ACTIVATION}")
    print(f"[model]   dropout        = {DROPOUT}")

    return classifier, backend


def train_model(classifier, X_train, y_train):
    """
    Train the DBN on the given data (RBM pre-training + backprop
    fine-tuning happen inside .fit()).

    Parameters
    ----------
    classifier : SupervisedDBNClassification
        The DBN model.
    X_train : np.ndarray
        Training feature vectors.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    classifier : SupervisedDBNClassification
        The trained model (modified in-place).
    """
    print("[train] Starting training — this may take several minutes …")
    classifier.fit(X_train, y_train)
    print("[train] Training complete.")
    return classifier


def predict(classifier, X):
    """
    Generate predictions for the given feature matrix.

    Parameters
    ----------
    classifier : SupervisedDBNClassification
        Trained DBN model.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    np.ndarray
        Predicted labels.
    """
    return classifier.predict(X)


def save_model(classifier):
    """
    Persist the trained model to disk.

    Tries the library's .save() method first; falls back to pickle.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)

    try:
        classifier.save(path)
    except AttributeError:
        with open(path, "wb") as f:
            pickle.dump(classifier, f)

    print(f"[save] Model saved → {path}")


def load_model(path=None):
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    path : str, optional
        Path to the .pkl file. Defaults to outputs/dbn_mnist_model.pkl.

    Returns
    -------
    object
        The deserialized DBN model.
    """
    if path is None:
        path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)

    with open(path, "rb") as f:
        model = pickle.load(f)

    print(f"[load] Model loaded ← {path}")
    return model
