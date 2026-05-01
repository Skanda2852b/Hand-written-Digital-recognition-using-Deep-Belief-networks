import os
import sys

from src.data_loader import load_mnist, preprocess
from src.model import build_model, train_model, predict, save_model, load_model
from src.evaluate import evaluate, full_report, compute_confusion_matrix
from src.visualize import generate_all_figures
from src.utils import print_section

def main():
    print("============================================================")
    print("  Handwritten Digit Recognition using Deep Belief Networks")
    print("  VTU 6th Semester Mini Project")
    print("============================================================")
    
    print_section(1, "Loading and preprocessing MNIST dataset")
    X_train_full, y_train_full, X_test, y_test = load_mnist()
    data = preprocess(X_train_full, y_train_full, X_test, y_test)
    
    model_path = os.path.join("outputs", "dbn_mnist_model.pkl")
    if os.path.exists(model_path):
        print_section(2, "Loading Pre-trained DBN Model")
        classifier = load_model(model_path)
    else:
        print_section(2, "Building DBN Model")
        classifier, backend = build_model()
        
        print_section(3, "Training DBN Model")
        classifier = train_model(classifier, data["X_train"], data["y_train"])
        
        print_section(3.5, "Saving Model")
        save_model(classifier)
    
    print_section(4, "Evaluating Model")
    y_pred, accuracy = evaluate(classifier, data["X_test"], data["y_test"], "test")
    report = full_report(data["y_test"], y_pred)
    
    print_section(5, "Generating Visualizations")
    cm = compute_confusion_matrix(data["y_test"], y_pred)
    generate_all_figures(cm, data["X_test"], data["y_test"], y_pred, accuracy)
    
    # Model is saved immediately after training if it was built from scratch
    
    print("\nPROJECT EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
