# Handwritten Digit Recognition using Deep Belief Networks

This project is a VTU 6th Semester Mini Project that demonstrates how to recognize handwritten digits (from the MNIST dataset) using a Deep Belief Network (DBN). 

## 🚀 Features
- Uses Restricted Boltzmann Machines (RBMs) to pre-train hidden layers.
- Fine-tunes the network using backpropagation.
- Automatically handles the downloading and preprocessing of the MNIST dataset.
- Generates beautiful visualizations including confusion matrices, sample predictions, and accuracy charts.

## 🛠️ Prerequisites
Make sure you have the following installed on your machine:
- **Python 3.10 or 3.11** 
- **Git**

## 💻 Local Setup Instructions

Follow these steps to set up the project on your local machine:

### 1. Clone the repository
Open your terminal and run:
```bash
git clone https://github.com/Skanda2852b/Hand-written-Digital-recognition-using-Deep-Belief-networks.git
cd Hand-written-Digital-recognition-using-Deep-Belief-networks
```

### 2. Run the automated setup
We have provided an automated setup script that creates a virtual environment and cleanly installs all necessary dependencies (including the custom `deep-belief-network` package from GitHub).

**For Windows (PowerShell):**
```powershell
.\setup.ps1
```

*(Note: Depending on your internet speed, the setup may take a few minutes as it downloads large packages like TensorFlow).*

### 3. Activate the environment
Once the setup is complete, activate your newly created virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

## 🏃‍♂️ Running the Project

To train the model and generate the output visualizations, simply run:
```bash
python main.py
```

### What happens when you run it?
1. **Training:** If a saved model is not found, the script will automatically start the pre-training and fine-tuning phases. 
2. **Instant Loading:** If you have already trained the model previously, it will automatically find your `outputs/dbn_mnist_model.pkl` file, skip the long training process, and instantly generate the charts.
3. **Outputs:** All charts, graphs, and the saved model file will be safely placed in the `outputs/` directory.

## 📁 Project Structure
- `main.py` - The main entry point to run the pipeline.
- `src/config.py` - Hyperparameter configurations (edit this to change layers, epochs, learning rate).
- `src/model.py` - Logic for building, training, and saving the DBN.
- `src/data_loader.py` - Handles downloading and splitting the MNIST dataset.
- `outputs/` - Generated charts and the saved `.pkl` model file.
