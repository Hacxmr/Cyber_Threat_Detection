# Cyber Threat Detection using Deep Learning

This repository contains a Colaboratory notebook (`Avishkar.ipynb`) that demonstrates the use of deep learning models for cyber threat detection using the UNSW-NB15 dataset.

## Notebook Overview

The notebook performs the following key tasks:

1.  **Data Loading and Preprocessing:**
    *   Loads the UNSW-NB15 dataset, which is split into multiple CSV files.
    *   Extracts feature names from a separate CSV file to correctly label the dataset columns.
    *   Handles duplicate column names by appending suffixes.
    *   Encodes categorical features using `LabelEncoder`.
    *   Normalizes numerical features using `StandardScaler`.
    *   Splits the data into training and testing sets.
    *   Reshapes the data to be suitable for CNN and LSTM models.

2.  **Deep Learning Model Implementation:**
    *   Defines and implements four different deep learning architectures:
        *   **DNN (Deep Neural Network):** A basic feed-forward neural network.
        *   **CNN (Convolutional Neural Network):** Utilizes `Conv1D` and `MaxPooling1D` layers for feature extraction.
        *   **LSTM (Long Short-Term Memory):** A recurrent neural network model suited for sequential data.
        *   **BiLSTM (Bidirectional Long Short-Term Memory):** An LSTM model that processes sequences in both forward and backward directions.
    *   Compiles each model with the Adam optimizer, categorical cross-entropy loss, and accuracy as a metric.

3.  **Model Training and Evaluation:**
    *   Trains each of the defined models on the preprocessed data.
    *   Evaluates the trained models on the test set using accuracy.
    *   Presents a summary of the performance metrics (Accuracy, Precision, Recall, F1-Score, Validation Accuracy, Training Time) in a table.

4.  **KDD Cup 99 Dataset Analysis (Exploratory):**
    *   Includes exploratory cells to load and inspect the KDD Cup 99 dataset, demonstrating basic data loading and type inspection.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Open the notebook:** Open `Avishkar.ipynb` in Google Colaboratory or a compatible Jupyter environment.
3.  **Install Dependencies:** The notebook uses `tensorflow`, `pandas`, `numpy`, and `scikit-learn`. These are typically pre-installed in Colab, but if running locally, you might need to install them:
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib seaborn tabulate
    ```
4.  **Dataset:** Ensure the UNSW-NB15 dataset files (`UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`, `UNSW-NB15_3.csv`, `UNSW-NB15_4.csv`, and `NUSW-NB15_features.csv`) are placed in the `/content/UNSW_NB15/` directory (or adjust the `folder` variable in the notebook accordingly). Also ensure the KDD dataset file (`kdd_dataset.csv`) is available at the specified path.

## Model Performance Summary

The notebook provides a table summarizing the performance of the trained models after 5 epochs:

| Metric                    | MLP     | CNN    | LSTM    |
| :------------------------ | :------ | :----- | :------ |
| Accuracy                  | 100%    | 100%   | 100%    |
| Precision (Attack)        | 0.93    | 0.92   | 0.89    |
| Recall (Attack)           | 0.94    | 0.95   | 0.94    |
| F1-Score (Attack)         | 0.94    | 0.94   | 0.92    |
| Val Accuracy (Last Epoch) | 99.73%  | 99.74% | 99.64%  |
| Training Time (5 Epochs)  | ~50 sec | ~7 min | ~22 min |

*Note: The exact performance metrics may vary slightly due to the stochastic nature of neural network training.*

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
