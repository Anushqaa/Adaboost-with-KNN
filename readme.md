# KNN with Adaboosting

This repository contains Python code for a K-nearest neighbors (KNN) classifier implemented with Adaboosting. K-nearest neighbors is a simple yet effective machine learning algorithm for classification tasks. Adaboosting is used to enhance the performance of the KNN classifier.

## Getting Started

These instructions will help you understand the code and run it on your local machine for testing purposes.

### Prerequisites

To run the code, you need:

- Python 3.x
- Numpy
- Sklearn

You can install Numpy and Sklearn using pip:

```bash
pip install numpy
pip install scikit-learn
```

### Code Overview

The repository consists of the following components:

- `knn_adaboost.py`: Python code for the K-nearest neighbors classifier with Adaboosting.
- `AdaKNN.ipynb` : Jupyter notebook for examples.

### Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Anushqaa/Adaboost-with-KNN.git
```

2. Run the Python script for the desired dataset and import:

```bash
python adaknn.py
```
```python
import adaknn
```

3. Access and use its functions and classes:

Examples for usage in [file]('Adaboost with KNN\AdaKNN.ipynb')

The script will load the dataset, train the KNN classifier with Adaboosting, and print the accuracy and recall scores.

## Datasets

### Iris Dataset

The Iris dataset is a well-known dataset in the machine learning community, consisting of three classes of iris plants, each with 50 instances. The dataset includes four numeric attributes and is commonly used for classification tasks.

### Breast Cancer Dataset

The Breast Cancer dataset contains features extracted from digitized images of breast cancer biopsies. It includes 30 numeric attributes and two classes: malignant and benign. This dataset is used for binary classification.

### Wine Dataset

The Wine dataset consists of 13 numeric attributes related to chemical analysis of wines, along with three classes. It is used for multi-class classification tasks.

## Acknowledgments

- The Iris, Breast Cancer, and Wine datasets used in this repository are commonly used datasets available in the scikit-learn library.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to use and modify the code for your own projects or educational purposes. If you find this repository helpful, please consider giving it a star!