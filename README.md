# ML Algorithms from Scratch with NumPy 🧠📊

This project implements several popular machine learning algorithms **from scratch using NumPy**, without relying on external ML libraries like scikit-learn. The algorithms are tested on a **heart disease dataset** and evaluated based on their **accuracy, time complexity, and space complexity**. 

## 📋 Features

- Implementation of **six key machine learning algorithms** from scratch:
  - Random Forest 🌲
  - Decision Tree 🌳
  - K-Nearest Neighbors (KNN) 👥
  - Logistic Regression 📈
  - Support Vector Machine (SVM) 🔗
  - Naive Bayes 🤔
- Comparison of performance metrics:
  - **Accuracy**: How well the model predicts outcomes.
  - **Time Complexity**: How long each algorithm takes to execute.
  - **Space Complexity**: Memory usage for each algorithm.
- Visualizations for easy comparison:
  - Accuracy 📊
  - Time Complexity ⏱️
  - Space Complexity 💾

## 📁 Dataset

The dataset used is a **Heart Disease Dataset**, which contains features relevant to predicting heart disease.

---

## 🛠️ Algorithms Implemented

Each algorithm is implemented from scratch, without external libraries like `scikit-learn`. Here's a quick overview of the algorithms:

1. **Random Forest**:
   - Combines multiple decision trees to improve accuracy.
   - Hyperparameter: Number of trees (`n_trees`).

2. **Decision Tree**:
   - Splits the data based on features to make predictions.
   - Hyperparameter: Maximum tree depth (`max_depth`).

3. **K-Nearest Neighbors (KNN)**:
   - Classifies data based on the majority vote of `k` nearest neighbors.
   - Hyperparameter: Number of neighbors (`k`).

4. **Logistic Regression**:
   - A statistical model for binary classification.
   - Hyperparameter: Learning rate (`lr`).

5. **Support Vector Machine (SVM)**:
   - Classifies data by finding the optimal hyperplane that separates classes.

6. **Naive Bayes**:
   - Based on Bayes' theorem with strong independence assumptions.

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ml-algorithms-from-scratch.git
   ```

2. Navigate to the project directory:
   ```bash
   cd ml-algorithms-from-scratch
   ```

3. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```

4. Run the Python script:
   ```bash
   python main.py
   ```

5. Visualize the performance of the algorithms with the generated plots:
   - Accuracy 📊
   - Time Complexity ⏱️
   - Space Complexity 💾

## 📌 Folder Structure

```
📁 ml-algorithms-from-scratch
├── 📄 main.py            # Main script to run the models
├── 📄 RandomForest.py    # Random Forest implementation
├── 📄 DecisionTree.py    # Decision Tree implementation
├── 📄 KNN.py             # K-Nearest Neighbors implementation
├── 📄 LogisticRegression.py # Logistic Regression implementation
├── 📄 SVM.py             # Support Vector Machine implementation
├── 📄 NaiveBayes.py      # Naive Bayes implementation
└── 📄 heart.csv          # Heart Disease Dataset
```
### 🌟 **Show Your Support**
If you find this project helpful, give it a ⭐ on GitHub and share it with others! 😊
