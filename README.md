## 📊 Drug Classification – Evaluation of ML Models & Scaling Techniques

This project explores various machine learning models and data scaling techniques to predict drug types based on patient attributes. It includes **EDA**, **data preprocessing**, **model training**, and **comparative evaluation** using metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

---

### 🔗 Dataset

**Source**: [Kaggle – Drug Classification Dataset](https://www.kaggle.com/datasets/prathamtripathi/drug-classification/data)
The dataset contains medical attributes of patients and the drug prescribed for them.

---

### 🧠 Features Used

| Feature       | Description                                    |
| ------------- | ---------------------------------------------- |
| `Age`         | Patient age (converted to category)            |
| `Sex`         | Gender (`M`/`F`)                               |
| `BP`          | Blood Pressure level (`LOW`, `NORMAL`, `HIGH`) |
| `Cholesterol` | Cholesterol level (`NORMAL`, `HIGH`)           |
| `Na_to_K`     | Sodium to Potassium ratio (float)              |
| `Drug`        | **Target** drug class (`DrugY`, `drugA`, etc.) |

---

### 🔍 Exploratory Data Analysis (EDA)

* Gender and Drug distribution
* BP and Cholesterol impact
* Value counts and relationships
* Visualizations using **Seaborn** and **Matplotlib**

---

### ⚙️ Data Preprocessing

* Encoding categorical features using `LabelEncoder`
* Age binning into 4 categories (Young, Adult, Middle-aged, Old)
* Scaling features using:

  * `MinMaxScaler`
  * `StandardScaler`
  * `RobustScaler`
  * `MaxAbsScaler`

---

### 🤖 Models Used

| Model                    | Description                   |
| ------------------------ | ----------------------------- |
| `Logistic Regression`    | Linear baseline model         |
| `K-Nearest Neighbors`    | Distance-based classification |
| `Support Vector Machine` | Hyperplane-based separation   |
| `Decision Tree`          | Tree-based rule learning      |

---

### 📈 Evaluation Metrics

Each model was evaluated on:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* `classification_report` for top 3 models

---

### 🏆 Top Performing Models (by F1-Score)

| Model | Scaler   | Accuracy | Precision | Recall | F1-Score |
| ----- | -------- | -------- | --------- | ------ | -------- |
| SVM   | standard | 0.975    | 0.976     | 0.975  | 0.974    |
| SVM   | robust   | 0.975    | 0.976     | 0.975  | 0.974    |
| KNN   | robust   | 0.925    | 0.935     | 0.925  | 0.911    |

> 🔎 See full table in the notebook.

---

### 📁 Files

| File                                                   | Description                   |
| ------------------------------------------------------ | ----------------------------- |
| `drug-classification-multiple-ml-models-scalers.ipynb` | Full notebook with all steps  |
| `data/drug200.csv`                                     | The dataset used for training |

---

### ✅ How to Run

1. Clone the repo
   `git clone https://github.com/YOUR_USERNAME/drug-classification-ml-evaluation.git`
2. Open the notebook in Jupyter or Kaggle
3. Run all cells

---

### 🙌 Author

Made with Amr Belal for learning & experimentation.
Feel free to fork, star ⭐, or suggest improvements!

