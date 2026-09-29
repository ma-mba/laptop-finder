# Laptop Finder

### Machine-Learning-Based Laptop Recommendation and Gaming Performance Prediction System

Laptop Finder is a machine-learning-based recommendation system that helps users select suitable laptops based on their **budget, tasks, RAM requirements, portability, and gaming needs**.

The system combines **requirement-based filtering, gaming FPS prediction, and Learning-to-Rank** to produce personalized laptop recommendations.

---

## Features

* Requirement-based laptop filtering
* Budget-based selection
* RAM and weight constraints
* Gaming performance prediction
* Laptop ranking using Learning-to-Rank
* Laptop comparison
* Recommendation explanations
* Custom PC compatibility checker
* Interactive Streamlit interface
* CSV result export

---

## System Workflow

```text
User Requirements
        ↓
Requirement Processing
        ↓
Candidate Laptop Filtering
        ↓
Gaming FPS Prediction
        ↓
Feature Engineering
        ↓
Learning-to-Rank Model
        ↓
Ranked Laptop Recommendations
```

---

## Machine Learning Models

### 1. FPS Prediction Model

A **LightGBM Regression model** is used to estimate gaming performance in terms of Frames Per Second (FPS).

The model uses hardware and game-related benchmark features to predict FPS.

**Evaluation metrics:**

* Mean Absolute Error (MAE)
* R²
* Percentage of predictions within ±10%
* Percentage of predictions within ±20%

The reported unseen-GPU validation results achieved:

* **MAE:** 9.506 FPS
* **R²:** 0.945
* **Within ±10%:** 66.707%
* **Within ±20%:** 89.518%

> FPS predictions are estimates and can vary depending on resolution, graphics settings, drivers, thermal conditions, and hardware configuration.

### 2. Laptop Ranking Model

A **LightGBM Learning-to-Rank model using LambdaRank** is used to rank eligible laptops.

The ranking model considers features such as:

* CPU performance
* GPU performance
* RAM
* Predicted FPS
* Price
* Value
* Weight
* Storage
* Requirement satisfaction

The ranking model was evaluated using:

* **NDCG@5**
* **Precision@5**

---

## Datasets

The project uses two main datasets:

### Laptop Specification and Price Dataset

Contains laptop product information such as:

* Brand and model
* CPU
* GPU
* RAM
* Storage
* Weight
* Display
* Price
* Other specifications

### Gaming FPS Benchmark Dataset

Contains benchmark information used to train the FPS prediction model, including hardware, games, and observed FPS.

### Dataset Setup

The datasets are not included in this repository because of their large size.

Place the required datasets in:

```text
data/
├── raw/
└── processed/
```

Refer to the project documentation for the required dataset names and preprocessing steps.

---

## Project Structure

```text
laptop-finder/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── fps_model.joblib
│   └── ranker.joblib
│
├── src/
│   ├── app.py
│   ├── core.py
│   ├── fps.py
│   └── pc_builder.py
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **LightGBM**
* **Joblib**
* **Matplotlib**
* **Streamlit**

---

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd laptop-finder
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the required datasets in the appropriate `data/` directories.

---

## Running the Application

Run the Streamlit application:

```bash
streamlit run src/app.py
```

The application provides:

* Top laptop recommendations
* Laptop comparison
* Gaming FPS estimates
* Custom PC compatibility checking
* Downloadable recommendation results

---

## Important Notes

* The system uses **actual laptop prices**; it does not predict laptop prices.
* The value score is a rule-based specification-to-price heuristic.
* Ranking relevance labels are generated using defined project rules rather than real user click or purchase data.
* FPS predictions are estimates rather than guaranteed real-world benchmark results.
* When a requested game is unavailable in the benchmark dataset, a mapped proxy game may be used for estimation.

---

## Future Scope

* Real-time laptop price and specification updates
* Larger and more diverse gaming benchmark datasets
* User feedback-based personalization
* Personalized recommendation models
* Improved natural-language requirement processing
* Advanced PC component recommendation
* Automatic model updating
* Web deployment

---

## Project Objective

The main objective of Laptop Finder is to combine **machine learning with requirement-based decision making** to make laptop selection easier by translating technical specifications and gaming benchmarks into understandable recommendations.

---
