# AutoML Engine — Machine Learning Core for AI Applications

[![CI](https://github.com/satter0827/automl-engine/actions/workflows/ci.yml/badge.svg)](
https://github.com/satter0827/automl-engine/actions/workflows/ci.yml
)

[日本語版はこちら](README.ja.md)

---

## Overview

This project is a Python-based AutoML engine designed for integration into AI applications.
It supports classification, regression, clustering, and anomaly detection, and allows programmatic control over training, evaluation, and explainability.
The engine is provided as an importable package and includes LLM-based interpretation of training results. UI and UX are out of scope.

This library is primarily intended for engineers who embed machine learning capabilities into production or business systems.

---

## Design Principles

* Do not impose fixed I/O or persistence mechanisms
* Separate responsibilities by task
* Assume orchestration and control from higher-level layers

---

## Key Features

This project provides an end-to-end AutoML framework covering classification, regression, clustering, and anomaly detection, with reproducible training control, model evaluation and optimization, explainability, and LLM-based result summarization.

* **AutoML Execution**
  Automatically executes end-to-end workflows—from model training to evaluation—for classification, regression, clustering, and anomaly detection tasks.

* **Explicit Control of Training and Model Selection**
  Algorithms, preprocessing steps, random seeds, and other parameters can be explicitly specified in code to ensure reproducibility.

* **Metric Computation and Model Comparison**
  Computes task-appropriate evaluation metrics and supports comparison across multiple models to facilitate best-model selection.

* **Hyperparameter Optimization**
  Supports multiple search strategies, such as Grid Search and Optuna, enabling flexible and efficient hyperparameter optimization.

* **Explainability**
  Uses SHAP and LIME to compute feature contributions and their impact on model predictions.

* **Structured Output of Training Results**
  Returns models, evaluation metrics, feature importances, and related artifacts in a structured, reusable format.

* **LLM-Based Result Summarization**
  Summarizes training results and evaluation outcomes in natural language using large language models.

---

## Scope and Non-Goals

This project intentionally focuses on the core responsibilities of an AutoML engine.
The following areas are explicitly out of scope by design:

* **User Interface Layer**
  No UI or web dashboards are provided. The engine is intended to be embedded into applications or services via code.

* **Production-Grade MLOps Infrastructure**
  CI/CD pipelines, model monitoring, and online inference platforms are expected to be handled externally based on deployment requirements.

* **Data and Artifact Persistence**
  Storage formats and destinations for datasets, models, and evaluation results are not prescribed and are left to the integrating system.

* **Distributed or Large-Scale Computing Optimization**
  The engine is optimized for single-machine execution. Extensions for distributed or cluster environments are left to the user.

* **Domain-Specific Preprocessing and Feature Engineering**
  To maintain generality, domain-dependent preprocessing and feature engineering are expected to be implemented by the user.

* **Automated Business Decision-Making**
  The engine supports model construction and decision support. Final business decisions are assumed to be made by users.

---

## Differentiation from Other Libraries

This engine is designed as an **AutoML core intended to be embedded into other libraries or applications**.

* **Focused on the AutoML Core**
  Provides core capabilities such as preprocessing, training, evaluation, optimization, and explainability, while assuming higher-layer implementations.

* **Controllable and Transparent AutoML**
  Algorithms, metrics, and search strategies are explicitly configurable to ensure transparency and reproducibility.

* **API Designed for Integration**
  Built as a library to be called from other systems, rather than as a standalone application.

* **Task-Specific Engine Architecture**
  Provides dedicated Engine implementations tailored to the characteristics of each task.

* **Explainability and Summarization as First-Class Features**
  Feature attribution and natural-language summarization are provided as standard capabilities.

---

## Intended Use Cases
This project is designed for use cases ranging from integration into business applications to PoC and analytics platforms, result explanation and sharing, and validation of machine learning pipeline design.

* **Embedding Machine Learning into Business Applications**
  Integrating AutoML capabilities as a library within existing business systems or AI applications.

* **AutoML Execution Engine for Analytics Platforms and PoCs**
  Rapid experimentation and evaluation for model selection and validation.

* **Systems for Explaining and Sharing Training Results**
  Using LLM-based summaries to communicate model behavior and evaluation results to non-technical stakeholders.

* **Learning and Validation of ML Pipeline Design**
  Explicitly designing and validating preprocessing, training, evaluation, and explainability pipelines.

---

## Supported Tasks

* Classification
* Regression
* Clustering
* Anomaly Detection

---

## Architecture Overview

* Task-specific Engine layer
* Pipeline-based preprocessing, training, and evaluation
* Pluggable hyperparameter search strategies
* Independent modules for explainability and result summarization

---

## Installation

```bash
pip install automl-engine
```

---

## Quick Start

Dedicated Engine classes are provided for each task, offering a unified API for training, inference, and explanation.

Note: This is a minimal example. Preprocessing, evaluation metrics, and search strategies can be configured as needed.

```python
import automl_engine as ae

engine = ae.ClassificationEngine()

engine.fit(
    X=train_data,
    y=target,
)

predictions = engine.predict(test_data)
probabilities = engine.predict_proba(test_data)
explanations = engine.explain(test_data)
summary = engine.summarize()
```

```python
import automl_engine as ae

engine = ae.RegressionEngine()

engine.fit(
    X=train_data,
    y=target,
)

predictions = engine.predict(test_data)
explanations = engine.explain(test_data)
summary = engine.summarize()
```

---

## LLM-Based Result Interpretation

The engine can generate natural-language summaries using LLMs based on training results, evaluation metrics, and important features.

---

## Roadmap

* Batch execution support via CLI
* Refined interfaces for integration into API services
* Extension points for persistence of training results

---

## License

MIT License

---