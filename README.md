# Cardiovascular Disease Zero-Shot Classification with Decider

This project evaluates the zero-shot decision-making capability of [`Mapika/decider-0.8b`](https://huggingface.co/Mapika/decider-0.8b) for identifying cardiovascular disease from patient clinical profiles in the MACCR dataset.

The model is prompted with a structured patient profile and asked:

> Does this patient case describe a cardiovascular disease?

The model returns a probability for the `Yes` response, which is evaluated against the `Cardiovascular Diseases` label in the dataset.

## Dataset

The analysis uses the **MACCRs** dataset (https://doi.org/10.6084/m9.figshare.c.4220324) stored as:

```text
../data/MACCRs.tsv
```

The dataset contains 3,100 patient cases.

The target variable is:

```text
Cardiovascular Diseases
```

The class distribution is:

| Class | Cases | Proportion |
| ----- | ----: | ---------: |
| 0     | 2,133 |      68.8% |
| 1     |   967 |      31.2% |

The following ten clinical fields are used to construct the patient profiles:

* Age
* Gender
* Life Style
* Family History
* Social History
* Medical/Surgical History
* Signs and Symptoms
* Comorbidities
* Diagnostic Techniques and Procedures
* Pathology

Other disease-category fields and metadata are not included in the patient profile to avoid directly providing the target information to the model.

Missing values are replaced with:

```text
Data not available
```

## Model

The project uses:

```text
Mapika/decider-0.8b
```

The model is loaded using the `decider-ai` package:

```python
from decider.infer import Decider

d = Decider("Mapika/decider-0.8b")
```

No model training or fine-tuning is performed. The model is used in a zero-shot setting.

## Patient Profile Construction

The ten clinical fields are combined into a structured text representation for each patient.

For example:

```text
Age: 65
Gender: male
Life Style: former smoker
Family History: ...
Social History: ...
Medical/Surgical History: ...
Signs and Symptoms: ...
Comorbidities: ...
Diagnostic Techniques and Procedures: ...
Pathology: ...
```

This profile is supplied to the model as the `state`.

## Decision Question

The same decision question is applied to every patient:

```text
Does this patient case describe a cardiovascular disease?
```

The question is defined using the `noul` decision type:

```python
question = {
    "cardiovascular": {
        "question": "Does this patient case describe a cardiovascular disease?",
        "type": "noul"
    }
}
```

The returned `noul` value is treated as the model's probability for the `Yes` response and stored as `y_prob`.

## Analysis Workflow

The notebook follows these steps:

1. Load the MACCR dataset.
2. Replace missing values.
3. Select the clinical variables used as model input.
4. Define the cardiovascular disease target.
5. Construct a text-based patient profile for each case.
6. Load `decider-0.8b`.
7. Run a small 10-case sanity check using five positive and five negative cases.
8. Run inference on all 3,100 cases.
9. Save the resulting probabilities.
10. Evaluate the probability outputs using PR-AUC.
11. Examine precision and recall across probability thresholds.
12. Evaluate binary predictions at an exploratory threshold of 0.5.
13. Inspect high-confidence false-positive and false-negative cases.

## Sanity Check

Before running inference on the full dataset, five positive and five negative cases are sampled:

```python
positives = X[y['Cardiovascular Diseases'] == 1]
negatives = X[y['Cardiovascular Diseases'] == 0]
```

Five cases from each class are then evaluated manually to verify that the model produces reasonable probability outputs and that the inference pipeline is functioning correctly.

This small sample is used only as a sanity check and is not treated as a performance evaluation.

## Full-Dataset Inference

The model is applied to all 3,100 patient profiles using `tqdm` to monitor progress.

For each case, the following information is retained:

* `index`: original dataset index
* `y_true`: observed cardiovascular disease label
* `y_prob`: model probability for the `Yes` response

The results are saved to:

```text
../output/mccr_raw_probabilities.tsv
```

## Evaluation

### PR-AUC

The primary threshold-independent evaluation metric is **Precision-Recall Area Under the Curve (PR-AUC)**.

```python
pr_auc = average_precision_score(
    results_df["y_true"],
    results_df["y_prob"]
)
```

For this analysis:

```text
PR-AUC = 0.8896
```

PR-AUC is calculated directly from the model probabilities and does not require selecting a classification threshold.

### Exploratory Threshold Analysis

A precision-recall curve is generated to examine how precision and recall change across probability thresholds.

An exploratory threshold of **0.5** is also used to convert the model probabilities into binary predictions:

```python
y_pred = (results_df["y_prob"] >= 0.5).astype(int)
```

At this threshold, the observed performance is:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8710 |
| Precision | 0.7570 |
| Recall    | 0.8635 |
| F1        | 0.8068 |

The corresponding confusion matrix is:

```text
[[1865, 268],
 [ 132, 835]]
```

where the rows represent the observed class and the columns represent the predicted class.

Thus:

* True negatives: 1,865
* False positives: 268
* False negatives: 132
* True positives: 835

The 0.5 threshold is treated as an **exploratory threshold**, rather than as a threshold optimized using the dataset.

## Error Analysis

False positives are cases where:

```text
y_true = 0
y_prob >= 0.5
```

False negatives are cases where:

```text
y_true = 1
y_prob < 0.5
```

The analysis identifies the highest-confidence false positives and false negatives and retrieves their original patient profiles for qualitative inspection.

This is intended to examine the types of clinical profiles associated with incorrect predictions and to identify potential inconsistencies or ambiguities in the dataset labels.

## Important Considerations

This experiment evaluates a pretrained decision model in a **zero-shot setting**. The model is not trained or fine-tuned using the MACCR data.

Therefore, the analysis does not involve a conventional training, validation, and test workflow. The main evaluation focuses on the model's probability outputs across the full dataset.

The reported threshold-based metrics depend on the selected threshold. PR-AUC provides a threshold-independent assessment of the ranking of positive cases.

The dataset labels are treated as the reference labels for evaluation. However, qualitative inspection of errors is important because some clinical profiles may contain explicit cardiovascular manifestations even when their corresponding dataset label is negative.

## Requirements

The main Python packages used are:

```text
pandas
numpy
decider-ai
tqdm
scikit-learn
```

They can be installed with:

```bash
uv pip install pandas numpy decider-ai tqdm scikit-learn
```

## Output

The main generated output is:

```text
../output/mccr_raw_probabilities.tsv
```

with the following columns:

```text
index
y_true
y_prob
```

This file contains the raw model probabilities and can be reused for additional threshold analysis or visualization without rerunning model inference.

==================

**DISCLAIMER**: I started this project with the intent of exploring system one models to check how can they be integrated within decision workflows. By no means the codes in their current state are complete. I plan to include more models and perform a proper evaluation of the models.  
