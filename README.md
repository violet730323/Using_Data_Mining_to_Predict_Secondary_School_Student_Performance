# Predicting Secondary School Student Performance

---

## Overview

This project replicates and extends **Cortez & Silva (2008)** — *"Using Data Mining to Predict Secondary School Student Performance"* — using the same Portuguese secondary school dataset. The original paper's key limitation was its heavy reliance on midterm grades (G1, G2) as predictors. We deliberately excluded those grades to ask a more actionable question:

> **Can social, behavioral, and demographic variables alone predict how a student will perform — or how much they will improve — without any prior grade information?**

---

## Dataset

**Source:** [UCI Machine Learning Repository — Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)  
**Original paper:** Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. *EUROSIS.*

The dataset covers students from two public secondary schools in the Alentejo region of Portugal, with records for two subjects:

| Subject | Raw n | After cleaning |
|---|---|---|
| Mathematics | 357 | ~175 |
| Portuguese | 649 | ~343 |

**Features (30 variables):** Demographics (age, sex, address), family background (parental education & occupation, family size), lifestyle (alcohol consumption, going out, free time), academic support (study time, tutoring, internet access), and historical failures/absences. Grades G1 and G2 were **excluded** from all models.

---

## Research Questions

| Model | Target Variable | Task | Question |
|---|---|---|---|
| `tree_social` / `rf_social` | G3 (final grade, 0–20) | Regression | What score will this student ultimately achieve? |
| `tree_move` / `rf_move` | G3_progress = G3 − G2 | Regression / Classification | Will this student improve or decline from midterm to final? |
| `nn_model` | Pass/Fail (G3 ≥ 10) | Binary classification | Will this student pass — using behavior alone? |
| SVM models | Declining / Stable / Improving | 3-class classification | Can we classify grade trajectory? |
| PCA + tree | G3_progress (on PCs) | Regression on components | What compound behavioral dimensions predict movement? |

---

## Data Cleaning

All models share a common preprocessing pipeline that addresses limitations in the original paper:

- **Removed G3 = 0 students** (n = 39): These represent structural dropouts, not real academic decline. Including them distorts all models.
- **Capped absences at 30**: The raw data contains extreme outliers (up to 75 absences) that skew predictions.
- **Excluded G1 and G2**: Forces models to rely solely on behavioral and social features — the paper's baseline without grades had RMSE = 4.46; we aimed to beat that.
- **Engineered G3_progress = G3 − G2**: A new feature capturing grade trajectory from midterm to final.
- **Binary-encoded yes/no variables**: For cleaner tree splits.
- **Removed G3 before predicting G3_progress**: Prevents data leakage in movement models.

Final merged dataset: **n = 343** students (after filtering).

---

## Methods

### 1. Decision Tree (DT)
Two trees were built on separate subject datasets (Math and Portuguese):

- **`tree_social`**: Regresses G3 on social/behavioral variables. Cross-validation via `cv.tree()` selected optimal pruning (5 terminal nodes). RMSE = **2.878** (vs. paper baseline of 4.46).
- **`tree_move`**: Predicts G3_progress. Found that `famrel` (family relationship quality) is the strongest predictor of grade movement. Math tree splits on `failures`; Portuguese tree was a single node — social variables cannot predict movement in Portuguese.

Key finding: **`failures`** (past class failures) is the strongest single predictor of final grade across both subjects and both trees.

### 2. Random Forest (RF)
Built using the same two-model structure as DT, with 500 trees and cross-validated RMSE evaluation:

- **`rf_social`** (regression): Math RMSE = **2.86** · Portuguese RMSE = **2.155** — both far below the paper's 4.46 baseline (Portuguese improves by 52%).
- **`rf_move`** (3-class classification): Math Kappa = **−0.011** · Portuguese Kappa = **0.089** — both near zero, confirming grade movement direction is not predictable from social variables.

Critical insight: `failures` importance drops from **38.1%** in `rf_social` (Portuguese) to **4.9%** in `rf_move` — it predicts *where* a student lands, but not *which direction* they will move.

### 3. Support Vector Machine (SVM)
Classified grade trajectory (Declining / Stable / Improving) using both `e1071` and `kernlab` packages. Severe class imbalance caused initial models to predict only the majority class ("Stable"). Improvements included:

- Combined Walc + Dalc into a single alcohol index
- Added G2 as a feature
- Applied SMOTE to correct class imbalance
- 10-fold CV repeated 3 times

Best result: Accuracy = **55%** (Error Rate = 0.45), modest but meaningfully above the collapsed-class baseline.

### 4. Neural Network (NN)
Binary classification (Pass/Fail, G3 ≥ 10) using `nnet` with a single hidden layer:

- Architecture tuned via 5-fold CV over node size (5–20) and decay (0.001–0.5)
- Inverse-frequency class weighting (~3× penalty for missed Fails) to handle imbalance
- CV-optimal threshold selected to maximize Kappa

Results: Accuracy = **72.1%** · Sensitivity = **78.4%** (catches most at-risk students) · ROC-AUC ≈ 0.74 · Kappa = 0.296. The high sensitivity reflects a deliberate design trade-off: flag more students at risk rather than optimize raw accuracy.

### 5. Principal Component Analysis (PCA)
Applied to both subject datasets to explore the underlying structure of behavioral variables:

- 14 PCs required to explain ~80% of variance — student profiles are genuinely multidimensional; no single factor dominates.
- **PC1** (Academic Engagement): loaded by study time, failures, parental education, alcohol use.
- **PC2** (Social/Lifestyle): loaded by going out, weekend alcohol, parental education.
- PCA tree on Math (RMSE = **0.803**) found meaningful splits on PC10 and PC8; Portuguese PCA tree found only one marginal split — reinforcing the subject-specific resilience finding.

---

## Key Findings

| Finding | Detail |
|---|---|
| Social variables **can** predict final grade | rf_social beats paper's 4.46 baseline in both subjects without any midterm grades |
| Grade **movement is not predictable** | Kappa ≈ 0 across all models for G3_progress — short-term changes are driven by unmeasured factors |
| `failures` is a level indicator, not a trajectory indicator | 38.1% importance in rf_social drops to 4.9% in rf_move (Portuguese) |
| Math is **sensitive**, Portuguese is **resilient** | Math responds to compound behavioral pressure; Portuguese grade movement resists prediction from this dataset |
| School support is a **struggle proxy** | `schoolsup` predicts lower grades — it marks pre-existing difficulty, not intervention benefit |
| Family relationships drive improvement | `famrel ≥ 4.5` (excellent) predicts +0.46 grade improvement on average |
| Parental occupation interaction | `Mjob × Fjob` interaction found by DT — not surfaced in original paper |

---

## Quantitative Summary

| Model | Metric | Value | Paper Baseline |
|---|---|---|---|
| rf_social (Math) | CV RMSE | 2.86 | 4.46 |
| rf_social (Portuguese) | CV RMSE | 2.155 | 4.46 |
| tree_social (merged) | RMSE | 2.878 | 4.46 |
| rf_move (Math) | Kappa | −0.011 | — |
| rf_move (Portuguese) | Kappa | 0.089 | — |
| nn_model (Pass/Fail) | Sensitivity | 78.4% | — |
| SVM (best) | Accuracy | 55.0% | — |
| PCA tree (Math) | RMSE | 0.803 | — |

---

## Repository Structure

```
├── student-mat.csv              # Math dataset (Cortez et al.)
├── student-por.csv              # Portuguese dataset (Cortez et al.)
├── student-merge.R              # Original merge script from Cortez et al.
├── student.txt                  # Variable descriptions
├── Team13_FinalProject.Rmd      # Full R Markdown analysis (all models)
└── FinalPresentation_Team13.pptx  # Presentation slides
```

---

## How to Run

1. Open `Team13_FinalProject.Rmd` in RStudio.
2. Ensure the following R packages are installed:

```r
install.packages(c("randomForest", "tree", "e1071", "kernlab", 
                   "nnet", "caret", "DMwR2", "ggplot2", "dplyr"))
```

3. Place `student-mat.csv` and `student-por.csv` in the same directory as the `.Rmd` file.
4. Knit or run chunk-by-chunk.

---

## Limitations

- Dataset is limited to two schools in Portugal — findings may not generalize across different cultural or socioeconomic contexts.
- Even with improvements, Portuguese grade movement remains largely unpredictable from behavioral variables alone.
- RMSE of ~2.9 on a 0–20 scale still represents substantial unexplained variance; unmeasured factors (exam difficulty, teacher grading, personal events) likely dominate short-term grade fluctuations.

---

## Reference

Cortez, P., & Silva, A. M. G. (2008). Using data mining to predict secondary school student performance. In *Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008)* (pp. 5–12). Porto, Portugal: EUROSIS.
