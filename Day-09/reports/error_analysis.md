# Day 09: Error Analysis and Explainability

## 1. Overall Model Performance

                       Model  Accuracy  Precision  Recall  F1 Score
         Logistic Regression     0.804      0.793   0.667     0.724
Balanced Logistic Regression     0.804      0.730   0.783     0.755
               Decision Tree     0.765      0.755   0.580     0.656

## 2. Error Analysis

- Total errors across all models: 112
- False Positives: 45
- False Negatives: 67

## 3. Group-wise Analysis

Performance was evaluated across Sex, Passenger Class, and Age Group.

                       Model    Group Category  Samples  Accuracy  Precision  Recall  F1 Score
         Logistic Regression      Sex     male      118     0.805      0.600   0.125     0.207
         Logistic Regression      Sex   female       61     0.803      0.811   0.956     0.878
         Logistic Regression   Pclass        3      100     0.800      0.600   0.500     0.545
         Logistic Regression   Pclass        2       34     0.882      0.944   0.850     0.895
         Logistic Regression   Pclass        1       45     0.756      0.850   0.680     0.756
         Logistic Regression AgeGroup    19-35       74     0.797      0.826   0.633     0.717
         Logistic Regression AgeGroup    36-60       28     0.893      1.000   0.700     0.824
         Logistic Regression AgeGroup  Unknown       40     0.775      0.615   0.667     0.640
         Logistic Regression AgeGroup     0-18       31     0.774      0.786   0.733     0.759
         Logistic Regression AgeGroup      60+        6     0.833      1.000   0.500     0.667
Balanced Logistic Regression      Sex     male      118     0.814      0.562   0.375     0.450
Balanced Logistic Regression      Sex   female       61     0.787      0.776   1.000     0.874
Balanced Logistic Regression   Pclass        3      100     0.790      0.560   0.583     0.571
Balanced Logistic Regression   Pclass        2       34     0.882      0.944   0.850     0.895
Balanced Logistic Regression   Pclass        1       45     0.778      0.742   0.920     0.821
Balanced Logistic Regression AgeGroup    19-35       74     0.824      0.758   0.833     0.794
Balanced Logistic Regression AgeGroup    36-60       28     0.857      0.800   0.800     0.800
Balanced Logistic Regression AgeGroup  Unknown       40     0.775      0.600   0.750     0.667
Balanced Logistic Regression AgeGroup     0-18       31     0.742      0.733   0.733     0.733
Balanced Logistic Regression AgeGroup      60+        6     0.833      1.000   0.500     0.667
               Decision Tree      Sex     male      118     0.763      0.300   0.125     0.176
               Decision Tree      Sex   female       61     0.770      0.860   0.822     0.841
               Decision Tree   Pclass        3      100     0.780      0.583   0.292     0.389
               Decision Tree   Pclass        2       34     0.882      0.944   0.850     0.895
               Decision Tree   Pclass        1       45     0.644      0.696   0.640     0.667
               Decision Tree AgeGroup    19-35       74     0.784      0.850   0.567     0.680
               Decision Tree AgeGroup    36-60       28     0.750      0.714   0.500     0.588
               Decision Tree AgeGroup  Unknown       40     0.750      0.583   0.583     0.583
               Decision Tree AgeGroup     0-18       31     0.806      0.909   0.667     0.769
               Decision Tree AgeGroup      60+        6     0.500      0.333   0.500     0.400

## 4. Permutation Feature Importance

Feature  Importance_Mean  Importance_Std
    Sex           0.2840          0.0633
 Pclass           0.0969          0.0286
  SibSp           0.0263          0.0095
  Parch           0.0153          0.0031
    Age           0.0119          0.0211

Permutation importance measures how much model performance changes when a feature is shuffled. It indicates model dependence, not causation.

## 5. Two Proposed Improvements

1. Investigate groups with lower F1 scores and collect or review more representative training examples for those groups.
2. Experiment with decision thresholds and compare precision and recall before selecting a suitable threshold.

## 6. Important Caveat

Feature importance shows association with model predictions, not proof of causation. Group metrics may also be less reliable for groups with very few samples.
