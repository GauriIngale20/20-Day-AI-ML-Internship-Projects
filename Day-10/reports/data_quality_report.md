# Titanic Dataset - Data Quality Report

Dataset Shape: (891, 15)

## Missing Values

|             |   Missing_Count |   Missing_Percentage |
|:------------|----------------:|---------------------:|
| survived    |               0 |             0        |
| pclass      |               0 |             0        |
| sex         |               0 |             0        |
| age         |             177 |            19.8653   |
| sibsp       |               0 |             0        |
| parch       |               0 |             0        |
| fare        |               0 |             0        |
| embarked    |               2 |             0.224467 |
| class       |               0 |             0        |
| who         |               0 |             0        |
| adult_male  |               0 |             0        |
| deck        |             688 |            77.2166   |
| embark_town |               2 |             0.224467 |
| alive       |               0 |             0        |
| alone       |               0 |             0        |

## Duplicate Records

107

## Data Types

|             | Data Type   |
|:------------|:------------|
| survived    | int64       |
| pclass      | int64       |
| sex         | str         |
| age         | float64     |
| sibsp       | int64       |
| parch       | int64       |
| fare        | float64     |
| embarked    | str         |
| class       | str         |
| who         | str         |
| adult_male  | bool        |
| deck        | str         |
| embark_town | str         |
| alive       | str         |
| alone       | bool        |