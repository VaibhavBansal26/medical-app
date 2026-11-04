# EDA summary

Rows: 920  |  prevalence: 55.3%

## Missingness
|          |   missing |
|:---------|----------:|
| ca       |     0.664 |
| thal     |     0.528 |
| slope    |     0.336 |
| chol     |     0.22  |
| fbs      |     0.098 |
| oldpeak  |     0.067 |
| trestbps |     0.065 |
| thalach  |     0.06  |
| exang    |     0.06  |
| restecg  |     0.002 |
| age      |     0     |
| sex      |     0     |
| cp       |     0     |

## Prevalence by site
| site        |   size |   mean |
|:------------|-------:|-------:|
| cleveland   |    303 |  0.459 |
| hungarian   |    294 |  0.361 |
| switzerland |    123 |  0.935 |
| va          |    200 |  0.745 |

## Numeric features vs target
| feature   |   median_healthy |   median_disease |   AUC_univariate |       p |
|:----------|-----------------:|-----------------:|-----------------:|--------:|
| age       |               51 |            57    |            0.331 | 1.3e-18 |
| trestbps  |              130 |           130    |            0.437 | 0.0013  |
| chol      |              233 |           248    |            0.427 | 0.0007  |
| thalach   |              151 |           128    |            0.731 | 9e-32   |
| oldpeak   |                0 |             1.05 |            0.277 | 8.5e-32 |

## Categorical features vs target
| feature   |   cramers_v |       p |
|:----------|------------:|--------:|
| sex       |       0.305 | 2.5e-20 |
| cp        |       0.54  | 7e-58   |
| fbs       |       0.139 | 6e-05   |
| restecg   |       0.113 | 0.0029  |
| exang     |       0.461 | 6.4e-42 |
| slope     |       0.381 | 5.1e-20 |
