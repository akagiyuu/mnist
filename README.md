# Mnist using burn

## Using normal convolution
- Metrics

| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Loss     | 0.088    | 20       | 1.218    | 1        |
| Train | Accuracy | 67.485   | 1        | 97.422   | 20       |
| Valid | Loss     | 0.053    | 20       | 0.476    | 1        |
| Valid | Accuracy | 90.510   | 1        | 98.290   | 20       |

- Model size: 8mb

## Using depth-wise convolution and point-wise convolution
- Metrics

| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Loss     | 0.200    | 20       | 2.090    | 1        |
| Train | Accuracy | 41.332   | 1        | 94.250   | 20       |
| Valid | Loss     | 0.122    | 20       | 1.786    | 1        |
| Valid | Accuracy | 80.680   | 1        | 96.340   | 20       |

- Model size: 2mb
