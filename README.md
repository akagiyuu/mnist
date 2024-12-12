# Mnist using burn

## Using normal convolution
- Metrics

| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Loss     | 0.088    | 20       | 1.218    | 1        |
| Train | Accuracy | 67.485   | 1        | 97.422   | 20       |
| Valid | Loss     | 0.053    | 20       | 0.476    | 1        |
| Valid | Accuracy | 90.510   | 1        | 98.290   | 20       |
| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |

- Model size: 8mb

## Using depth-wise convolution and point-wise convolution
- Metrics

|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 40.452   | 1        | 94.288   | 20       |
| Train | Loss     | 0.197    | 20       | 2.075    | 1        |
| Valid | Accuracy | 76.370   | 1        | 96.440   | 20       |
| Valid | Loss     | 0.120    | 20       | 1.756    | 1        |

- Model size: 2.8mb
