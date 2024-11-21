f1 per class

|        Modell       |   n_paramter   | n_layers | hidden_size | intermediate_size | f1 train | f1 eval | lr (best pick) |
|---------------------|----------------|----------|-------------|-------------------|----------|---------|----------------|
| Activated Attention |  34,352,258    | 2        | 628         | /                 | 0.736    | 0.7025  | 1e-4           |
| Deberta base        | 139,215,390    | 12       | 768         | 3072              | 0.768    | 0.680   | 4e-5           |
