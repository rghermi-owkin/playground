# Playground
My personal playground.

## Code organization

```
playground
├── conf                                            # Contains configuration files for experiments.
├── README.md
├── playground                                      # Main code.
│   ├── data                                        # Contains data-related code.
│   │   ├── loading                                 # Contains data loading functions.
│   │   │   ├── tcga.py
│   │   │   ├── __init__.py
│   │   ├── preprocessing                           # Contains pre-processing functions.
│   │   │   ├── preprocess_histo.py
│   │   │   ├── preprocess_exp.py
│   │   │   ├── preprocess_mut.py
│   │   │   ├── preprocess_cnv.py
│   │   │   ├── __init__.py
│   │   ├── datasets                                # Contains torch.utils.data.Dataset classes.
│   │   │   ├── __init__.py
│   ├── transforms                                  # Contains transformation functions.
│   │   ├── __init__.py
│   ├── models                                      # Contains models.
│   │   ├── mean_pool.py
│   │   ├── chowder.py
│   │   ├── deep_mil.py
│   │   ├── __init__.py
│   ├── losses                                      # Contains losses.
│   │   ├── bce_with_logits_loss.py
│   │   ├── cross_entropy_loss.py
│   │   ├── mse_loss.py
│   │   ├── cox_loss.py
│   │   ├── smooth_cindex_loss.py
│   │   ├── __init__.py
│   ├── metrics                                     # Contains metrics.
│   │   ├── classification_metrics.py
│   │   ├── regression_metrics.py
│   │   ├── survival_metrics.py
│   │   ├── __init__.py
│   ├── trainers                                    # Contains trainers.
│   │   ├── sklearn_trainer.py
│   │   ├── lifelines_trainer.py
│   │   ├── torch_trainer.py
│   │   ├── __init__.py
│   ├── utils                                       # Contains utilitary functions.
│   │   ├── loading.py
│   │   ├── logging.py
│   │   ├── plotting.py
│   │   ├── __init__.py
│   ├── constants.py
│   ├── __init__.py
├── tools                                           # Contains tools.
├── tests                                           # Contains tests.
```
