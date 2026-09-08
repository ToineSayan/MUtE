#!/usr/bin/python
import torch
import random
import numpy as np

from erasers import *
from configs import *
from data.utils import load_data
from eval.erasure import nonlinear_probing
from eval.utility import utility_MSE
from eval.fairness import demographic_parity, TPR_RMS, TPR_corr_coef, train_clfs

import sys
import time


# config_path = "configs/biasbios_MUtE.yml"
config_path = "configs/DIAL_MUtE.yml"


# Load the configuration
# ------------------------------------------------------------------
config = load_config(config_path=config_path)
print(config)

seed = config.get('seed', None)
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Load the dataset
# ------------------------------------------------------------------
x, z, y, x_val, z_val, y_val, x_test, z_test, y_test, z_values, y_values = load_data(config['dataset'])

# print the shape of the data
print("Dataset information:")
print("-" * 40)
print(f"{'Name':<20}: {config['dataset']}")
print(f"{'Train':<20}: {x.shape[0]:<8} samples")
print(f"{'Validation':<20}: {x_val.shape[0] if x_val is not None else '--':<8} samples")
print(f"{'Test':<20}: {x_test.shape[0]:<8} samples")
print(f"{'Input dimension d':<20}: {x.shape[1]:<8}")
print(f"{'Concept Z':<20}: {z_values} ({len(z_values)} unique values)")
majority_class = np.bincount(z_test).argmax()
data_proportion_majority_class = np.mean(z_test == majority_class)
print(f"{'Majority class':<20}: {z_values[majority_class]} ({data_proportion_majority_class*100:.2f}% of the test data)")
print(f"{'Task labels Y':<20}: {y_values} ({len(y_values)} unique values)")
majority_class_y = np.bincount(y_test).argmax() if y_test.size > 0 else None
if majority_class_y is not None:
    data_proportion_majority_class_y = np.mean(y_test == majority_class_y) if y_test is not None and y_test.size > 0 else None
    print(f"{'Majority class Y':<20}: {y_values[majority_class_y]} ({data_proportion_majority_class_y*100:.2f}% of the test data)")




####################################################################
# Training of the erasure model
####################################################################

# Load the erasure model
# ------------------------------------------------------------------
print("\nModel params:")
print("-" * 40)

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
model_name = config["erasure_model"]["name"]
model_type = str_to_class(model_name)

erasure_model = model_type(**config["erasure_model_init_params"])


print(f"{'Erasure model':<20}: {model_name}")
for k,v in config["erasure_model_init_params"].items():
    print(f"{k:<20}: {v}")


# Fit the erasure model and transform the training data
# ------------------------------------------------------------------
print("\nModel fitting and data transformation:")
print("-" * 40)


# measure the time taken to fit the erasure model and transform the data
erasure_model.optimal_erasure = False
start_time = time.time()
x_orig =erasure_model.fit_transform(x,z)
fit_time = time.time() - start_time
print(f"Time taken to fit the erasure model: {fit_time:.2f} seconds")

# measure the time taken to transform the data using the fitted erasure model
start_time = time.time()
x_test_orig = erasure_model.transform(x_test, z_test)
end_time = time.time()
print(f"Time taken to transform the test data: {end_time - start_time:.2f} seconds")



# Surrogate Model


from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

pca = PCA(whiten=True)
x_ = pca.fit_transform(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_ = torch.from_numpy(x_).float().to(device)
x_orig = torch.from_numpy(x_orig).float().to(device)
surrogate_model = MLP(input_size=x_.shape[1], output_size=x_orig.shape[1], hidden_size=2*x_.shape[1], num_layers=3, activation='tanh', layer_norm=True).to(device)
print(surrogate_model)
print(f"Number of parameters: {surrogate_model.count_parameters()}")

surrogate_model.train_model(
    x_train=x_[2000:],
    y_train=x_orig[2000:],
    x_val=x_[:2000],
    y_val=x_orig[:2000],
    learning_rate=0.001,
    num_epochs=10,
    batch_size=2048,
    patience=50,
    device=device
)

surrogate_model.eval()
surrogate_model = surrogate_model.to('cpu')
x_ = x_.cpu()
x_orig = x_orig.detach().cpu().numpy()

with torch.no_grad():
    x_ = surrogate_model(x_).numpy()

# measure the time taken to transform the data using the fitted erasure model
start_time = time.time()
x_test_ = pca.transform(x_test)
x_test_ = torch.from_numpy(x_test_).float()
with torch.no_grad():
    x_test_ = surrogate_model(x_test_).numpy()
end_time = time.time()
print(f"Time taken to transform the test data using the surrogate model: {end_time - start_time:.2f} seconds")



print("Evaluation of the surrogate model on transformed data")

# Model evaluation
# ------------------------------------------------------------------

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



print("\nModel evaluation:")
print("-" * 40)
nprobing = 5
max_length = 30
print(f"\n{'Evaluation':<{max_length}}: Utility MSE ↓")
print(f"{'optimal MUtE*':<{max_length}}: {utility_MSE(x, x_orig):.4f} (approx. 0 expected)")
print(f"{'surrogate':<{max_length}}: {utility_MSE(x, x_):.6f}") # => 0.075 - 0.084 (approx)


print(f"\n{'Evaluation':<{max_length}}: Probing accuracy  ↓ (nonlinear probe)")
train_acc, test_acc, train_std, test_std = nonlinear_probing(x_orig, z, x_test_orig, z_test, nprobing=nprobing, random_state=seed)
print(f"{'optimal MUtE*':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
train_acc, test_acc, train_std, test_std = nonlinear_probing(x_orig, z, x_test_, z_test, nprobing=nprobing, random_state=seed)
print(f"{'surrogate':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")

# Downstream task evaluation (predicting y)
if y.size != 0:
    print(f"\n{'Evaluation':<{max_length}}: Probing accuracy for predicting y ↓ (nonlinear probe)")
    (train_acc, test_acc, train_std, test_std), clfs_orig = train_clfs(x_orig, y, x_test_orig, y_test, ntrain=nprobing, max_iter=10000, ignore_warnings=True, return_clfs=True, random_state=seed)
    print(f"{'optimal MUtE*':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
    (train_acc, test_acc, train_std, test_std), clfs_erased = train_clfs(x_orig, y, x_test_, y_test, ntrain=nprobing, max_iter=10000, ignore_warnings=True, return_clfs=True, random_state=seed)
    print(f"{'surrogate':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")

    # Fairness evaluation
    print(f"\n{'Evaluation':<{max_length}}: Demographic Parity ↓")
    dp_mean, dp_std = demographic_parity(x_test_orig, y_test, z_test, clfs_orig)
    print(f"{'optimal MUtE*':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")
    dp_mean, dp_std = demographic_parity(x_test_, y_test, z_test, clfs_erased)
    print(f"{'surrogate':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")

    print(f"\n{'Evaluation':<{max_length}}: TPR-Gap RMS ↓")
    mean_rms, std_rms = TPR_RMS(x_test_orig, y_test, z_test, 0, 1, clfs_orig)
    print(f"{'optimal MUtE*':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
    mean_rms, std_rms = TPR_RMS(x_test_, y_test, z_test, 0, 1, clfs_erased)
    print(f"{'surrogate':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")

    print(f"\n{'Evaluation':<{max_length}}: TPR-Gap correlation coefficient ↓")
    mean_rms, std_rms = TPR_corr_coef(x_test_orig, y_test, z_test, 0, 1, clfs_orig)
    print(f"{'optimal MUtE*':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
    mean_rms, std_rms = TPR_corr_coef(x_test_, y_test, z_test, 0, 1, clfs_erased)
    print(f"{'surrogate':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")


print("\n" + "-" * 40)
print("-" * 40)
print("-" * 40)

