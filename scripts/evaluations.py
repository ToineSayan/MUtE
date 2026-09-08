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

# config_path = "configs/GloVe_KRaM.yml"
# config_path = "configs/GloVe_FaRM.yml"
# config_path = "configs/GloVe_nonLEOPARD.yml"
# config_path = "configs/GloVe_TACO.yml"
config_path = "configs/GloVe_MUtE.yml"

# config_path = "configs/biasbios_KRaM.yml"
# config_path = "configs/biasbios_FaRM.yml"
# config_path = "configs/biasbios_nonLEOPARD.yml"
# config_path = "configs/biasbios_TACO.yml"
# config_path = "configs/biasbios_MUtE.yml"

# config_path = "configs/DIAL_KRaM.yml"
# config_path = "configs/DIAL_FaRM.yml"
# config_path = "configs/DIAL_nonLEOPARD.yml"
# config_path = "configs/DIAL_TACO.yml"
# config_path = "configs/DIAL_MUtE.yml"

# config_path = "configs/jigsaw_KRaM.yml"
# config_path = "configs/jigsaw_FaRM.yml"
# config_path = "configs/jigsaw_nonLEOPARD.yml"
# config_path = "configs/jigsaw_TACO.yml"
# config_path = "configs/jigsaw_MUtE.yml"



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

if not model_name == 'nonLEOPARD':
    erasure_model = model_type(**config["erasure_model_init_params"])
else:
    erasure_model = model_type(**config["erasure_model_init_params"], X_init=x, Z_init=z)

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
erasure_model.fit(x,z)
fit_time = time.time() - start_time
print(f"Time taken to fit the erasure model: {fit_time:.2f} seconds")

# measure the time taken to transform the data using the fitted erasure model
start_time = time.time()
x_ = erasure_model.transform(x)
end_time = time.time()
print(f"Time taken to transform the training data: {end_time - start_time:.2f} seconds")
x_test_ = erasure_model.transform(x_test)


# # Uncomment to test LEACE
# from concept_erasure import LeaceEraser
# x_tensor = torch.from_numpy(x)
# z_tensor = torch.from_numpy(z)
# x_test_tensor = torch.from_numpy(x_test)
# z_test_tensor = torch.from_numpy(z_test)
# eraser = LeaceEraser.fit(x_tensor, z_tensor)
# x_ = eraser(x_tensor).detach().cpu().numpy()
# x_test_ = eraser(x_test_tensor).detach().cpu().numpy()


# # Uncomment to test with original representations
# x_ = x
# x_test_ = x_test

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
print(f"{'original':<{max_length}}: {utility_MSE(x, x):.4f} (approx. 0 expected)") 
print(f"{'erased':<{max_length}}: {utility_MSE(x, x_):.6f}") 
print(f"{'constant baseline':<{max_length}}: {utility_MSE(x, np.ones_like(x)):.4f}") 



print(f"\n{'Evaluation':<{max_length}}: Probing accuracy  ↓ (nonlinear probe)")
train_acc, test_acc, train_std, test_std = nonlinear_probing(x, z, x_test, z_test, nprobing=nprobing, random_state=seed)
print(f"{'original':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
train_acc, test_acc, train_std, test_std = nonlinear_probing(x_, z, x_test_, z_test, nprobing=nprobing, random_state=seed)
print(f"{'erased':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
train_acc, test_acc, train_std, test_std = nonlinear_probing(np.ones_like(x), z, np.ones_like(x_test), z_test, nprobing=nprobing, random_state=seed)
print(f"{'constant baseline':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f}) (approx. {data_proportion_majority_class*100:.2f} expected)")


# Downstream task evaluation (predicting y)
if y.size != 0:
    print(f"\n{'Evaluation':<{max_length}}: Probing accuracy for predicting y ↓ (nonlinear probe)")
    (train_acc, test_acc, train_std, test_std), clfs_orig = train_clfs(x, y, x_test, y_test, ntrain=nprobing, max_iter=10000, ignore_warnings=True, return_clfs=True, random_state=seed)
    print(f"{'original':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
    (train_acc, test_acc, train_std, test_std), clfs_erased = train_clfs(x_, y, x_test_, y_test, ntrain=nprobing, max_iter=10000, ignore_warnings=True, return_clfs=True, random_state=seed)
    print(f"{'erased':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
    (train_acc, test_acc, train_std, test_std), clfs_cst = train_clfs(np.ones_like(x), y, np.ones_like(x_test), y_test, ntrain=nprobing, max_iter=10000, ignore_warnings=True, return_clfs=True, random_state=seed)
    print(f"{'constant baseline':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f}) (approx. {data_proportion_majority_class*100:.2f} expected)")

    # Fairness evaluation
    print(f"\n{'Evaluation':<{max_length}}: Demographic Parity ↓")
    dp_mean, dp_std = demographic_parity(x_test, y_test, z_test, clfs_orig)
    print(f"{'original':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")
    dp_mean, dp_std = demographic_parity(x_test_, y_test, z_test, clfs_erased)
    print(f"{'erased':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")
    dp_mean, dp_std = demographic_parity(np.ones_like(x_test), y_test, z_test, clfs_cst)
    print(f"{'constant baseline':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f}) (approx. 0 expected)")

    print(f"\n{'Evaluation':<{max_length}}: TPR-Gap RMS ↓")
    mean_rms, std_rms = TPR_RMS(x_test, y_test, z_test, 0, 1, clfs_orig)
    print(f"{'original':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
    mean_rms, std_rms = TPR_RMS(x_test_, y_test, z_test, 0, 1, clfs_erased)
    print(f"{'erased':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
    mean_rms, std_rms = TPR_RMS(np.ones_like(x_test), y_test, z_test, 0, 1, clfs_cst)
    print(f"{'constant baseline':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f}) (approx. 0 expected)")

    print(f"\n{'Evaluation':<{max_length}}: TPR-Gap correlation coefficient ↓")
    mean_rms, std_rms = TPR_corr_coef(x_test, y_test, z_test, 0, 1, clfs_orig)
    print(f"{'original':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
    mean_rms, std_rms = TPR_corr_coef(x_test_, y_test, z_test, 0, 1, clfs_erased)
    print(f"{'erased':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
    mean_rms, std_rms = TPR_corr_coef(np.ones_like(x_test), y_test, z_test, 0, 1, clfs_cst)
    print(f"{'constant baseline':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f}) (approx. 0 expected)")


print("\n" + "-" * 40)
print("-" * 40)
print("-" * 40)

