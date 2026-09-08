#!/usr/bin/python
import torch
import random
import numpy as np

from erasers import *
from configs import *
from data.utils import load_data
from eval.fairness import demographic_parity, TPR_RMS, train_clfs


config_path = "configs/biasbios_MUtE.yml"
# config_path = "configs/DIAL_MUtE.yml"



# Load the configuration
# ------------------------------------------------------------------
config = load_config(config_path=config_path)
print(config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
# try:
#     majority_class_y = np.bincount(y).argmax()
#     data_proportion_majority_class_y = np.mean(y == majority_class_y)
#     print(f"{'Majority class Y':<20}: {y_values[majority_class_y]} ({data_proportion_majority_class_y*100:.2f}% of the data)")
# except:
#     print(f"{'Majority class Y':<20}: None")



####################################################################
# Data augmentation 
####################################################################


nprobing = 5
max_length = 30
num_training_samples = 50000

x_extracted = x[-num_training_samples:]
z_extracted = z[-num_training_samples:]
x = x[:num_training_samples]
z = z[:num_training_samples]
y = y[:num_training_samples]


# Here we train a classifier to predic y from x (will be biased)
print(f"\n{'Evaluation':<{max_length}}: Probing accuracy for predicting y ↓ (nonlinear probe)")
(train_acc, test_acc, train_std, test_std), clfs_orig = train_clfs(x, y, x_test, y_test, ntrain=nprobing, max_iter=10000, ignore_warnings=True, return_clfs=True, random_state=seed)
print(f"{'original':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
print(f"{'Evaluation':<{max_length}}: TPR-Gap RMS ↓")
mean_rms, std_rms = TPR_RMS(x_test, y_test, z_test, 0, 1, clfs_orig)
print(f"{'original':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
print(f"{'Evaluation':<{max_length}}: Demographic Parity ↓")
dp_mean, dp_std = demographic_parity(x_test, y_test, z_test, clfs_orig)
print(f"{'original':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")

# the training set will contain 2*x.shape[0] samples
# let's define the permutation of the indices for shuffling the training set and the target labels for all experiments
indices = np.random.permutation(2*x.shape[0])
y_cf_train = np.hstack([y, y])[indices]


# Load the erasure model
# ------------------------------------------------------------------
print("\nTest 1: Z available along Y at training time (optimal erasure)")
print("-" * 40)


erasure_model = MUtE(**config["primary_erasure_model_init_params"])
erasure_model.optimal_erasure = True # we consider that Z is available along Y
erasure_model.fit(x, z)

x_switch = erasure_model.inverse_transform(
    erasure_model.transform(x, z),
    1-z
)


x_cf_train = np.vstack([x, x_switch])[indices]

# Here we train a classifier to predic y from x (will be biased)
print(f"\n{'Evaluation':<{max_length}}: Probing accuracy for predicting y ↓ (nonlinear probe)")
(train_acc, test_acc, train_std, test_std), clfs_orig = train_clfs(x_cf_train, y_cf_train, x_test, y_test, ntrain=nprobing, max_iter=100, ignore_warnings=True, return_clfs=True, random_state=seed)
print(f"{'original':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
print(f"{'Evaluation':<{max_length}}: TPR-Gap RMS ↓")
mean_rms, std_rms = TPR_RMS(x_test, y_test, z_test, 0, 1, clfs_orig)
print(f"{'original':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
print(f"{'Evaluation':<{max_length}}: Demographic Parity ↓")
dp_mean, dp_std = demographic_parity(x_test, y_test, z_test, clfs_orig)
print(f"{'original':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")



# Load the erasure model
# ------------------------------------------------------------------
print("\nTest 2: Z not available along Y (empirical erasure)")
print("-" * 40)

erasure_model = MUtE(**config["primary_erasure_model_init_params"])
erasure_model.optimal_erasure = False # we consider that Z is not available along Y
erasure_model.fit(x_extracted, z_extracted)

x_switch = erasure_model.inverse_transform(
    erasure_model.transform(x),
    1-erasure_model.class_clf.predict(x) 
)

x_cf_train = np.vstack([x, x_switch])[indices]


print(f"\n{'Evaluation':<{max_length}}: Probing accuracy for predicting y ↓ (nonlinear probe)")
(train_acc, test_acc, train_std, test_std), clfs_orig = train_clfs(x_cf_train, y_cf_train, x_test, y_test, ntrain=nprobing, max_iter=100, ignore_warnings=True, return_clfs=True, random_state=seed)
print(f"{'original':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
print(f"{'Evaluation':<{max_length}}: TPR-Gap RMS ↓")
mean_rms, std_rms = TPR_RMS(x_test, y_test, z_test, 0, 1, clfs_orig)
print(f"{'original':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
print(f"{'Evaluation':<{max_length}}: Demographic Parity ↓")
dp_mean, dp_std = demographic_parity(x_test, y_test, z_test, clfs_orig)
print(f"{'original':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")



# Load the erasure model
# ------------------------------------------------------------------
print("\nTest 3: Z available along Y (mean difference translation to generate CFRs)")
print("-" * 40)



mean_diff_translation = x[z==1].mean(axis=0) - x[z==0].mean(axis=0)
x_switch = x.copy()
x_switch[z==0] += mean_diff_translation
x_switch[z==1] -= mean_diff_translation


x_cf_train = np.vstack([x, x_switch])[indices]

print(f"\n{'Evaluation':<{max_length}}: Probing accuracy for predicting y ↓ (nonlinear probe)")
(train_acc, test_acc, train_std, test_std), clfs_orig = train_clfs(x_cf_train, y_cf_train, x_test, y_test, ntrain=nprobing, max_iter=100, ignore_warnings=True, return_clfs=True, random_state=seed)
print(f"{'original':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
print(f"{'Evaluation':<{max_length}}: TPR-Gap RMS ↓")
mean_rms, std_rms = TPR_RMS(x_test, y_test, z_test, 0, 1, clfs_orig)
print(f"{'original':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
print(f"{'Evaluation':<{max_length}}: Demographic Parity ↓")
dp_mean, dp_std = demographic_parity(x_test, y_test, z_test, clfs_orig)
print(f"{'original':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")


# Load the erasure model
# ------------------------------------------------------------------
print("\nTest 4: Z available along Y (linear OT)")
print("-" * 40)

import ot

lin_ot_01 = ot.da.LinearTransport(reg=1e-2).fit(Xs=x[z==0], Xt=x[z==1])
lin_ot_10 = ot.da.LinearTransport(reg=1e-2).fit(Xs=x[z==1], Xt=x[z==0])

x_switch = np.empty_like(x)
x_switch[z==0] = lin_ot_01.transform(x[z==0])
x_switch[z==1] = lin_ot_10.transform(x[z==1])

x_cf_train = np.vstack([x, x_switch])[indices]


# Here we train a classifier to predic y from x (will be biased)
print(f"\n{'Evaluation':<{max_length}}: Probing accuracy for predicting y ↓ (nonlinear probe)")
(train_acc, test_acc, train_std, test_std), clfs_orig = train_clfs(x_cf_train, y_cf_train, x_test, y_test, ntrain=nprobing, max_iter=100, ignore_warnings=True, return_clfs=True, random_state=seed)
print(f"{'original':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
print(f"{'Evaluation':<{max_length}}: TPR-Gap RMS ↓")
mean_rms, std_rms = TPR_RMS(x_test, y_test, z_test, 0, 1, clfs_orig)
print(f"{'original':<{max_length}}: {mean_rms:.4f} ({std_rms:.4f})")
print(f"{'Evaluation':<{max_length}}: Demographic Parity ↓")
dp_mean, dp_std = demographic_parity(x_test, y_test, z_test, clfs_orig)
print(f"{'original':<{max_length}}: {dp_mean:.4f} ({dp_std:.4f})")

print("\nDone.")









