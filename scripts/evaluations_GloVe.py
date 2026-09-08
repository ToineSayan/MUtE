#!/usr/bin/python
import torch
import random
import numpy as np

from erasers import *
from configs import *
from data.utils import load_data, load_ws3353
from eval.erasure import nonlinear_probing, mdl, privacy
from eval.utility import utility_MSE, neighborhood_overlap, similarity_correlation

import sys
import time

config_path = "configs/GloVe_KRaM.yml"
# config_path = "configs/GloVe_FaRM.yml"
# config_path = "configs/GloVe_MUtE.yml"
# config_path = "configs/GloVe_nonLEOPARD.yml"





# Load the configuration
# ------------------------------------------------------------------
config = load_config(config_path=config_path)
print(config)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
model_name = config["primary_erasure_model"]["name"]
model_type = str_to_class(model_name)
erasure_model = model_type(**config["primary_erasure_model_init_params"])

print(f"{'Erasure model':<20}: {model_name}")
for k,v in config["primary_erasure_model_init_params"].items():
    print(f"{k:<20}: {v}")

print("\n")
print(erasure_model)

# Fit the erasure model and transform the training data
# ------------------------------------------------------------------
print("\nModel fitting and data transformation:")
print("-" * 40)

x_tensor = torch.from_numpy(x)
z_tensor = torch.from_numpy(z)
x_test_tensor = torch.from_numpy(x_test)
z_test_tensor = torch.from_numpy(z_test)

# measure the time taken to fit the erasure model and transform the data
erasure_model.optimal_erasure = False
start_time = time.time()
# erasure_model.fit(x_tensor, z_tensor)
erasure_model.fit(x, z)
fit_time = time.time() - start_time
print(f"Time taken to fit the erasure model: {fit_time:.2f} seconds")

# measure the time taken to transform the data using the fitted erasure model
start_time = time.time()
# x_ = erasure_model.transform(x_tensor).detach().cpu().numpy()
x_ = erasure_model.transform(x)
end_time = time.time()
print(f"Time taken to transform the training data: {end_time - start_time:.2f} seconds")
# x_test_ = erasure_model.transform(x_test_tensor).detach().cpu().numpy()
x_test_ = erasure_model.transform(x_test)


# Model evaluation
# ------------------------------------------------------------------

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

max_length = 30


print("\nGloVe-specific evaluation:")

print(f"\n{'Evaluation':<{max_length}}: Neighborhood Overlap ↓ (50%/10%/1% nearest neighbors)")
# print(f"{'original':<{max_length}}: {round(neighborhood_overlap(x, x, k=0.5), 2)} / {round(neighborhood_overlap(x, x, k=0.1), 2)} / {round(neighborhood_overlap(x, x, k=0.01), 2)}")
print(f"{'erased':<{max_length}}: {round(neighborhood_overlap(x, x_, k=0.5), 2)} / {round(neighborhood_overlap(x, x_, k=0.1), 2)} / {round(neighborhood_overlap(x, x_, k=0.01), 2)}")

print(f"\n{'Evaluation':<{max_length}}: Similarity correlation ↓ (test WS353 - Spearman correlation between cosine similarity and human similarity ratings)")
x1, x2, y_sim = load_ws3353()
# print(f"{'original':<{max_length}}: {similarity_correlation(x1, x2, y_sim)[0]:.4f}")
print(f"{'erased':<{max_length}}: {similarity_correlation(erasure_model.transform(x1), erasure_model.transform(x2), y_sim)[0]:.4f}")

print(np.linalg.matrix_rank(x_))

print("\nModel evaluation:")
print("-" * 40)
nprobing = 5
max_length = 30
print(f"\n{'Evaluation':<{max_length}}: Utility MSE ↓")
print(f"{'original':<{max_length}}: {utility_MSE(x, x):.4f} (approx. 0 expected)") # => 0.075 - 0.084 (approx)
print(f"{'erased':<{max_length}}: {utility_MSE(x, x_):.4f}") # => 0.075 - 0.084 (approx)
# print(f"{'constant baseline':<{max_length}}: {utility_MSE(x, np.ones_like(x)):.4f}") # => 0.142, which is the MSE of the original data with a constant vector, which shows that the constant baseline also destroys the information in the data and achieves a similar utility as random noise.


print(f"\n{'Evaluation':<{max_length}}: Privacy ↓")
print(f"{'original':<{max_length}}: {privacy(x, z):.4f}")
print(f"{'erased':<{max_length}}: {privacy(x_, z):.4f}")
# print(f"{'constant baseline':<{max_length}}: {privacy(np.ones_like(x), z):.4f} (approx. 0 expected)") 


print(f"\n{'Evaluation':<{max_length}}: Probing accuracy  ↓ (nonlinear probe)")
train_acc, test_acc, train_std, test_std = nonlinear_probing(x, z, x_test, z_test, nprobing=nprobing, random_state=seed)
print(f"{'original':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
train_acc, test_acc, train_std, test_std = nonlinear_probing(x_, z, x_test_, z_test, nprobing=nprobing, random_state=seed)
print(f"{'erased':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f})")
# train_acc, test_acc, train_std, test_std = nonlinear_probing(np.ones_like(x), z, np.ones_like(x_test), z_test, nprobing=nprobing, random_state=seed)
# print(f"{'constant baseline':<{max_length}}: train: {train_acc*100:.2f} ({train_std*100:.2f}), test: {test_acc*100:.2f} ({test_std*100:.2f}) (approx. {data_proportion_majority_class*100:.2f} expected)")














quit()




# # short routine to evaluate when the model should stop
# print("Accuracy of nonlinear probing (original data):", nonlinear_probing(x, z, x_test, z_test, nprobing=nprobing))
# erasure_model.nsteps = 0
# erasure_model.fit(x_tensor, z_tensor)
# x_ = erasure_model.transform(x_tensor).detach().cpu().numpy()
# x_test_ = erasure_model.transform(x_test_tensor).detach().cpu().numpy()
# print("Accuracy of nonlinear probing (after rough erasure):", nonlinear_probing(x_, z, x_test_, z_test, nprobing=nprobing))
# print(f"Utility MSE ↓ (after rough erasure):", utility_MSE(x, x_))
# for i in range(100):
#     _ = erasure_model.partial_fit_transform(torch.from_numpy(x_), z_tensor, iteration=i)
#     x_ = erasure_model.transform(x_tensor).detach().cpu().numpy()
#     x_test_ = erasure_model.transform(x_test_tensor).detach().cpu().numpy()

#     if (i+1) % 50 == 0:
#         acc = nonlinear_probing(x_, z, x_test_, z_test, nprobing=nprobing)
#         print(f"Accuracy of nonlinear probing (after {i+1} steps of erasure):", acc)
#         print(f"Utility MSE ↓ (after {i+1} steps of erasure):", utility_MSE(x, x_))







quit()

x_primary_erased = erasure_model.fit_transform(x_tensor, z_tensor).detach().cpu().numpy()
# x_test_primary_erased = erasure_model.transform(x_test_tensor, z_test_tensor).detach().cpu().numpy()
# x_primary_erased = erasure_model.fit_transform(x_tensor).detach().cpu().numpy()
x_test_primary_erased = erasure_model.transform(x_test_tensor).detach().cpu().numpy()

x_tensor, z_tensor = None, None # free up memory   








quit()






# Load the erasure model
# ------------------------------------------------------------------
if config['model']['name'] == 'fullerasure':
    erasure_model = ConditionalAutoencoderEraser(**config["erasure_model_init_params"])
# erasure_model = eraser_class(**config["erasure_model_init_params"])

# print(erasure_model.__dict__)



# Fit the erasure model and transform the training data
# ------------------------------------------------------------------

# Transform numpy arrays to torch tensors and move to device
x_tensor = torch.from_numpy(x)
z_tensor = torch.from_numpy(z)

x_erased = erasure_model.fit_transform(x_tensor, z_tensor)



# # Save the model and the erased data for future use (e.g. for training a surrogate model, or for evaluating the erasure on
# torch.save(erasure_model, config["output_dir"] + '/ItGauss_erasure_model.pt')
# # erasure_model = torch.load(config["output_dir"] + '/ItGauss_erasure_model.pt', weights_only=False)
# np.save(config["output_dir"] + '/ItGauss_x_erased.npy', x_erased.detach().cpu().numpy())


# # Train the surrogate model on the erased data
# # ------------------------------------------------------------------
# surrogate_model = DataGenerator(
#     input_dim=x_erased.shape[1],
#     num_classes=len(z_values),
#     **config["surrogate_model_init_params"]
#     ).to(device)

# surrogate_model.fit(x_tensor, x_erased, z_tensor)
# # Note: the best model is saved during training using early stopping, so we don't need to save it again here

x_tensor, z_tensor = None, None # free up memory
x_erased = x_erased.detach().cpu().numpy()

# x_erased = np.load(config["output_dir"] + '/ItGauss_x_erased.npy')


# x_erased = erasure_model.transform(
#     torch.from_numpy(x).to(device),
#     torch.from_numpy(z).to(device)
#     ).detach().cpu().numpy()
# quit()
# x_test_erased = erasure_model.transform(
#     torch.from_numpy(x_test).to(device),
#     torch.from_numpy(z_test).to(device)
#     ).detach().cpu().numpy()


# create a random transformation baseline for comparison
x_erased_random = np.random.randn(*x_erased.shape)

# Eval the erasure on train data
# ------------------------------------------------------------------

nsamples_eval = 100000
print("MDL:", mdl(x_erased, z, max_iter=10000))
# print("MDL (random transformation baseline):", mdl(x_erased_random, z, max_iter=10000)) # approx 400kB

# Eval the erasure on train data
# ------------------------------------------------------------------
# print("Privacy (original data):", privacy(x, z)) # privacy 1.0
print("Privacy (erased data):", privacy(x_erased, z))
# print("Privacy (random transformation baseline):", privacy(np.random.randn(*x_erased.shape), z)) # privacy 0.068 approx

# Eval the utility preservation of the erasure
# ------------------------------------------------------------------
print("Utility MSE:", utility_MSE(x, x_erased))
print("Utility MSE (random transformation baseline):", utility_MSE(x, x_erased_random))
print("Neighborhood Overlap (k=0.10):", round(neighborhood_overlap(x, x_erased, k=0.1), 2))




quit()

# x_erased_surrogate = surrogate_model.transform(
#     torch.from_numpy(x),
#     torch.from_numpy(z)
#     ).detach().cpu().numpy()

# x_reconstructed = surrogate_model.inverse_transform(
#     torch.from_numpy(x_erased),
#     torch.from_numpy(z)
#     ).detach().cpu().numpy()


# print("Utility MSE (surrogate):", utility_MSE(x, x_erased_surrogate))
# print("MSE of reconstruction (surrogate):", ((x - x_reconstructed)**2).mean())



# Load a saved surrogate model and evaluate its reconstruction quality
# ------------------------------------------------------------------

surrogate_model = DataGenerator(
    input_dim=x_erased.shape[1],
    num_classes=len(z_values),
    **config["surrogate_model_init_params"]
    )
surrogate_model.load_state_dict(torch.load('best_model.pt'))

x_erased_surrogate = surrogate_model.transform(
    torch.from_numpy(x),
    torch.from_numpy(z)
    ).detach().cpu().numpy()

x_test_erased_surrogate = surrogate_model.transform(
    torch.from_numpy(x_test),
    torch.from_numpy(z_test)
    ).detach().cpu().numpy()

x_reconstructed = surrogate_model.inverse_transform(
    torch.from_numpy(x_erased),
    torch.from_numpy(z)
    ).detach().cpu().numpy()

print("Utility MSE (surrogate):", utility_MSE(x, x_erased_surrogate))
print("MSE of reconstruction (surrogate):", ((x - x_reconstructed)**2).mean())

print("MDL:", mdl(x_erased_surrogate[mask], z[mask], max_iter=10000))
print("Privacy (erased data):", privacy(x_erased_surrogate[mask], z[mask]))

# mask_test = z_test != 2
mask_test = np.ones_like(z_test, dtype=bool) # for the probing evaluations, we keep all the data (including neutral words), since we want to evaluate how much information about the concept is still present in the erased data, even for the neutral words
print("Nonlinear probing (original data):", nonlinear_probing(x[mask], z[mask], x_test[mask_test], z_test[mask_test], nprobing=5))
print("Nonlinear probing (erased data from surrogate):", nonlinear_probing(x_erased_surrogate[mask], z[mask], x_test_erased_surrogate[mask_test], z_test[mask_test], nprobing=5))
print("Nonlinear probing (erased data from erasure model):", nonlinear_probing(x_erased[mask], z[mask], x_test_erased_surrogate[mask_test], z_test[mask_test], nprobing=5))
