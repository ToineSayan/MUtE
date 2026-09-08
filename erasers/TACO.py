"""
TaCo -- Targeted Concept Erasure
=================================
Standalone re-implementation of the method described in:

    "TaCo: Targeted Concept Erasure Prevents Non-Linear Classifiers From
    Detecting Protected Attributes" (Jourdan et al., 2023)
    https://arxiv.org/pdf/2312.06499
    Official repo: https://github.com/fanny-jourdan/TaCo

Given representations X (N, d) and a sensitive/protected attribute Z (N,)
that you want to erase, TaCo proceeds in 3 steps:

  (I)   Concept Discovery : factorize A = X into A ~= U @ W (U: per-sample
        concept coefficients, W: concept directions) using SVD / PCA / ICA.

  (II)  Concept Ranking   : train a non-linear probe (MLP) that predicts Z
        from X, and use Sobol total-order sensitivity indices to measure how
        much each concept (i.e. each column of U / row of W) contributes to
        that probe's output. If you also give a downstream task label Y that
        you want to *keep* predictable, a second MLP + Sobol pass measures
        each concept's importance for Y, and concepts are ranked by the
        angle between "importance for Z" and "importance for Y" (concepts
        useful for Z but useless for Y are removed first).

  (III) Concept Erasure   : drop the k most Z-informative concepts and
        reconstruct a "clean" representation matrix from the remaining ones.
        Because this is a genuine dimensionality reduction (not just hiding
        information from a linear probe), no downstream model -- linear or
        non-linear -- can recover the erased information from X_clean.

Dependencies: numpy, torch, scipy (>=1.7, for scipy.stats.qmc.Sobol), scikit-learn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as skPCA, FastICA
from sklearn.metrics import accuracy_score


# =============================================================================
# STEP 0 -- a small MLP used both as the "adversary" whose ability to predict
# Z we want to destroy, and (optionally) as the classifier for a downstream
# task Y whose accuracy we want to preserve.
# =============================================================================

class MLP(nn.Module):
    """Two-layer perceptron: d -> hidden -> num_classes, returning raw logits."""

    def __init__(self, in_dim, num_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # logits, no softmax


def train_mlp(X_train, y_train, X_val, y_val, num_classes, device="cpu",
              epochs=100, batch_size=256, lr=1e-3, hidden=128, verbose=False):
    """Train an MLP classifier by minimizing cross-entropy."""
    model = MLP(X_train.shape[1], num_classes, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

    n = X_train_t.shape[0]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X_train_t[batch_idx])
            loss = F.cross_entropy(logits, y_train_t[batch_idx])
            loss.backward()
            optimizer.step()
        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            model.eval()
            with torch.no_grad():
                val_acc = (model(X_val_t).argmax(1) == y_val_t).float().mean().item()
            print(f"    epoch {epoch + 1}/{epochs} - val acc: {val_acc:.3f}")

    model.eval()
    return model


# =============================================================================
# STEP (I) -- CONCEPT DISCOVERY
# Factorize the representation matrix A (N, d) into A ~= U @ W with
# U (N, r) the concept coefficients and W (r, d) the concept basis.
# =============================================================================

def decompose(A, method="PCA", n_components=20, seed=0):
    """
    A: np.ndarray (n_samples, d)
    Returns (U, W, decomposer) with U: (n_samples, r), W: (r, d), such that A ~= U @ W.
    decomposer: the fitted sklearn object (or dict for SVD) to apply to new data.
    method: one of "SVD", "PCA", "ICA" (matches the 3 decompositions used in
    the paper; PCA was found to give the best fairness/accuracy trade-off).
    """
    if method == "SVD":
        # Truncated SVD: A = U0 * S * V0^T, keep the r largest singular values.
        U0, S, VT = np.linalg.svd(A, full_matrices=False)
        U = U0[:, :n_components]
        W = np.diag(S[:n_components]) @ VT[:n_components, :]
        # For SVD, store info needed for projection; note: SVD transform is data-specific
        decomposer = {"method": "SVD", "n_components": n_components}
    elif method == "PCA":
        pca = skPCA(n_components=n_components)
        U = pca.fit_transform(A)
        W = pca.components_
        decomposer = pca
    elif method == "ICA":
        ica = FastICA(n_components=n_components, random_state=seed, max_iter=1000)
        U = ica.fit_transform(A)
        W = ica.mixing_.T
        decomposer = ica
    else:
        raise ValueError(f"Unknown decomposition method: {method!r}")
    return U, W, decomposer


def decompose_new_data(X_new, decomposer, W):
    """
    Apply a fitted decomposer to new data X_new.
    
    X_new: np.ndarray (n_samples, d)
    decomposer: fitted sklearn object (PCA, ICA) or dict with method info
    W: (r, d) basis matrix from decompose()
    
    Returns U_new: (n_samples, r) concept coefficients for new data
    """
    if isinstance(decomposer, dict) and decomposer["method"] == "SVD":
        # For SVD, we cannot directly project new data without storing full info
        # This is a limitation; SVD is dataset-specific.
        raise NotImplementedError(
            "SVD decomposition cannot be applied to new data without storing full "
            "SVD info. Use PCA or ICA for transform() on new data."
        )
    elif hasattr(decomposer, "transform"):
        # PCA or ICA: use the fitted transformer
        return decomposer.transform(X_new)
    else:
        raise ValueError(f"Unknown decomposer type: {type(decomposer)}")


# =============================================================================
# STEP (II) -- CONCEPT RANKING (Sobol total-order sensitivity indices)
# For every concept i, we ask: "if I mask this concept's coefficient, how
# much does the probe's output change?" -- averaged over many random masks
# of *all* concepts (to also capture interaction effects), following the
# Jansen (1999) estimator of total Sobol indices, exactly as in the paper.
# =============================================================================

def _sobol_sequence_masks(num_components, nb_design):
    """
    Quasi-Monte-Carlo (Sobol sequence) sampling of the replicated-design
    A / B / C matrices used by the Jansen estimator.
    Returns an array of shape (nb_design * (2 + num_components), num_components)
    with values in [0, 1).
    """
    sampler = qmc.Sobol(num_components * 2, scramble=False)
    # nb_design need not be a power of 2 for our purposes; suppress the
    # informational warning scipy emits about balance properties.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sampling_ab = sampler.random(nb_design).astype(np.float32)
    sampling_a, sampling_b = sampling_ab[:, :num_components], sampling_ab[:, num_components:]

    # Replicated design C: for each concept i, copy A but swap in B's i-th column.
    replicated_c = np.array([sampling_a.copy() for _ in range(num_components)])
    for i in range(num_components):
        replicated_c[i, :, i] = sampling_b[:, i]
    replicated_c = replicated_c.reshape((-1, num_components))

    return np.concatenate([sampling_a, sampling_b, replicated_c], axis=0)


def _jansen_total_sobol(outputs, nb_design, num_components):
    """Jansen (1999) estimator of total-order Sobol' indices."""
    sampling_a = outputs[:nb_design]
    replication_c = np.array([
        outputs[nb_design * 2 + nb_design * i: nb_design * 2 + nb_design * (i + 1)]
        for i in range(num_components)
    ])

    mu_a = np.mean(sampling_a)
    var = np.sum((sampling_a - mu_a) ** 2) / (len(sampling_a) - 1)
    if var == 0:
        return np.zeros(num_components, dtype=np.float32)

    stis = [
        np.sum((sampling_a - replication_c[i]) ** 2.0) / (2 * nb_design * var)
        for i in range(num_components)
    ]
    return np.array(stis, dtype=np.float32)


def _concept_perturbation(a, u, masks, W, torch_model, device):
    """
    For a single sample (a=A[i], u=U[i]), evaluate the probe's output
    margin (top logit minus 2nd logit) after perturbing (masking) subsets
    of concepts, following the paper's perturbation operator:

        U~ = U ⊙ M            (Hadamard product with a random {0,1}-ish mask)
        A~ = U~ @ W + (A - U @ W)     (residual re-injected so we stay in-distribution)
    """
    delta = a - u @ W                 # reconstruction residual (info not in U,W)
    keep = 1.0 - masks                # masks in [0,1) -> keep-ratio in (0,1]
    u_masked = u[None, :] * keep      # perturbed concept coefficients
    a_perturbed = u_masked @ W + delta[None, :]

    with torch.no_grad():
        a_perturbed_t = torch.as_tensor(a_perturbed, dtype=torch.float32, device=device)
        logits = torch_model(a_perturbed_t).cpu().numpy()

    top2 = np.sort(logits, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]  # highest logit - second highest logit
    return margin


def sobol_concept_importance(A, U, W, torch_model, num_components,
                              sobol_nb_design=50, n_samples=None, device="cpu"):
    """
    Estimate, for every concept i, its total Sobol index w.r.t. the probe's
    output margin, averaged over n_samples data points. Larger index =>
    concept i matters more for what `torch_model` predicts.

    A: (n_samples, d) original representations
    U: (n_samples, num_components) concept coefficients
    W: (num_components, d) concept basis
    torch_model: callable, torch.Tensor (n, d) -> torch.Tensor (n, num_classes) logits
    """
    torch_model.eval()
    if n_samples is not None:
        A = A[:n_samples]
        U = U[:n_samples]

    masks = _sobol_sequence_masks(num_components, sobol_nb_design)

    importances = []
    for a, u in zip(A, U):
        outputs = _concept_perturbation(a, u, masks, W, torch_model, device)
        stis = _jansen_total_sobol(outputs, sobol_nb_design, num_components)
        importances.append(stis)

    return np.mean(importances, axis=0)  # shape (num_components,)


# =============================================================================
# STEP (III) -- CONCEPT ERASURE
# =============================================================================

def rank_concepts(importance_sensitive, importance_task=None):
    """
    Decide the removal order of concepts.

    If `importance_task` (importance for the label Y you want to keep) is
    given, concepts are ranked by the angle
        angle_i = arctan(importance_task_i / importance_sensitive_i)
    (as in the paper, Fig. 2): concepts with the SMALLEST angle are the ones
    that are important for Z but not for Y, and are removed first -- this
    is the best fairness/utility trade-off.

    If `importance_task` is None (no downstream task to preserve), concepts
    are simply ranked by their importance for Z, most important first.
    """
    eps = 1e-12
    if importance_task is not None:
        angle = np.arctan(importance_task / (importance_sensitive + eps)) * 180.0 / np.pi
        order = np.argsort(angle)                    # ascending: most "Z-only" concepts first
    else:
        order = np.argsort(-importance_sensitive)     # descending importance for Z
    return order


def erase_concepts(U, W, order, n_remove):
    """
    Remove the `n_remove` concepts listed first in `order` and reconstruct
    the cleaned representation matrix X_clean = U_kept @ W_kept.

    Returns (X_clean, keep_mask) where keep_mask (r,) is True for the
    concepts that were kept.
    """
    to_remove = order[:n_remove]
    keep_mask = np.ones(W.shape[0], dtype=bool)
    keep_mask[to_remove] = False

    X_clean = U[:, keep_mask] @ W[keep_mask, :]
    return X_clean, keep_mask


# =============================================================================
# TACO CLASS - sklearn-like interface with fit() and transform()
# =============================================================================

class TACO:
    """
    Targeted Concept Erasure for erasing sensitive/protected attributes from
    learned representations while preserving downstream task utility.
    
    Implements a sklearn-like interface with fit() and transform() methods.
    
    Parameters
    ----------
    method_decompose : {"PCA", "SVD", "ICA"}, default="PCA"
        Decomposition method for concept discovery.
    num_components : int, default=20
        Number of concepts to extract.
    n_concepts_to_remove : int, default=3
        Number of most Z-informative concepts to erase.
    sobol_nb_design : int, default=50
        Sobol sequence design size for sensitivity analysis.
    sobol_n_samples : int, default=2000
        Number of samples for Sobol importance estimation.
    test_size : float, default=0.2
        Fraction of data to use as test set (only for training probes).
    device : str or None, default=None
        Device for PyTorch ("cuda" or "cpu"). Defaults to CUDA if available.
    epochs : int, default=100
        Training epochs for MLP probes.
    seed : int, default=0
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress messages.
    
    Attributes
    ----------
    decomposer_ : fitted sklearn object or dict
        The fitted decomposition model (PCA, ICA, or SVD info dict).
    W_ : np.ndarray of shape (num_components, d)
        The learned concept basis.
    order_ : np.ndarray of shape (num_components,)
        Concept removal order (indices of concepts to remove, sorted by importance).
    keep_mask_ : np.ndarray of shape (num_components,), dtype=bool
        Boolean mask indicating which concepts were kept.
    importance_Z_ : np.ndarray of shape (num_components,)
        Sobol importance of each concept for predicting Z.
    importance_Y_ : np.ndarray of shape (num_components,) or None
        Sobol importance of each concept for predicting Y (if Y was provided).
    acc_Z_before_ : float
        Accuracy of probe predicting Z from original X (before erasure).
    acc_Z_after_ : float
        Accuracy of probe predicting Z from erased X_clean (after erasure).
    """
    
    def __init__(self, method_decompose="PCA", num_components=20,
                 n_concepts_to_remove=3, sobol_nb_design=50,
                 sobol_n_samples=2000, test_size=0.2, device=None,
                 epochs=100, seed=0, verbose=True):
        self.method_decompose = method_decompose
        self.num_components = num_components
        self.n_concepts_to_remove = n_concepts_to_remove
        self.sobol_nb_design = sobol_nb_design
        self.sobol_n_samples = sobol_n_samples
        self.test_size = test_size
        self.device = device
        self.epochs = epochs
        self.seed = seed
        self.verbose = verbose
        
        # Fitted attributes
        self.decomposer_ = None
        self.W_ = None
        self.order_ = None
        self.keep_mask_ = None
        self.importance_Z_ = None
        self.importance_Y_ = None
        self.acc_Z_before_ = None
        self.acc_Z_after_ = None
    
    def fit(self, X, Z_sensitive, Y=None):
        """
        Learn the concept decomposition and ranking.
        
        Performs steps (I) Concept Discovery and (II) Concept Ranking from
        the TaCo pipeline.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Representation matrix to debias.
        Z_sensitive : np.ndarray of shape (n_samples,)
            Sensitive/protected attribute to erase.
        Y : np.ndarray of shape (n_samples,) or None, default=None
            Optional downstream task label to preserve while erasing Z.
            Recommended for better fairness/utility trade-off.
        
        Returns
        -------
        self : TACO
            Returns self for method chaining.
        """
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        N, d = X.shape
        num_components = min(self.num_components, d, N - 1)
        if self.n_concepts_to_remove >= num_components:
            raise ValueError("n_concepts_to_remove must be < num_components")
        
        # ---- split (only needed to train/evaluate the probing classifiers) ----
        idx_train, idx_test = train_test_split(
            np.arange(N), test_size=self.test_size, random_state=self.seed, 
            stratify=Z_sensitive)
        X_train, X_test = X[idx_train], X[idx_test]
        Z_train, Z_test = Z_sensitive[idx_train], Z_sensitive[idx_test]
        
        # -------------------------------------------------------------------
        # (I) CONCEPT DISCOVERY: factorize X ~= U @ W
        # -------------------------------------------------------------------
        if self.verbose:
            print(f"[1/3] Decomposing X with {self.method_decompose} into "
                  f"{num_components} concepts...")
        U, W, decomposer = decompose(
            X, method=self.method_decompose, n_components=num_components, 
            seed=self.seed)
        U_train = U[idx_train]
        
        # Train the non-linear adversary c_Z predicting Z from X
        classes_Z = np.unique(Z_sensitive)
        z_map = {c: i for i, c in enumerate(classes_Z)}
        Z_train_idx = np.array([z_map[z] for z in Z_train])
        Z_test_idx = np.array([z_map[z] for z in Z_test])
        
        if self.verbose:
            print("      training the non-linear probe for Z...")
        clf_Z = train_mlp(X_train, Z_train_idx, X_test, Z_test_idx,
                          num_classes=len(classes_Z), device=device, 
                          epochs=self.epochs)
        
        with torch.no_grad():
            preds = clf_Z(torch.as_tensor(X_test, dtype=torch.float32, 
                          device=device)).argmax(1).cpu().numpy()
        acc_Z_before = accuracy_score(Z_test_idx, preds)
        
        # Optionally train a probe for the downstream task Y to preserve
        clf_Y = None
        if Y is not None:
            classes_Y = np.unique(Y)
            y_map = {c: i for i, c in enumerate(classes_Y)}
            Y_train_idx = np.array([y_map[y] for y in Y[idx_train]])
            Y_test_idx = np.array([y_map[y] for y in Y[idx_test]])
            if self.verbose:
                print("      training the non-linear probe for Y...")
            clf_Y = train_mlp(X_train, Y_train_idx, X_test, Y_test_idx,
                              num_classes=len(classes_Y), device=device, 
                              epochs=self.epochs)
        
        # -------------------------------------------------------------------
        # (II) CONCEPT RANKING: Sobol total-order indices for Z (and Y)
        # -------------------------------------------------------------------
        if self.verbose:
            print("[2/3] Estimating Sobol importance of each concept for Z"
                  + (" and Y..." if clf_Y is not None else "..."))
        importance_Z = sobol_concept_importance(
            X_train, U_train, W, clf_Z, num_components=num_components,
            sobol_nb_design=self.sobol_nb_design, 
            n_samples=self.sobol_n_samples, device=device)
        
        importance_Y = None
        if clf_Y is not None:
            importance_Y = sobol_concept_importance(
                X_train, U_train, W, clf_Y, num_components=num_components,
                sobol_nb_design=self.sobol_nb_design,
                n_samples=self.sobol_n_samples, device=device)
        
        order = rank_concepts(importance_Z, importance_Y)
        
        # ---- sanity check: evaluate on test set ----
        # We compute erase_concepts to get keep_mask for the diagnostic
        _, keep_mask = erase_concepts(U, W, order, self.n_concepts_to_remove)
        
        X_clean_train, X_clean_test = (
            U_train[:, keep_mask] @ W[keep_mask, :],
            U[idx_test][:, keep_mask] @ W[keep_mask, :]
        )
        clf_Z_after = train_mlp(X_clean_train, Z_train_idx, X_clean_test, 
                                Z_test_idx, num_classes=len(classes_Z), 
                                device=device, epochs=self.epochs)
        with torch.no_grad():
            preds_after = clf_Z_after(
                torch.as_tensor(X_clean_test, dtype=torch.float32, 
                device=device)
            ).argmax(1).cpu().numpy()
        acc_Z_after = accuracy_score(Z_test_idx, preds_after)
        
        # Store fitted attributes
        self.decomposer_ = decomposer
        self.W_ = W
        self.order_ = order
        self.keep_mask_ = keep_mask
        self.importance_Z_ = importance_Z
        self.importance_Y_ = importance_Y
        self.acc_Z_before_ = acc_Z_before
        self.acc_Z_after_ = acc_Z_after
        
        return self
    
    def transform(self, X):
        """
        Apply concept erasure to new data.
        
        Performs step (III) Concept Erasure: removes the k most Z-informative
        concepts learned during fit() and reconstructs the cleaned
        representation.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New representation matrix to transform.
        
        Returns
        -------
        X_clean : np.ndarray of shape (n_samples, n_features)
            Representation with erased concepts.
        """
        if self.decomposer_ is None:
            raise ValueError("TACO must be fitted before calling transform(). "
                           "Call fit() first.")
        
        if self.verbose:
            print(f"[3/3] Removing the {self.n_concepts_to_remove} most "
                  "Z-informative concepts...")
        
        # Decompose new data using the fitted decomposer
        U_new = decompose_new_data(X, self.decomposer_, self.W_)
        
        # Apply erasure using the learned keep_mask
        X_clean = U_new[:, self.keep_mask_] @ self.W_[self.keep_mask_, :]
        
        return X_clean
    
    def fit_transform(self, X, Z_sensitive, Y=None):
        """
        Fit the model and transform the training data in one step.
        
        Equivalent to calling fit() followed by transform() on the same data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Representation matrix to debias.
        Z_sensitive : np.ndarray of shape (n_samples,)
            Sensitive/protected attribute to erase.
        Y : np.ndarray of shape (n_samples,) or None, default=None
            Optional downstream task label to preserve while erasing Z.
        
        Returns
        -------
        X_clean : np.ndarray of shape (n_samples, n_features)
            Debiased representation with erased concepts.
        """
        self.fit(X, Z_sensitive, Y)
        return self.transform(X)
    
    def get_results(self):
        """
        Get diagnostics from the fitted model.
        
        Returns
        -------
        dict with keys:
            "keep_mask" : boolean mask over the r concepts (True = kept)
            "order" : concept indices, most-to-least removed
            "importance_Z" : Sobol importance of each concept for predicting Z
            "importance_Y" : Sobol importance of each concept for predicting Y (or None)
            "acc_Z_before" : accuracy of MLP predicting Z from original X
            "acc_Z_after" : accuracy of MLP predicting Z from erased X_clean
        """
        if self.W_ is None:
            raise ValueError("TACO must be fitted before calling get_results(). "
                           "Call fit() first.")
        
        return {
            "keep_mask": self.keep_mask_,
            "order": self.order_,
            "importance_Z": self.importance_Z_,
            "importance_Y": self.importance_Y_,
            "acc_Z_before": self.acc_Z_before_,
            "acc_Z_after": self.acc_Z_after_,
        }


# =============================================================================
# BACKWARD COMPATIBILITY: taco_erase function wrapper
# =============================================================================

def taco_erase(X, Z, Y=None, method_decompose="PCA", num_components=20,
                n_concepts_to_remove=3, sobol_nb_design=50, sobol_n_samples=2000,
                test_size=0.2, device=None, epochs=100, seed=0, verbose=True):
    """
    Full TaCo pipeline: erase the concept Z from representations X.
    
    DEPRECATED: Use the TACO class instead for a more flexible interface.

    Args
    ----
    X : np.ndarray (N, d)         representations to debias
    Z : np.ndarray (N,)           sensitive/protected attribute to erase
    Y : np.ndarray (N,) or None   optional downstream task label to preserve
                                   while erasing Z (recommended -- without it,
                                   TaCo just removes the concepts most
                                   predictive of Z, with no utility trade-off)
    method_decompose : "PCA" | "SVD" | "ICA"
    num_components   : total number of concepts r extracted by the decomposition
    n_concepts_to_remove : number k < r of concepts erased from X
    sobol_nb_design, sobol_n_samples : Sobol estimator hyper-parameters
                                   (higher = less variance, more compute)
    test_size         : held-out fraction used to report probe accuracies
    epochs            : training epochs for the MLP probes

    Returns a dict with:
        "X_clean"      : (N, d) representation with the k concepts erased
        "keep_mask"    : boolean mask over the r concepts (True = kept)
        "order"        : concept indices, most-to-least removed
        "importance_Z" : Sobol importance of each concept for predicting Z
        "importance_Y" : Sobol importance of each concept for predicting Y (or None)
        "acc_Z_before" : accuracy of an MLP predicting Z from the ORIGINAL X
        "acc_Z_after"  : accuracy of a freshly retrained MLP predicting Z
                          from X_clean (should be much lower / near chance)
    """
    taco = TACO(
        method_decompose=method_decompose,
        num_components=num_components,
        n_concepts_to_remove=n_concepts_to_remove,
        sobol_nb_design=sobol_nb_design,
        sobol_n_samples=sobol_n_samples,
        test_size=test_size,
        device=device,
        epochs=epochs,
        seed=seed,
        verbose=verbose
    )
    X_clean = taco.fit_transform(X, Z, Y)
    
    return {
        "X_clean": X_clean,
        **taco.get_results()
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Toy example: a synthetic representation X where one direction leaks the
    # sensitive attribute Z and another direction encodes a task Y we'd like
    # to keep predictable.
    rng = np.random.RandomState(0)
    N, d = 10000, 768
    Y_task = rng.randint(0, 4, size=N)      # 4-class downstream task
    Z_sensitive = rng.randint(0, 2, size=N)  # binary sensitive attribute

    X = rng.randn(N, d).astype(np.float32) * 0.5
    X[:, 0] += 3.0 * Z_sensitive             # dim 0 strongly leaks Z
    X[:, 1] += 1.5 * Y_task                  # dim 1 encodes the task

    print("=" * 70)
    print("Example 1: Using the TACO class with fit_transform()")
    print("=" * 70)
    
    taco = TACO(
        method_decompose="PCA",
        num_components=20,
        n_concepts_to_remove=3,
        epochs=50,
        verbose=True
    )
    
    X_clean = taco.fit_transform(X, Z_sensitive, Y=Y_task)
    results = taco.get_results()
    
    print(f"\nAccuracy predicting Z BEFORE erasure: {results['acc_Z_before']:.3f}")
    print(f"Accuracy predicting Z AFTER  erasure: {results['acc_Z_after']:.3f}")
    print(f"Concepts kept: {results['keep_mask'].sum()} / {len(results['keep_mask'])}")
    
    print("\n" + "=" * 70)
    print("Example 2: Using fit() and transform() separately")
    print("=" * 70)
    
    # Split data manually for demonstration
    from sklearn.model_selection import train_test_split
    X_train, X_test, Z_train, Z_test, Y_train, Y_test = train_test_split(
        X, Z_sensitive, Y_task, test_size=0.3, random_state=42
    )
    
    # Fit on training data
    taco2 = TACO(method_decompose="PCA", num_components=20,
                 n_concepts_to_remove=3, epochs=50, verbose=False)
    taco2.fit(X_train, Z_train, Y=Y_train)
    
    # Transform test data
    X_test_clean = taco2.transform(X_test)

    print("Majority class baseline for Z:", max(np.bincount(Z_train)) / len(Z_train))
    
    print(f"\nTrained on {X_train.shape[0]} samples")
    print(f"Transformed {X_test.shape[0]} test samples")
    print(f"X_clean shape: {X_test_clean.shape}")
    print(f"\nAccuracy predicting Z BEFORE erasure: {taco2.acc_Z_before_:.3f}")
    print(f"Accuracy predicting Z AFTER  erasure: {taco2.acc_Z_after_:.3f}")