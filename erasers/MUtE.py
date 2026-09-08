import torch
from tqdm import tqdm
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier



class MarginalHistogramUniformization:
    """
    Performs marginal uniformization using PyTorch for GPU acceleration.
    Replaces scipy.stats.rv_histogram with torch-based interpolation.
    """
    name: str = "marghistuni"

    def __init__(self, alpha: float = 1e-10, bound_ext: float = 0.1, bins: int = 1000):
        self.alpha = alpha
        self.bound_ext = bound_ext
        self.bins = bins
        self.hists = None  # Stores CDF values for each dimension
        self.bin_edges = None
        self.device = None

    def fit(self, X: torch.Tensor):
        self.device = X.device
        _, n_features = X.shape
        self.hists = []
        self.bin_edges = []

        for i in range(n_features):
            iX = X[:, i]
            diff = iX.max() - iX.min()
            lower = iX.min() - self.bound_ext * diff
            upper = iX.max() + self.bound_ext * diff
            
            # Compute histogram on GPU
            counts = torch.histc(iX, bins=self.bins, min=lower.item(), max=upper.item())
            
            # Regularize PDF
            counts = counts + self.alpha
            
            # Compute CDF
            cdf = torch.cumsum(counts, dim=0)
            cdf = cdf / cdf[-1]
            # Prepended zero for interpolation
            cdf = torch.cat([torch.tensor([0.0], device=self.device), cdf])
            
            edges = torch.linspace(lower.item(), upper.item(), self.bins + 1, device=self.device)
            
            self.hists.append(cdf)
            self.bin_edges.append(edges)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        Q = torch.zeros_like(X)
        for i in range(X.shape[1]):
            # Use linear interpolation to find CDF value (analogous to scipy's cdf)
            Q[:, i] = self._interp(X[:, i], self.bin_edges[i], self.hists[i])
        return torch.clamp(Q, self.alpha, 1.0 - self.alpha)

    def inverse_transform(self, Q: torch.Tensor) -> torch.Tensor:
        X = torch.zeros_like(Q)
        for i in range(Q.shape[1]):
            # Invert interpolation (swap x and y)
            X[:, i] = self._interp(Q[:, i], self.hists[i], self.bin_edges[i])
        return X

    def _interp(self, x, xp, fp):
        """
        One-dimensional linear interpolation for monotonically increasing sample points.
        Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.
        
        Parameters
        ----------
        x : 1-D torch.Tensor
            The x-coordinates at which to evaluate the interpolated values.

        xp : 1-D sequence of floats
            The x-coordinates of the data points, must be increasing.

        fp : 1-D sequence of float or complex
            The y-coordinates of the data points, same length as `xp`.
        """
        # Find indices of bins
        idx = torch.searchsorted(xp, x.contiguous())
        idx = torch.clamp(idx, 1, len(xp) - 1)
        
        x_l, x_r = xp[idx-1], xp[idx]
        f_l, f_r = fp[idx-1], fp[idx]
        
        t = (x - x_l) / (x_r - x_l) 
        return f_l + t * (f_r - f_l)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class InverseGaussCDF:
    """GPU-accelerated Gaussian domain mapping using torch.distributions."""
    name: str = "invgausscdf"

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        # Standard Normal distribution in PyTorch
        self.normal = torch.distributions.Normal(0, 1)

    def fit(self, X=None, Z=None): return self

    def transform(self, X: torch.Tensor, Z=None) -> torch.Tensor:
        Q = torch.clamp(X, self.eps, 1.0 - self.eps)
        return self.normal.icdf(Q) # inverse cdf

    def inverse_transform(self, Q: torch.Tensor, Z=None) -> torch.Tensor:
        return self.normal.cdf(Q)

    def fit_transform(self, X, Z=None):
        return self.transform(X)

class PCARotation:
    """GPU PCA using torch.linalg.eigh."""
    name: str = "pca"

    def __init__(self):
        self.mean = None
        self.components = None

    def fit(self, X: torch.Tensor, Z=None):
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean
        # Compute Covariance matrix
        cov = torch.mm(X_centered.t(), X_centered) / (X.shape[0] - 1)
        # Eigen-decomposition (eigh is stable for symmetric matrices)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # Sort by eigenvalues descending
        idx = torch.argsort(eigenvalues, descending=True)
        self.components = eigenvectors[:, idx]

    def transform(self, X: torch.Tensor, Z=None) -> torch.Tensor:
        return torch.mm(X - self.mean, self.components)

    def inverse_transform(self, Q: torch.Tensor, Z=None) -> torch.Tensor:
        return torch.mm(Q, self.components.t()) + self.mean

    def fit_transform(self, X, Z=None):
        self.fit(X)
        return self.transform(X)



class RandomRotation:
    """GPU PCA using torch.linalg.eigh."""
    name: str = "random"

    def __init__(self):
        self.mean = None
        self.components = None

    def fit(self, X: torch.Tensor, Z=None):
        self.mean = torch.mean(X, dim=0)


        X_centered = X - self.mean
        # Compute Covariance matrix
        cov = torch.mm(X_centered.t(), X_centered) / (X.shape[0] - 1)
        # Eigen-decomposition (eigh is stable for symmetric matrices)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)


        dim = X.shape[1]
        device = X.device
        # 1. Generate a random square matrix from a standard normal distribution
        # This happens directly on the GPU.
        H = torch.randn(dim, dim, device=device)
        
        # 2. Perform QR decomposition
        # Q is an orthogonal matrix (Q^T * Q = I)
        Q, R = torch.linalg.qr(H)
        
        # 3. Correct for the signs to ensure a uniform distribution (Haar measure)
        # This prevents the distribution from being biased by the QR algorithm logic
        d = torch.diag(R).sign()
        Q *= d
        
        self.components = Q

    def transform(self, X: torch.Tensor, Z=None) -> torch.Tensor:
        return torch.mm(X - self.mean, self.components)

    def inverse_transform(self, Q: torch.Tensor, Z=None) -> torch.Tensor:
        return torch.mm(Q, self.components.t()) + self.mean

    def fit_transform(self, X, Z=None):
        self.fit(X)
        return self.transform(X)




class InverseRotation:
    name: str = "inverse_rotation"

    def __init__(self, R=None):
        self.mean = None
        self.components = R

    def fit(self, X: torch.Tensor, Z=None):
        self.mean = torch.mean(X, dim=0)
        if self.components is None:
            # If no rotation matrix is provided, we cannot fit an inverse rotation, so we just set components to identity.
            self.components = torch.eye(X.shape[1], device=X.device)
        return None

    def transform(self, X: torch.Tensor, Z=None) -> torch.Tensor:
        return torch.mm(X - self.mean, self.components.t()) # apply the inverse rotation (which is the transpose for orthogonal matrices)

    def inverse_transform(self, Q: torch.Tensor, Z=None) -> torch.Tensor:
        return torch.mm(Q, self.components) + self.mean

    def fit_transform(self, X, Z=None):
        self.fit(X)
        return self.transform(X)





class ClassConditionalMarginalHistogramUniformization:
    def __init__(self, alpha: float = 1e-10, bound_ext: float = 0.1, bins='auto'):
        self.cc_mapping = {}
        self.alpha = alpha
        self.bound_ext = bound_ext
        self.bins = bins if isinstance(bins, int) else 1000 # default to 1000 bins if 'auto' is specified, since torch.histc does not support 'auto' binning like numpy.histogram

    def fit(self, x, z):
        classes = torch.unique(z)
        for i in classes:
            label = i.item()
            self.cc_mapping[label] = MarginalHistogramUniformization(alpha=self.alpha, bound_ext=self.bound_ext, bins=self.bins)
            self.cc_mapping[label].fit(x[z == i])
        return self

    def transform(self, x, z):
        x_out = torch.zeros_like(x)
        for label, mapper in self.cc_mapping.items():
            mask = (z == label)
            if mask.any():
                x_out[mask] = mapper.transform(x[mask])
        return x_out

    def inverse_transform(self, x, z):
        x_out = torch.zeros_like(x)
        for label, mapper in self.cc_mapping.items():
            mask = (z == label)
            if mask.any():
                x_out[mask] = mapper.inverse_transform(x[mask])
        return x_out

    def fit_transform(self, x, z):
        self.fit(x, z)
        return self.transform(x, z)



class MUtE:
    def __init__(self, 
                 optimal_erasure=False, 
                 rotation='pca', 
                 nsteps=10, 
                 rotation_scheme='alternate',
                 alpha=1e-10, 
                 bound_ext=0.1, 
                 eps=1e-5, 
                 bins='auto', # 1000 by default 
                 max_samples_classifier_fit=10000,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 seed=42):
        self.optimal_erasure = optimal_erasure
        self.unique_classes = None
        self.mappings = []
        self.rotation = rotation # to remove
        self.nsteps = nsteps

        if (isinstance(rotation_scheme, str) and rotation_scheme in ['alternate', 'global']):
            self.rotation_scheme = rotation_scheme
        else:
            raise ValueError(f"Invalid rotation_scheme: {rotation_scheme}. Must be 'alternate' or 'global'.")
        
        if bins == 'auto':
            self.bins = 1000 # default to 1000 bins if 'auto' is specified, since torch.histc does not support 'auto' binning like numpy.histogram
        elif isinstance(bins, int):
            self.bins = bins
        else:
            raise ValueError(f"Invalid bins value: {bins}. Must be 'auto' or an integer.")
        
        self.alpha = alpha
        self.bound_ext = bound_ext
        self.eps = eps
        self.info_losses = [] # to remove
        
        
        self.device = device
        self.seed = seed
        

        # classifier to rout the class-conditional transformations, only used if optimal_erasure is set to False (i.e. under the empirical MUtE approach). 
        # This classifier will be trained on the original data before fitting the erasure model, and will be used to predict the class labels of the input data for the class-conditional transformations during the iterative gaussianization process. 
        # If optimal_erasure is set to True, we assume access to the ground truth class labels for the input data, and we rely on them for the class-conditional transformations, without training a class predictor.
        self.class_clf = None 
        self.max_num_samples_classifier_fit = max_samples_classifier_fit # to speed up the fitting of the class predictor, we will limit the number of samples used for fitting the class predictor to this number.
        self.class_clf_fitted = False # flag to indicate whether the class predictor has been fitted or not, to avoid refitting it multiple times if fit_transform is called multiple times with optimal_erasure set to False.

    def __str__(self):
        model_card = f'''
        MUtE (Maximum Utility-preserving Erasure) model with the following hyperparameters:
        Parameters:
        - optimal_erasure: {self.optimal_erasure} Whether to use optimal erasure (True) or empirical MUtE with a class predictor (False).
        - rotation: {self.rotation} Type of rotation to apply at each step ('pca' or 'random').
        - nsteps: {self.nsteps} Number of iterations (steps) in the MUtE process.
        - rotation_scheme: {self.rotation_scheme} Scheme for fitting rotations ('alternate' or 'global').
        - alpha: {self.alpha} Regularization parameter for histogram uniformization.
        - bound_ext: {self.bound_ext} Extension factor for histogram binning.
        - eps: {self.eps} Small constant to avoid numerical issues in inverse Gaussian CDF.
        - bins: {self.bins} Number of bins for histogram uniformization (or 'auto' for default).
        - max_samples_classifier_fit: {self.max_num_samples_classifier_fit} Maximum number of samples to use when fitting the class predictor (only if optimal_erasure is False).
        - device: {self.device} Device to use for computations ('cuda' or 'cpu').
        - seed: {self.seed} Random seed for reproducibility.
        - class_clf_fitted: {self.class_clf_fitted} Whether the class predictor has been fitted or not (internal flag).
        '''
        return model_card
        
    def fit_rotation(self, x, z, iteration):
        '''
        Fits the rotation for the current iteration based on the specified rotation scheme.

        Parameters:
            x: input data
            z: class labels
            iteration: current iteration number
        Returns:
            rot: the fitted rotation object
        '''
        rot = PCARotation() if self.rotation == 'pca' else RandomRotation()
        if self.rotation_scheme == 'alternate': 
            rot.fit(x[z==self.unique_classes[iteration % len(self.unique_classes)].item()])
        elif self.rotation_scheme == 'global':
            rot.fit(x)
        else:
            raise ValueError(f"Invalid rotation_scheme: {self.rotation_scheme}. Must be 'alternate' or 'global'.")
        return rot

    def _get_backend(self, x):
        '''
        Detects if the input is a torch tensor or numpy array.
        Parameters:
            x: input data
        Returns:
            the torch or numpy module depending on the type of x
        '''
        if torch is not None and torch.is_tensor(x):
            return torch
        return np
    
    def _get_device(self, x):
        '''
        Detects the device of the input tensor (CPU or GPU).
        If numpy array is provided, returns 'cpu' since numpy does not support GPU tensors.
        Parameters:
            x: input tensor
        Returns:
            the device of the input tensor
        '''
        if torch is not None and torch.is_tensor(x):
            return x.device
        return 'cpu'
    
    def _set_backend_and_device(self, x):
        '''
        Sets the backend (torch or numpy) and device (CPU or GPU) for the input data.
        If the input is a torch tensor, it will be moved to the specified device if it's not already on it.
        If the input is a numpy array, it will be converted to a torch tensor and moved to the specified device.

        Parameters:
            x: input data (torch tensor or numpy array)
        Returns:
            x: input data as a torch tensor on the correct device
            backend: the torch or numpy module depending on the type of x
            device: the device of the input tensor
        '''
        backend = self._get_backend(x)
        device = self._get_device(x)
        if backend == torch:
            if x.device != self.device:
                x = x.to(self.device)
        else:
            x = torch.tensor(x, device=self.device)
        return x, backend, device

    def _set_back_to_original_backend(self, x, backend, device):
        '''
        Converts the input tensor back to the original backend (torch or numpy) if necessary.
        If the original backend was torch, it returns the tensor as is.
        If the original backend was numpy, it converts the tensor back to a numpy array.

        Parameters:
            x: input data as a torch tensor
            backend: the original backend of the input data (torch or numpy)
            device: the original device of the input tensor
        Returns:
            x: input data in the original backend format
        '''
        if backend == torch and torch.is_tensor(x):
            if x.device != device:
                x = x.to(device)
            return x
        if backend == np and torch.is_tensor(x):
            return x.cpu().numpy()
        if backend == torch and not torch.is_tensor(x):
            return torch.from_numpy(x).to(device)
        return x
     


    def fit_transform(self, x: torch.Tensor, z: torch.Tensor, class_predictor=None):
        '''
        Fits the MUtE model to the input data (x, z) and returns the transformed data. 
        This method performs the full iterative gaussianization process, including fitting the class predictor 
        if optimal_erasure is set to False, and applying the sequence of transformations at each iteration.

        Parameters:
            x: input data (features)
            z: input data (class labels)
        Returns:
            x_: the transformed data after applying the MUtE process
        '''
        # get the backend and device for the input data and transform x to a torch tensor on the correct device if it's not already
        x_, x_backend, x_device = self._set_backend_and_device(x)
        z, _, _ = self._set_backend_and_device(z)

        if self.optimal_erasure == False: # we use the empirical MUtE approach, and we need to fit the class predictor if it hasn't been fitted yet
            if class_predictor is not None:
                self.set_class_predictor(class_predictor)
            else:
                self.fit_class_predictor(x_, z) 
            z_hat = self._get_z_hat(x_, z=None)
        else:
            z_hat = z.clone()
        z_hat = z_hat.to(self.device)

        # set the unique classes based on the input z
        self.unique_classes = torch.unique(z)

        iterations = tqdm(range(self.nsteps))
        for _, iteration in enumerate(iterations, 0):
            x_ = self.fit_one_iteration(x_, z, z_hat, iteration)
            iterations.set_description(f"Iteration {iteration+1}/{self.nsteps} complete.")


        return self._set_back_to_original_backend(x_, x_backend, x_device)

    def fit_one_iteration(self, x: torch.Tensor, z: torch.Tensor, z_hat: torch.Tensor, iteration = -1):
        '''
        Performs one step of the iterative gaussianization process, by adding one rotation, one marginal transformation, and one gaussianization.
        Parameters:
            x: input data (features) as a torch tensor on the correct device
            z: input data (class labels) as a torch tensor on the correct device
            z_hat: predicted class labels for the input data if optimal_erasure is False, or ground truth class labels if optimal_erasure is True (as a torch tensor on the correct device)
            iteration: current iteration number (used for fitting the rotation)
        Returns:
            x_: the transformed data after applying one step of the MUtE process
        '''
        x_, x_backend, x_device = self._set_backend_and_device(x)
        z, _, _ = self._set_backend_and_device(z)
        z_hat, _, _ = self._set_backend_and_device(z_hat)

        rot = self.fit_rotation(x_, z, iteration) # fit the first rotation before the loop to initialize the global rotation matrix
        x_ = rot.transform(x_)
        self.mappings.append(rot)
        # Add a marginal transformation + gaussianization at each iteration
        unif = ClassConditionalMarginalHistogramUniformization(alpha=self.alpha, bound_ext=self.bound_ext, bins=self.bins)
        unif.fit(x_, z) # fit the class-conditional marginal histogram uniformization on the current data x_ and ground truth class labels z
        x_ = unif.transform(x_, z_hat)  # apply the class-conditional marginal histogram uniformization using the predicted class labels z_hat if optimal_erasure is False, or ground truth class labels z if optimal_erasure is True
        self.mappings.append(unif)
        gauss = InverseGaussCDF(eps=self.eps)
        x_ = gauss.transform(x_)
        self.mappings.append(gauss)

        return self._set_back_to_original_backend(x_, x_backend, x_device)

    def fit(self, x: torch.Tensor, z: torch.Tensor, class_predictor=None):
        _ = self.fit_transform(x, z, class_predictor=class_predictor)

    def transform(self, x, z=None):
        x, x_backend, x_device = self._set_backend_and_device(x)
        if z is not None:
            z, _, _ = self._set_backend_and_device(z)
        
        z = self._get_z_hat(x, z)

        for m in self.mappings:
            if isinstance(m, (PCARotation, InverseGaussCDF, InverseRotation)):
                x = m.transform(x)
            else:
                x = m.transform(x, z)
        return self._set_back_to_original_backend(x, x_backend, x_device)

    def _get_z_hat(self, x, z=None):
        if self.optimal_erasure == False: # we use the empirical MUtE approach
            if z is not None:
                # raise a warning if z is provided but will not be used for the transform, since optimal erasure is set to False and the class predictor will be used to predict the classes of the input data for the class-conditional transformations.
                warnings.warn("optimal_erasure is set to False, under the empirical MUtE approach, the ground truth class labels z are assumed unavailable and won't be used. Set z to None if you don't want to disable this warning.")
            # we use the class predictor
            z_hat = self.class_predictor(x)
        else: # we use the optimal erasure approach MUtE (i.e. we assume access to the ground truth class labels for the input data, and we rely on them)
            if z is None:
                raise ValueError("optimal_erasure is set to True, no class predictor has been trained. Provide the target class labels z for transform.")
            z_hat = z
        return z_hat

    def inverse_transform(self, q, z_target):
        q, q_backend, q_device = self._set_backend_and_device(q)
        z_target, _, _ = self._set_backend_and_device(z_target)

        for m in reversed(self.mappings):
            if isinstance(m, (PCARotation, InverseGaussCDF, InverseRotation)):
                q = m.inverse_transform(q)
            else:
                q = m.inverse_transform(q, z_target)

        return self._set_back_to_original_backend(q, q_backend, q_device)
 
    

    def class_predictor(self, x):
        x_device = self._get_device(x)
        x_backend = self._get_backend(x)
        x = x.to('cpu') if x_backend == torch else x
        
        if self.class_clf is None:
            raise ValueError("No class predictor trained or provided. Please set or fit a class predictor.")
        z_hat = torch.from_numpy(self.class_clf.predict(x))
        return z_hat.to(x_device) if x_backend == torch else z_hat.numpy()
    
    def fit_class_predictor(self, x, z):
        x = x.to('cpu') if torch.is_tensor(x) else x
        z = z.to('cpu') if torch.is_tensor(z) else z

        self.class_clf = MLPClassifier(random_state=self.seed, max_iter=1000, early_stopping=True, validation_fraction=0.1)
        self.class_clf.fit(x[:self.max_num_samples_classifier_fit], z[:self.max_num_samples_classifier_fit])
        self.class_clf_fitted = True
        print(f"Class predictor trained with accuracy (on train set): {self.class_clf.score(x, z)}")

    def set_class_predictor(self, clf):
        if not hasattr(clf, 'predict'):
            raise ValueError("Provided class predictor does not have a 'predict' method.")
        self.class_clf = clf
        self.class_clf_fitted = True


  














