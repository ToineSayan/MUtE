import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np




class RBF(torch.nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth_heuristic=None): 
        """
        Parameters
        ----------
        n_kernels : int, default=5
            Number of kernels in the mixture.
        
        mul_factor : int, default=2
            Multiplicative factor used to calculate the bandwith multipliers when relying on a heuristic estimation for the bandwidth.
            The multipliers are mul_factor^i with i ranging from [-⌊n_kernels /2⌋, ..., n_kernels-1-⌊n_kernels /2⌋].
            This parameter has no effect if `bandwidth` is set to ``None``

        bandwidth_heuristic : str, default=None
            Defines the bandwith heuristic to use among ["median", "mean"].
            Set it to None to use custom bandwidths.
        """
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2) 
        self.bandwidth_heuristic = bandwidth_heuristic
        self.current_bandwidths = None 

    def get_bandwidth(self, L2_distances):
        """
        Parameters
        ----------
        L2_distances : tensor
            number of bandwidth 
        """
        if self.bandwidth_heuristic == "median":
            return L2_distances[L2_distances > 0].median()
        elif self.bandwidth_heuristic == "mean":
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        elif self.bandwidth_heuristic is None:
            return None

    def forward(self, X, log_bw=None):
        """
        Parameters
        ----------
        X : tensor
            data
        
        bw : tensor, default=None
            bandwidths to use when not relying on a bandwidth heuristic 
        """
        L2_distances = torch.cdist(X, X) ** 2
        if self.bandwidth_heuristic is not None:
            bw = self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(X.device)
        else:
            bw = torch.nn.functional.softplus(log_bw) # softplus for positivity
        return torch.exp(-L2_distances[None, ...] / (bw)[:, None, None]).sum(dim=0)





class MMDLoss(torch.nn.Module):
    
    def __init__(self, kernel_type='RBF', n_kernels=5, bandwidth_heuristic =None):
        """
        Parameters
        ----------
        kernel_type : str, default="RBF"
            Type of kernel to use to calculate the MMD. Only RBF implemented for now.
            RBF: Radial Basis Function, a universal kernel.
        
        n_kernels : int, default=5,
            Number of kernels in the mixture.

        bandwidth_heuristic : str, default=None
            Defines the bandwith heuristic among ["median", "mean"] to use along with the RBF kernel.
            Set it to None to use custom bandwidths.
        """
        super().__init__()
        if kernel_type == "RBF":
            self.kernel_bandwidth = bandwidth_heuristic
            self.n_kernels = n_kernels
            self.kernel = RBF(n_kernels=n_kernels, bandwidth_heuristic=bandwidth_heuristic)



    def forward(self, X, Y, log_bw=None, bw=None):
        """
        Calculation of the unbiased MMD between two distributions P and Q using samples.

        Parameters
        ----------
        X : tensor of shape (n,d)
            Set of samples drawn from a distribution P.
        
        Y : tensor of shape (m,d)
            Set of samples drawn from a distribution Q.

        Returns:
        ----------
        Unbiased MMD estimation between P and Q.
        """
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        n, m = X_size, K.shape[0] - X_size

        XX = K[:X_size, :X_size].sum()/(n*(n-1))
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].sum()/(m*(m-1))
        return torch.sqrt(XX - 2 * XY + YY)







class PairwiseQuadraticMMD(MMDLoss):
    """
    Pairwise MMD Loss
    """
    def __init__(self, kernel_type='RBF', n_kernels=5, bandwidth_heuristic=None):
        """
        Parameters
        ----------
        kernel_type : str, default="RBF"
            Type of kernel to use to calculate the MMD. Only RBF implemented for now.
            RBF: Radial Basis Function, a universal kernel.
        
        n_kernels : int, default=5,
            Number of kernels in the mixture.

        bandwidth_heuristic : str, default=None
            Defines the bandwith heuristic among ["median", "mean"] to use along with the RBF kernel.
            Set it to None to use custom bandwidths.
        """
        super().__init__(kernel_type, n_kernels, bandwidth_heuristic)

    def forward(self, fX, Z, log_bw = None, log_loss=False):
        distributions_ids = torch.unique(Z)
        couples = [(distributions_ids[i], distributions_ids[j]) for i in range(len(distributions_ids)) for j in range(i+1, len(distributions_ids))]
        
        if not log_loss:
            loss_mmd = sum([super().forward(fX[Z == z1], fX[Z == z2], log_bw)**2 for z1,z2 in couples])/len(couples)
        else:
            L2_distances = torch.cdist(fX, fX) ** 2
            bw = L2_distances[L2_distances > 0].median()


            # l = [torch.log(super().forward(fX[Z == z1], fX[Z == z2], log_bw)**2 + 1e-10) for z1,z2 in couples]

            l = [torch.log(super().forward(fX[Z == z1], fX[Z == z2], bw=bw)**2 + 1e-10) for z1,z2 in couples]



            loss_mmd = sum(l)/len(couples) 

        # loss_mmd = sum([super().forward(fX[Z == z1], fX[Z == z2], log_bw)**2 for z1,z2 in couples])/len(couples)
        return loss_mmd




class nonLEOPARD(nn.Module):
    
    def __init__(
            self, 
            input_size, 
            rank, # rank of the projection to learn

            batch_size = 'auto', # if auto, batch size in the number of samples (full batch learning) 
            learning_rate = 1.0e-3, # initial learning rate
            num_epochs = 1000,
            scheduler_milestones = [],
            scheduler_gamma = 0.1,
            cascaded_training = False,
            device = 'cuda' if torch.cuda.is_available() else 'cpu', 
            seed = None,

            learnable_sigmas = False, # True if bandwidth parameters sigma for RBF calculation in MMD are learned
            # init_median_evaluation_samples = None, # Number of samples used to evaluate the median value of the sigmas
            num_sigmas = 5, # number of kernels in the mixture for MMD
            sigma_heuristic = "mean",
            gamma = 100, # gamma coefficient for loss calculation
            use_log_erasure_loss = False, # use the log of the erasure loss in the calculation of the loss
            
            X_init =  None, # Mandatory if cascaded training to calculate the orthogonal projection to erase the linear signal or if sigmas are learnable to init the bandwidth parameters.
            Z_init = None, # Mandatory if cascaded training 

            ):
        super().__init__()
        self.input_size = input_size 
        self.rank = rank
        # training params
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.scheduler_milestones = scheduler_milestones
        self.scheduler_gamma = scheduler_gamma
        self.cascaded_training = cascaded_training
        self.device = device
        self.seed = seed
        # loss calculation params
        self.num_sigmas = num_sigmas
        self.sigma_heuristic = sigma_heuristic
        self.gamma = gamma
        self.use_log_erasure_loss = use_log_erasure_loss


        # Sanity checks
        # Check if cascaded training is enabled and if X_init and Z_init are provided
        if cascaded_training and (X_init is None or Z_init is None):
            raise ValueError("X_init and Z_init must be provided for cascaded training.")
        # Check if the rank is less than the input size if not cascaded training.
        if not cascaded_training and rank > input_size:
            raise ValueError("Rank must be less than or equal to input size for non-cascaded training.")
        # Check if the rank is less than the input size minus 'the number of unique labels in Z_init minus 1' if cascaded training.
        if cascaded_training:
            num_unique_labels = len(torch.unique(Z_init)) if torch.is_tensor(Z_init) else len(np.unique(Z_init))
            if rank > input_size - (num_unique_labels - 1):
                raise ValueError(f"Rank must be less than or equal to input size minus 'the number of unique labels in Z_init minus 1' (here {input_size - (num_unique_labels - 1)}) for cascaded training.")


        if X_init is not None and Z_init is not None:
            X_init, _, _ = self._set_backend_and_device(X_init)
            Z_init, _, _ = self._set_backend_and_device(Z_init)

        if cascaded_training:
            self.Ep_lin = self.linear_erasure(X_init, Z_init)
            self.U = nn.Parameter(torch.eye(self.Ep_lin.shape[1], rank, requires_grad=True))
        else:
            self.Ep_lin = None
            self.U = nn.Parameter(torch.eye(input_size, rank, requires_grad=True))
        self.U_final = None

        self.pq_mmd = PairwiseQuadraticMMD(kernel_type="RBF", n_kernels=num_sigmas, bandwidth_heuristic=sigma_heuristic)
        

    def erasure_loss(self, x_erased, labels):
        """
        Compute the erasure loss.

        Args:
            x_erased (torch.Tensor): The erased representations.
            labels (torch.Tensor): The corresponding labels.
        Returns:
            torch.Tensor: The computed erasure loss.
        """
        erasure_loss = self.pq_mmd(x_erased, labels, log_bw=None, log_loss=False) 
        return torch.log(erasure_loss) if self.use_log_erasure_loss else erasure_loss
    
    def projection_loss(self):
        """
        Compute the projection loss.
        """
        return torch.norm(self.U.T @ self.U - torch.eye(self.rank, device=self.U.device), p=2)**2

        
    def get_loss(self, x_erased, labels):
        """
        Compute the total loss, including erasure and projection losses.

        Args:
            x_erased (torch.Tensor): The erased representations.
            labels (torch.Tensor): The corresponding labels.
        Returns:
            tuple: A tuple containing the total loss, erasure loss, projection loss.
        """
        erasure_loss = self.erasure_loss(x_erased, labels)
        projection_loss = self.projection_loss()
        loss = erasure_loss + (self.gamma/(self.rank**2))*projection_loss

        return loss, erasure_loss, projection_loss
    

    def post_training_update(self):
        """
        After training, update the projection matrix P to ensure it is orthogonal and idempotent.
        """
        with torch.no_grad():
            # Update P
            _, U = torch.linalg.eigh(self.U @ self.U.T) # use eigh because UU.T is symmetrical / eigenvalues are sorted in ascending order
            self.U_final = U[:,-self.rank:]
            
            P = self.get_projector()
            print('Symmetry ||P - P.T||:', torch.norm(P - P.T))
            print('Idempotence ||P @ P - P||:', torch.norm(P @ P - P))
            P_approx = self.get_projector(use_approx=True)
            print('Closeness to P_approx ||P - U@U.T||:', torch.norm(P-P_approx)) 

    def fit(self, X, Z):
        """
        Fit the model to the given data. 
        Learns the projection matrix that erases the information about the labels from the input representations.

        Args:
            X (torch.Tensor): The input representations.
            Z (torch.Tensor): The corresponding labels.
        Returns:
            None
        """
        X, _, _ = self._set_backend_and_device(X)
        Z, _, _ = self._set_backend_and_device(Z)

        # Erase the linear signal if cascaded training is enabled
        X = X @ self.Ep_lin if self.cascaded_training else X
    
        self = self.to(self.device)
        self.train()

        if self.seed is not None:
            torch.manual_seed(self.seed) 


        # Initialize the optimizer and the scheduler
        if len(list(self.parameters())) > 0:
            # weight decay is 0.0 in nonLEOPARD not to interfere with the orthogonality constraint of the projection matrix.
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0) 
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.scheduler_milestones, gamma=self.scheduler_gamma)

        # Initialize the data loader
        batch_size = len(X) if self.batch_size == "auto" else self.batch_size
        dataset = TensorDataset(X, Z)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


        # Training loop
        from tqdm import tqdm
        iterations = tqdm(range(self.num_epochs))
        
        for _, epoch in enumerate(iterations, 0):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = inputs @ self.U @ self.U.T
                loss, erasure_loss, projection_loss = self.get_loss(outputs, labels)

                loss.backward()
                optimizer.step() 

                # Training status display
                iterations.set_description(f"[train] loss: {loss}, erasure loss: {erasure_loss}, projection loss: {projection_loss}")
            scheduler.step() # update the scheduler at the end of each epoch

        self.eval()
        # After training, update the projection matrix P to ensure it is orthogonal and idempotent.
        self.post_training_update()
        self = self.to('cpu')



    def linear_erasure(self, X, Z):
        """
        Erases the linear signal from the input representations using LEACE.

        Args:
            X (torch.Tensor): The input representations.
            Z (torch.Tensor): The corresponding labels.
        Returns:
            Ep (torch.Tensor): The projection matrix that erases the linear signal.
        """
        from concept_erasure import LeaceEraser
        Z_1hot = torch.nn.functional.one_hot(Z)
        P_leace = LeaceEraser.fit(X, Z_1hot).P # oblique projection 
        _,_,Vh = torch.linalg.svd(P_leace)
        Ep = Vh[:-Z_1hot.shape[1]+1,:].T 
        return Ep
        

    def forward(self, X, use_approx=False):
        """
        Forward pass of the model. Projects the input representations onto the learned subspace.
        
        Args:
            X (torch.Tensor): The input representations.
            use_approx (bool): Whether to use the approximate projector (U @ U.T)
                               or the final projector (U_final @ U_final.T) for the projection.
        Returns:
            torch.Tensor: The projected input representations.
        """
        X, x_backend, x_device = self._set_backend_and_device(X)
        X_ = X @ self.get_projector(use_approx=use_approx).to(X.device)
        return self._set_back_to_original_backend(X_, x_backend, x_device)
    
    def transform(self, X, use_approx=False):
        """
        Forward pass of the model. Projects the input representations onto the learned subspace.

        Args:
            X (torch.Tensor): The input representations.
            use_approx (bool): Whether to use the approximate projector (U @ U.T)
                               or the final projector (U_final @ U_final.T) for the projection.
        Returns:
            torch.Tensor: The projected input representations.
        """
        return self.forward(X, use_approx=use_approx)
    
    def get_projector(self, use_approx=False):
        """
        Returns the projection matrix.

        Args:
            use_approx (bool): Whether to return the approximate projector (U @ U.T)
                               or the final projector (U_final @ U_final.T).
        Returns:
            torch.Tensor: The projection matrix (U @ U.T or U_final @ U_final.T).
        """
        # Use U_final by default if it has been calculated
        if (self.U_final is not None) and not use_approx:
            U = self.U_final
        else:
            U = self.U
        # If cascaded setting, first project in a subspace with no linear signal
        if self.Ep_lin is not None:
            U = self.Ep_lin.to(U.device) @ U
                   
        return U @ U.T
    
    def count_parameters(self):
        """
        Returns the number of trainable parameters in the model.

        Returns:
            int: The number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _get_backend(self, x):
        '''
        Detects if the input is a torch tensor or numpy array.

        Args:
            x (torch.Tensor or np.ndarray): input data
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

        Args:
            x (torch.Tensor or np.ndarray): input data
        Returns:
            torch.device or str: the device of the input tensor
        '''
        if torch is not None and torch.is_tensor(x):
            return x.device
        return 'cpu'
    
    def _set_backend_and_device(self, x):
        '''
        Sets the backend (torch or numpy) and device (CPU or GPU) for the input data.
        If the input is a torch tensor, it will be moved to the specified device if it's not already on it.
        If the input is a numpy array, it will be converted to a torch tensor and moved to the specified device.

        Args:
            x (torch.Tensor or np.ndarray): input data
        Returns:
            x (torch.Tensor): input data as a torch tensor on the correct device
            backend (module): the torch or numpy module depending on the type of x
            device (torch.device or str): the device of the input tensor
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

        Args:
            x (torch.Tensor): input data as a torch tensor
            backend (module): the original backend of the input data (torch or numpy)
            device (torch.device or str): the original device of the input tensor
        Returns:
            x (torch.Tensor or np.ndarray): input data in the original backend format
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





if __name__ == "__main__":
    input_size = 500
    X= torch.randn(10000, input_size)
    Z = torch.randint(0, 3, (10000,))
    # Example usage
    rank = 495
    model = nonLEOPARD(
        input_size=input_size, 
        rank=rank,
        num_epochs=100,
        batch_size=4096,
        learning_rate=1e-3,
        sigma_heuristic="median",
        cascaded_training=True,
        X_init=X,
        Z_init=Z
    )
    print("Number of trainable parameters:", model.count_parameters())
    
    model.fit(X, Z)

    







