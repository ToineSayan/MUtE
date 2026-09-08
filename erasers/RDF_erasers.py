import torch
from erasers import *
from tqdm import tqdm
import math

import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class RDF_loss(torch.nn.Module):
    """
    A generic class that defines RDF computation for use in various losses.
    """
    def __init__(self, eps_squared=0.5, sphere_radius=1.0):
        """
        eps (float): epsilon in RDF
        n_features (int): number of features
        sphere_radius (float): radius of the sphere on which representations are projected before RDF calculation
        """
        super().__init__()
        self.eps = math.sqrt(eps_squared)
        self.sphere_radius = sphere_radius

    def scale(self, X):
        n_features = X.shape[1]
        scale_fn = torch.nn.LayerNorm(normalized_shape=n_features, elementwise_affine=False)
        scale_coef = self.sphere_radius/math.sqrt(n_features)
        return scale_fn(X)*scale_coef # on the unit sphere

    def rate(self, X, n_total=-1):
        """
        Empirical Discriminative Loss.
        """
        device = X.device
        n, d = X.shape
        # In some decompositions of RDF calculation, the value associated with the 
        # number of observations n may differ from the number of observations in X
        if n_total > 0:
            n = n_total
        I = torch.eye(d).to(device)
        scalar = d / (n * (self.eps ** 2))
        logdet = torch.logdet(I + scalar * (X.T@X))
        return 0.5*logdet / math.log(2)
        

    def kernelized_rate(self, X, Z = None, kernel=None):
        device = X.device
        n, d = X.shape


        if Z is None:
            K = torch.ones((n,n)).to(device)
        else:
            K = torch.zeros((n,n)).to(device)
            for z_val in list(torch.unique(Z)):
                mask = Z == z_val  # Masque booléen pour Z
                condition = mask.unsqueeze(0) & mask.unsqueeze(1)

                if kernel is None:
                    K[condition] = 1.0
                elif kernel == 'rbf':
                    diff = X.unsqueeze(1) - X.unsqueeze(0)
                    norm = -0.5*torch.norm(diff, dim=2)**2
                    K[condition] = torch.exp(norm[condition])
        I = torch.eye(n).to(device)
        scalar = d / (n * (self.eps ** 2))
        logdet = (0.5 / math.log(2)) * torch.logdet(I + scalar * (X@X.T) * K)
        return logdet

    def forward(self, X, Z=None, n_total=-1):
        X = self.scale(X)
        if Z is not None:
            loss_value = sum([self.rate(X[Z==z], n_total=n_total) for z in torch.unique(Z)])
        else:
            loss_value = self.rate(X, n_total=n_total)
        self.loss_infos = {"loss":loss_value.item()}
        return loss_value

        


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------




class KRaM_Loss_categorical(RDF_loss):
    """
    Implementation of the KRaM loss for a categorical concept defined in the article 
    'Robust Concept Erasure via Kernelized Rate-Distortion Maximization' (eq. 4)

    R(f(X)|K) - lambda*|R(f(X)) - alpha*R(X)|

    Note: coef. alpha is not in the article
    """
    def __init__(self, lambda_=0.7):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, fX, X, Z, *args):
        # Scale the inputs before RDF calculation

        fX = self.scale(fX)
        X = self.scale(X)

        lRDF = self.kernelized_rate(fX,Z) 
        if self.lambda_ > 0.0:
            b = self.kernelized_rate(X) # R(X)
            RZ = self.kernelized_rate(fX) # R(f(X))
            control = abs(RZ-b)  
        else:
            control = torch.tensor(0.0)

        loss_full = lRDF - self.lambda_*control 
        return -loss_full # minus to maximize the loss during training




class FaRM_Loss_categorical(RDF_loss):
    """
    Implementation of the FaRM loss (unconstrained) for a categorical concept defined in the article 
    'Learning Fair Representations via Rate-Distortion Maximization' (eq. 3)

    R(f(X)|Pi^g) + R(f(X))
    """
    def __init__(self):
        super().__init__()

    def forward(self, fX, X, Z, **args):
        # Scale the inputs before RDF calculation
        fX = self.scale(fX)
        n = X.size(dim=0)
        lRDF = sum([len(Z[Z==z])*self.rate(fX[Z==z]) for z in torch.unique(Z)])/n # R(f(X)|Pi^g)
        RZ = self.rate(fX) # R(f(X))
        loss_full = lRDF + RZ
        return -loss_full # minus to maximize the loss during training



#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Training loop for FaRM and KRaM

def train(
    model,
    erasure_criterion,
    x,
    z,
    batch_size,
    learning_rate, 
    weight_decay,
    num_epochs,
    scheduler_milestones,
    scheduler_gamma,
    device,
    seed = None,
    **kwargs
):
    model = model.to(device)
    model.train()

    if seed is not None:
        torch.manual_seed(seed) 

    # Initialize the optimizer and the scheduler
    if len(list(model.parameters())) > 0:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)

    # Initialize the data loader
    # indices_X = torch.arange(n).to(device)
    dataset = TensorDataset(x, z)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) # drop_last to ignore incomplete batches

    # Training loop
    from tqdm import tqdm
    iterations = tqdm(range(num_epochs))
    
    for _, epoch in enumerate(iterations, 0):
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = erasure_criterion(outputs, inputs, labels) # KRaM

            loss.backward()
            optimizer.step() 

        scheduler.step() # update the scheduler at the end of each epoch





class FaRM:
    def __init__(self, io_size, hidden_size=None, num_layers=3, n_epochs=100, batch_size=512, learning_rate=1e-3, weight_decay=1e-5, scheduler_milestones=[], scheduler_gamma=0.1, seed=None, device=None):

        self.io_size = io_size
        self.hidden_size = hidden_size if hidden_size is not None else io_size
        self.num_layers = num_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_milestones = scheduler_milestones
        self.scheduler_gamma = scheduler_gamma
        self.seed = seed
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.model = MLP(
            input_size=self.io_size, 
            hidden_size=self.hidden_size,
            output_size=self.io_size,
            num_layers=self.num_layers,
            activation='relu',
            layer_norm=False,
            )
        
        self.loss_fn = FaRM_Loss_categorical()
        

    def fit_transform(self, x: torch.Tensor, z: torch.Tensor):
        self.fit(x, z)
        x_ = self.transform(x, z)
        return x_

    def fit(self, x: torch.Tensor, z: torch.Tensor):
        x, _, _ = self._set_backend_and_device(x)
        z, _, _ = self._set_backend_and_device(z)


        self.model = self.model.to(self.device)
        self.model.train()

        if self.seed is not None:
            torch.manual_seed(self.seed) 

        # Initialize the optimizer and the scheduler
        if len(list(self.model.parameters())) > 0:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.scheduler_milestones, gamma=self.scheduler_gamma)

        # Initialize the data loader
        # indices_X = torch.arange(n).to(device)
        dataset = TensorDataset(x, z)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) # drop_last to ignore incomplete batches

        # Training loop
        from tqdm import tqdm
        iterations = tqdm(range(self.n_epochs))
        
        for _, epoch in enumerate(iterations, 0):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs, labels
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, inputs, labels) # KRaM

                loss.backward()
                optimizer.step() 

            scheduler.step() # update the scheduler at the end of each epoch

        self.model.to('cpu')
        self.model.eval()
        return None

    def transform(self, x, z = None):
        x, x_backend, x_device = self._set_backend_and_device(x)
        x = x.to("cpu") if torch.is_tensor(x) else torch.from_numpy(x).to(self.device)
        self.model.eval()
        with torch.no_grad():
            x_ = self.model.forward(x)
        return self._set_back_to_original_backend(x_, x_backend, x_device)

    def inverse_transform(self, x, z):
        return None
    
    def generate_counterfactual(self, x, target_z, z_init = None):
        return None
    

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



class KRaM:
    def __init__(self, io_size, hidden_size=None, num_layers=3, n_epochs=100, batch_size=512, learning_rate=1e-3, weight_decay=1e-5, scheduler_milestones=[], scheduler_gamma=0.1, seed=None, device=None):

        self.io_size = io_size
        self.hidden_size = hidden_size if hidden_size is not None else io_size
        self.num_layers = num_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_milestones = scheduler_milestones
        self.scheduler_gamma = scheduler_gamma
        self.seed = seed
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = MLP(
            input_size=self.io_size, 
            hidden_size=self.hidden_size,
            output_size=self.io_size,
            num_layers=self.num_layers,
            activation='relu',
            layer_norm=False,
            )
        
        self.loss_fn = KRaM_Loss_categorical()
        

    def fit_transform(self, x: torch.Tensor, z: torch.Tensor):
        self.fit(x, z)
        x_ = self.transform(x, z)
        return x_

    def fit(self, x: torch.Tensor, z: torch.Tensor):

        x, _, _ = self._set_backend_and_device(x)
        z, _, _ = self._set_backend_and_device(z)

        self.model = self.model.to(self.device)
        self.model.train()

        if self.seed is not None:
            torch.manual_seed(self.seed) 

        # Initialize the optimizer and the scheduler
        if len(list(self.model.parameters())) > 0:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.scheduler_milestones, gamma=self.scheduler_gamma)

        # Initialize the data loader
        dataset = TensorDataset(x, z)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) # drop_last to ignore incomplete batches

        # Training loop
        from tqdm import tqdm
        iterations = tqdm(range(self.n_epochs))
        
        for _, epoch in enumerate(iterations, 0):
            for _, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs, labels
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, inputs, labels) # KRaM

                loss.backward()
                optimizer.step() 

            scheduler.step() # update the scheduler at the end of each epoch

        self.model.to('cpu')
        self.model.eval()

        return None

    def transform(self, x, z = None):
        x, x_backend, x_device = self._set_backend_and_device(x)
        x = x.to("cpu") if torch.is_tensor(x) else torch.from_numpy(x).to(self.device)
        self.model.eval()
        with torch.no_grad():
            x_ = self.model.forward(x)
        return self._set_back_to_original_backend(x_, x_backend, x_device)

    def inverse_transform(self, x, z):
        return None
    
    def generate_counterfactual(self, x, target_z, z_init = None):
        return None

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




if __name__ == "__main__":
    # Example usage
    x = torch.randn(1000, 20)  # 1000 samples, 20 features
    z = torch.randint(0, 5, (1000,))  # 5 classes

    farm = FaRM(io_size=20, hidden_size=50, num_layers=3, n_epochs=10)
    x_erased = farm.fit_transform(x, z)


    print("Difference between original and erased representations:", torch.norm(x - x_erased).item())