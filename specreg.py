import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralRegularizer(nn.Module):

    def __init__(
        self,
        model: nn.Module,
        k: int = 3,
        n_power_iterations: int = 1,
        eps: float = 1e-12,
        store_vectors: bool = False,
    ):
        super().__init__()
        self.model = model
        self.k = k
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.store_vectors = store_vectors

        self._cached_uv = {}
        
    def forward(self) -> torch.Tensor:
        reg_loss = torch.zeros(1, device=next(self.model.parameters()).device)

        for module in self.model.modules():
            if len(list(module.children())) > 0:
                continue

            W = getattr(module, 'weight', None)
            b = getattr(module, 'bias', None)

            if W is not None:
                if W.dim() >= 2:
                    W_mat = W.reshape(W.shape[0], -1)
                    sigma = self._power_iter_singular_value(W_mat)
                    reg_loss += ( (sigma**self.k) - 1.0 )**2
                else:

                    w_norm = torch.norm(W, 2)
                    reg_loss += (w_norm**(2 * self.k))
                  
            if b is not None:
                # The "spectral norm" of a vector is its L2 norm
                b_norm = torch.norm(b, 2)
                # => (||b||_2^(2k)).
                reg_loss += (b_norm**(2 * self.k))

        return reg_loss

    def _power_iter_singular_value(self, W_mat: torch.Tensor) -> torch.Tensor:
        # Identify param by its "data_ptr" or by object id
        param_id = W_mat.data_ptr()
        
        with torch.no_grad():
            # If storing vectors, try to fetch them from the cache
            if self.store_vectors and param_id in self._cached_uv:
                u, v = self._cached_uv[param_id]
                # Validate shapes
                if u.shape[0] != W_mat.shape[0]:
                    u = None
                if v.shape[0] != W_mat.shape[1]:
                    v = None
            else:
                u, v = None, None

            if u is None or v is None:
                u = torch.randn(W_mat.size(0), device=W_mat.device, dtype=W_mat.dtype)
                v = torch.randn(W_mat.size(1), device=W_mat.device, dtype=W_mat.dtype)

            u = F.normalize(u, dim=0, eps=self.eps)
            v = F.normalize(v, dim=0, eps=self.eps)

            # Power iteration
            for _ in range(self.n_power_iterations):
                # v <- W^T u
                v = torch.mv(W_mat.t(), u)
                v_norm = v.norm(2)
                v = v / (v_norm + self.eps)

                # u <- W v
                u = torch.mv(W_mat, v)
                u_norm = u.norm(2)
                u = u / (u_norm + self.eps)
            sigma = torch.norm(torch.mv(W_mat, v), 2)

            # Cache (u, v) if desired
            if self.store_vectors:
                self._cached_uv[param_id] = (u, v)

        return sigma
