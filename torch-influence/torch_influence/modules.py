import glob
import json
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse.linalg as L
from tqdm import tqdm
import torch
from torch import nn
from torch._tensor import Tensor
from torch.utils import data

from torch_influence.base import BaseInfluenceModule, BaseObjective


class AutogradInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    by directly forming and inverting the risk Hessian matrix using :mod:`torch.autograd`
    utilities.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        check_eigvals: if ``True``, this initializer checks that the damped risk Hessian
            is positive definite, and raises a :mod:`ValueError` if it is not. Otherwise,
            no check is performed.

    Warnings:
        This module scales poorly with the number of model parameters :math:`d`. In
        general, computing the Hessian matrix takes :math:`\mathcal{O}(nd^2)` time and
        inverting it takes :math:`\mathcal{O}(d^3)` time, where :math:`n` is the size
        of the training dataset.
    """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            check_eigvals: bool = False
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        d = flat_params.shape[0]
        hess = 0.0

        for batch, batch_size in self._loader_wrapper(train=True):
            def f(theta_):
                self._model_reinsert_params(self._reshape_like_params(theta_))
                return self.objective.train_loss(self.model, theta_, batch)

            hess_batch = torch.autograd.functional.hessian(f, flat_params).detach()
            hess = hess + hess_batch * batch_size

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)
            hess = hess / len(self.train_loader.dataset)
            hess = hess + damp * torch.eye(d, device=hess.device)

            if check_eigvals:
                eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
                logging.info("hessian min eigval %f", np.min(eigvals).item())
                logging.info("hessian max eigval %f", np.max(eigvals).item())
                if not bool(np.all(eigvals >= 0)):
                    raise ValueError()

            self.inverse_hess = torch.inverse(hess)

    def inverse_hvp(self, vec):
        return self.inverse_hess @ vec


class CGInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the method of (truncated) Conjugate Gradients (CG).

    This module relies :func:`scipy.sparse.linalg.cg()` to perform CG.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        **kwargs: keyword arguments which are passed into the "Other Parameters" of
            :func:`scipy.sparse.linalg.cg()`.
    """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            gnh: bool = False,
            **kwargs
    ):
        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )

        self.damp = damp
        self.gnh = gnh
        self.cg_kwargs = kwargs

    def inverse_hvp(self, vec):
        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        def hvp_fn(v):
            v = torch.tensor(v, requires_grad=False, device=self.device, dtype=vec.dtype)

            hvp = 0.0
            for batch, batch_size in self._loader_wrapper(train=True):
                hvp_batch = self._hvp_at_batch(batch, flat_params, vec=v, gnh=self.gnh)
                hvp = hvp + hvp_batch.detach() * batch_size
            hvp = hvp / len(self.train_loader.dataset)
            hvp = hvp + self.damp * v

            return hvp.cpu().numpy()

        d = vec.shape[0]
        linop = L.LinearOperator((d, d), matvec=hvp_fn)
        ihvp = L.cg(A=linop, b=vec.cpu().numpy(), **self.cg_kwargs)[0]

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return torch.tensor(ihvp, device=self.device)


class LiSSAInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the Linear time Stochastic Second-Order Algorithm (LiSSA).

    At a high level, LiSSA estimates an inverse-Hessian vector product
    by using truncated Neumann iterations:

    .. math::
        \mathbf{H}^{-1}\mathbf{v} \approx \frac{1}{R}\sum\limits_{r = 1}^R
        \left(\sigma^{-1}\sum_{t = 1}^{T}(\mathbf{I} - \sigma^{-1}\mathbf{H}_{r, t})^t\mathbf{v}\right)

    Here, :math:`\mathbf{H}` is the risk Hessian matrix and :math:`\mathbf{H}_{r, t}` are
    loss Hessian matrices over batches of training data drawn randomly with replacement (we
    also use a batch size in ``train_loader``). In addition, :math:`\sigma > 0` is a scaling
    factor chosen sufficiently large such that :math:`\sigma^{-1} \mathbf{H} \preceq \mathbf{I}`.

    In practice, we can compute each inner sum recursively. Starting with
    :math:`\mathbf{h}_{r, 0} = \mathbf{v}`, we can iteratively update for :math:`T` steps:

    .. math::
        \mathbf{h}_{r, t} = \mathbf{v} + \mathbf{h}_{r, t - 1} - \sigma^{-1}\mathbf{H}_{r, t}\mathbf{h}_{r, t - 1}

    where :math:`\mathbf{h}_{r, T}` will be equal to the :math:`r`-th inner sum.

    Args:
        model: the model of interest.
        objective: an implementation of :class:`BaseObjective`.
        train_loader: a training dataset loader.
        test_loader: a test dataset loader.
        device: the device on which operations are performed.
        damp: the damping strength :math:`\lambda`. Influence functions assume that the
            risk Hessian :math:`\mathbf{H}` is positive-definite, which often fails to
            hold for neural networks. Hence, a damped risk Hessian :math:`\mathbf{H} + \lambda\mathbf{I}`
            is used instead, for some sufficiently large :math:`\lambda > 0` and
            identity matrix :math:`\mathbf{I}`.
        repeat: the number of trials :math:`R`.
        depth: the recurrence depth :math:`T`.
        scale: the scaling factor :math:`\sigma`.
        gnh: if ``True``, the risk Hessian :math:`\mathbf{H}` is approximated with
            the Gauss-Newton Hessian, which is positive semi-definite.
            Otherwise, the risk Hessian is used.
        debug_callback: a callback function which is passed in :math:`(r, t, \mathbf{h}_{r, t})`
            at each recurrence step.
     """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            repeat: int,
            depth: int,
            scale: float,
            gnh: bool = False,
            debug_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            cache_name: Optional[str] = "influence_cache.json",
    ):

        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            cache_name=cache_name, 
        )

        self.damp = damp
        self.gnh = gnh
        self.repeat = repeat
        self.depth = depth
        self.scale = scale
        self.debug_callback = debug_callback

    def inverse_hvp(self, vec):

        params = self._model_make_functional()
        flat_params = self._flatten_params_like(params)

        ihvp = 0.0

        with tqdm(total=self.repeat) as pbar:
            for r in range(self.repeat):

                h_est = vec.clone()

                for t, (batch, _) in enumerate(self._loader_wrapper(sample_n_batches=self.depth, train=True)):
                    #if (t+1) % 50 == 0:
                    #    print(f"at the {t}th depth")

                    hvp_batch = self._hvp_at_batch(batch, flat_params, vec=h_est, gnh=self.gnh)

                    with torch.no_grad():
                        hvp_batch = hvp_batch + self.damp * h_est
                        h_est = vec + h_est - hvp_batch / self.scale

                    if self.debug_callback is not None:
                        self.debug_callback(r, t, h_est)

                ihvp = ihvp + h_est / self.scale
                pbar.update(1)

        with torch.no_grad():
            self._model_reinsert_params(self._reshape_like_params(flat_params), register=True)

        return ihvp / self.repeat


class IdentityInfluenceModule(BaseInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the identity Hessian.
     """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            cache_name: Optional[str] = "influence_cache.json",
    ):

        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            cache_name=cache_name
        )

    def inverse_hvp(self, vec):
        return vec


class LiSSAInfluenceModuleWithGradientStorage(LiSSAInfluenceModule):
    r"""An influence module that computes inverse-Hessian vector products
    using the identity Hessian.
     """

    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device,
            damp: float,
            repeat: int,
            depth: int,
            scale: float,
            gnh: bool = False,
            debug_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            cache_name: Optional[str] = "influence_cache.json",
            gradient_save_name: Optional[str] = "gradient_cache.json"
    ):

        super().__init__(
            model=model,
            objective=objective,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            damp=damp,
            repeat=repeat,
            depth=depth,
            scale=scale,
            gnh=gnh,
            debug_callback=debug_callback,
            cache_name=cache_name,
        )

        self.gradients = []
        self.gradient_save_name = gradient_save_name
        self.gradient_files = glob.glob(self.gradient_save_name)
        if not self.gradient_files:
            self.gradients = self.calculate_gradients(batch_size=1, train_idxs=range(len(self.train_dataloader.dataset)))

    def calculate_gradients(self, batch_size, train_idxs):
        gradients = []
        for grad_z, _ in self._loss_grad_loader_wrapper(batch_size=batch_size, subset=train_idxs, train=True):
            gradients.append(grad_z)
        return gradients

    def save_gradients_in_batches(self, batch_size, train_idxs, save_batch_size):
        accumulated_gradients = []
        count = 1
        with tqdm(total=len(train_idxs)) as pbar:
            for grad_z, _ in self._loss_grad_loader_wrapper(batch_size=batch_size, subset=train_idxs, train=True):
                accumulated_gradients.append(grad_z.cpu())

                # When batch_size is reached or it's the last data point, save the accumulated gradients
                if count % save_batch_size == 0 or count == len(train_idxs):
                    batch_index = count // save_batch_size
                    if count % save_batch_size != 0:  # Adjust for the last batch if it's smaller than batch_size
                        batch_index += 1
                    file_path = f"{self.gradient_save_name}_batch_{batch_index}.pt"
                    torch.save(torch.stack(accumulated_gradients), file_path)
                    print(f"Saved {len(accumulated_gradients)} gradients to {file_path}")

                    # Reset the accumulated gradients list for the next batch
                    accumulated_gradients = []
                pbar.update(1)
                count += 1

    def influences(self, train_idxs: List[int], test_idxs: List[int], stest: Tensor | None = None) -> Tensor:
        stest = self.stest(test_idxs) if (stest is None) else stest.to(self.device)
        scores = []
        num_chunks = 3

        json.dump(
            stest.tolist(),
            Path(self.cache_file_name).open('w'),
            indent=2
        )
        gradient_chunks = 1

        with tqdm(total=len(self.gradient_files)) as pbar:

            for chunk in range(1, num_chunks + 1):
                print(f"processing chunk {chunk}")
                batch_index = 1
                while True:
                    file_path = Path(self.gradient_save_name).parent / f"{chunk}chunk_batch_{batch_index}.pt"
                    if not file_path.exists():
                        break  # No more batches in this chunk
                    gradients = torch.load(str(file_path), map_location=stest.device)
                    
                    for grad_z in gradients.chunk(gradient_chunks):
                        batched_scores = grad_z@ stest
                        scores.append(batched_scores.cpu())
                    pbar.update(1)
                    batch_index += 1

        return torch.cat(scores) / len(self.train_loader.dataset)
