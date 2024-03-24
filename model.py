import torch
import numpy as np
import tensorly as tl
from tqdm import tqdm
from scipy.linalg import *


device = torch.device('cuda:0')

def ORPTC(M, omega, R=None, maxiter=None, tol=None):
    N = M.ndim
    omega_col0 = omega[:, 0]
    omega_col1 = omega[:, 1]
    omega_col2 = omega[:, 2]
    if N == 4:
        omega_col3 = omega[:, 3]

    if R is None:
        ru = int(0.1 * M.shape[0])
        ri = int(0.1 * M.shape[1])
        rt = int(M.shape[-1])
        if M.ndim == 3:
            R = [ru, ri, rt]
        elif M.ndim == 4:
            R = [ru, ri, ri, rt]
    if maxiter is None:
        maxiter = 3
    if tol is None:
        tol = 1e-2

    rse = np.zeros(maxiter)  # record the error
    Loop = 1
    print('Running orptc function ...')

    U_cell = [torch.zeros((M.shape[i], R[i]), requires_grad=False) for i in range(N)]
    M = M
    X_old = M.clone()  # X_old is tensor
    X = M.clone()

    while Loop:
        for iteration in tqdm(range(0, maxiter)):
        # for iteration in range(maxiter):
            for j in range(N):
                dim_ls = list(range(N))
                del dim_ls[j]
                M_rand_dim = int(torch.prod(torch.tensor([M.shape[dim] for dim in dim_ls])))
                M_rand = torch.rand(M_rand_dim, R[j], dtype=torch.float32)

                try:
                    Cmat = torch.matmul(torch.tensor(tl.base.unfold(X.cpu().numpy(), mode=j)).to(device), M_rand.to(device))
                except torch.cuda.OutOfMemoryError:
                    Cmat = torch.zeros((M.shape[j], R[j]))
                    # X_unfold_temp = torch.tensor(tl.base.unfold(X.cpu().numpy(), mode=j)).to(device)
                    X_unfold_temp = tl.base.unfold(X.cpu().numpy(), mode=j)
                    for n in range(X_unfold_temp.shape[0]):
                        try:
                            Cmat[n] = torch.matmul(torch.tensor(X_unfold_temp[n]).to(device), M_rand.to(device))
                        except torch.cuda.OutOfMemoryError:
                            Cmat[n] = torch.matmul(torch.tensor(X_unfold_temp[n]).cpu(), M_rand.cpu())

                u, _, _ = torch.svd(Cmat.to(device))
                U_cell[j].data = u[:, :R[j]] @ u[:, :R[j]].t()

            torch.cuda.empty_cache()

            for k in range(N):
                try:
                    X_unfold = torch.tensor(tl.base.unfold(X.cpu().numpy(), mode=k)).to(device)
                except torch.cuda.OutOfMemoryError:
                    X_unfold = torch.tensor(tl.base.unfold(X.cpu().numpy(), mode=k))
                # if N == 3:
                #     try:
                #         _prod = torch.zeros_like(X_unfold)
                #     except:
                #         _prod = torch.zeros_like(X_unfold).cpu()
                # if N == 4:
                #     _prod = torch.zeros_like(X_unfold).cpu()

                # if N == 3:
                #     for n in range(X_unfold.shape[0]):
                #         try:
                #             _prod[n] = torch.matmul(U_cell[k].data[n].to(device), X_unfold.to(device))
                #         except torch.cuda.OutOfMemoryError:
                #             _prod[n] = torch.matmul(U_cell[k].data[n].cpu(), X_unfold)
                # if N == 4:
                #     for n in range(X_unfold.shape[0]):
                #         _prod[n] = torch.matmul(U_cell[k].data[n].cpu(), X_unfold.cpu())

                try:
                    _prod = torch.zeros_like(X_unfold)
                except:
                    _prod = torch.zeros_like(X_unfold).cpu()
                for n in range(X_unfold.shape[0]):
                    try:
                        _prod[n] = torch.matmul(U_cell[k].data[n].to(device), X_unfold.to(device))
                    except torch.cuda.OutOfMemoryError:
                        _prod[n] = torch.matmul(U_cell[k].data[n].cpu(), X_unfold)

                X = torch.tensor(tl.base.fold(_prod.cpu().numpy(), k, M.shape))

            torch.cuda.empty_cache()

            if N == 3:
                X[omega_col0, omega_col1, omega_col2] = M[omega_col0, omega_col1, omega_col2]
            if N == 4:
                X[omega_col0, omega_col1, omega_col2, omega_col3] = M[omega_col0, omega_col1, omega_col2, omega_col3]

            # convergence condition
            try:
                rse[iteration] = torch.norm(X.to(device) - X_old.to(device)) / torch.norm(X_old.to(device))
            except torch.cuda.OutOfMemoryError:
                rse[iteration] = torch.norm(X - X_old) / torch.norm(X_old)
            # print_for_writing('rse:', rse[iteration])
            if (rse[iteration] < tol) or (iteration == maxiter-1):
                # print_for_writing('Convergence')
                Loop = 0
                break

            X_old = X.clone()
            torch.cuda.empty_cache()
    return X, rse


