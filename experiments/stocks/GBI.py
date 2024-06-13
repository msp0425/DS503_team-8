import torch
import cvxpy as cp
import numpy as np




def create_cvxpy_problem(
        r_upper,
        r_lower,
        r_mean
    ):

        horizon = 19
        num_asset = 3
        params = init_GBI_params()
        eta_var = cp.Variable((horizon,num_asset))
        zeta_var = cp.Variable((horizon,num_asset))
        xi_var = cp.Variable((horizon+1,num_asset+1))
        c_var = cp.Variable((horizon))

        constraints = [eta_var >= 0, 
                       zeta_var >= 0, 
                       xi_var >= 0,
                       c_var >=0,
                       c_var <= params['Goal1'],
                       xi_var[1:,:-1] == xi_var[:-1,:-1] - eta_var + zeta_var, 
                       xi_var[0,-1] == 100,
                       xi_var[0,:-1] == 0,
                       cp.max(xi_var[1:,:], axis = 1) <= cp.sum(xi_var[1:, :], axis = 1)*0.5,
                       xi_var[1:,-1] <= xi_var[:-1,-1]
                                        + cp.sum((1-params['mu']) * cp.multiply(((1+r_lower)*torch.cumprod(1+r_mean, dim=0)),
                                        eta_var/torch.cumprod((params['Rf']+1) * torch.ones([20,3]),axis=0)[:-1,:])
                                        - (1+params['nu']) * cp.multiply(((1+r_upper)*torch.cumprod(1+r_mean, dim=0)),
                                        zeta_var/torch.cumprod((params['Rf']+1) * torch.ones([20,3]),axis=0)[:-1,:]),axis=1)
                                        - (c_var/torch.cumprod(((params['Rf']+1) * torch.ones([19])),axis=0))]
        objective = cp.Maximize( cp.sum(c_var/torch.cumprod(((params['Rf']+1) * torch.ones([19])),axis=0)))
        problem = cp.Problem(objective, constraints)

        result_1 = problem.solve(solver=cp.ECOS)
        c_0 = c_var.value


        eta_var2 = cp.Variable((horizon,num_asset))
        zeta_var2 = cp.Variable((horizon,num_asset))
        xi_var2 = cp.Variable((horizon+1,num_asset+1))
        c_var2 = cp.Variable((horizon))
        c_var1 = torch.tensor(c_0)

        constraints2 = [eta_var2 >= 0, 
                       zeta_var2 >= 0, 
                       xi_var2 >= 0,
                       c_var2 <= params['Goal2'] + c_var1,
                       c_var2 >= c_var1,
                       xi_var2[1:,:-1] == xi_var2[:-1,:-1] - eta_var2 + zeta_var2, 
                       xi_var2[0,-1] == 100,
                       xi_var2[0,:-1] == 0,
                       cp.max(xi_var2[1:,:], axis = 1) <= cp.sum(xi_var2[1:, :], axis = 1)*0.5,
                       xi_var2[1:,-1] <= xi_var2[:-1,-1]
                                        + cp.sum((1-params['mu']) * cp.multiply(((1+r_lower)*torch.cumprod(1+r_mean, dim=0)),
                                        eta_var2/torch.cumprod((params['Rf']+1) * torch.ones([20,3]),axis=0)[:-1,:])
                                        - (1+params['nu']) * cp.multiply(((1+r_upper)*torch.cumprod(1+r_mean, dim=0)),
                                        zeta_var2/torch.cumprod((params['Rf']+1) * torch.ones([20,3]),axis=0)[:-1,:]),axis=1)
                                        - (c_var2/torch.cumprod(((params['Rf']+1) * torch.ones([19])),axis=0))]
        objective2 = cp.Maximize( cp.sum(c_var2/torch.cumprod(((params['Rf']+1) * torch.ones([19])),axis=0)))
        problem2 = cp.Problem(objective2, constraints2)  

        result_2 = problem2.solve(solver=cp.ECOS)




        return result_2 , c_var2.value, eta_var2.value, zeta_var2.value, xi_var2.value

def init_GBI_params():
        params = {}

        # Ordering costs
        params['mu'] = 0.05
        params['nu'] = 0.05

        # Goal amounts
        params['Goal1'] = torch.tensor([0, 10, 0, 0, 10, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0])
        params['Goal2'] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 10, 0,0,0,0,0,100,0,0,0,0])

        # Invest amount
        params['Invest'] = torch.tensor([100, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0])

        # Risk free
        params['Rf'] = torch.tensor(0.01)

        return params

def get_objective(
        r, Z
    ):
        horizon = 19
        num_asset = 3
        params = init_GBI_params()
        eta_var3, zeta_var3, xi_var3_ = Z
        eta_var3 = torch.tensor(eta_var3)
        zeta_var3 = torch.tensor(zeta_var3)
        xi_var3_ = torch.tensor(xi_var3_)
        c_var3 = cp.Variable((horizon))
        xi_var3 = cp.Variable((horizon+1,num_asset+1))

        constraints3 = [c_var3 <= params['Goal1'],
                        c_var3 >= 0,
                        xi_var3[:,:-1] == xi_var3_[:,:-1],
                        xi_var3[0, -1] == 100,
                        xi_var3 >= 0,
                       xi_var3[1:,-1] <= xi_var3[:-1,-1]
                                        + cp.sum((1-params['mu']) * cp.multiply(((1+r)*torch.cumprod(1+r, dim=0)),
                                        eta_var3/torch.cumprod((params['Rf']+1) * torch.ones([20,3]),axis=0)[:-1,:])
                                        - (1+params['nu']) * cp.multiply(((1+r)*torch.cumprod(1+r, dim=0)),
                                        zeta_var3/torch.cumprod((params['Rf']+1) * torch.ones([20,3]),axis=0)[:-1,:]),axis=1)
                                        - (c_var3/torch.cumprod(((params['Rf']+1) * torch.ones([19])),axis=0))]
        objective3 = cp.Maximize( cp.sum(c_var3/torch.cumprod(((params['Rf']+1) * torch.ones([19])),axis=0)))
        problem3 = cp.Problem(objective3, constraints3)

        result_3 = problem3.solve(solver=cp.ECOS)
        if problem3.status == 'infeasible':
            return torch.tensor([0])
        elif problem3.status.upper() == 'UNKNOWN':
            return torch.tensor([0])

        c_1 = c_var3.value
        if c_1 is None:
            return torch.tensor([0])
        c_var4 = cp.Variable((horizon))
        xi_var4 = cp.Variable((horizon+1,num_asset+1))

        constraints4 = [c_var4 <= params['Goal2'] + c_1,
                        c_var4 >= c_1,
                        xi_var4[:,:-1] == xi_var3_[:,:-1],
                        xi_var4[0, -1] == 100,
                        xi_var4 >= 0,
                       xi_var4[1:,-1] <= xi_var4[:-1,-1]
                                        + cp.sum((1-params['mu']) * cp.multiply(((1+r)*torch.cumprod(1+r, dim=0)),
                                        eta_var3/torch.cumprod((params['Rf']+1) * torch.ones([20,3]),axis=0)[:-1,:])
                                        - (1+params['nu']) * cp.multiply(((1+r)*torch.cumprod(1+r, dim=0)),
                                        zeta_var3/torch.cumprod((params['Rf']+1) * torch.ones([20,3]),axis=0)[:-1,:]),axis=1)
                                        - (c_var4/torch.cumprod(((params['Rf']+1) * torch.ones([19])),axis=0))]
        objective4 = cp.Maximize( cp.sum(c_var4/torch.cumprod(((params['Rf']+1) * torch.ones([19])),axis=0)))
        problem4 = cp.Problem(objective4, constraints4)  

        result_4 = problem4.solve(solver=cp.ECOS)




        return result_4