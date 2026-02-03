import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm


"""This module implements a Model Predictive Controller (MPC) for a DC motor"""
class MPC_Controller:
    def __init__(self, A, B, C, D, Q, R, T, ref, u_min, u_max, du_min, du_max, Np=10, Nc=4, x0=None, discretize=False):
        #### Model Predictive Controller Parameter####
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.Q = Q
        self.R = R
        self.ref = ref.reshape(-1, 1)

        self.u_min = u_min
        self.u_max = u_max
        self.du_min = du_min
        self.du_max = du_max

        self.x = np.zeros((self.A.shape[0],1), dtype=float)
        self.u = np.zeros((self.B.shape[1],1), dtype=float)
        self.y = np.zeros((self.C.shape[0],1), dtype=float)
        self.T = T

        ### Discretize the system
        # self.Ad = expm(self.A * T)
        # self.Bd = np.linalg.solve(A, (self.Ad - np.eye(A.shape[0]))) @ B
        if discretize:
            # Exact discretization of (A,B)
            n = A.shape[0]
            m = B.shape[1]

            M = np.block([
                [A,               B],
                [np.zeros((m, n)), np.zeros((m, m))]
            ])

            Md = expm(M * T)

            self.Ad = Md[:n, :n]
            self.Bd = Md[:n, n:n+m]
            #print("Ad Matrix", self.Ad)

        self.Np = Np         # Prediction horizon
        self.Nc = Nc         # Control Horizon

    def Observer_Matrix(self):
        O = []

        for i in range(self.Np):
            block = self.C @ np.linalg.matrix_power(self.Ad, i+1)
            O.append(block)

        return np.vstack(O)
    
    def Toeplitz_Matrix(self):

        ny = self.C.shape[0]      # outputs
        nu = self.B.shape[1]      # inputs

        Np = self.Np              # prediction horizon
        Nc = self.Nc              # control horizon

        M = np.zeros((Np * ny, Nc * nu))

        # Precompute all needed powers of A
        AdPowers = [np.linalg.matrix_power(self.Ad, k) for k in range(Np)]

        # Precompute G_k = C A^k B
        G = [self.C @ AdPowers[k] @ self.Bd for k in range(Np)]

        # Fill Toeplitz structure
        for i in range(Np):
            for j in range(Nc):
                if i >= j:
                    block = G[i - j]
                    M[i*ny:(i+1)*ny, j*nu:(j+1)*nu] = block
        return M
    
    def build_cost_matrices(self):
        O = self.Observer_Matrix()
        M = self.Toeplitz_Matrix()

        ny = self.C.shape[0]
        nu = self.B.shape[1]
        Np = self.Np
        Nc = self.Nc

        # Block-diagonal Q_bar and R_bar
        Q_bar = block_diag(*[self.Q for _ in range(Np)])
        R_bar = block_diag(*[self.R for _ in range(Nc)])

        # Stacked reference vector (Np*ny × 1)
        r_vec = np.tile(self.ref.reshape(-1, 1), (self.Np, 1))

        # Current state
        x0 = self.x

        # Quadratic cost matrices
        H = M.T @ Q_bar @ M + R_bar
        f = M.T @ Q_bar @ (O @ x0 - r_vec)

        return H, f

    def build_du_matrix(self):
        Nc = self.Nc
        nu = self.B.shape[1]

        # Size is (Nc*nu, Nc*nu)
        D = np.zeros((Nc * nu, Nc * nu))

        for i in range(Nc * nu):

            
            D[i, i] = 1
            if i > 0:
                D[i, i-1] = -1

        return D

    def build_constraints(self, U):
        nu = self.B.shape[1]
        Nc = self.Nc

        # Magnitude constraints
        Umin = np.tile(self.u_min.reshape(-1,1), (Nc, 1))
        Umax = np.tile(self.u_max.reshape(-1,1), (Nc, 1))

        #previous before changing for quadcopter
        # Umin = np.tile(self.u_min, (Nc * nu, 1))
        # Umax = np.tile(self.u_max, (Nc * nu, 1))    

        constraints = [U >= Umin, U <= Umax]

        # Rate constraints (if enabled)
        if self.du_min is not None and self.du_max is not None:
            D = self.build_du_matrix()

            # d0 vector (Nc*nu x 1)
            d0 = np.zeros((Nc * nu, 1))
            d0[0:nu, :] = self.u   # only first block uses previous input

            delta_u_min = np.tile(self.du_min, (Nc * nu, 1))
            delta_u_max = np.tile(self.du_max, (Nc * nu, 1))

            DeltaU = D @ U - d0

            constraints += [
                DeltaU >= delta_u_min,
                DeltaU <= delta_u_max
            ]

        return constraints
    
    def solve_mpc(self):
        H, f = self.build_cost_matrices()

        nu = self.B.shape[1]
        Nc = self.Nc

        U = cp.Variable((Nc * nu, 1))

        # Build constraints
        constraints = self.build_constraints(U)

        objective = cp.Minimize(0.5 * cp.quad_form(U, cp.psd_wrap(H)) + f.T @ U)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False, polish=False)
        # print("status:", prob.status)
        # print("U is finite:", np.isfinite(U.value).all() if U.value is not None else None)

        if U.value is None:
            raise RuntimeError("MPC optimization failed.")

        return U.value

    def step(self):
        U_opt = self.solve_mpc()

        nu = self.B.shape[1]

        # Extract first control move (shape (1,1))
        u0 = U_opt[:nu].reshape((nu, 1))

        # Convert to clean Python scalar if single input
        #u0_scalar = float(u0.item())

        # Store in consistent (1,1) array form
        #self.u = np.array([[u0_scalar]], dtype=float)

        self.u = u0

        # State update
        # self.x = self.Ad @ self.x + self.Bd @ self.u
        # self.y = self.C @ self.x

        return self.x, self.y, self.u.item() if self.u.size == 1 else self.u

    def simulate(self, steps):
        xs, ys, us = [], [], []
        for _ in range(steps):
            x, y, u = self.step()
            xs.append(x)
            ys.append(y)
            us.append(u)
        return np.array(xs), np.array(ys), np.array(us)
