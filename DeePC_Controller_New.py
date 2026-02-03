import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm

class DeePC_Controller:
    def __init__(self, u_d, y_d, u_ini, y_ini, T_ini, N, Q, R, ref, lambda_g, lambda_y, lambda_u, u_min, u_max, du_min, du_max):
        self.u_d = u_d
        self.y_d = y_d
        self.u_ini = u_ini
        self.y_ini = y_ini

        self.Q = Q
        self.R = R
        self.ref = ref.reshape(-1, 1)
        self.lambda_g = lambda_g
        self.lambda_y = lambda_y
        self.lambda_u = lambda_u

        self.u_min = u_min
        self.u_max = u_max
        self.du_min = du_min
        self.du_max = du_max

        self.M = u_d.shape[1]    # input dimension
        self.P = y_d.shape[1]    # output dimension
        self.T = u_d.shape[0]    # total data length
        self.T_ini = T_ini    # initial trajectory length
        self.N = N         # prediction horizon
        self.L = self.T_ini + self.N   # total window length

        self.U_p, self.U_f, self.Y_p, self.Y_f= self.behavior_matrix()

        print("DEBUG: u_d shape:", u_d.shape)
        print("DEBUG: y_d shape:", y_d.shape)
        print("DEBUG: T =", self.T)
        print("Expected len(g) = ", self.T - self.T_ini - self.N + 1)


    def hankel_matrix(self, data):
        data = np.asarray(data)
        # make SISO data into (T,1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        T, m = data.shape                  # m = dimension of each datapoint
        window = self.T_ini + self.N                 # rows per block column (L)
        shifts = T - window + 1            # number of hankel columns

        # allocate hankel matrix: (window*m) rows, 'shifts' cols
        H = np.zeros((window * m, shifts))

        for i in range(shifts):
            # extract window of length (T_ini+N), shape = (window, m)
            block = data[i:i+window, :]

            # flatten into column vector (stacked output)
            H[:, i] = block.flatten(order='C')

        return H
    
    def behavior_matrix(self):
        U_h = self.hankel_matrix(self.u_d)
        Y_h = self.hankel_matrix(self.y_d)

        # split into past and future
        U_p = U_h[:self.T_ini * self.u_d.shape[1], :]
        U_f = U_h[self.T_ini * self.u_d.shape[1]:, :]

        Y_p = Y_h[:self.T_ini * self.y_d.shape[1], :]
        Y_f = Y_h[self.T_ini * self.y_d.shape[1]:, :]

        return U_p, U_f, Y_p, Y_f
    
    def check_PE(self):
        print("\n=== DeePC Persistent Excitation Check ===")

        checks = [
            ("U_p", self.U_p, self.M*self.T_ini, "PE of order T_ini (MOST IMPORTANT)"),
            ("Y_p", self.Y_p, self.P*self.T_ini, "output past consistency"),
            ("U_f", self.U_f, self.M*self.N,    "future input coverage"),
            ("Y_f", self.Y_f, self.P*self.N,    "future output coverage")
        ]

        for name, H, req, desc in checks:
            r = np.linalg.matrix_rank(H)
            status = "PASS" if r >= req else "FAIL"
            print(f"{name}: rank={r:3d}  req={req:3d}  → {status} ({desc})")

        print("=========================================\n")

    def build_du_matrix(self):
        Nc = self.N
        nu = self.M
        D = np.zeros((Nc * nu, Nc * nu))

        for i in range(Nc * nu):
            D[i, i] = 1
            if i > 0:
                D[i, i - 1] = -1

        return D
    
    def opt_solve(self):
        #assert y_future.shape == (self.P * self.N,)
        g = cp.Variable(shape=(self.T - self.T_ini - self.N + 1))
        slack_y = cp.Variable(shape=(self.P * self.T_ini))
        slack_u = cp.Variable(shape=(self.M * self.T_ini))

        # Build variables
        A = np.vstack([self.U_p, self.Y_p])
        b = cp.hstack([self.u_ini + slack_u, self.y_ini + slack_y])
        constraints = [A @ g == b]

        u_future = self.U_f @ g
        y_future = cp.reshape(self.Y_f @ g, (self.P * self.N), order='C')    # (P*N, 1)
        constraints += [u_future <= self.u_max]
        constraints += [u_future >= self.u_min]
        if self.du_min is not None and self.du_max is not None:
            D = self.build_du_matrix()   # (N*M, N*M)

            # d0 = [u(k-1); 0; 0; ...] expressed using uini
            prev_u_expr = self.u_ini[-1]                        # scalar (expression)
            zeros_tail = np.zeros((self.N * self.M - 1,)) # (N*M-1,)
            d0 = cp.hstack([prev_u_expr, zeros_tail])     # (N*M,)

            DeltaU = D @ u_future - d0

            du_max_vec = self.du_max * np.ones((self.N * self.M,))
            du_min_vec = self.du_min * np.ones((self.N * self.M,))

            constraints += [
                DeltaU <= du_max_vec,
                DeltaU >= du_min_vec
            ]

        # u = cp.reshape(u, (self.N, self.M), order = 'C')
        # y = cp.reshape(y, (self.N, self.P), order = 'C')
        regularizers = self.lambda_g * cp.norm(g, p=1)
        regularizers += self.lambda_y * cp.norm(slack_y, p=1)
        regularizers += self.lambda_u * cp.norm(slack_u, p=1)

        Q_bar = block_diag(*[self.Q for _ in range(self.N)])
        R_bar = block_diag(*[self.R for _ in range(self.N)])
        _objective = cp.quad_form(y_future - self.ref, Q_bar)
        _objective += cp.quad_form(u_future, R_bar)
        #print("y_future shape:", y_future.shape)
        #print("ref shape:", self.ref.shape)
        _objective += regularizers
        objective = cp.Minimize(_objective)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False, polish=False)
        # print("DeePC status:", problem.status)
        # print("g.value:", type(g.value), getattr(g.value, "shape", None))

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"DeePC optimization failed with status: {problem.status}")

        g_val = np.asarray(g.value).reshape(-1)
        # print("g_star shape after reshape:", g_val.shape)
        return g_val
    
    def step_deepc(self):
        g_star = self.opt_solve()
        u_future = self.U_f @ g_star
        u_control = u_future[0]   # first control input
        return u_control

    def update(self, u_ini_new, y_ini_new, ref_new=None):
        assert y_ini_new.shape == (self.T_ini, self.P)
        assert u_ini_new.shape == (self.T_ini, self.M)
        # flatten past trajectories
        self.u_ini = u_ini_new.reshape(-1)
        self.y_ini = y_ini_new.reshape(-1)

        # update reference
        if ref_new is not None:
            # ref_new is (P,1) or (P,)
            ref_flat = ref_new.reshape(-1)         # shape (P,)
            ref_stack_new = np.tile(ref_flat, self.N)  # shape (P*N,)
            self.ref = ref_stack_new
            #print("Updated ref shape:", self.ref.shape)
