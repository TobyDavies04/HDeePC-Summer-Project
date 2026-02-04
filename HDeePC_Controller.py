import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag, expm

##################################
##### HDeePC Controller Class ####
##################################

class HDeePC_Controller:
    ###
    def __init__(self, Ac, Ak, Ay, Bk, Cc, Ck, Cy, Dk, u_d, yu_d, u_ini, y_uini,
                  N, Q, R, ref, u_min, u_max, du_min, du_max, lambda_g, lambda_y, 
                  lambda_u, NP, calculate_Ay_Cy):
        self.Ac = Ac
        self.Ak = Ak
        self.Bk = Bk
        self.Cc = Cc
        self.Ck = Ck
        self.Dk = Dk
        if calculate_Ay_Cy:
            self.Ay, self.Cy = self.compute_Ay_Cy(Ac, Cc, Ck, Dk)
        else:
            self.Ay = Ay
            self.Cy = Cy

        self.nk = NP[1]   # numbe of states from known part of plant
        self.pk = NP[3]  # numbe of outputs from known part of the plant
        self.xk_init = np.zeros((self.nk,))
        self.u_d = u_d
        self.yu_d = yu_d
        self.u_ini = u_ini
        self.y_uini = y_uini
        self.ny = 2
        self.Q = np.array(Q).reshape(-1, 1)
        self.R = np.array(R).reshape(-1, 1)

        self.lambda_g = lambda_g
        self.lambda_y = lambda_y
        self.lambda_u = lambda_u

        self.u_min = u_min
        self.u_max = u_max
        self.du_min = du_min
        self.du_max = du_max

        self.M = u_d.shape[1]    # input dimension
        self.pu = NP[2]    # output unknown dimension (pu)
        self.Pt = self.pu + self.pk   # total output dimension
        self.T = u_d.shape[0]    # total data length
        self.T_ini = u_ini.shape[0]    # initial trajectory length
        self.N = N         # prediction horizon
        self.U_p, self.U_f, self.Y_up, self.Y_uf = self.behavior_matrix()
        self.check_PE()
        # Store full reference (4x1, for example)
        self.ref_full = ref.reshape(-1, 1)  # shape (Pt, 1)

        # Split into unknown / known output references
        if self.pu > 0:
            self.ref_u = np.tile(self.ref_full[:self.pu], (self.N, 1))
            Q_u_vec = self.Q[:self.pu]
        else:
            self.ref_u = None
            Q_u_vec = np.array([])

        if self.pk > 0:
            self.ref_k = np.tile(self.ref_full[self.pu:self.pu + self.pk], (self.N, 1))
            Q_k_vec = self.Q[self.pu:self.pu + self.pk]
        else:
            self.ref_k = None
            Q_k_vec = np.array([])
        self.Q_u_bar = block_diag(*[np.diagflat(Q_u_vec) for _ in range(self.N)]) if self.pu > 0 else None
        self.Q_k_bar = block_diag(*[np.diagflat(Q_k_vec) for _ in range(self.N)]) if self.pk > 0 else None


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
        self.U_h = self.hankel_matrix(self.u_d)
        self.Y_h = self.hankel_matrix(self.yu_d)

        # split into past and future
        U_p = self.U_h[:self.T_ini * self.u_d.shape[1], :]
        U_f = self.U_h[self.T_ini * self.u_d.shape[1]:, :]

        Y_up = self.Y_h[:self.T_ini * self.yu_d.shape[1], :]
        Y_uf = self.Y_h[self.T_ini * self.yu_d.shape[1]:, :]

        return U_p, U_f, Y_up, Y_uf
    
    def check_PE(self):
        print("\n=== DeePC Persistent Excitation Check ===")

        checks = [
            ("U_p", self.U_p, self.M*self.T_ini, "PE of order T_ini (MOST IMPORTANT)"),
            ("Y_p", self.Y_up, self.pu*self.T_ini, "output past consistency"),
            ("U_f", self.U_f, self.M*self.N,    "future input coverage"),
            ("Y_f", self.Y_uf, self.pu*self.N,    "future output coverage")
        ]

        for name, H, req, desc in checks:
            r = np.linalg.matrix_rank(H)
            status = "PASS" if r >= req else "FAIL"
            print(f"{name}: rank={r:3d}  req={req:3d}  → {status} ({desc})")

        print("=========================================\n")

    def compute_Ay_Cy(self, Ac, Cc, Cu, Du=None):
        """
        Compute Ay and Cy such that:
            Ay @ Cu = Ac,   Ay @ Du = 0
            Cy @ Cu = Cc,   Cy @ Du = 0
        """

        p_u, n_u = Cu.shape
        n_k = Ac.shape[0]
        p_k = Cc.shape[0]

        m = Du.shape[1]

        # Build Z = [Cu Du]
        Z = np.hstack([Cu, Du])  # (p_u, n_u + m)

        # Build RHS
        RHS_A = np.hstack([Ac, np.zeros((n_k, m))])
        RHS_C = np.hstack([Cc, np.zeros((p_k, m))])

        # Solve least squares: Ay Z ≈ RHS_A → Ay = RHS_A Z⁺
        Z_pinv = np.linalg.pinv(Z)

        Ay = RHS_A @ Z_pinv
        Cy = RHS_C @ Z_pinv

        # Diagnostics
        print("Ay residual:", np.linalg.norm(Ay @ Z - RHS_A))
        print("Cy residual:", np.linalg.norm(Cy @ Z - RHS_C))

        return Ay, Cy


#3rd opt function, reshaped variables
    def opt_problemx(self):

        # Mode flags
        use_deepc = self.pu > 0
        use_mpc = self.pk > 0

        # === Decision variables ===
        if use_deepc:
            g = cp.Variable((self.T - self.T_ini - self.N + 1, 1))
            y_u = cp.Variable((self.N * self.pu, 1))
            slack_y = cp.Variable((self.pu * self.T_ini, 1))
            slack_u = cp.Variable((self.M * self.T_ini, 1))
        else:
            g = None
            y_u = None
            slack_y = None
            slack_u = None

        if use_mpc:
            x_k = cp.Variable((self.nk, self.N + 1))
            y_k = cp.Variable((self.N * self.pk, 1))
        else:
            x_k = None
            y_k = None

        u = cp.Variable((self.N * self.M, 1))

        constraints = []

        # === DeePC behavioral constraints ===
        if use_deepc:
            A = np.vstack([self.U_p, self.Y_up, self.U_f, self.Y_uf])
            b = cp.vstack([self.u_ini + slack_u,
                        self.y_uini + slack_y,
                        u,
                        y_u])
            constraints += [A @ g == b]

        # # === MPC dynamics constraints ===
        if use_mpc:
            constraints += [x_k[:, 0:1] == self.xk_init.reshape(-1,1)]
            for i in range(self.N):
                u_i = u[i*self.M:(i+1)*self.M]        # (M,1)
                x_i = x_k[:, i:i+1]                  # (nk,1)
                x_ip1 = x_k[:, i+1:i+2]   # slices for inputs and unknown outputs
                if use_deepc:
                    y_u_i = y_u[i*self.pu:(i+1)*self.pu]  # (pu,1)
                    if i == 100:
                        print("y_u_i shape:", y_u_i.shape)
                        print("Constraint 1 RHS shape: ", (self.Ay @ y_u_i + self.Ak @ x_i + self.Bk @ u_i).shape)
                        print("x_ip1 shape: ", x_ip1.shape)
                        print("Constraint 1 shape", (x_ip1 == self.Ay @ y_u_i + self.Ak @ x_i + self.Bk @ u_i).shape)
                        print("Constraint 2 RHS shape: ", (self.Cy @ y_u_i + self.Ck @ x_i + self.Dk @ u_i).shape)
                        print("y_k slice shape: ", y_k[i*self.pk:(i+1)*self.pk].shape)
                        print("Constraint 2 shape", (y_k[i*self.pk:(i+1)*self.pk] == self.Cy @ y_u_i + self.Ck @ x_i + self.Dk @ u_i).shape)
                    constraints += [
                        x_ip1 == self.Ay @ y_u_i + self.Ak @ x_i + self.Bk @ u_i,
                        y_k[i*self.pk:(i+1)*self.pk] == self.Cy @ y_u_i + self.Ck @ x_i + self.Dk @ u_i
                    ]

                else:                 
                    constraints += [
                        x_ip1 == self.Ak @ x_i + self.Bk @ u_i,
                        y_k[i*self.pk:(i+1)*self.pk] == self.Ck @ x_i + self.Dk @ u_i
                    ]

        # === Input constraints ===
        if self.u_min is not None:
            constraints += [u >= self.u_min]
        if self.u_max is not None:
            constraints += [u <= self.u_max]

        # === Input rate constraints ===
        if self.du_min is not None or self.du_max is not None:
            du = u[self.M:, :] - u[:-self.M, :]
            if self.du_min is not None:
                constraints += [du >= self.du_min]
            if self.du_max is not None:
                constraints += [du <= self.du_max]

        # === Regularization ===
        regularizers = 0
        if use_deepc:
            if self.lambda_g > 0:
                regularizers += self.lambda_g * cp.norm(g, 1)
            if self.lambda_y > 0:
                regularizers += self.lambda_y * cp.norm(slack_y, 1)
            if self.lambda_u > 0:
                regularizers += self.lambda_u * cp.norm(slack_u, 1)

        # === Cost function ===
        cost = 0
        Q_bar = block_diag(*[self.Q for _ in range(self.N)])
        R_bar = block_diag(*[self.R for _ in range(self.N)])
        # Build output stack
        if use_deepc and use_mpc:
            y_stack = cp.vstack([y_u, y_k])
        elif use_deepc:
            y_stack = y_u
        elif use_mpc:
            y_stack = y_k
        else:
            raise RuntimeError("Neither DeePC nor MPC active — invalid configuration.")

        ref_stack = self.ref.flatten(order='F').reshape(-1, 1)
        cost += cp.quad_form(y_stack - ref_stack, Q_bar)
        cost += cp.quad_form(u, R_bar)
        cost += regularizers

        # === Solve ===
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(
            solver=cp.OSQP,
            verbose=False,
            max_iter=500000,
            eps_abs=1e-4,
            eps_rel=1e-4,
            polish=True
        )

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError("HDeePC optimization failed: " + problem.status)
        
        # Tracking Costs
        J_track = cp.quad_form(y_stack - ref_stack, Q_bar).value
        J_input = cp.quad_form(u, R_bar).value
        J_reg = regularizers.value if use_deepc else 0

        print("Tracking cost:", J_track)
        print("Input cost:", J_input)
        print("Regularization cost:", J_reg)

        return u.value[:self.M].reshape(-1)

    def opt_problem(self):

        # Mode flags
        use_deepc = self.pu > 0
        use_mpc = self.pk > 0

        # === Decision variables ===
        if use_deepc:
            g = cp.Variable((self.T - self.T_ini - self.N + 1, 1))
            y_u = cp.Variable((self.N * self.pu, 1))
            slack_y = cp.Variable((self.pu * self.T_ini, 1))
            slack_u = cp.Variable((self.M * self.T_ini, 1))
        else:
            g = None
            y_u = None
            slack_y = None
            slack_u = None

        if use_mpc:
            x_k = cp.Variable((self.nk, self.N + 1))
            y_k = cp.Variable((self.N * self.pk, 1))
        else:
            x_k = None
            y_k = None

        u = cp.Variable((self.N * self.M, 1))

        constraints = []

        # === DeePC behavioral constraints ===
        if use_deepc:
            A = np.vstack([self.U_p, self.Y_up, self.U_f, self.Y_uf])
            b = cp.vstack([
                self.u_ini + slack_u,
                self.y_uini + slack_y,
                u,
                y_u
            ])
            constraints += [A @ g == b]

        # === MPC dynamics constraints ===
        if use_mpc:
            constraints += [x_k[:, 0:1] == self.xk_init.reshape(-1, 1)]
            for i in range(self.N):
                u_i = u[i*self.M:(i+1)*self.M]
                x_i = x_k[:, i:i+1]
                x_ip1 = x_k[:, i+1:i+2]

                if use_deepc:
                    y_u_i = y_u[i*self.pu:(i+1)*self.pu]
                    constraints += [
                        x_ip1 == self.Ay @ y_u_i + self.Ak @ x_i + self.Bk @ u_i,
                        y_k[i*self.pk:(i+1)*self.pk] == self.Cy @ y_u_i + self.Ck @ x_i + self.Dk @ u_i
                    ]
                else:
                    constraints += [
                        x_ip1 == self.Ak @ x_i + self.Bk @ u_i,
                        y_k[i*self.pk:(i+1)*self.pk] == self.Ck @ x_i + self.Dk @ u_i
                    ]

        # === Input constraints ===
        if self.u_min is not None:
            constraints += [u >= self.u_min]
        if self.u_max is not None:
            constraints += [u <= self.u_max]

        # === Input rate constraints ===
        if self.du_min is not None or self.du_max is not None:
            du = u[self.M:, :] - u[:-self.M, :]
            if self.du_min is not None:
                constraints += [du >= self.du_min]
            if self.du_max is not None:
                constraints += [du <= self.du_max]

        # === Regularization ===
        regularizers = 0
        if use_deepc:
            if self.lambda_g > 0:
                regularizers += self.lambda_g * cp.norm(g, 1)
            if self.lambda_y > 0:
                regularizers += self.lambda_y * cp.norm(slack_y, 1)
            if self.lambda_u > 0:
                regularizers += self.lambda_u * cp.norm(slack_u, 1)

        # === Cost function ===
        cost = 0

        # Unknown output tracking cost (DeePC part)
        if use_deepc:
            #Q_u_bar = block_diag(*[self.Q for _ in range(self.N)])
            cost += cp.quad_form(y_u - self.ref_u, self.Q_u_bar)

        # Known output tracking cost (MPC part)
        if use_mpc:
            #Q_k_bar = block_diag(*[self.Q for _ in range(self.N)])
            cost += cp.quad_form(y_k - self.ref_k, self.Q_k_bar)

        # Input cost
        #R_bar = block_diag(*[self.R for _ in range(self.N)])
        R_bar = block_diag(*[np.diagflat(self.R) for _ in range(self.N)])
        # print("Shape u:", u.shape)
        # print("Shape r_bar:", R_bar.shape)
        cost += cp.quad_form(u, R_bar)

        # Regularization
        cost += regularizers

        # === Solve ===
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(
            solver=cp.OSQP,
            verbose=False,
            max_iter=500000,
            eps_abs=1e-4,
            eps_rel=1e-4,
            polish=True
        )

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError("HDeePC optimization failed: " + problem.status)

        # === Cost breakdown (debug) ===
        self.J_track = 0
        if use_deepc:
            self.J_track += cp.quad_form(y_u - self.ref_u, self.Q_u_bar).value
        if use_mpc:
            self.J_track += cp.quad_form(y_k - self.ref_k, self.Q_k_bar).value

        self.J_input = cp.quad_form(u, R_bar).value
        self.J_reg = regularizers.value if use_deepc else 0

        # print("Tracking cost:", J_track)
        # print("Input cost:", J_input)
        # print("Regularization cost:", J_reg)

        return u.value[:self.M].reshape(-1)


    def update(self, u_ini_new, y_ini_new, xk_init_new, ref_new):
        self.u_ini = u_ini_new.reshape(-1, 1)
        self.y_uini = y_ini_new.reshape(-1, 1)
        self.xk_init = xk_init_new.reshape(-1)

        if ref_new is not None:
            self.ref_full = ref_new.reshape(-1, 1)

            if self.pu > 0:
                self.ref_u = np.tile(self.ref_full[:self.pu], (self.N, 1))
            else:
                self.ref_u = None

            if self.pk > 0:
                self.ref_k = np.tile(self.ref_full[self.pu:self.pu + self.pk], (self.N, 1))
            else:
                self.ref_k = None

