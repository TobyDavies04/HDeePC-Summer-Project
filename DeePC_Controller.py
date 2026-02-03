import numpy as np
import cvxpy as cp
from typing import NamedTuple, Tuple, Optional, List, Union
from cvxpy import Expression, Variable, Problem, Parameter
from cvxpy.constraints.constraint import Constraint
from scipy.linalg import block_diag, expm

class OptimizationProblemVariables(NamedTuple):
    """
    Class used to store all the variables used in the optimization
    problem
    """
    u_ini: Union[Variable, Parameter]
    y_ini: Union[Variable, Parameter]
    ref_param: Union[Variable, Parameter]
    # u: Union[Variable, Parameter]
    # y: Union[Variable, Parameter]
    g: Union[Variable, Parameter]
    slack_y: Union[Variable, Parameter]
    # slack_u: Union[Variable, Parameter]


class OptimizationProblem(NamedTuple):
    """
    Class used to store the elements an optimization problem
    :param problem_variables:   variables of the opt. problem
    :param constraints:         constraints of the problem
    :param objective_function:  objective function
    :param problem:             optimization problem object
    """
    variables: OptimizationProblemVariables
    constraints: List[Constraint]
    objective_function: Expression
    problem: Problem


class DeePC_Controller:
    def __init__(self, u_d, y_d, u_ini, y_ini, T_ini, N, Q, R, ref, lambda_g, lambda_y, u_min, u_max, du_min, du_max):
        self.u_d = u_d
        self.y_d = y_d
        self.u_ini = u_ini
        self.y_ini = y_ini

        self.Q = Q
        self.R = R
        self.ref = ref.reshape(-1, 1)
        self.lambda_g = lambda_g
        self.lambda_y = lambda_y

        self.u_min = u_min
        self.u_max = u_max
        self.du_min = du_min
        self.du_max = du_max

        self.M = u_d.shape[1]    # input dimension
        self.P = y_d.shape[1]    # output dimension
        print("P:", self.P)
        self.T = u_d.shape[0]    # total data length
        self.T_ini = T_ini    # initial trajectory length
        self.N = N         # prediction horizon
        self.L = self.T_ini + self.N   # total window length

    def hankel_matrix(self, data):
        data = np.asarray(data)

        # make SISO data into (T,1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        T, m = data.shape                  # m = dimension of each datapoint
        window = self.T_ini + self.N                 # rows per block column (L)
        shifts = T - window + 1            # number of Hankel columns

        # allocate Hankel matrix: (window*m) rows, 'shifts' cols
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
    
    def opt_setup(self):

        # Build variables
        uini = cp.Parameter(shape=(self.M * self.T_ini), name='u_ini')
        yini = cp.Parameter(shape=(self.P * self.T_ini), name='y_ini')
        ref_param = cp.Parameter(shape=(self.P * self.N), name="ref")
        # u = cp.Variable(shape=(self.M * self.N), name='u')
        # y = cp.Variable(shape=(self.P * self.N), name='y')
        g = cp.Variable(shape=(self.T - self.T_ini - self.N + 1), name='g')
        slack_y = cp.Variable(shape=(self.T_ini * self.P), name='slack_y')
        #slack_u = cp.Variable(shape=(self.T_ini * self.M), name='slack_u')

        self.U_p, self.U_f, self.Y_p, self.Y_f= self.behavior_matrix()
        self.check_PE()
        A = np.vstack([self.U_p, self.Y_p])
        b = cp.hstack([uini, yini + slack_y])
        constraints = [A @ g == b]

        u_future = self.U_f @ g
        y_future = cp.reshape(self.Y_f @ g, (self.P * self.N), order='C')    # (P*N, 1)
        #print("y_future shape:", y_future.shape)
        constraints += [u_future <= self.u_max]
        constraints += [u_future >= self.u_min]
        # ADD RATE CONSTRAINTS
        # if self.du_min is not None and self.du_max is not None:
        #     D = self.build_du_matrix()   # (N*M, N*M)

        #     # d0 = [u(k-1); 0; 0; ...] expressed using uini
        #     prev_u_expr = uini[-1]                        # scalar (expression)
        #     zeros_tail = np.zeros((self.N * self.M - 1,)) # (N*M-1,)
        #     d0 = cp.hstack([prev_u_expr, zeros_tail])     # (N*M,)

        #     DeltaU = D @ u - d0

        #     du_max_vec = self.du_max * np.ones((self.N * self.M,))
        #     du_min_vec = self.du_min * np.ones((self.N * self.M,))

        #     constraints += [
        #         DeltaU <= du_max_vec,
        #         DeltaU >= du_min_vec
        #     ]

        # u = cp.reshape(u, (self.N, self.M), order = 'C')
        # y = cp.reshape(y, (self.N, self.P), order = 'C')
        #first try without input/output constraints

        regularizers = self.lambda_g * cp.norm(g, p=1)
        regularizers += self.lambda_y * cp.norm(slack_y, p=1)


        Q_bar = block_diag(*[self.Q for _ in range(self.N)])
        R_bar = block_diag(*[self.R for _ in range(self.N)])
        # print("Q_bar shape:", Q_bar.shape)
        # print("R_bar shape:", R_bar.shape)


        #ref_stack = np.tile(self.ref, (self.N, 1))
        _objective = cp.quad_form(y_future - ref_param, Q_bar)
        _objective += cp.quad_form(u_future, R_bar)
        # _objective = cp.quad_form(y_future - self.ref_param, Q_bar)
        # _objective += cp.quad_form(u_future, R_bar)
        _objective += regularizers
        objective = cp.Minimize(_objective)
        problem = cp.Problem(objective, constraints)
        opt_prob = OptimizationProblem(
            variables=OptimizationProblemVariables(
            u_ini=uini, 
            y_ini=yini,
            ref_param = ref_param, 
            # u=u, 
            # y=y, 
            g=g, 
            slack_y=slack_y, 
            # slack_u=slack_u
            ),
            constraints=constraints,
            objective_function=objective,
            problem=problem
        )

        return opt_prob
    
    def solve_opt(self, opt_problem: OptimizationProblem):
        opt_problem.problem.solve(solver=cp.OSQP, warm_start=False, verbose=False, max_iter=20000)
        if opt_problem.problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"DeePC optimization failed: {opt_problem.problem.status}")

        # get optimal u from the variable, non-condensed form
        # u_star = opt_problem.variables.u.value   
        # u0 = u_star[:self.M]                     # first input (M elements)

        #Condnensed form
        g_star = opt_problem.variables.g.value
        u_future = self.U_f @ g_star

        return float(u_future[0])   # return as scalar

    
    def update(self, opt_problem: OptimizationProblem, u_ini_new, y_ini_new, ref_new=None):
        assert y_ini_new.shape == (self.T_ini, 2)
        assert u_ini_new.shape == (self.T_ini, 1)
        # flatten past trajectories
        opt_problem.variables.u_ini.value = u_ini_new.reshape(-1)
        opt_problem.variables.y_ini.value = y_ini_new.reshape(-1)

        # update reference
        if ref_new is not None:
            # ref_new is (P,1) or (P,)
            ref_flat = ref_new.reshape(-1)         # shape (P,)
            ref_stack_new = np.tile(ref_flat, self.N)  # shape (P*N,)
            opt_problem.variables.ref_param.value = ref_stack_new
