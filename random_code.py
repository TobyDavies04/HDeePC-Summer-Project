import numpy as np

def hankel_matrix(data, T_ini, N):
    data = np.asarray(data)

    # make SISO data into (T,1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T, m = data.shape                  # m = dimension of each datapoint
    window = T_ini + N                 # rows per block column (L)
    shifts = T - window + 1            # number of Hankel columns

    # allocate Hankel matrix: (window*m) rows, 'shifts' cols
    H = np.zeros((window * m, shifts))

    for i in range(shifts):
        # extract window of length (T_ini+N), shape = (window, m)
        block = data[i:i+window, :]

        # flatten into column vector (stacked output)
        H[:, i] = block.flatten(order='C')

    return H

T_data = 800
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
x_data = np.load("x_history_IP.npy")
u_data = np.load("u_history_IP.npy")
print("U Data", u_data)
u_d = np.array(u_data).reshape(-1, 1)   # (T,-1)
u_d = u_d[-T_data:]
print("u_d data:", u_d)
print("u_data shape:", u_d.shape)
y_d = np.array(x_data[-T_data:, :])
print("Y Data", y_d)
#print("X Data", x_data[500:750])
u_hankel = hankel_matrix(u_data, 15, 10)
x_hankel = hankel_matrix(y_d, 15, 10)
#print("U Hankel", u_hankel)
print("X Hankel", x_hankel)


# print("Data:\n", data[-3: , :0])
# T_ini = 2
# N = 2
# H = hankel_matrix(data, T_ini, N)
# print("Hankel Matrix:\n", H)