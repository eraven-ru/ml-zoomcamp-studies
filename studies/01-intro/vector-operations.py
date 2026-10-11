import numpy as np

u = np.array([0, 1, 2])

v = np.array([3, 2, 1])

U = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

V = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

def vec_vec_multiplication(u, v):
    assert u.shape[0] == v.shape[0]

    n = u.shape[0]
    result = 0
    for i in range(n):
        result += u[i] * v[i]

    return result

print("vec_vec_multiplication(u, v) ", vec_vec_multiplication(u, v))
print("u.dot(v) ", u.dot(v))

def mat_vec_multiplication(U, v):
    assert U.shape[1] == v.shape[0]

    n_rows = U.shape[0]
    result = np.zeros(n_rows)
    for i in range(n_rows):
        result[i] = vec_vec_multiplication(U[i], v)

    return result

print("mat_vec_multiplication(u, v) ", mat_vec_multiplication(U, v))
print("U.dot(v) ", U.dot(v))

def mat_mat_multiplication(U, V):
    assert U.shape[1] == V.shape[0]

    n_rows = U.shape[0]
    n_cols = V.shape[1]

    result = np.zeros((n_rows, n_cols))
    for i in range(n_cols):
        result[:, i] = mat_vec_multiplication(U, V[:, i])

    return result

print("mat_mat_multiplication(u, v) ", mat_mat_multiplication(U, V))
print("U.dot(V) ", U.dot(V))

# Identity matrix - I (U * I == U); np.eye(dimension)
identity_mat = np.eye(5)

# Inverse matrix U[[-1]] * U == I
Us = np.array([[1., 7.],
 [4., 2.]])
Us_inv = np.linalg.inv(Us)
print(Us_inv.dot(Us))



