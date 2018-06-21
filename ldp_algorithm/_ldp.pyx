import numpy as np

# Import special compile-time information about the numpy module
# (Stored in numpy.pxd bundled with Cython)
cimport numpy as np

# Fix datatype for arrays
DTYPE = np.uint8

# Assigns a corresponding compile-time type to DTYPE_t
# For every type in the numpy module there's a corresponding 
# compile-time type with a _t-suffix
ctypedef np.uint8_t DTYPE_t


def bilinear_interpolation(np.ndarray matrix, np.float32_t y, np.float32_t x):
    cdef int intY = int(y)
    cdef int yn[2]
    yn[:] = [intY, intY + 1]

    cdef int intX = int(x)
    cdef int xn[2]
    xn[:] = [intY, intY + 1]

    cdef np.float32_t fat[4]
    fat = [
        xn[1] - x,
        yn[1] - y,
        x - xn[0],
        y - yn[1]
    ]

    try:
		# In this application, the denominator (xn [2] - xn [1]) * (yn [2] - yn [1]) is always equal to 1
        return (
            matrix[yn[0], xn[0]] * fat[0] * fat[1] +
            matrix[yn[0], xn[1]] * fat[1] * fat[2] + 
            matrix[yn[1], xn[0]] * fat[0] * fat[3] +
            matrix[yn[1], xn[1]] * fat[2] * fat[3]
        )
    except IndexError: 
        return 0



    