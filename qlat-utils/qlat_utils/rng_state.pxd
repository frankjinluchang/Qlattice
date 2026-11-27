from . cimport everything as cqlat_utils

cdef class RngState:
    cdef cqlat_utils.RngState xx
    # Internal implementation details of rand_arr()/u_rand_arr()/g_rand_arr().
    # Deliberately cdef (no Python binding): the public u_rand_fill/g_rand_fill
    # family was removed in qlat-issue-3.
    cdef void _rand_fill_u_long(self, cqlat_utils.ULong[:] arr)
    cdef void _u_rand_fill_real_d(self, double[:] arr, double lower, double upper)
    cdef void _g_rand_fill_real_d(self, double[:] arr, double center, double sigma)
