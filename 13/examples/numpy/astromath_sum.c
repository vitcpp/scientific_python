#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL astromath_ARRAY_API

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/ndarrayobject.h>
#include <stdio.h>
#include <assert.h>

/* Функция сложения двух чисел */
PyObject* astromath_sum(PyObject* self, PyObject* args)
{
    long result = 0;
    PyArrayObject* input = NULL;
    npy_intp* pshape;
    npy_intp* pstrides;
    int ndim, i;
    void* pdata;
    int dtype = 0;

    if (!PyArg_ParseTuple(args, "O", &input))
        return NULL;

    if (!PyArray_Check(input))
        return NULL;

    dtype    = PyArray_TYPE(input);
    ndim     = PyArray_NDIM(input);
    pshape   = PyArray_SHAPE(input);
    pdata    = PyArray_DATA(input);
    pstrides = PyArray_STRIDES(input);

    printf("dtype: %s\n", (
        dtype == NPY_INT64  ? "INT64"  :
        dtype == NPY_UINT64 ? "UINT64" : "UNDEF"));

    printf("ndim: %d\n", ndim);

    if (pshape && ndim > 0)
    {
        for (i = 0; i < ndim; ++i)
            printf("shape[%d]: %d\n", i, (int)pshape[i]);
    }

    {
        uint64_t* pd = (uint64_t*)pdata;
        for (i = 0; i < 100; ++i)
            result += pd[i];
    }

    return PyLong_FromLong(result);
}
