#include <Python.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <omp.h>

/* Функция сложения двух чисел */
PyObject* astromath_sum(PyObject* self, PyObject* args)
{
    long count = 0;
    double result = 0;

    /* Парсинг аргументов из PyObject в типы языка С. */
    if (!PyArg_ParseTuple(args, "l", &count))
        return NULL;

    for (long i = 0; i < count; ++i)
        result += sin(i) + cos(i);

    return PyFloat_FromDouble(result);
}
