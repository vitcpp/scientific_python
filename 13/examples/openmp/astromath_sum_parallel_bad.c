#include <Python.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <omp.h>

/* Функция сложения двух чисел */
PyObject* astromath_sum_parallel_bad(PyObject* self, PyObject* args)
{
    long count = 0;
    double result = 0;

    /* Парсинг аргументов из PyObject в типы языка С. */
    if (!PyArg_ParseTuple(args, "l", &count))
        return NULL;

    Py_BEGIN_ALLOW_THREADS

    #pragma omp parallel
    {
        // Локальные вычисления в потоке
        double localres = 0;
        #pragma omp for
        for (long i = 0; i < count; ++i)
            localres += sin(i) + cos(i);

        // Обновить общую переменную
        result += localres;
    }

    Py_END_ALLOW_THREADS

    return PyFloat_FromDouble(result);
}
