#include <Python.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <omp.h>

/* Функция сложения двух чисел */
PyObject* astromath_sum_parallel_good(PyObject* self, PyObject* args)
{
    long count = 0;
    double result = 0;
    omp_lock_t result_lock;

    /* Парсинг аргументов из PyObject в типы языка С. */
    if (!PyArg_ParseTuple(args, "l", &count))
        return NULL;

    omp_init_lock(&result_lock);

    Py_BEGIN_ALLOW_THREADS

    #pragma omp parallel
    {
        // Локальные вычисления в потоке
        double localres = 0;
        #pragma omp for
        for (long i = 0; i < count; ++i)
            localres += sin(i) + cos(i);

        // Обновить общую переменную
        //omp_set_lock(&result_lock);
        //#pragma imp atomic
        #pragma omp critical
        {
            result += localres;
        }
        //omp_unset_lock(&result_lock);
    }

    Py_END_ALLOW_THREADS

    omp_destroy_lock(&result_lock);

    return PyFloat_FromDouble(result);
}
