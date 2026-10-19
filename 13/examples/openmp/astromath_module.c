#include "astromath_sum.h"
#include <Python.h>
#include <stdio.h>

extern PyObject* astromath_sum_parallel_bad(PyObject* self, PyObject* args);
extern PyObject* astromath_sum_parallel_good(PyObject* self, PyObject* args);

/* Список методов модуля */
static PyMethodDef astromath_methods[] =
{
    { "sum",  astromath_sum_parallel_good, METH_VARARGS, "Сложение."},
    { NULL, NULL, 0, NULL }
};

/* Описание модуля */
static struct PyModuleDef astromath_module =
{
    PyModuleDef_HEAD_INIT,
    "astromath",
    "Модуль astromath",
    -1,
    astromath_methods
};

/* Функция, которую вызывает питон для загрузки модуля */
PyMODINIT_FUNC
PyInit_astromath(void)
{
    PyObject *m;

    m = PyModule_Create(&astromath_module);
    if (m == NULL)
        return NULL;

    return m;
}
