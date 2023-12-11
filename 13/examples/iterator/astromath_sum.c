#include <Python.h>
#include <stdio.h>
#include <assert.h>

/* Функция сложения двух чисел */
PyObject* astromath_sum(PyObject* self, PyObject* args)
{
    long result = 0;
    PyObject* obj;
    PyObject* item;
    PyObject* iter;

    if (!PyArg_ParseTuple(args, "O", &obj))
    {
        PyErr_SetString(PyExc_RuntimeError, "Неверный тип аргумента");
        return NULL;
    }

    if ((iter = PyObject_GetIter(obj)) == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "Объект неитерабельного типа");
        return NULL;
    }

    while ((item = PyIter_Next(iter)) != NULL)
    {
        if (!PyLong_Check(item))
        {
            PyErr_SetString(PyExc_RuntimeError, "Неверный тип аргумента в последовательности");
            return NULL;
        }

        result += PyLong_AsLong(item);

        Py_DECREF(item);
    }

    Py_DECREF(iter);

    return PyLong_FromLong(result);
}
