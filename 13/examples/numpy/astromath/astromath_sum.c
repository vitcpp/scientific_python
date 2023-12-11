#include <Python.h>
#include <stdio.h>
#include <assert.h>

/* Функция сложения двух чисел */
PyObject* astromath_sum(PyObject* self, PyObject* args)
{
    long result = 0;
    Py_ssize_t nargs;

    if (!PyTuple_Check(args))
    {
        PyErr_SetString(PyExc_RuntimeError, "Неверный тип аргумента");
        return NULL;
    }

    nargs = PyTuple_Size(args);

    if (nargs == 1)
    {
        PyObject* arg = PyTuple_GetItem(args, 0);

        if (!PySequence_Check(arg))
        {
            PyErr_SetString(PyExc_RuntimeError, "Неверный тип аргумента");
            return NULL;
        }

        for (Py_ssize_t i = 0, size = PySequence_Size(arg); i < size; ++i)
        {
            PyObject* seqitem = PySequence_GetItem(arg, i);

            if (!PyLong_Check(seqitem))
            {
                PyErr_SetString(PyExc_RuntimeError, "Неверный тип аргумента в последовательности");
                return NULL;
            }

            result += PyLong_AsLong(seqitem);
        }
    }
    else
    {
        long arg1, arg2;

        if (!PyArg_ParseTuple(args, "ll", &arg1, &arg2))
            return NULL;

        result = arg1 + arg2;
    }

    return PyLong_FromLong(result);
}
