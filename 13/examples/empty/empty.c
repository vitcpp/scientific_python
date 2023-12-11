#include <Python.h>

PyMODINIT_FUNC
PyInit_empty(void)
{
    PyErr_SetString(PyExc_RuntimeError, "Не готово еще!");
    PyErr_Print();
	return NULL;
}
