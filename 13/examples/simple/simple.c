#include <Python.h>
#include <stdio.h>

/* Функция сложения двух чисел */
static PyObject* simple_add(PyObject* self, PyObject* args)
{
    long arg1, arg2, result;

    /* Парсинг аргументов из PyObject в типы языка С. */
    if (!PyArg_ParseTuple(args, "ll", &arg1, &arg2))
        return NULL;

    /* Выполнение операции. */
    result = arg1 + arg2;

    /* Возврат PyObject со значением result. */
    return PyLong_FromLong(result);
}

/* Список методов модуля */
static PyMethodDef simple_methods[] =
{
    { "add",  simple_add, METH_VARARGS, "Сложение двух чисел."},
    { NULL, NULL, 0, NULL }
};

/* Описание модуля */
static struct PyModuleDef simple_module =
{
    PyModuleDef_HEAD_INIT,
    "simple",
    "Это тестовый модуль с именем simple",
    -1,
    simple_methods
};

/* Функция, которую вызывает питон для загрузки модуля */
PyMODINIT_FUNC
PyInit_simple(void)
{
    PyObject *m;

    m = PyModule_Create(&simple_module);
    if (m == NULL)
        return NULL;

    return m;
}
