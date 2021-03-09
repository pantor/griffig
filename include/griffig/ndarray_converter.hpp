#pragma once
// borrowed in spirit from https://github.com/yati-sagade/opencv-ndarray-conversion
// MIT License

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>


#if PY_VERSION_HEX >= 0x03000000
    #define PyInt_Check PyLong_Check
    #define PyInt_AsLong PyLong_AsLong
#endif


static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...) {
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}


class PyAllowThreads {
    PyThreadState* _state;

public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads() {
        PyEval_RestoreThread(_state);
    }
};


class PyEnsureGIL {
    PyGILState_STATE _state;

public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL() {
        PyGILState_Release(_state);
    }
};


class NumpyAllocator: public cv::MatAllocator {
public:
    NumpyAllocator() {
        stdAllocator = cv::Mat::getStdAllocator();
    }

    ~NumpyAllocator() {}

    cv::UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const {
        cv::UMatData* u = new cv::UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for (int i = 0; i < dims - 1; i++) {
            step[i] = (size_t)_strides[i];
        }
        step[dims-1] = CV_ELEM_SIZE(type);
        u->size = sizes[0]*step[0];
        u->userdata = o;
        return u;
    }

#if CV_VERSION_MAJOR >= 4
    cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, cv::AccessFlag flags, cv::UMatUsageFlags usageFlags) const
#else
    cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, cv::UMatUsageFlags usageFlags) const
#endif
    {
        if (data != 0) {
            CV_Error(cv::Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case
            return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for ( i = 0; i < dims; i++) {
            _sizes[i] = sizes[i];
        }
        if (cn > 1) {
            _sizes[dims++] = cn;
        }
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if (!o) {
            CV_Error_(cv::Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        }
        return allocate(o, dims0, sizes, type, step);
    }

#if CV_VERSION_MAJOR >= 4
    bool allocate(cv::UMatData* u, cv::AccessFlag accessFlags, cv::UMatUsageFlags usageFlags) const
#else
    bool allocate(cv::UMatData* u, int accessFlags, cv::UMatUsageFlags usageFlags) const
#endif
    {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(cv::UMatData* u) const {
        if (!u) {
            return;
        }
            
        PyEnsureGIL gil;
        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);
        if (u->refcount == 0) {
            PyObject* o = (PyObject*)u->userdata;
            Py_XDECREF(o);
            delete u;
        }
    }

    const cv::MatAllocator* stdAllocator;
};


struct NDArrayConverter {
    // must call this first, or the other routines don't work!
    static bool init_numpy() {
        import_array1(false);
        return true;
    }

    static bool toMat(PyObject* o, cv::Mat &m);
    static PyObject* toNDArray(const cv::Mat& mat);
};


NumpyAllocator g_numpyAllocator;


bool NDArrayConverter::toMat(PyObject *o, cv::Mat &m) {
    bool allowND = true;
    if (!o || o == Py_None) {
        if (!m.data) {
            m.allocator = &g_numpyAllocator;
        }
        return true;
    }

    if (PyInt_Check(o)) {
        double v[] = {static_cast<double>(PyInt_AsLong((PyObject*)o)), 0., 0., 0.};
        m = cv::Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if (PyFloat_Check(o)) {
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = cv::Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if (PyTuple_Check(o)) {
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = cv::Mat(sz, 1, CV_64F);
        for (i = 0; i < sz; i++) {
            PyObject* oi = PyTuple_GET_ITEM(o, i);
            if (PyInt_Check(oi)) {
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            } else if (PyFloat_Check(oi)) {
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            } else {
                failmsg("return value is not a numerical tuple");
                m.release();
                return false;
            }
        }
        return true;
    }

    if (!PyArray_Check(o)) {
        failmsg("return value is not a numpy array, neither a scalar");
        return false;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if (type < 0) {
        if (typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG) {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        } else {
            failmsg("return value data type = %d is not supported", typenum);
            return false;
        }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM) {
        failmsg("return value dimensionality (=%d) is too high", ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1];
    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for (int i = ndims-1; i >= 0 && !needcopy; i--) {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        // the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
        if ( (i == ndims-1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _sizes[i] > 1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }

    if (ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2]) {
        needcopy = true;
    }

    if (needcopy) {
        if (needcast) {
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        }
        else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    // Normalize strides in case NPY_RELAXED_STRIDES is set
    size_t default_step = elemsize;
    for (int i = ndims - 1; i >= 0; --i) {
        size[i] = (int)_sizes[i];
        if (size[i] > 1) {
            step[i] = (size_t)_strides[i];
            default_step = step[i] * size[i];
        } else {
            step[i] = default_step;
            default_step *= size[i];
        }
    }

    // handle degenerate case
    if (ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if (ismultichannel) {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if (ndims > 2 && !allowND) {
        failmsg("return value has more than 2 dimensions");
        return false;
    }

    m = cv::Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
    m.addref();

    if (!needcopy) {
        Py_INCREF(o);
    }
    m.allocator = &g_numpyAllocator;
    return true;
}

PyObject* NDArrayConverter::toNDArray(const cv::Mat& m) {
    if (!m.data) {
        Py_RETURN_NONE;
    }

    cv::Mat temp, *p = (cv::Mat*)&m;
    if (!p->u || p->allocator != &g_numpyAllocator) {
        temp.allocator = &g_numpyAllocator;
        try {
            PyAllowThreads allowThreads;
            m.copyTo(temp);
        } catch (const cv::Exception &e) {
            PyErr_SetString(opencv_error, e.what());
            return 0;
        }
        p = &temp;
    }
    PyObject* o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    return o;
}


namespace pybind11 { namespace detail {
template<>
struct type_caster<cv::Mat> {
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    bool load(handle src, bool) {
        return NDArrayConverter::toMat(src.ptr(), value);
    }

    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        return handle(NDArrayConverter::toNDArray(m));
    }
};

}} // namespace pybind11::detail
