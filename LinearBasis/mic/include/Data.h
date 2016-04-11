#ifndef DATA_H
#define DATA_H

#include <algorithm>
#include <assert.h>
#include <mic_runtime.h>
#include <vector>

#define MIC_ERROR_CHECK(x) do { micError_t err = x; if (( err ) != micSuccess ) { \
	printf ("Error %d at %s :%d \n" , err, __FILE__ , __LINE__ ) ; exit(-1);\
}} while (0)

template<typename T>
class Vector
{
	std::vector<T> data;
	T* device_data;

	// Denotes that host array items have been possibly changed,
	// and we need to reload new data into device memory.
	bool dirty;

public :
	Vector() : dirty(false), device_data(NULL) { }

	~Vector()
	{
		if (device_data)
			MIC_ERROR_CHECK(micFree(device_data));
	}

	Vector(int dim) : dirty(false), device_data(NULL)
	{
		data.resize(dim);
		MIC_ERROR_CHECK(micMallocAligned((void**)&device_data,
			AVX_VECTOR_SIZE * sizeof(T), dim * sizeof(T)));
	}

	T* getData()
	{
		if (dirty)
		{
			MIC_ERROR_CHECK(micMemcpy(device_data, &data[0], data.size() * sizeof(T), micMemcpyHostToDevice));
			dirty = false;
		}
		return device_data;
	}

	T& operator()(int x)
	{
		assert(x < data.size());
		dirty = true;
		return data[x];
	}

	const T& operator()(int x) const
	{
		assert(x < data.size());
		return data[x];
	}
	
	int length() { return data.size(); }
	
	void resize(int length)
	{
		if (length >= data.size())
		{
			MIC_ERROR_CHECK(micFree(device_data));
			MIC_ERROR_CHECK(micMallocAligned((void**)&device_data,
				AVX_VECTOR_SIZE * sizeof(T), length * sizeof(T)));
		}
		data.resize(length);
	}
};

template<typename T>
class Matrix
{
	std::vector<T> data;
	T* device_data;
	int dimY, dimX;
	
	// Denotes that host array items have been possibly changed,
	// and we need to reload new data into device memory.
	bool dirty;

public :
	Matrix() : dimY(0), dimX(0), dirty(false), device_data(NULL) { }
	
	~Matrix()
	{
		if (device_data)
			MIC_ERROR_CHECK(micFree(device_data));
	}

	Matrix(int dimY_, int dimX_) : dimY(dimY_), dimX(dimX_), dirty(false), device_data(NULL)
	{
		data.resize(dimY_ * dimX_);
		MIC_ERROR_CHECK(micMallocAligned((void**)&device_data,
			AVX_VECTOR_SIZE * sizeof(T), dimY_ * dimX_ * sizeof(T)));
	}

	T* getData()
	{
		if (dirty)
		{
			MIC_ERROR_CHECK(micMemcpy(device_data, &data[0], data.size() * sizeof(T), micMemcpyHostToDevice));
			dirty = false;
		}
		return device_data;
	}
	
	T& operator()(int y, int x)
	{
		int index = x + dimX * y;
		assert(index < data.size());
		dirty = true;
		return data[index];
	}

	const T& operator()(int x, int y) const
	{
		int index = x + dimX * y;
		assert(index < data.size());
		return data[index];
	}
	
	int dimy() { return dimY; }

	int dimx() { return dimX; }
	
	void resize(int dimY_, int dimX_)
	{
		dimY = dimY_; dimX = dimX_;

		if (dimY_ * dimX_ >= data.size())
		{
			MIC_ERROR_CHECK(micFree(device_data));
			MIC_ERROR_CHECK(micMallocAligned((void**)&device_data,
				dimY_ * dimX_ * sizeof(T), AVX_VECTOR_SIZE * sizeof(T)));
		}
		data.resize(dimY_ * dimX_);
	}
	
	void fill(T value)
	{
		std::fill(data.begin(), data.end(), value);
		dirty = true;
	}
};

class Interpolator;

class Data
{
	int dim, vdim, nno, TotalDof, Level;
	Matrix<real> index;
	Matrix<real> surplus, surplus_t;

	friend class Interpolator;

public :
	int getNno() const;

	void load(const char* filename);

	Data();
};

#endif // DATA_H

