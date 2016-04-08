#ifndef DATA_H
#define DATA_H

#include <algorithm>
#include <assert.h>
#include <vector>

template<typename T>
class Vector
{
	std::vector<T> data;

public :
	Vector() { }

	Vector(int dim) { data.resize(dim); }

	T* getData() { return &data[0]; }

	T& operator()(int x)
	{
		assert(x < data.size());
		return data[x];
	}
	
	int length() { return data.size(); }
	
	void resize(int length) { data.resize(length); }
};

template<typename T>
class Matrix
{
	std::vector<T> data;
	int dimX, dimY;

public :
	Matrix() : dimX(0), dimY(0) { }

	Matrix(int dimX_, int dimY_) : dimX(dimX_), dimY(dimY_)
	{
		data.resize(dimX_ * dimY_);
	}

	T* getData() { return &data[0]; }
	
	T& operator()(int x, int y)
	{
		int index = x + dimX * y;
		assert(index < data.size());
		return data[index];
	}

	int dimx() { return dimX; }
	
	int dimy() { return dimY; }
	
	void resize(int dimX_, int dimY_)
	{
		dimX = dimX_; dimY = dimY_;
		data.resize(dimX_ * dimY_);
	}
	
	void fill(T value)
	{
		std::fill(data.begin(), data.end(), value);
	}
};

class Interpolator;

class Data
{
	int dim, vdim, nno, TotalDof, Level;
	Matrix<int> index;
	Matrix<real> surplus, surplus_t;

	friend class Interpolator;

public :
	void load(const char* filename);

	Data();
};

#endif // DATA_H

