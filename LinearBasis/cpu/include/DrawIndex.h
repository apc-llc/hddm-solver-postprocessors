#ifndef DRAW_INDEX_H
#define DRAW_INDEX_H

#include <vector>

struct Index
{
	int i, j;
	int rowind;

	Index() : i(0), j(0), rowind(0) { }
	
	Index(int i_, int j_, int rowind_) : i(i_), j(j_), rowind(rowind_) { }
};

void DrawIndex(int dim, int vdim, std::vector<Index>& indexes, const char* filename);

#endif // DRAW_INDEX_H

