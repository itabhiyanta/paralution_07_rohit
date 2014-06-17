// *************************************************************************
//
//    PARALUTION   www.paralution.com
//
//    Copyright (C) 2012-2014 Dimitar Lukarski
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// *************************************************************************



// PARALUTION version 0.7.0 


#ifndef PARALUTION_UTILS_MATH_FUNCTIONS_HPP_
#define PARALUTION_UTILS_MATH_FUNCTIONS_HPP_

namespace paralution {

/// Return absolute float value
float paralution_abs(const float val);
/// Return absolute double value
double paralution_abs(const double val);
/// Return absolute int value
int paralution_abs(const int val);


int bubmap_create(const double *, int *, const int, const int, const int, const int, int*, const int);
int makebubmap(const int , const int, const int, const int , const double *, int *, const int , const int);
int instore(int *, int , int *, int);//linear search of unique values
int fixbubmap(int *, int, int);
int calclvlstval(int, int, int, int, double *, int *, int, int, int);
int get_vecnum(int rowid, int xdim, int ydim, int zdim, int novecni_x_,
	       int novecni_y_, int novecni_z_);
int top(const int, const int, const int);

void swap_int(int *in1, int *in2, int *tmp);
void swap_dbl(double *in1, double *in2, double *tmp);
void quick_sort(int *cols, int  *rows, double *vals, int low, int high);
}

#endif // PARALUTION_UTILS_MATH_FUNCTIONS_HPP_

