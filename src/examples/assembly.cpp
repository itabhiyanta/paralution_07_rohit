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


#include <iostream>
#include <cstdlib>

#include <paralution.hpp>

using namespace paralution;

int main(int argc, char* argv[]) {

  init_paralution();

  if (argc > 1)
    set_omp_threads_paralution(atoi(argv[1]));

  info_paralution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalMatrix<double> mat;

  // rows
  int i[11] = {0, 1, 2, 3, 4, 5, 1, 2, 4, 3, 3 };

  // cols
  int j[11] = {0, 1, 2, 3, 4, 5, 2, 2, 5, 2, 2 };

  // values
  double a[11] = {2.3, 3.5, 4.7, 6.3, 0.4, 4.3, 6.2, 4.4, 4.6, 0.7, 4.8 };

  mat.Assemble(i, j, a, 11, "A");
  mat.info();

  rhs.Assemble(i, a, 11, "x");
  rhs.info();

  x.Allocate("rhs", mat.get_ncol());
  x.Zeros();

  // Check if the data is correct
  if (( mat.Check() == false ) ||
      ( x.Check() == false ) ||
      ( rhs.Check() == false )) {
    std::cout << "The PARALUTION objects are not ok" << std::endl;
    exit(1);
  }

  // Linear Solver
  GMRES<LocalMatrix<double>, LocalVector<double>, double > ls;

  ls.SetOperator(mat);

  ls.Build();

  ls.Solve(rhs, &x);

  stop_paralution();

  return 0;
}
