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

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_paralution();

  if (argc > 2) {
    set_omp_threads_paralution(atoi(argv[2]));
  } 

  info_paralution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  // Linear Solver
  GMRES<LocalMatrix<double>, LocalVector<double>, double > ls;

  // Preconditioner
  AS<LocalMatrix<double>, LocalVector<double>, double > p; // Additive Schwarz
  //  RAS<LocalMatrix<double>, LocalVector<double>, double > p; // Restricted Additive Schwarz

  // Second level preconditioners
  Solver<LocalMatrix<double>, LocalVector<double>, double > **p2;

  int n = 2;
  int *size;
  size = new int[n];

  p2 = new Solver<LocalMatrix<double>, LocalVector<double>, double >*[n];

  for (int i=0; i<n; ++i) {
    size[i] = mat.get_nrow() / n ;   

    MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > *mc;
    mc = new MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double >;
    p2[i] = mc;

  }

  double tick, tack;
  
  rhs.Ones();
  x.Zeros(); 

  p.Set(n, 
        4,
        p2);
  
  ls.SetOperator(mat);
  ls.SetPreconditioner(p);
  //  ls.Verbose(2);

  ls.Build();

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  ls.MoveToAccelerator();
  
  mat.info();

  tick = paralution_time();

  ls.Solve(rhs, &x);

  tack = paralution_time();
  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  stop_paralution();

  delete[] size;
  delete[] p2;

  return 0;
}
