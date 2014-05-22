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

  // read from file 
  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  MixedPrecisionDC<LocalMatrix<double>, LocalVector<double>, double,
                   LocalMatrix<float>, LocalVector<float>, float> mp;

  CG<LocalMatrix<float>, LocalVector<float>, float> cg;
  MultiColoredILU<LocalMatrix<float>, LocalVector<float>, float> p;

  double tick, tack;

  rhs.Ones();
  x.Zeros();

  // setup a lower tol for the inner solver
  cg.SetPreconditioner(p);
  cg.Init(1e-5, 1e-2, 1e+20,
          100000);

  // setup the mixed-precision DC
  mp.SetOperator(mat);
  mp.Set(cg);

  mp.Build();

  tick = paralution_time();

  mp.Solve(rhs, &x);

  tack = paralution_time();

  std::cout << "Solver execution:" << (tack-tick)/1000000 << " sec" << std::endl;

  stop_paralution();

  return 0;
}
