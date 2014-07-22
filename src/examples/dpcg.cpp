// *************************************************************************
//
//    PARALUTION   www.paralution.com
//
//    Copyright (C) 2012-2013 Dimitar Lukarski
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

#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include <paralution.hpp>
using namespace std;
using namespace paralution;

// #define GUUS
// #define SCALIN
#define GPURUN
#define BUBFLO
#define MATDIA
int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> <initial_guess> <rhs> [Num threads]" << std::endl;
    exit(1);
  }

  init_paralution();

//   if (argc > 4) {
//     set_omp_threads_paralution(atoi(argv[5]));
//   } 
  set_omp_threads_paralution(8);
  
  info_paralution();

  struct timeval now;
  double tick, tack, b=0.0f,s=0.0f, lprep=0.0f, sol_norm, diff_norm, ones_norm;
  double *phi_ptr=NULL;
  int *bubmap_ptr=NULL, phisize, maxbmap, setlssd, lvst_offst;
  int xdim, ydim, zdim, defvex_perdirec, defvex_perdirec_y, defvex_perdirec_z;
  DPCG<LocalMatrix<double>, LocalVector<double>, double > ls;
#ifdef BUBFLO  
  xdim=atoi(argv[5]);	ydim=atoi(argv[5]);	zdim=atoi(argv[5]);
  setlssd=atoi(argv[6]);
  defvex_perdirec=atoi(argv[7]);
  lvst_offst=atoi(argv[8]);
  phisize=(xdim+2*lvst_offst)*(ydim+2*lvst_offst)*(zdim+2*lvst_offst);
#endif  
  LocalVector<double> x;
  LocalVector<double>refsol;
  LocalVector<double>refones;
  LocalVector<double>chk_r;
  LocalVector<double> rhs;
  LocalMatrix<double> mat;
  LocalVector<double> Dinvhalf_min;
  LocalVector<double> Dinvhalf_plus;
#ifdef GUUS  
  LocalMatrix<double> Zin;
#endif  
  mat.ReadFileMTX(std::string(argv[1]));
  mat.info();
#ifdef GUUS  
  Zin.ReadFileMTX(std::string(argv[2]));
  Zin.info();
  
#endif  
  x.Allocate("x", mat.get_nrow());
  refsol.Allocate("refsol", mat.get_nrow());
  refones.Allocate("refones", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());
  chk_r.Allocate("chk_r", mat.get_nrow());
#ifdef BUBFLO
   x.ReadFileASCII(std::string(argv[2]));
#endif   
  rhs.ReadFileASCII(std::string(argv[3]));
#ifdef GUUS
  x.SetRandom(0.0,1.0,1000);
  refsol.ReadFileASCII(std::string(argv[4]));
  refones.Ones();
#endif  

  
  //refsol.Ones();
// 
//   // Uncomment for GPU

  
  gettimeofday(&now, NULL);
  tick = now.tv_sec*1000000.0+(now.tv_usec);

#ifdef BUBFLO  
  if(setlssd){
    LocalVector<double> phi;
    LocalVector<int> bubmap;
    phi.Allocate("PHI", phisize);
    bubmap.Allocate("bubmap",mat.get_nrow());
    phi.ReadFileASCII(std::string(argv[4]));
    
    bubmap.LeaveDataPtr(&bubmap_ptr);
    phi.LeaveDataPtr(&phi_ptr);

    bubmap_create(phi_ptr, bubmap_ptr, xdim, xdim, xdim, mat.get_nrow(), &maxbmap, lvst_offst);
    phi.Clear();
    
  }
  ls.Setxdim(xdim);
  ls.SetNVectors_eachdirec(defvex_perdirec, defvex_perdirec, defvex_perdirec);
  ls.Set_alldims(xdim, xdim, xdim);
  ls.Setlvst_offst(lvst_offst);
  ls.SetNVectors(defvex_perdirec);
  ls.SetZlssd(setlssd);
#endif  
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   lprep=(tack-tick)/1000000;
//   std::cout << "levelset_prep" << lprep << " sec" << std::endl;
  // Linear Solver
//   return 0;
 
#ifdef SCALIN
  
  mat.ExtractInverseDiagonal_sqrt(&Dinvhalf_min, -1);
  mat.ExtractInverseDiagonal_sqrt(&Dinvhalf_plus, 1);
  
  mat.DiagonalMatrixMult(Dinvhalf_min);
  mat.DiagonalMatrixMult_fromL(Dinvhalf_min);
  
  //x.PointWiseMult(Dinvhalf_plus);
  rhs.PointWiseMult(Dinvhalf_min);
//   rhs.Scale(0.3);
#endif
#ifdef GUUS  
   ls.SetZ(Zin);
#endif   
  ls.SetOperator(mat);
  ls.Init(0.0, 1e-6, 1e8, 200000);
//  ls.RecordResidualHistory();
  

#ifdef BUBFLO
  ls.MakeZ_CSR(); // requires xdim_ and novecni_ and zlssd_ to be set
  if(setlssd)
    ls.MakeZLSSD(bubmap_ptr, maxbmap); // bubmap must be ready and maxbmap available
#endif    
//   
#ifdef GPURUN  
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  chk_r.MoveToAccelerator();
  Dinvhalf_min.MoveToAccelerator();
  Dinvhalf_plus.MoveToAccelerator();
  
#endif  
  
  ls.Build();
#ifdef MATDIA  
   mat.ConvertToDIA();
#endif  

  gettimeofday(&now, NULL);
  tack = now.tv_sec*1000000.0+(now.tv_usec);
  b=(tack-tick)/1000000;
  std::cout << "Building:" << b+lprep << " sec" << std::endl;
  
//   ls.Verbose(2);

  mat.info();

  gettimeofday(&now, NULL);
  tick = now.tv_sec*1000000.0+(now.tv_usec);

  ls.Solve(rhs, &x);

  gettimeofday(&now, NULL);
  tack = now.tv_sec*1000000.0+(now.tv_usec);
  s=(tack-tick)/1000000;
  std::cout << "Solver execution:" << s << " sec" << std::endl;
  std::cout << "Total execution:" << s+b << " sec" << std::endl;
  
#ifdef SCALIN
  x.PointWiseMult(Dinvhalf_min);
#endif

//   

  
#ifdef GUUS  
//   x.WriteFileASCII("x_solution_shell_inv_neumann.rec");
  //ls.RecordHistory("res__ongpu_tns.rec");
  x.MoveToHost();
  x.WriteFileASCII("x_neumann.rec");
  x.MoveToAccelerator();
  sol_norm=x.Norm();
  mat.Apply(x, &chk_r); 
  chk_r.ScaleAdd(double(-1.0), rhs);
  cout<<"\n Real Residual Norm is "<<chk_r.Norm();
  cout<<"\n Norm of Solution is "<<sol_norm<<endl;
  cout<<"\n Norm of Reference Solution is "<<refsol.Norm()<<endl;
  cout<<"\n Norm of Ones is "<<refones.Norm()<<endl;
  x.MoveToHost();
  refones.AddScale(x,(double)-1.0f);
  x.AddScale(refsol,(double)-1.0f);
  diff_norm=x.Norm();
  ones_norm=refones.Norm();
  cout<<"\n Relative Norm of Calculated Solution w.r.t. Reference is "<<((double)diff_norm/(double)sol_norm)<<endl;
  cout<<"\n Relative Norm of Calculated Solution w.r.t. Ones is "<<((double)ones_norm/(double)sol_norm)<<endl;
#endif  
  ls.Clear();
  


  stop_paralution();

  return 0;
}
