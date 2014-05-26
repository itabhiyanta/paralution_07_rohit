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

#define GPURUN	4
// #define MATDIA	1
// #define SCALIN	55
#define GUUS	2
// #define	BUBFLO	3
using namespace std;
using namespace paralution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> <initial_guess> <rhs> [Num threads]" << std::endl;
    exit(1);
  }

  init_paralution();

//   if (argc > 4) {
//     set_omp_threads_paralution(atoi(argv[]));
//   } 
  set_omp_threads_paralution(8);
  info_paralution();

  struct timeval now;
  double tick, tack, b,s, sol_norm, diff_norm, ones_norm;
  double *phi_ptr=NULL;
  int *bubmap_ptr=NULL, phisize, maxbmap, setlssd, lvst_offst;
  int xdim, ydim, zdim, defvex_perdirec;
#ifdef BUBFLO
  xdim=atoi(argv[5]);
  setlssd=atoi(argv[6]);
  defvex_perdirec=atoi(argv[7]);
  lvst_offst=atoi(argv[8]);
  phisize=(xdim+2*lvst_offst)*(ydim+2*lvst_offst)*(zdim+2*lvst_offst);
#endif  
  LocalVector<double> x;
  
  LocalVector<double> rhs;
  LocalMatrix<double> mat;
  LocalVector<double> Dinvhalf_min;
  LocalVector<double> Dinvhalf_plus;
#ifdef GUUS  
  LocalMatrix<double> Zin;
  LocalVector<double> refsol;
  LocalVector<double> refones;
#endif  
  mat.ReadFileMTX(std::string(argv[1]));
  mat.info();
#ifdef GUUS  
  Zin.ReadFileMTX(std::string(argv[2]));
  Zin.info();
  refsol.Allocate("refsol", mat.get_nrow());
  refones.Allocate("refones", mat.get_nrow());
  //refsol.Ones();
  refsol.ReadFileASCII(std::string(argv[4]));
  refones.Ones();
#endif  
  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());
  
  
  // Linear Solver
  DPCG<LocalMatrix<double>, LocalVector<double>, double > ls;
  MultiElimination<LocalMatrix<double>, LocalVector<double>, double > p;
  Jacobi<LocalMatrix<double>, LocalVector<double>, double > j_p;
  MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double > mcilu_p;
  ILU<LocalMatrix<double>, LocalVector<double>, double > ilu_p;
  MultiColoredSGS<LocalMatrix<double>, LocalVector<double>, double > mcsgs_p;
  FSAI<LocalMatrix <double>, LocalVector<double>, double > fsai_p ;
  SPAI<LocalMatrix <double>, LocalVector <double>, double > spai_p ;
  


#ifdef GPURUN  
  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
#endif  
  
#ifdef SCALIN
  mat.ExtractInverseDiagonal_sqrt(&Dinvhalf_min, -1);
  mat.ExtractInverseDiagonal_sqrt(&Dinvhalf_plus, 1);
  
  mat.DiagonalMatrixMult(Dinvhalf_min);
  mat.DiagonalMatrixMult_fromL(Dinvhalf_min);
  
  //x.PointWiseMult(Dinvhalf_plus);
  rhs.PointWiseMult(Dinvhalf_min);
#endif
  
    /////////////////////////////////////////////////////////////////  
//   std::cout << "-----------------------------------------------" << std::endl;
//   std::cout << "DPCG solver MCSGS" << std::endl;
// #ifdef GUUS
//   rhs.ReadFileASCII(std::string(argv[3]));
//   x.SetRandom(0.0,1.0,1000);
//   ls.SetZ(Zin);
// #endif
//   
// #ifdef BUBFLO  
//   x.ReadFileASCII(std::string(argv[2]));
//   rhs.ReadFileASCII(std::string(argv[3]));
// #endif
// 
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
//   
// #ifdef BUBFLO  
//   if(setlssd){
//     LocalVector<double> phi;
//     LocalVector<int> bubmap;
//     phi.Allocate("PHI", phisize);
//     bubmap.Allocate("bubmap",mat.get_nrow());
//     phi.ReadFileASCII(std::string(argv[4]));
//     
//     bubmap.LeaveDataPtr(&bubmap_ptr);
//     phi.LeaveDataPtr(&phi_ptr);
// 
//     x.SetRandom(0.0,1.0,1000);
//     bubmap_create(phi_ptr, bubmap_ptr, xdim, xdim, xdim, mat.get_nrow(), &maxbmap, lvst_offst);
//     phi.Clear();
//     
//   }
//   ls.Setxdim(xdim);
//   ls.SetNVectors(defvex_perdirec);
//   ls.Setlvst_offst(lvst_offst);
//   ls.SetZlssd(setlssd);
//   mat.ConvertToCSR();  
// #endif
//   
//   ls.SetOperator(mat);
//   ls.SetPreconditioner (mcsgs_p) ;
//   mcsgs_p.SetPrecondMatrixFormat(HYB);
//   
//  
//   ls.Init(0.0, 1e-6, 1e8, 200000);
// #ifdef BUBFLO  	
//   ls.MakeZ_CSR(); // requires xdim_ and novecni_ and zlssd_ to be set
//   if(setlssd)
//     ls.MakeZLSSD(bubmap_ptr, maxbmap); // bubmap must be ready and maxbmap available	
// #endif
//     
//   
// 
//   ls.Build();
// #ifdef MATDIA  
//   mat.ConvertToDIA();
// #endif  
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   b=(tack-tick)/1000000;
//   std::cout << "Building:" << b << " sec" << std::endl;
//   
// 
//   mat.info();
// 
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
// 
//   ls.Solve(rhs, &x);
// 
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   s= (tack-tick)/1000000;
//   std::cout << "Solver execution:" << s << " sec" << std::endl;
//   std::cout << "Total execution:" << s+b << " sec" << std::endl;
// #ifdef GUUS 
//   x.MoveToHost();
//   sol_norm=x.Norm();
//   cout<<"\n Norm of Solution is "<<sol_norm<<endl;
//   cout<<"\n Norm of Reference Solution is "<<refsol.Norm()<<endl;
//   refones.AddScale(x,(double)-1.0f);
//   x.AddScale(refsol,(double)-1.0f);
//   
//   
//   diff_norm=x.Norm();
//   ones_norm=refones.Norm();
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Reference is "<<((double)diff_norm/(double)sol_norm)<<endl;
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Ones is "<<((double)ones_norm/(double)sol_norm)<<endl;
// #endif  
//   //x.WriteFileASCII("x_solution1e3shell_ilu01.rec");
//   ls.Clear();
  /////////////////////////////////////////////////////////////////  
 
  /////////////////////////////////////////////////////////////////  
//   std::cout << "-----------------------------------------------" << std::endl;
//   std::cout << "DPCG solver FSAI" << std::endl;
// #ifdef GUUS  
//   rhs.ReadFileASCII(std::string(argv[3]));
//   x.SetRandom(0.0,1.0,1000);
//   ls.SetZ(Zin);
// #endif
// 
//   
// #ifdef BUBFLO  
//   x.ReadFileASCII(std::string(argv[2]));
//   rhs.ReadFileASCII(std::string(argv[3]));
// #endif
// 
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
//   
// #ifdef BUBFLO  
//   if(setlssd){
//     LocalVector<double> phi;
//     LocalVector<int> bubmap;
//     phi.Allocate("PHI", phisize);
//     bubmap.Allocate("bubmap",mat.get_nrow());
//     phi.ReadFileASCII(std::string(argv[4]));
//     
//     bubmap.LeaveDataPtr(&bubmap_ptr);
//     phi.LeaveDataPtr(&phi_ptr);
// 
//     //x.SetRandom(0.0,1.0,1000);
//     bubmap_create(phi_ptr, bubmap_ptr, xdim, xdim, xdim, mat.get_nrow(), &maxbmap, lvst_offst);
//     phi.Clear();
//     
//   }
//   ls.Setxdim(xdim);
//   ls.SetNVectors(defvex_perdirec);
//   ls.SetZlssd(setlssd);
//   mat.ConvertToCSR();  
// #endif    
//   fsai_p.Init (2) ;
//   
//   ls.SetOperator(mat);
//   ls.SetPreconditioner (fsai_p) ;
//   fsai_p.SetPrecondMatrixFormat(HYB);
//   
//  
//   ls.Init(0.0, 1e-6, 1e8, 200000);
// 
//   
//   
// #ifdef BUBFLO  
//   ls.MakeZ_CSR(); // requires xdim_ and novecni_ and zlssd_ to be set
//   if(setlssd)
//     ls.MakeZLSSD(bubmap_ptr, maxbmap); // bubmap must be ready and maxbmap available
// #endif  
//   
// 
//   ls.Build();
// #ifdef MATDIA  
//   mat.ConvertToDIA();
// #endif
//   
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   b=(tack-tick)/1000000;
//   std::cout << "Building:" << b << " sec" << std::endl;
//   
// 
// //   mat.info();
// 
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
// 
//   ls.Solve(rhs, &x);
// 
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   s= (tack-tick)/1000000;
//   std::cout << "Solver execution:" << s << " sec" << std::endl;
//   std::cout << "Total execution:" << s+b << " sec" << std::endl;
// #ifdef GUUS
//     x.MoveToHost();
//   sol_norm=x.Norm();
//   cout<<"\n Norm of Solution is "<<sol_norm<<endl;
//   cout<<"\n Norm of Reference Solution is "<<refsol.Norm()<<endl;
//   refones.AddScale(x,(double)-1.0f);
//   x.AddScale(refsol,(double)-1.0f);
//   
//   
//   diff_norm=x.Norm();
//   ones_norm=refones.Norm();
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Reference is "<<((double)diff_norm/(double)sol_norm)<<endl;
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Ones is "<<((double)ones_norm/(double)sol_norm)<<endl;
// #endif  
//   //x.WriteFileASCII("x_solution1e3shell_ilu01.rec");
//   ls.Clear();
// //   
  
///////////////////////////////////////////////////////////////  
//   std::cout << "-----------------------------------------------" << std::endl;
//   std::cout << "DPCG solver ILU-p" << std::endl;
// #ifdef GUUS  
//   rhs.ReadFileASCII(std::string(argv[3]));
//   x.SetRandom(0.0,1.0,1000);
//   ls.SetZ(Zin);
// #endif
//   
// #ifdef BUBFLO  
//    x.ReadFileASCII(std::string(argv[2]));
//   rhs.ReadFileASCII(std::string(argv[3]));
// #endif  
//   
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
//   
// #ifdef BUBFLO
//   if(setlssd){
//     LocalVector<double> phi;
//     LocalVector<int> bubmap;
//     phi.Allocate("PHI", phisize);
//     bubmap.Allocate("bubmap",mat.get_nrow());
//     phi.ReadFileASCII(std::string(argv[4]));
//     
//     bubmap.LeaveDataPtr(&bubmap_ptr);
//     phi.LeaveDataPtr(&phi_ptr);
// 
//     //x.SetRandom(0.0,1.0,1000);
//     bubmap_create(phi_ptr, bubmap_ptr, xdim, xdim, xdim, mat.get_nrow(), &maxbmap, lvst_offst);
//     phi.Clear();
//     
//   }
//   ls.Setxdim(xdim);
//   ls.SetNVectors(defvex_perdirec);
//   ls.SetZlssd(setlssd);
//   mat.ConvertToCSR();  
// #endif
//   
//   ilu_p.Init(0);
//   ls.SetOperator(mat);
//   ls.SetPreconditioner(ilu_p);
//   ls.Init(0.0, 1e-6, 1e8, 20000);
// 
// #ifdef BUBFLO  
//   ls.MakeZ_CSR(); // requires xdim_ and novecni_ and zlssd_ to be set
//   if(setlssd)
//     ls.MakeZLSSD(bubmap_ptr, maxbmap); // bubmap must be ready and maxbmap available
// #endif  
//   
//   ls.Build();
// #ifdef MATDIA  
//   mat.ConvertToDIA();
// #endif  
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   b=(tack-tick)/1000000;
//   std::cout << "Building:" << b << " sec" << std::endl;
//   
//   ls.Verbose(2);
// //   mat.info();
// 
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
//   
//   ls.Solve(rhs, &x);
// 
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   s= (tack-tick)/1000000;
//   std::cout << "Solver execution:" << s << " sec" << std::endl;
//   std::cout << "Total execution:" << s+b << " sec" << std::endl;
// 
// #ifdef SCALIN
//   x.PointWiseMult(Dinvhalf_min);
// #endif
//   x.MoveToHost();
// //   x.WriteFileASCII("x_solution_shell_scal.rec");
// #ifdef GUUS
//   sol_norm=x.Norm();
//   cout<<"\n Norm of Solution is "<<sol_norm<<endl;
//   cout<<"\n Norm of Reference Solution is "<<refsol.Norm()<<endl;
//   refones.AddScale(x,(double)-1.0f);
//   x.AddScale(refsol,(double)-1.0f);
//   
//   x.MoveToHost();
//   diff_norm=x.Norm();
//   ones_norm=refones.Norm();
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Reference is "<<((double)diff_norm/(double)sol_norm)<<endl;
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Ones is "<<((double)ones_norm/(double)sol_norm)<<endl;
// #endif  
//   
//   ls.Clear();
  
/////////////////////////////////////////////////////////////////
//   std::cout << "-----------------------------------------------" << std::endl;
//   std::cout << "DPCG solver ME-ILU-J" << std::endl;
// #ifdef GUUS  
//   rhs.ReadFileASCII(std::string(argv[3]));
//   x.SetRandom(0.0,1.0,1000);
//   ls.SetZ(Zin);
// #endif  
// 
// #ifdef BUBFLO
//    x.ReadFileASCII(std::string(argv[2]));
//    rhs.ReadFileASCII(std::string(argv[3]));
// #endif
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
//  
// #ifdef BUBFLO   
//   if(setlssd){
//     LocalVector<double> phi;
//     LocalVector<int> bubmap;
//     phi.Allocate("PHI", phisize);
//     bubmap.Allocate("bubmap",mat.get_nrow());
//     phi.ReadFileASCII(std::string(argv[4]));
//     
//     bubmap.LeaveDataPtr(&bubmap_ptr);
//     phi.LeaveDataPtr(&phi_ptr);
// 
//     //x.SetRandom(0.0,1.0,1000);
//     bubmap_create(phi_ptr, bubmap_ptr, xdim, xdim, xdim, mat.get_nrow(), &maxbmap, lvst_offst);
//     phi.Clear();
//     
//   }
//   ls.Setxdim(xdim);
//   ls.SetNVectors(defvex_perdirec);
//   ls.SetZlssd(setlssd);
//   mat.ConvertToCSR();
// #endif
//   p.Init(j_p, 1);
//   
//   ls.SetOperator(mat);
//   ls.SetPreconditioner(p);
//   
//   
//   
//   
//   ls.Init(0.0, 1e-6, 1e8, 200000);
// #ifdef BUBFLO  
//   ls.MakeZ_CSR(); // requires xdim_ and novecni_ and zlssd_ to be set
//   if(setlssd)
//     ls.MakeZLSSD(bubmap_ptr, maxbmap); // bubmap must be ready and maxbmap available
// #endif  
//   
//   ls.Build();
// #ifdef MATDIA  
//   mat.ConvertToDIA();
// #endif  
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   b=(tack-tick)/1000000;
//   std::cout << "Building:" << b << " sec" << std::endl;
//   
// 
//   mat.info();
// 
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
// 
//   ls.Solve(rhs, &x);
// 
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   s= (tack-tick)/1000000;
//   std::cout << "Solver execution:" << s << " sec" << std::endl;
//   std::cout << "Total execution:" << s+b << " sec" << std::endl;
// #ifdef GUUS  
//   x.MoveToHost();
//   sol_norm=x.Norm();
//   cout<<"\n Norm of Solution is "<<sol_norm<<endl;
//   cout<<"\n Norm of Reference Solution is "<<refsol.Norm()<<endl;
//   refones.AddScale(x,(double)-1.0f);
//   x.AddScale(refsol,(double)-1.0f);
//   
//   
//   diff_norm=x.Norm();
//   ones_norm=refones.Norm();
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Reference is "<<((double)diff_norm/(double)sol_norm)<<endl;
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Ones is "<<((double)ones_norm/(double)sol_norm)<<endl;
// #endif  
//   //x.WriteFileASCII("x_solution1e3shell_ilu01.rec");
//   ls.Clear();

/////////////////////////////////////////////////////////////////  
//   std::cout << "-----------------------------------------------" << std::endl;
//   std::cout << "DPCG solver ME-ILU-SGS" << std::endl;
// #ifdef GUUS  
//   rhs.ReadFileASCII(std::string(argv[3]));
//   x.SetRandom(0.0,1.0,1000);
//   ls.SetZ(Zin);
// #endif
//   
// #ifdef BUBFLO  
//   x.ReadFileASCII(std::string(argv[2]));
//   rhs.ReadFileASCII(std::string(argv[3]));
// #endif
//   
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
// 
// #ifdef BUBFLO  
//   if(setlssd){
//     LocalVector<double> phi;
//     LocalVector<int> bubmap;
//     phi.Allocate("PHI", phisize);
//     bubmap.Allocate("bubmap",mat.get_nrow());
//     phi.ReadFileASCII(std::string(argv[4]));
//     
//     bubmap.LeaveDataPtr(&bubmap_ptr);
//     phi.LeaveDataPtr(&phi_ptr);
// 
//     //x.SetRandom(0.0,1.0,1000);
//     bubmap_create(phi_ptr, bubmap_ptr, xdim, xdim, xdim, mat.get_nrow(), &maxbmap, lvst_offst);
//     phi.Clear();
//     
//   }
//   ls.Setxdim(xdim);
//   ls.SetNVectors(defvex_perdirec);
//   ls.SetZlssd(setlssd);
//   mat.ConvertToCSR();  
// #endif  
//   p.Init(mcsgs_p, 1);
//   ls.SetOperator(mat);
//   ls.SetPreconditioner(p);
// 
//   ls.Init(0.0, 1e-6, 1e8, 200000);
// //   ls.SetNVectors(4);
// #ifdef BUBFLO  
//   ls.MakeZ_CSR(); // requires xdim_ and novecni_ and zlssd_ to be set
//   if(setlssd)
//     ls.MakeZLSSD(bubmap_ptr, maxbmap); // bubmap must be ready and maxbmap available
// #endif
//     
//   
//   ls.Build();
// #ifdef MATDIA  
//   mat.ConvertToDIA();
// #endif  
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   b=(tack-tick)/1000000;
//   std::cout << "Building:" << b << " sec" << std::endl;
//   
// 
// //   mat.info();
// 
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
// 
//   ls.Solve(rhs, &x);
// 
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   s= (tack-tick)/1000000;
//   std::cout << "Solver execution:" << s << " sec" << std::endl;
//   std::cout << "Total execution:" << s+b << " sec" << std::endl;
// #ifdef GUUS  
// x.MoveToHost();
//   sol_norm=x.Norm();
//   cout<<"\n Norm of Solution is "<<sol_norm<<endl;
//   cout<<"\n Norm of Reference Solution is "<<refsol.Norm()<<endl;
//   refones.AddScale(x,(double)-1.0f);
//   x.AddScale(refsol,(double)-1.0f);
//   
//   
//   diff_norm=x.Norm();
//   ones_norm=refones.Norm();
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Reference is "<<((double)diff_norm/(double)sol_norm)<<endl;
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Ones is "<<((double)ones_norm/(double)sol_norm)<<endl;
// #endif  
//   //x.WriteFileASCII("x_solution1e3shell_ilu01.rec");
//   ls.Clear();
// 
// /////////////////////////////////////////////////////////////////  
//   
// /////////////////////////////////////////////////////////////////  
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "DPCG solver ME-ILU-ILU(0,1)" << std::endl;
#ifdef GUUS  
  rhs.ReadFileASCII(std::string(argv[3]));
  x.SetRandom(0.0,1.0,1000);
  ls.SetZ(Zin);
#endif
#ifdef BUBFLO  
  x.ReadFileASCII(std::string(argv[2]));
  rhs.ReadFileASCII(std::string(argv[3]));
#endif
  
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

    //x.SetRandom(0.0,1.0,1000);
    bubmap_create(phi_ptr, bubmap_ptr, xdim, xdim, xdim, mat.get_nrow(), &maxbmap, lvst_offst);
    phi.Clear();
    
  }
  ls.Setxdim(xdim);
  ls.SetNVectors(defvex_perdirec);
  ls.SetZlssd(setlssd);
  mat.ConvertToCSR();    
#endif  
//   mcilu_p.Init(0);
//   
//   p.Init(mcilu_p, 1, 0.0);
  mcilu_p.Set(0);
  p.Set(mcilu_p, 2, 0.4);

  ls.SetOperator(mat);
  ls.SetPreconditioner(p);
//   p.SetPrecondMatrixFormat(HYB);
  ls.Init(0.0, 1e-6, 1e8, 200000);
#ifdef BUBFLO  
//   ls.SetNVectors(4);
  ls.MakeZ_CSR(); // requires xdim_ and novecni_ and zlssd_ to be set
  if(setlssd)
    ls.MakeZLSSD(bubmap_ptr, maxbmap); // bubmap must be ready and maxbmap available
#endif    
//   
  
  ls.Build();
#ifdef MATDIA  
  mat.ConvertToDIA();
#endif  
  gettimeofday(&now, NULL);
  tack = now.tv_sec*1000000.0+(now.tv_usec);
  b=(tack-tick)/1000000;
  std::cout << "Building:" << b << " sec" << std::endl;
// 
//   ls.Verbose(2);
  mat.info();

  gettimeofday(&now, NULL);
  tick = now.tv_sec*1000000.0+(now.tv_usec);

  ls.Solve(rhs, &x);
  
  gettimeofday(&now, NULL);
  tack = now.tv_sec*1000000.0+(now.tv_usec);
  s= (tack-tick)/1000000;
  std::cout << "Solver execution:" << s << " sec" << std::endl;
  std::cout << "Total execution:" << s+b << " sec" << std::endl;
#ifdef GUUS  
x.MoveToHost();
  sol_norm=x.Norm();
  cout<<"\n Norm of Solution is "<<sol_norm<<endl;
  cout<<"\n Norm of Reference Solution is "<<refsol.Norm()<<endl;
  refones.AddScale(x,(double)-1.0f);
  x.AddScale(refsol,(double)-1.0f);
  
  
  diff_norm=x.Norm();
  ones_norm=refones.Norm();
  cout<<"\n Relative Norm of Calculated Solution w.r.t. Reference is "<<((double)diff_norm/(double)sol_norm)<<endl;
  cout<<"\n Relative Norm of Calculated Solution w.r.t. Ones is "<<((double)ones_norm/(double)sol_norm)<<endl;
  //x.WriteFileASCII("x_solution1e3shell_ilu01.rec");
#endif  
  ls.Clear();

// /////////////////////////////////////////////////////////////////    
//   std::cout << "-----------------------------------------------" << std::endl;
//   std::cout << "DPCG solver ILU(0,1)" << std::endl;
// #ifdef GUUS  
//   rhs.ReadFileASCII(std::string(argv[3]));
//   x.SetRandom(0.0,1.0,1000);
//   ls.SetZ(Zin);
// #endif  
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
// 
// #ifdef BUBFLO  
//   x.ReadFileASCII(std::string(argv[2]));
//   rhs.ReadFileASCII(std::string(argv[3]));
//   if(setlssd){
//     LocalVector<double> phi;
//     LocalVector<int> bubmap;
//     phi.Allocate("PHI", phisize);
//     bubmap.Allocate("bubmap",mat.get_nrow());
//     phi.ReadFileASCII(std::string(argv[4]));
//     
//     bubmap.LeaveDataPtr(&bubmap_ptr);
//     phi.LeaveDataPtr(&phi_ptr);
// 
//     //x.SetRandom(0.0,1.0,1000);
//     bubmap_create(phi_ptr, bubmap_ptr, xdim, xdim, xdim, mat.get_nrow(), &maxbmap, lvst_offst);
//     phi.Clear();
//     
//   }
//   ls.Setxdim(xdim);
//   ls.SetNVectors(defvex_perdirec);
//   ls.SetZlssd(setlssd);
//   mat.ConvertToCSR();
// #endif  
//   
//   mcilu_p.Init(0);
//   ls.SetOperator(mat);
//   ls.SetPreconditioner(mcilu_p);
//   
//   ls.Init(0.0, 1e-6, 1e8, 200000);
// //   ls.SetNVectors(4);
// #ifdef BUBFLO  
//   ls.MakeZ_CSR(); // requires xdim_ and novecni_ and zlssd_ to be set
//   if(setlssd)
//     ls.MakeZLSSD(bubmap_ptr, maxbmap); // bubmap must be ready and maxbmap available	
// #endif
//     
//   
//   ls.Build();
// #ifdef MATDIA  
//   mat.ConvertToDIA();
// #endif  
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   b=(tack-tick)/1000000;
//   std::cout << "Building:" << b << " sec" << std::endl;
//   
// 
// //   mat.info();
// 
//   gettimeofday(&now, NULL);
//   tick = now.tv_sec*1000000.0+(now.tv_usec);
// 
//   ls.Solve(rhs, &x);
// 
//   gettimeofday(&now, NULL);
//   tack = now.tv_sec*1000000.0+(now.tv_usec);
//   s= (tack-tick)/1000000;
//   std::cout << "Solver execution:" << s << " sec" << std::endl;
//   std::cout << "Total execution:" << s+b << " sec" << std::endl;
// #ifdef GUUS  
// x.MoveToHost();
//   sol_norm=x.Norm();
//   cout<<"\n Norm of Solution is "<<sol_norm<<endl;
//   cout<<"\n Norm of Reference Solution is "<<refsol.Norm()<<endl;
//   refones.AddScale(x,(double)-1.0f);
//   x.AddScale(refsol,(double)-1.0f);
//   
//   
//   diff_norm=x.Norm();
//   ones_norm=refones.Norm();
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Reference is "<<((double)diff_norm/(double)sol_norm)<<endl;
//   cout<<"\n Relative Norm of Calculated Solution w.r.t. Ones is "<<((double)ones_norm/(double)sol_norm)<<endl;
// //   x.WriteFileASCII("x_solution1e3shell_ilu01.rec");
// #endif  
//   ls.Clear();
/////////////////////////////////////////////////////////////////    
  stop_paralution();

  return 0;
}
