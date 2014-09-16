#ifndef PARALUTION_DEFLATION_DPCG_FOR_DG_HPP_
#define PARALUTION_DEFLATION_DPCG_FOR_DG_HPP_
#include "../solver.hpp"
#include "../krylov/cg.hpp"
#include "dpcg.hpp"
#include "../multigrid/multigrid_amg.hpp"
// #include <paralution.hpp>

// #define PCG_INNR
#define DPCG_INNR
// #define ILU_PREC_INNR
// #define AMG_PREC_INNR
#define DEFVEX_PERDIREC	16
namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
class DPCG_FOR_DG : public IterativeLinearSolver<OperatorType, VectorType, ValueType> {
  
public:
  DPCG_FOR_DG();
  virtual ~DPCG_FOR_DG();
  virtual void Print(void) const;
  virtual void Build(void);
  virtual void Clear(void);
  virtual void SetA0_and_m(LocalMatrix<ValueType> &A0, const int);
protected:
  virtual void SolveNonPrecond_(const VectorType &rhs,
                                VectorType *x);
  virtual void SolvePrecond_(const VectorType &rhs,
                             VectorType *x);
  
  virtual void SetZ(LocalMatrix<ValueType> &Z);
  virtual void PrintStart_(void) const;
  virtual void PrintEnd_(void) const;
  virtual void MoveToHostLocalData_(void);
  virtual void MoveToAcceleratorLocalData_(void);
  
private:
  OperatorType A0_, Z_;
  VectorType r_, w_;
  VectorType p_, y_;

  VectorType Dinv_;

  VectorType Qb_, Ptx_;
  VectorType w1_, w2_, w3_, w4_;

  int novecni_, zlssd_, xdim_, ydim_;
  int novecni_x_, novecni_y_;
  int val_lvst_offst_;
  int size_A0_, m_, A0_nrows_;
#ifdef PCG_INNR
  CG<OperatorType, VectorType, ValueType> ls_inner_;
#endif
#ifdef DPCG_INNR
  DPCG<OperatorType, VectorType, ValueType > ls_inner_;
#endif  
  Solver<OperatorType, VectorType, ValueType> *precond_inner_;
  
  
  
};
  

}
#endif // PARALUTION_DEFLATION_DPCG_FOR_DG_HPP_