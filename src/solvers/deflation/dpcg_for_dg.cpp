#include "dpcg_for_dg.hpp"
#include "../iter_ctrl.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/local_matrix.hpp"

#include "../../base/global_stencil.hpp"
#include "../../base/local_stencil.hpp"

#include "../../base/global_vector.hpp"
#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/math_functions.hpp"
#include "../../utils/time_functions.hpp"

#include "../preconditioners/preconditioner.hpp"
#include "omp.h"
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <cstring>
using namespace std;
namespace paralution {


  template <class OperatorType, class VectorType, typename ValueType>
DPCG_FOR_DG<OperatorType, VectorType, ValueType>::DPCG_FOR_DG() {

}

template <class OperatorType, class VectorType, typename ValueType>
DPCG_FOR_DG<OperatorType, VectorType, ValueType>::~DPCG_FOR_DG() {
  this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::Print(void) const {
  
  if (this->precond_ == NULL) { 
    
    LOG_INFO("DPCG_FOR_DG solver");
    
  } else {
    
    LOG_INFO("DPCG_FOR_DG solver, with preconditioner:");
    this->precond_->Print();

  }

  
}


template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  if (this->precond_ == NULL) { 

    LOG_INFO("DPCG_FOR_DG (non-precond) linear solver starts");

  } else {

    LOG_INFO("DPCG_FOR_DG solver starts, with preconditioner:");
    this->precond_->Print();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

  if (this->precond_ == NULL) { 

    LOG_INFO("DPCG_FOR_DG (non-precond) ends");

  } else {

    LOG_INFO("DPCG_FOR_DG ends");

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::SetA0_and_m(LocalMatrix<ValueType> &A0_in,
							       const int val_m){
//   assert(this->op_ != NULL);
    this->A0_.CopyFrom(A0_in);
    std::cout<<"No. of non-zeros in A0 is "<<this->A0_.get_nnz()<<endl;
    this->A0_nrows_= this->A0_.get_nrow();
    assert(val_m > 0);
    this->m_ = val_m;
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::Build(void) {

   if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);
  assert(this->op_->get_nrow() == this->op_->get_ncol());
  assert(this->op_->get_nrow() > 0);

  //int ncol = novecni_*novecni_*novecni_ - 1; 

  this->r_.CloneBackend(*this->op_);
  this->r_.Allocate("r", this->op_->get_ncol());

  this->p_.CloneBackend(*this->op_);
  this->p_.Allocate("p", this->op_->get_nrow());
  
  this->y_.CloneBackend(*this->op_);
  this->y_.Allocate("y", this->op_->get_nrow());

  this->w_.CloneBackend(*this->op_);
  this->w_.Allocate("w", this->op_->get_nrow());
  
  this->w1_.CloneBackend(*this->op_);
  this->w1_.Allocate("w1", this->op_->get_nrow());
  
  this->w2_.CloneBackend(*this->op_);
  this->w2_.Allocate("w2", this->A0_nrows_);
  
  this->w3_.CloneBackend(*this->op_);
  this->w3_.Allocate("w3", this->A0_nrows_);
  
  this->w4_.CloneBackend(*this->op_);
  this->w4_.Allocate("w4", this->op_->get_nrow());
  
  this->Qb_.CloneBackend(*this->op_);
  this->Qb_.Allocate("Qb", this->op_->get_nrow());

  this->Ptx_.CloneBackend(*this->op_);
  this->Ptx_.Allocate("Ptx", this->op_->get_nrow());
  
  this->Dinv_.CloneBackend(*this->op_);  
  this->op_->ExtractInverseDiagonal(&this->Dinv_);
  
  this->ls_inner_.Init(0,1e-6,1e+8,2000);

  
  this->A0_.CloneBackend(*this->op_);
//   ILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ilu_p;
//   ilu_p.Set(0);
//   this->ls_inner_.SetPreconditioner(ilu_p);  
  this->ls_inner_.SetOperator(this->A0_);
  

  this->ls_inner_.Build();
//  this->ls_inner_.Verbose(2);

  
  if (this->precond_ != NULL) {
    this->precond_->SetOperator(*this->op_);
    this->precond_->Build();
  } 
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::Clear(void) {

  if (this->build_ == true) {
    
    if (this->precond_ != NULL) {
      this->precond_->Clear();
      this->precond_   = NULL;
    }

    this->r_.Clear();
    this->p_.Clear();
    this->y_.Clear();
    this->w_.Clear();
    
    this->w1_.Clear();
    this->w2_.Clear();
    this->w3_.Clear();
    this->w4_.Clear();
    
    this->Qb_.Clear();
    this->Ptx_.Clear();
    this->Dinv_.Clear();
    this->A0_.Clear();
    this->ls_inner_.Clear();
    this->iter_ctrl_.Clear();
    
   
    this->build_ = false;
  }
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  if (this->build_ == true) {

    this->r_.MoveToHost();
    this->p_.MoveToHost();
    this->y_.MoveToHost();
    this->w_.MoveToHost();
    
    this->w1_.MoveToHost();
    this->w2_.MoveToHost();
    this->w3_.MoveToHost();
    this->w4_.MoveToHost();
    
    this->Qb_.MoveToHost();
    this->Ptx_.MoveToHost();
    this->Dinv_.MoveToHost();
    this->ls_inner_.MoveToHost();
    if (this->precond_ != NULL)
      this->precond_->MoveToHost();
    this->ls_inner_.MoveToHost();
    this->A0_.MoveToHost();
  }
    
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  if (this->build_ == true) {

    this->r_.MoveToAccelerator();
    this->p_.MoveToAccelerator();
    this->y_.MoveToAccelerator();
    this->w_.MoveToAccelerator();
    
    this->w1_.MoveToAccelerator();
    this->w2_.MoveToAccelerator();
    this->w3_.MoveToAccelerator();
    this->w4_.MoveToAccelerator();
    
    this->Qb_.MoveToAccelerator();
    this->Ptx_.MoveToAccelerator();
    this->Dinv_.MoveToAccelerator();
   
    if (this->precond_ != NULL)
      this->precond_->MoveToAccelerator();
    this->ls_inner_.MoveToAccelerator();
    this->A0_.MoveToAccelerator();
  } 
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType &rhs,
                                                              VectorType *x) {
  
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType &rhs,
                                                            VectorType *x) {
   assert(x != NULL);
   assert(x != &rhs);
   assert(this->op_  != NULL);
   assert(this->precond_ != NULL);
   assert(this->build_ == true);

   const OperatorType *op = this->op_;

   VectorType *r = &this->r_;
   VectorType *p = &this->p_;
   VectorType *y = &this->y_;
   VectorType *w = &this->w_;

   VectorType *w1 = &this->w1_;
   VectorType *w2 = &this->w2_;
   VectorType *w3 = &this->w3_;
   VectorType *w4 = &this->w4_;
    
   VectorType *Qb = &this->Qb_;
   VectorType *Ptx = &this->Ptx_;

   ValueType beta, alpha;
   ValueType rho, rho_old;
   ValueType res_norm = 0.0, b_norm = 1.0;
   ValueType check_residual = 0.0;  
   double tick, tack, time_innrsolve=0.0f, time_multR_Rt=0.0f;
   int m_local=this->m_;
  
//    this->A0_.info();
   /*** making Qb ***/
   tick = paralution_time();
   rhs.multiply_with_R(*w2,m_local);
   tack = paralution_time();
   time_multR_Rt+=(tack-tick)/1000000;
//    w3->Zeros();
   
   tick = paralution_time();
   this->ls_inner_.Solve(*w2, w3);
   tack = paralution_time();
   time_innrsolve+=(tack-tick)/1000000;
//   cout<<"Norm of w3 after solve is "<<  this->Norm(*w3)<<endl; 
//     w4->Zeros();
    tick = paralution_time();
    w3->multiply_with_Rt(*w4,m_local);
    tack = paralution_time();
    time_multR_Rt+=(tack-tick)/1000000;
    
    Qb->CopyFrom(*w4,0,0,this->op_->get_nrow());
//    /*** making Ptx ***/
    op->Apply(*x,Ptx);
//     w2->Zeros();
    Ptx->multiply_with_R(*w2,m_local);
//     w3->Zeros();
    tick = paralution_time();
    this->ls_inner_.Solve(*w2,w3);
    tack = paralution_time();
    time_innrsolve+=(tack-tick)/1000000;
//     w4->Zeros();
    tick = paralution_time();
    w3->multiply_with_Rt(*w4,m_local);
    tack = paralution_time();
    time_multR_Rt+=(tack-tick)/1000000;
    
    x->AddScale(*w4,(ValueType)-1.0);
    x->ScaleAdd((ValueType)1.0, *Qb); // here we have x=Qb+(I-AQ)^{t}X 
//    /*** BEGINNING DPCG***/
//    // initial residual = b - Ax
    op->Apply(*x, r); 
    r->ScaleAdd(ValueType(-1.0), rhs);
//    // initial residual for the interation control
//    // = |res|
//    //  init_residual = this->Norm(*r);
//    // apply deflation
    //y:=omega*M^{-1}r // omega kept as 1
//    //y:= y + Q*(r-Ay)
    this->precond_->SolveZeroSol(*r, y);

    op->Apply(*y,w1);
    w1->ScaleAdd((ValueType)-1.0f, *r);
//     w2->Zeros();	
    tick = paralution_time();
    w1->multiply_with_R(*w2,m_local);
    tack = paralution_time();
    time_multR_Rt+=(tack-tick)/1000000;
//     w3->Zeros();
    tick = paralution_time();
    this->ls_inner_.Solve(*w2,w3);
    tack = paralution_time();
    time_innrsolve+=(tack-tick)/1000000;
//     w4->Zeros();
    tick = paralution_time();
    w3->multiply_with_Rt(*w4,m_local);
    tack = paralution_time();
    time_multR_Rt+=(tack-tick)/1000000;
    
    y->ScaleAdd((ValueType)1.0f,*w4);
    ////p=y
    p->CopyFrom(*y,0,0,this->op_->get_nrow());
//    // initial residual for the interation control
//    // = |res| / |b|
    res_norm = this->Norm(*r);
    b_norm = this->Norm(rhs);
    this->iter_ctrl_.InitResidual(b_norm);
//    // w = Ap
    op->Apply(*p, w);
// 
    rho=r->Dot(*y);
//    // alpha = rho / (p,w)
    alpha=rho/p->Dot(*w);
//    // x = x + alpha * p
    x->AddScale(*p, alpha);
//    // r = r - alpha * w
    r->AddScale(*w, alpha*((ValueType)-1.0f));
// 
    res_norm = this->Norm(*r);
    check_residual = res_norm; 
// 
    while (!this->iter_ctrl_.CheckResidual(check_residual)) {
//      //Apply deflation
// 
      this->precond_->SolveZeroSol(*r, y);
//     
      op->Apply(*y,w1);
      w1->ScaleAdd((ValueType)-1.0f, *r);
//       w2->Zeros();	
      tack = paralution_time();
      w1->multiply_with_R(*w2,m_local);
      tack = paralution_time();
      time_multR_Rt+=(tack-tick)/1000000;
      
//       w3->Zeros();
      tick = paralution_time();
      this->ls_inner_.Solve(*w2,w3);
      tack = paralution_time();
      time_innrsolve+=(tack-tick)/1000000;
//       w4->Zeros();
      tick = paralution_time();
      w3->multiply_with_Rt(*w4,m_local);
      tack = paralution_time();
      time_multR_Rt+=(tack-tick)/1000000;
      
      y->ScaleAdd((ValueType)1.0f,*w4);
//      
      rho_old = rho;
//      // rho = (r,y)
      rho = r->Dot(*y);
// 
      beta = rho / rho_old;
//      // p = p + beta * y
      p->ScaleAdd(beta, *y);
//      // w = Ap
      op->Apply(*p, w);
//        // at this point save alpha and beta
//  //   lanczos<<setprecision(16)<<alpha<<" "<<setprecision(16)<<beta<<" "<<endl;
//      // alpha = rho / (p,w)
      alpha=rho/p->Dot(*w);
//      // x = x + alpha * p
      x->AddScale(*p, alpha);
//      // r = r - alpha * w
      r->AddScale(*w, alpha*((ValueType)-1.0f));
// 
      res_norm = this->Norm(*r);
      check_residual = res_norm; 
    }

    cout<<"Time in solving "<<time_innrsolve<<" secs"<<endl;
    cout<<"Time in multiplying with R and Rt "<<time_multR_Rt<<" secs"<<endl;
    
}

template class DPCG_FOR_DG< LocalMatrix<double>, LocalVector<double>, double >;
template class DPCG_FOR_DG< LocalMatrix<float>,  LocalVector<float>, float >;

}
