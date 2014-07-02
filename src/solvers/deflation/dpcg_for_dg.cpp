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
  //assert(this->op_ != NULL);
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
    this->A0_.MoveToHost();
    if (this->precond_ != NULL)
      this->precond_->MoveToHost();
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
    this->A0_.MoveToAccelerator();
    if (this->precond_ != NULL)
      this->precond_->MoveToAccelerator();
  } 
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType &rhs,
                                                              VectorType *x) {
  
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG_FOR_DG<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType &rhs,
                                                            VectorType *x) {
}

template class DPCG_FOR_DG< LocalMatrix<double>, LocalVector<double>, double >;
template class DPCG_FOR_DG< LocalMatrix<float>,  LocalVector<float>, float >;

}