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


#ifndef PARALUTION_GPU_MATRIX_HYB_HPP_
#define PARALUTION_GPU_MATRIX_HYB_HPP_

#include "../base_matrix.hpp"
#include "../base_vector.hpp"
#include "../matrix_formats.hpp"

namespace paralution {

template <typename ValueType>
class GPUAcceleratorMatrixHYB : public GPUAcceleratorMatrix<ValueType> {
  
public:

  GPUAcceleratorMatrixHYB();
  GPUAcceleratorMatrixHYB(const Paralution_Backend_Descriptor local_backend);
  virtual ~GPUAcceleratorMatrixHYB();

  inline int get_ell_max_row(void) const { return this->mat_.ELL.max_row; }
  inline int get_ell_nnz(void) const { return this->ell_nnz_; }
  inline int get_coo_nnz(void) const { return this->coo_nnz_; }

  virtual void info(void) const;
  virtual unsigned int get_mat_format(void) const{ return HYB; }

  virtual void Clear(void);
  virtual void AllocateHYB(const int ell_nnz, const int coo_nnz, const int ell_max_row, 
                           const int nrow, const int ncol);


  virtual bool ConvertFrom(const BaseMatrix<ValueType> &mat);

  virtual void CopyFrom(const BaseMatrix<ValueType> &mat);
  virtual void CopyFromAsync(const BaseMatrix<ValueType> &mat);
  virtual void CopyTo(BaseMatrix<ValueType> *mat) const;
  virtual void CopyToAsync(BaseMatrix<ValueType> *mat) const;

  virtual void CopyFromHost(const HostMatrix<ValueType> &src);
  virtual void CopyFromHostAsync(const HostMatrix<ValueType> &src);
  virtual void CopyToHost(HostMatrix<ValueType> *dst) const;
  virtual void CopyToHostAsync(HostMatrix<ValueType> *dst) const;

  virtual void Apply(const BaseVector<ValueType> &in, BaseVector<ValueType> *out) const; 
  virtual void ApplyAdd(const BaseVector<ValueType> &in, const ValueType scalar,
                        BaseVector<ValueType> *out) const; 
  
private:
  
  MatrixHYB<ValueType, int> mat_;
  int ell_nnz_;
  int coo_nnz_;

  friend class BaseVector<ValueType>;  
  friend class AcceleratorVector<ValueType>;  
  friend class GPUAcceleratorVector<ValueType>;  

};


}

#endif // PARALUTION_GPU_MATRIX_HYB_HPP_

