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


#ifndef PARALUTION_LOCAL_STENCIL_HPP_
#define PARALUTION_LOCAL_STENCIL_HPP_

#include "operator.hpp"
#include "local_vector.hpp"

namespace paralution {

template <typename ValueType>
class LocalVector;
template <typename ValueType>
class GlobalVector;

template <typename ValueType>
class GlobalStencil;


// Local Stencil
template <typename ValueType>
class LocalStencil : public Operator<ValueType> {
  
public:

  LocalStencil();
  virtual ~LocalStencil();

  inline int get_ndim(void) const { return this->ndim_; }
  virtual int get_nrow(void) const;
  virtual int get_ncol(void) const;

  virtual void Apply(const LocalVector<ValueType> &in, LocalVector<ValueType> *out) const; 
  virtual void ApplyAdd(const LocalVector<ValueType> &in, const ValueType scalar, 
                        LocalVector<ValueType> *out) const; 

protected:

  virtual bool is_host(void) const {return true;};
  virtual bool is_accel(void) const {return true;};


private:
  
  std::string object_name_ ;

  //  BaseStencil<ValueType> *stencil_;


  int ndim_;
  int *dim_;

  friend class LocalVector<ValueType>;  
  friend class GlobalVector<ValueType>;  
  friend class GlobalStencil<ValueType>;  
  
};


}

#endif // PARALUTION_LOCAL_STENCIL_HPP_
