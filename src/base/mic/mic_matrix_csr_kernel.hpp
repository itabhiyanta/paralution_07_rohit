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


#ifndef PARALUTION_MIC_MATRIX_CSR_KERNEL_HPP_
#define PARALUTION_MIC_MATRIX_CSR_KERNEL_HPP_

#include "../matrix_formats_ind.hpp"

namespace paralution {

template <typename ValueType>
void spmv_csr(const int mic_dev, 
	      const int *row, const int *col, const ValueType *val,
	      const int nrow,
	      const ValueType *in, ValueType *out) {
  

#pragma offload target(mic:mic_dev)		    \
  in(row:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(col:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(val:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(in:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)	   
#pragma omp parallel for
    for (int ai=0; ai<nrow; ++ai) {
      out[ai] = ValueType(0.0);
      for (int aj=row[ai]; aj<row[ai+1]; ++aj)
	out[ai] += val[aj] * in[ col[aj] ];
    }
}

template <typename ValueType>
void spmv_add_csr(const int mic_dev, 
		  const int *row, const int *col, const ValueType *val,
		  const int nrow,
		  const ValueType scalar,
		  const ValueType *in, ValueType *out) {
  

#pragma offload target(mic:mic_dev)		    \
  in(row:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(col:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(val:length(0) MIC_REUSE MIC_RETAIN)	    \
  in(in:length(0) MIC_REUSE MIC_RETAIN)		    \
  in(out:length(0) MIC_REUSE MIC_RETAIN)	    
#pragma omp parallel for
  for (int ai=0; ai<nrow; ++ai)
    for (int aj=row[ai]; aj<row[ai+1]; ++aj)
      out[ai] += scalar*val[aj] * in[ col[aj] ];

}

}

#endif // PARALUTION_BASE_MATRIX_CSR_KERNEL_HPP_
