// *************************************************************************
//
//    This code is developed and maintained by TU Delft. 
//    The deflation solver (files src/solvers/deflation/dpcg.hpp and
//    src/solvers/deflation/dpcg.cpp) are released under GNU LESSER GENERAL 
//    PUBLIC LICENSE (LGPL v3)
//
//    Copyright (C) 2013 Kees Vuik (TU Delft)
//    Delft University of Technology, the Netherlands
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as 
//    published by the Free Software Foundation, either version 3 of the 
//    License, or (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public 
//    License along with this program.  
//    If not, see <http://www.gnu.org/licenses/>.
//
// *************************************************************************

#include "dpcg.hpp"
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
#include <assert.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <cstring>
using namespace std;
namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
DPCG<OperatorType, VectorType, ValueType>::DPCG() {

  LOG_DEBUG(this, "DPCG::DPCG()",
            "default constructor");

  this->novecni_ = 2; 

}

template <class OperatorType, class VectorType, typename ValueType>
DPCG<OperatorType, VectorType, ValueType>::~DPCG() {

  LOG_DEBUG(this, "DPCG::~DPCG()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::Print(void) const {
  
  if (this->precond_ == NULL) { 
    
    LOG_INFO("DPCG solver");
    
  } else {
    
    LOG_INFO("PDPCG solver, with preconditioner:");
    this->precond_->Print();

  }

  
}


template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  if (this->precond_ == NULL) { 

    LOG_INFO("DPCG (non-precond) linear solver starts");

  } else {

    LOG_INFO("PDPCG solver starts, with preconditioner:");
    this->precond_->Print();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

  if (this->precond_ == NULL) { 

    LOG_INFO("DPCG (non-precond) ends");

  } else {

    LOG_INFO("PDPCG ends");

  }

}


template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::Setlvst_offst(const int val_lvst_offst) {

  assert(val_lvst_offst > 0);
  this->val_lvst_offst_ = val_lvst_offst;

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::SetNVectors(const int novecni) {
  LOG_DEBUG(this, "DPCG::SetNVectors()",
            novecni);

  assert(novecni > 0);
  this->novecni_ = novecni;

}
template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::SetNVectors_eachdirec(const int novecnix,
								      const int novecniy,
								      const int novecniz) {
//   LOG_DEBUG(this, "DPCG::SetNVectors_eachdirec()",
//             novecnix, novecniy, novecniz);

  assert(novecnix > 0);
  this->novecni_x_ = novecnix;
  assert(novecniy > 0);
  this->novecni_y_ = novecniy;
  assert(novecniz > 0);
  this->novecni_z_ = novecniz;

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::SetZlssd(const int val_zlssd) {

  //assert(val_zlssd > 0);
  this->zlssd_ = val_zlssd;

}
template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::Setxdim(const int val_xdim) {

  assert(val_xdim > 0);
  this->xdim_ = val_xdim;

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::Set_alldims(const int val_xdim,
							    const int val_ydim,
							    const int val_zdim) {

  assert(val_xdim > 0);
  this->xdim_ = val_xdim;
  assert(val_ydim > 0);
  this->ydim_ = val_ydim;
  assert(val_zdim > 0);
  this->zdim_ = val_zdim;

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::MakeZ_COO(void) 
{

  int ncol = novecni_*novecni_*novecni_ - 1; 
  int nrow = this->op_->get_nrow();
  int part = nrow / (ncol+1);
  int nrows, ncols, j, k, didx_x, didx_y, didx_z;
  int s, gridarea2d, griddiminx, dom_idx_x, dom_idx_y, dom_idx_z, domarea2d, domsize;
 
  int numvecs = ncol;
  nrows= nrow;
  ncols= numvecs;	
  part=nrows/(numvecs+1); 
  this->Z_.ConvertToCOO();
  this->Z_.AllocateCOO("Z",nrows-part,nrows,ncols);
  
  int *Z_row = NULL;
  int *Z_col = NULL;
  ValueType *Z_val = NULL;
  
  this->Z_.LeaveDataPtrCOO(&Z_row, &Z_col, &Z_val);
  
  s=xdim_/novecni_;
  domsize=s*s*s;  griddiminx=xdim_/s; gridarea2d=griddiminx*griddiminx; domarea2d=s*s;
  for(j=0;j<numvecs;j++){
    didx_x=(j)%griddiminx;    didx_y=(j%gridarea2d)/griddiminx; didx_z=(j)/gridarea2d;
    for(k=0;k<domsize;k++){
      dom_idx_x=(k)%s;      dom_idx_y=(k%domarea2d)/s;	dom_idx_z=(k)/domarea2d;
      Z_row[j*domsize+k]=didx_x*s+didx_y*griddiminx*domarea2d+didx_z*gridarea2d*domsize+ 
        //we are at the first ele of the right block
        //now within the block we have to position
        dom_idx_x+dom_idx_y*xdim_+dom_idx_z*xdim_*xdim_;
      Z_col[j*domsize+k]=j; 
      Z_val[j*domsize+k] = 1.0f;
    }
  }
  
  this->Z_.SetDataPtrCOO(&Z_row, &Z_col, &Z_val, "Z", nrows-part, nrow, ncol);
  this->Z_.ConvertToCSR();
  
}
/*
template <class OperatorType, class VectorType, typename ValueType>
  void DPCG<OperatorType, VectorType, ValueType>::MakeZ_CSR(void) {
  
    int nrow = this->op_->get_nrow();
    int nrows, ncols, part, i,j,column;
    int s, cntr=0, didx, didy, didz, nnz_Zsd, numvecs;
    s=xdim_/novecni_;
    int *Z_row_offset = NULL;  int *Z_col = NULL;  ValueType *Z_val = NULL;
    
    //this->Z_.ConvertToCSR();
   //cout<<"do we need to make Z lssd ? "<<zlssd_<<endl;
    if(zlssd_){
      numvecs = novecni_*novecni_*novecni_;
      nnz_Zsd=nrow;
    }
    else{
      numvecs = novecni_*novecni_*novecni_ - 1;
      part=nrow/(numvecs+1); 
      nnz_Zsd=nrow-part;
    }
    nrows= nrow;  ncols= numvecs;
    //cout<<"nrows ="<<nrows<<" ncols="<<numvecs<<" nnz_Zsd="<<nnz_Zsd<<endl;
    this->Z_.AllocateCSR("Z",nnz_Zsd,nrows,ncols);
    
    this->Z_.LeaveDataPtrCSR(&Z_row_offset, &Z_col, &Z_val);
    
    
    
    for(i=0,j=0;i<nrows;i++){
      // find z index
      didx=(i%xdim_)/s;   didy=((i/(xdim_*s))%novecni_);    didz=i/(xdim_*xdim_*s);
      column=didz*novecni_*novecni_+didy*novecni_+didx;
      //LOG_INFO("indices"<<didx<<" "<<didy<<" "<<didz);
      //if(didx==novecni_-1&&didy==novecni_-1&&didz==novecni_-1)
     if(column==numvecs && zlssd_!=1)
        {
          Z_row_offset[i+1]=Z_row_offset[i];
          cntr++;
        }//cout<<endl;
      else
        {
          Z_col[j]=column;
          Z_val[j] = 1.0f;
          Z_row_offset[i+1]=Z_row_offset[i]+1;
          j++;
        }
    }
    //LOG_INFO("Number of non-zeros in Z are "<<j<<" nrows-part is "<<nrows-part<<"discarded"<<cntr<<" should be equal to"<<nrows-part);
    Z_row_offset[i]=nnz_Zsd;
    this->Z_.SetDataPtrCSR(&Z_row_offset, &Z_col, &Z_val, "Z", nnz_Zsd, nrow, ncols);
     // at this point we have Z sub-domain, we need to have bubmap and then work with it  
    this->Z_.WriteFileMTX("Zcsr_old_sd.rec");
  }*/


template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::MakeZ_CSR(void) {

  int nrow = this->op_->get_nrow();	FILE *fp, *fp1;
  int nrows, ncols, part, i,j,column;
  int cntr=0, d_idx, d_idy, d_idz, nnz_Zsd, numvecs;
  int *Z_row_offset = NULL;  int *Z_col = NULL;  ValueType *Z_val = NULL;
  int z_coord, y_coord, x_coord, tempval;
  int nnz_x_last, nnz_y_last, nnz_z_last;
  //this->Z_.ConvertToCSR();
  //cout<<"do we need to make Z lssd ? "<<zlssd_<<endl;
  if(zlssd_){
    numvecs = novecni_x_ * novecni_y_ * novecni_z_;
    nnz_Zsd=nrow;
    tempval	=	xdim_ % novecni_x_;
    nnz_x_last	=	tempval > 0 ? (xdim_ / novecni_x_) + tempval : xdim_ / novecni_x_;
    tempval	=	ydim_ % novecni_y_;
    nnz_y_last	=	tempval > 0 ? (ydim_ / novecni_y_) + tempval : ydim_ / novecni_y_;
    tempval	= 	zdim_ % novecni_z_;
    nnz_z_last	= 	tempval > 0 ? (zdim_ / novecni_z_) + tempval : zdim_ / novecni_z_;
  }
  else{
    numvecs = novecni_x_ * novecni_y_ * novecni_z_ - 1;
    // calculate unknowns in the last cube
    tempval	=	xdim_ % novecni_x_;
    nnz_x_last	=	tempval > 0 ? (xdim_ / novecni_x_) + tempval : xdim_ / novecni_x_;
    tempval	=	ydim_ % novecni_y_;
    nnz_y_last	=	tempval > 0 ? (ydim_ / novecni_y_) + tempval : ydim_ / novecni_y_;
    tempval	= 	zdim_ % novecni_z_;
    nnz_z_last	= 	tempval > 0 ? (zdim_ / novecni_z_) + tempval : zdim_ / novecni_z_;
    part=nnz_x_last * nnz_y_last * nnz_z_last;//nrow/(numvecs+1); 
    nnz_Zsd=nrow-part;
  }
  nrows= nrow;  ncols= numvecs;
//   cout<<"nrows ="<<nrows<<" ncols="<<numvecs<<" nnz_Zsd="<<nnz_Zsd<<endl;
//   cout<<"Last domains' nnz in each direc are x="<<nnz_x_last<<" y="<<nnz_y_last<<" z="<<nnz_z_last<<endl;
  this->Z_.AllocateCSR("Z",nnz_Zsd,nrows,ncols);
  
  this->Z_.LeaveDataPtrCSR(&Z_row_offset, &Z_col, &Z_val);
//   fp=fopen("Zsd_record.rec","wt");
  Z_row_offset[0]=0;
//   fp1=fopen("Z_tobeset.rec","wt");
  for(i=0, j=0 ; i<nrows; i++)
    //working on the idea that each idex of the grid can e represented as
    // i= a * (xdim_ * ydim_) + b * xdim_ + c
    // where a --> z co-ord, b --> y co-ord, c --> x co-ord.
  {
    
    x_coord	=	i/(ydim_ * zdim_);	
    tempval	=	i - x_coord * ydim_ * zdim_;
    y_coord	=	(tempval>0)?(tempval)/zdim_:0;	
    tempval	=	tempval-y_coord * zdim_;
    z_coord	=	(tempval>0)?tempval:0;
    
    tempval	=	top(z_coord, zdim_, novecni_z_) ;
    d_idz	=	(tempval>0)?tempval:0;
    tempval	=	top(y_coord, ydim_, novecni_y_);	
    d_idy	=	(tempval>0)?tempval:0;
    tempval	=	top(x_coord, xdim_, novecni_x_);
    d_idx	=	(tempval>0)?tempval:0;
    
    
    column=d_idx * novecni_y_ * novecni_z_ + d_idy * novecni_z_ + d_idz;
   
//    fprintf(fp,"%d %d %d %d %d %d %d %d\n",i, column, d_idx, d_idy, d_idz, x_coord, y_coord, z_coord);
    if(column==numvecs && zlssd_!=1)
      {
        Z_row_offset[i+1]=Z_row_offset[i];
        cntr++;
      }
    else
      {
        Z_col[j]=column;
        Z_val[j] = 1.0f;
        Z_row_offset[i+1]=Z_row_offset[i]+1;
// 	fprintf(fp1,"%d %d %d %f %d %d\n",column, Z_val[j], Z_row_offset[i+1], i,j);
        j++;
      }
  }
  
//    fclose(fp1);
//    fclose(fp);
//   LOG_INFO("Number of non-zeros in Z are "<<j<<" nrows-part is "<<nrows-part<<" discarded "<<cntr<<" should be equal to "<<nrows-part);
  Z_row_offset[i]=nnz_Zsd;
  this->Z_.SetDataPtrCSR(&Z_row_offset, &Z_col, &Z_val, "Z", nnz_Zsd, nrow, ncols);
   // at this point we have Z sub-domain, we need to have bubmap and then work with it  
//   this->Z_.WriteFileMTX("Zcsr_new_sd.rec");
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::SetZ(LocalMatrix<ValueType> &Zin){
  //assert(this->op_ != NULL);
  

    this->Z_.CopyFrom(Zin);
    std::cout<<"No. of non-zeros in Z is "<<this->Z_.get_nnz()<<endl;

}

/*template <class OperatorType, class VectorType, typename ValueType>
  void DPCG<OperatorType, VectorType, ValueType>::MakeZLSSD(const int *bmap, const int maxbmap){
    // Z sub-domain and bubmap are assumed to have been made already 
    int *Zsubd_rows=NULL, *Zsubd_cols=NULL, i, complementval, rowid, j,k, s;
    int didx_x, didx_y, didx_z, gridarea2d, griddiminx, dom_idx_x, dom_idx_y, dom_idx_z;
    int domarea2d, domsize, numvecs, save_idx, rem_frm_w2;
    int nnz_w2=0, nnz_w1=0, col_ctr, nnz_orig, new_col_value;
    int *Zlssd_rows=NULL, *Zlssd_cols=NULL, maxcol_w1;
   FILE *fp, *fp1;
    ValueType *Zsubd_vals=NULL, *Zlssd_vals=NULL;
    //cout<<"The non zeros in Z are "<<this->Z_.get_nnz()<<endl;
   nnz_orig=this->Z_.get_nnz();
    // now we make w1 and w2
    this->Z_.ConvertToCOO();
    this->Z_.MoveToHost();
    this->Z_.LeaveDataPtrCOO(&Zsubd_rows, &Zsubd_cols, &Zsubd_vals);
    // getting arrays of Z into pointers.
    
    for(i=0;i<nnz_orig;i++){
      complementval=1-(bmap[Zsubd_rows[i]]>0?1:0);
    
      if(((ValueType)complementval)*Zsubd_vals[i])
        nnz_w1++;
    }//calculated number of non-zeros in w1
    
    // this count of nnz_w1 can now be used to make a new Z matrix
      s=xdim_/novecni_;	numvecs=novecni_*novecni_*novecni_;
    domsize=s*s*s;  griddiminx=xdim_/s; gridarea2d=griddiminx*griddiminx; domarea2d=s*s;
    for(j=0;j<numvecs;j++){
      didx_x=(j)%griddiminx;    didx_y=(j%gridarea2d)/griddiminx; didx_z=(j)/gridarea2d;
      for(k=0;k<domsize;k++){
        dom_idx_x=(k)%s;      dom_idx_y=(k%domarea2d)/s;	dom_idx_z=(k)/domarea2d;
        rowid=didx_x*s+didx_y*griddiminx*domarea2d+didx_z*gridarea2d*domsize+ 
  			  dom_idx_x+dom_idx_y*xdim_+dom_idx_z*xdim_*xdim_;
  	if(bmap[rowid]>0)
  	  nnz_w2++;
      }
    }
    cout<<"Number of non-zeros in w1 "<<nnz_w1<<" and in w2 is "<<nnz_w2<<" respectively."<<" Sum is ="<<nnz_w1+nnz_w2<<"."<<endl;
    //Allocating space for Z_lssd
    Zlssd_rows=new int[nnz_w1+nnz_w2];	Zlssd_cols=new int[nnz_w1+nnz_w2];
    Zlssd_vals= new ValueType[nnz_w1+nnz_w2];
    // use w1, w2 and Z_CSR_subdomain to generate Z_COO_LSSD
    col_ctr=0;
    for(i=0,j=0;i<nnz_orig;i++){
      complementval=1-(bmap[Zsubd_rows[i]]>0?1:0);
      if(((ValueType)complementval)*Zsubd_vals[i])
      {
        Zlssd_rows[j]=Zsubd_rows[i];	Zlssd_cols[j]=Zsubd_cols[i];	Zlssd_vals[j]=Zsubd_vals[i];
        j++;
        if(Zsubd_cols[i]>col_ctr)
  	col_ctr=Zsubd_cols[i];
      }
    }// we have included w1 in Zlssd
 fp=fopen("Z_lssd_old_w1.rec","wt");
 for(i=0;i<nnz_w1+nnz_w2;i++)
   fprintf(fp,"%d %d %0.2f\n",Zlssd_rows[i], Zlssd_cols[i], Zlssd_vals[i]);
 fclose(fp); 
    maxcol_w1=col_ctr+1;
    col_ctr=maxcol_w1;
   cout<<"Number of columns from w1 is "<<col_ctr<<". Entries added for w1 ="<<j<<"."<<endl;
    save_idx=j;// saving the index from the last set of vectors
    i=save_idx;
   
 //   fp=fopen("col_idxs_w2_old.rec","wt");
 //   fp1=fopen("domain_bubmap_old.rec","wt");
        s=xdim_/novecni_;	numvecs=novecni_*novecni_*novecni_;
    domsize=s*s*s;  griddiminx=xdim_/s; gridarea2d=griddiminx*griddiminx; domarea2d=s*s;
    for(j=0;j<numvecs;j++){
      didx_x=(j)%griddiminx;    didx_y=(j%gridarea2d)/griddiminx; didx_z=(j)/gridarea2d;
      for(k=0;k<domsize;k++){
        dom_idx_x=(k)%s;      dom_idx_y=(k%domarea2d)/s;	dom_idx_z=(k)/domarea2d;
        rowid=didx_x*s+didx_y*griddiminx*domarea2d+didx_z*gridarea2d*domsize+ 
  			  dom_idx_x+dom_idx_y*xdim_+dom_idx_z*xdim_*xdim_;
  	if(bmap[rowid]>0)
  	{
  	  Zlssd_rows[i]=rowid;	Zlssd_cols[i]=(bmap[rowid]-1)+j*maxbmap; Zlssd_vals[i]=(ValueType)1.0f;
 // 	  fprintf(fp,"%d \n",bmap[rowid]-1+j*maxbmap+maxcol_w1);
 // 	  fprintf(fp1,"%d %d %d\n",j, bmap[rowid], rowid);
  	  i++;
 // 	  if(bmap[rowid]-1+j+maxcol_w1>col_ctr)
  // 	  {
  // 	    col_ctr=bmap[rowid]-1+j*maxbmap+maxcol_w1;
  // 	 
  // 	  }
  	}
      }
    }
 //   fclose(fp);
 //   fclose(fp1);
 fp=fopen("Z_lssd_old_w1w2.rec","wt");
 for(i=0;i<nnz_w1+nnz_w2;i++)
   fprintf(fp,"%d %d %0.2f\n",Zlssd_rows[i], Zlssd_cols[i], Zlssd_vals[i]);
 fclose(fp); 
    //serialize values in Zlssd
    col_ctr=Zlssd_cols[save_idx];
    new_col_value=maxcol_w1;
    for (i = save_idx; i < nnz_w1+nnz_w2; i++){
          if(col_ctr!=Zlssd_cols[i])
  	{
  	  col_ctr=Zlssd_cols[i];
  	  new_col_value=new_col_value+1;
  	}
  	Zlssd_cols[i]=new_col_value;
    }
    //removing 1 column to avoid possibility that 1 belongs to Row space(Zlssd)
    rem_frm_w2=0;
    for (i = nnz_w1+nnz_w2-1; i >= 0; i--)
      if(Zlssd_cols[i]==new_col_value)
        rem_frm_w2++;
      
 fp=fopen("Z_lssd_old_w1w2_corr.rec","wt");
 for(i=0;i<nnz_w1+nnz_w2;i++)
   fprintf(fp,"%d %d %0.2f\n",Zlssd_rows[i], Zlssd_cols[i], Zlssd_vals[i]);
 fclose(fp); 
 
    cout<<"Number of columns after addig w2 (DPCG-LSSD) is "<<new_col_value<<"Non-zeros to be removed are= "<<rem_frm_w2<<endl;
    
    this->Z_.Clear();
    this->Z_.SetDataPtrCOO(&Zlssd_rows, &Zlssd_cols, &Zlssd_vals, "Zlssd", nnz_w1+nnz_w2-rem_frm_w2, this->op_->get_nrow(),new_col_value);
    //convert Z_COO LSSD to Z_CSR LSSD
    //this->ZLSSD_.ConvertToCSR();
    //this->Z_.WriteFileMTX("ZLSSD_2sd.rec");
    this->Z_.ConvertToCSR();
    this->Z_.WriteFileMTX("Zcsr_old_lssd.rec");
  //   this->Z_.MoveToAccelerator();
  }*/

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::MakeZLSSD(const int *bmap, const int maxbmap){
  // Z sub-domain and bubmap are assumed to have been made already 
  int *Zsubd_rows=NULL, *Zsubd_cols=NULL, i, complementval, j, vec_num,k;
  
  int save_idx, rem_frm_w2=0,posval, totalmax, colidx, expr;
  int nnz_w2=0, nnz_w1=0, col_ctr, nnz_orig, dim;
  int *Zlssd_rows=NULL, *Zlssd_cols=NULL, *numbub_pervec=NULL, maxcol_w1;
 FILE *fp, *fp1;
  ValueType *Zsubd_vals=NULL, *Zlssd_vals=NULL;
  //cout<<"The non zeros in Z are "<<this->Z_.get_nnz()<<endl;
  nnz_orig=this->Z_.get_nnz();
  // now we make w1 and w2
  this->Z_.ConvertToCOO();
  this->Z_.MoveToHost();
  this->Z_.LeaveDataPtrCOO(&Zsubd_rows, &Zsubd_cols, &Zsubd_vals);
  // getting arrays of Z into pointers.
  dim=this->op_->get_nrow();
  for(i=0;i<nnz_orig;i++){
    complementval=1-(bmap[Zsubd_rows[i]]>0?1:0);
  
    if(((ValueType)complementval)*Zsubd_vals[i])
      nnz_w1++;
  }//calculated number of non-zeros in w1
  
  // this count of nnz_w1 can now be used to make a new Z matrix
  for(i=0;i<dim; i++)
    if(bmap[i]>0)
      nnz_w2++;
    
//   cout<<"Number of non-zeros in w1 "<<nnz_w1<<" and in w2 is "<<nnz_w2<<" respectively."<<" Sum is ="<<nnz_w1+nnz_w2<<"."<<endl;
  //Allocating space for Z_lssd
  Zlssd_rows=new int[nnz_w1+nnz_w2];	Zlssd_cols=new int[nnz_w1+nnz_w2];
  Zlssd_vals= new ValueType[nnz_w1+nnz_w2];
  // use w1, w2 and Z_CSR_subdomain to generate Z_COO_LSSD
  col_ctr=0;
  for(i=0,j=0;i<nnz_orig;i++){
    complementval=1-(bmap[Zsubd_rows[i]]>0?1:0);
    if(((ValueType)complementval)*Zsubd_vals[i])
    {
      Zlssd_rows[j]=Zsubd_rows[i];	Zlssd_cols[j]=Zsubd_cols[i];	Zlssd_vals[j]=Zsubd_vals[i];
      j++;
      if(Zsubd_cols[i]>col_ctr)
	col_ctr=Zsubd_cols[i];
    }
  }// we have included w1 in Zlssd
  maxcol_w1=col_ctr+1;
  col_ctr=maxcol_w1;
//  cout<<"Number of columns from w1 is "<<col_ctr<<". Entries added for w1 ="<<j<<"."<<endl;
  save_idx=j;// saving the index from the last set of vectors
 
  numbub_pervec= new int[col_ctr*(maxbmap+1)]; 
// #pragma omp parallel for 
  for(j=0;j<col_ctr*(maxbmap+1);j++)
    numbub_pervec[j]=0;
  //calculate which levels in bmap exist in which vectors.
  // can be more than one level per vector or none
// #pragma omp parallel for 
  for(j=0;j<dim;j++)
    if(bmap[j]>0)
	{
	 vec_num=get_vecnum(j, xdim_, ydim_, zdim_, novecni_x_, novecni_y_, novecni_z_);
	 numbub_pervec[vec_num*(maxbmap+1)+bmap[j]-1]++;
	}
 // Assign each level in each vector a number in increasing order starting from 1.
 // This helps in counting how many levels per vector are there so the columns
 // can be added serially.
//    for(k=0;k<col_ctr;k++)
//    {
//      for(j=0;j<maxbmap;j++)
//	printf("%d ",numbub_pervec[k*(maxbmap+1)+j]);
//      printf("_+_-> %d ",numbub_pervec[k*(maxbmap+1)+j]);
//      printf("\n");
//    }
//  printf("\n*********************************\n");
  for(j=0;j<col_ctr;j++)
  {
   posval=1;
    for(k=0;k<maxbmap;k++)
      if(numbub_pervec[j*(maxbmap+1)+k]>0)
         {
           numbub_pervec[j*(maxbmap+1)+k]=posval;
           ++posval;
         }
         // calculate cumulative sum of each vector's number of levels and store in last index (maxbmap+1)
    //if(j>0)// not needed
      numbub_pervec[j*(maxbmap+1)+maxbmap]=numbub_pervec[(j-1)*(maxbmap+1)+maxbmap]+posval-1;

  }  
//   for(k=0;k<col_ctr;k++)
//   {
//     for(j=0;j<maxbmap;j++)
// 	printf("%d ",numbub_pervec[k*(maxbmap+1)+j]);
//    printf("_+_-> %d ",numbub_pervec[k*(maxbmap+1)+j]);
//     printf("\n");
//   }
//   

   // Finding maximum columns' value 
   // (cumulative sum of all columns possible in vectors + no. of levels in last vector)
   totalmax= maxcol_w1-1 + numbub_pervec[(maxcol_w1-1)*(maxbmap+1)+maxbmap];
//    printf("\n Total number of columns is %d",totalmax+1);
   // 1 is added to make it no. of columns rather than idex in C

//Make new columns ins Z_lssd minus the column with largest value    
//removing 1 column to avoid possibility that 1 belongs to Row space(Zlssd)
//#pragma omp parallel for  
//   fp=fopen("colids_w2_new.rec","wt");
  for(j=0,i=save_idx;j<dim;++j)
    if(bmap[j]>0)
	{
	  vec_num=get_vecnum(j, xdim_, ydim_, zdim_, novecni_x_, novecni_y_, novecni_z_);
	  expr=(((vec_num-1)*(maxbmap+1)+maxbmap)>0)?numbub_pervec[((vec_num-1)*(maxbmap+1)+maxbmap)]:0;
	  colidx = numbub_pervec[vec_num*(maxbmap+1)+(bmap[j]-1)] +
		    maxcol_w1-1 +
		    expr;
          if(colidx==totalmax)
            ++rem_frm_w2;
          else{ 
	  Zlssd_rows[i] = j;	Zlssd_cols[i]=colidx;
          
//          fprintf(fp,"(%d,%d) %d + %d + %d = %d\n",vec_num, bmap[j]-1,
// 						numbub_pervec[vec_num*(maxbmap+1)+(bmap[j]-1)],  maxcol_w1-1 ,
// 						 expr, Zlssd_cols[i]);
	  Zlssd_vals[i] = (ValueType)1.0f;
          ++i;
          }
       }
//   fclose(fp);
/* fp=fopen("Z_lssd_new_w1w2.rec","wt");
 for(i=0;i<nnz_w1+nnz_w2;i++)
   fprintf(fp,"%d %d %0.2f\n",Zlssd_rows[i], Zlssd_cols[i], Zlssd_vals[i]);
 fclose(fp);*/ 
//   printf("\n non-zeros removed are %d. originally were %d. Setting cols=%d",rem_frm_w2, nnz_w1+nnz_w2, totalmax);  
  this->Z_.Clear();
  this->Z_.SetDataPtrCOO(&Zlssd_rows, &Zlssd_cols, &Zlssd_vals, "Zlssd", nnz_w1+nnz_w2-rem_frm_w2, this->op_->get_nrow(),totalmax);
  //convert Z_COO LSSD to Z_CSR LSSD
  //this->ZLSSD_.ConvertToCSR();
  //this->Z_.WriteFileMTX("ZLSSD_2sd.rec");
  this->Z_.ConvertToCSR();
//   this->Z_.WriteFileMTX("Zcsr_new_lssd.rec");
//   this->Z_.MoveToAccelerator();
  delete [] numbub_pervec;
}
template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::Build(void) {

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
  
  this->q_.CloneBackend(*this->op_);
  this->q_.Allocate("q", this->op_->get_nrow());


    
  if (this->precond_ != NULL) {
    this->precond_->SetOperator(*this->op_);
    this->precond_->Build();
  } 
  else
  {
    this->LLtx_.CloneBackend(*this->op_);
    this->LLtx_.Allocate("LLtx", this->op_->get_nrow());

    this->LLtx2_.CloneBackend(*this->op_);
    this->LLtx2_.Allocate("LLtx2", this->op_->get_nrow());

    this->Dinv_.CloneBackend(*this->op_);

    this->op_->ExtractInverseDiagonal(&this->Dinv_);

    this->L_.CloneBackend(*this->op_);
    this->LT_.CloneBackend(*this->op_);

    this->op_->ExtractL(&this->L_, false);
    this->L_.DiagonalMatrixMult(this->Dinv_);

    this->LT_.CopyFrom(this->L_);
    this->LT_.Transpose();
    
    this->L_.ConvertToDIA();
    this->LT_.ConvertToDIA();
  }
  this->w_.CloneBackend(*this->op_);
  this->w_.Allocate("w", this->op_->get_nrow());

  
  this->Qb_.CloneBackend(*this->op_);
  this->Qb_.Allocate("Qb", this->op_->get_nrow());

  this->Ptx_.CloneBackend(*this->op_);
  this->Ptx_.Allocate("Ptx", this->op_->get_nrow());

  

//   //  this->MakeZ_COO();
//   if (this->Z_.get_nnz() == 0)
//   {
//     std::cout<<"assertion that there is no Z is false, making Z using functions"<<endl;
//     this->MakeZ_CSR();
//     if(zlssd_) // we have to make level set vectors now.
//     {
//       
//       this->MakeZLSSD();
//     }
//       
//   
//   }
  this->hat_.CloneBackend(*this->op_);
  this->hat_.Allocate("hat", this->Z_.get_ncol());

  this->intmed_.CloneBackend(*this->op_);
  this->intmed_.Allocate("intmed", this->Z_.get_ncol());

  this->ZT_.CopyFrom(this->Z_);
  this->ZT_.Transpose();

  this->E_.CloneBackend(*this->op_);
  this->Z_.CloneBackend(*this->op_);
  this->ZT_.CloneBackend(*this->op_);
  this->AZ_.CloneBackend(*this->op_);
  this->AZT_.CloneBackend(*this->op_);

  

  this->AZ_.MatrixMult(*this->op_, this->Z_);
//  std::cout<<"multiplied A with Z"<<endl;
  this->AZT_.CopyFrom(this->AZ_);
  this->AZT_.Transpose();

  this->E_.MatrixMult(this->ZT_, this->AZ_);
//  std::cout<<"calculated E"<<endl;
// 
//   // The Invert() should be on the host
  this->E_.MoveToHost();
//   this->E_.WriteFileMTX("E_shell_inv_old.rec");

  this->E_.ConvertToDENSE();
  this->E_.Invert();
//  this->E_.WriteFileMTX("E_inv_old_lssd.rec");
// 
//   // E_ goes back to the original backend
  this->E_.CloneBackend(*this->op_);
 // this->E_.info();
 // this->Z_.info();
 // this->op_->info();
//   this->ZT_.ConvertToELL();
//   this->AZ_.ConvertToHYB();
  //this->Z_.info();
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::Clear(void) {

  if (this->build_ == true) {
    
    if (this->precond_ != NULL) {
      this->precond_->Clear();
      this->precond_   = NULL;
    }
    else
    {
      this->Dinv_.Clear();
      this->LLtx_.Clear();
      this->LLtx2_.Clear();
      this->L_.Clear();
      this->LT_.Clear();
    }
    this->r_.Clear();
    this->w_.Clear();
    this->p_.Clear();
    this->q_.Clear();
    
    
    this->hat_.Clear();
    this->intmed_.Clear();
    this->Qb_.Clear();
    this->Ptx_.Clear();
    
    this->ZT_.Clear();
    this->E_.Clear();
    this->E_.MoveToHost();
    this->E_.ConvertToCSR();
    this->AZ_.Clear();
    
    this->Z_.Clear();
    this->AZT_.Clear();
    
    this->iter_ctrl_.Clear();
    
    /*this->Z_.info(); 
    this->E_.info();
    this->op_->info();
    
    this->r_.info();
    this->w_.info();
    this->p_.info();
    this->q_.info();
   
    this->hat_.info();
    this->intmed_.info();
    this->Qb_.info();
    this->Ptx_.info();
    
    this->ZT_.info();
    this->AZ_.info();
    this->AZT_.info();
      this->Dinv_.info();
      this->LLtx_.info();
      this->LLtx2_.info();
      this->L_.info();
      this->LT_.info();
    */
    this->build_ = false;
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  if (this->build_ == true) {

    this->r_.MoveToHost();
    this->w_.MoveToHost();
    this->p_.MoveToHost();
    this->q_.MoveToHost();
    
    
    this->hat_.MoveToHost();
    this->intmed_.MoveToHost();
    this->Qb_.MoveToHost();
    this->Ptx_.MoveToHost();
    
    this->ZT_.MoveToHost();
    this->E_.MoveToHost();
    this->AZ_.MoveToHost();
    this->Z_.MoveToHost();
    this->AZT_.MoveToHost();
    if (this->precond_ != NULL)
      this->precond_->MoveToHost();
    else
    {
      this->Dinv_.MoveToHost();
      this->LLtx_.MoveToHost();
      this->LLtx2_.MoveToHost();
      this->L_.MoveToHost();
      this->LT_.MoveToHost();
    
    }
    
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  if (this->build_ == true) {

    this->r_.MoveToAccelerator();
    this->w_.MoveToAccelerator();
    this->p_.MoveToAccelerator();
    this->q_.MoveToAccelerator();
        
    this->hat_.MoveToAccelerator();
    this->intmed_.MoveToAccelerator();
    this->Qb_.MoveToAccelerator();
    this->Ptx_.MoveToAccelerator();
    
    this->ZT_.MoveToAccelerator();
    this->E_.MoveToAccelerator();
    this->AZ_.MoveToAccelerator();
    this->Z_.MoveToAccelerator();
    this->AZT_.MoveToAccelerator();

    if (this->precond_ != NULL)
      this->precond_->MoveToAccelerator();
    else
    {
      this->Dinv_.MoveToAccelerator();
      this->LLtx_.MoveToAccelerator();
      this->LLtx2_.MoveToAccelerator();
      this->L_.MoveToAccelerator();
      this->LT_.MoveToAccelerator();
    }
    
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType &rhs,
                                                              VectorType *x) {
  ofstream lanczos;
//   lanczos.open("ritz_pcg_neumann_32_9bubb1e_1e10.rec",ios::out);
  assert(x != NULL);
  assert(x != &rhs);
  assert(this->op_  != NULL);
  assert(this->precond_  == NULL);
  assert(this->build_ == true);

  const OperatorType *op = this->op_;

  VectorType *r = &this->r_;
  VectorType *p = &this->p_;
  VectorType *q = &this->q_;
  VectorType *w = &this->w_;

  VectorType *Dinv = &this->Dinv_;
  VectorType *LLtx = &this->LLtx_;
  VectorType *LLtx2 = &this->LLtx2_;
  OperatorType *L = &this->L_;
  OperatorType *LT = &this->LT_;

  VectorType *hat = &this->hat_;
  VectorType *intmed = &this->intmed_;
  VectorType *Qb = &this->Qb_;
  VectorType *Ptx = &this->Ptx_;
  OperatorType *ZT = &this->ZT_;
  OperatorType *E = &this->E_;
  OperatorType *AZ = &this->AZ_;
  OperatorType *Z = &this->Z_;
  OperatorType *AZT = &this->AZT_;
 
  ValueType alpha, beta;
  ValueType rho, rho_old;
  ValueType res_norm = 0.0, b_norm = 1.0;
//  ValueType init_residual = 0.0;
  ValueType check_residual = 0.0;

  // initial residual = b - Ax
  op->Apply(*x, r); 
  r->ScaleAdd(ValueType(-1.0), rhs);

  // initial residual for the interation control
  // = |res|
  //  init_residual = this->Norm(*r);

  /// deflation
  // Z^{T}r

  ZT->Apply(*r, hat);

  E->Apply(*hat, intmed);
  AZ->Apply(*intmed, w);
  // r = r - w
  r->AddScale(*w, ValueType(-1.0));



  // initial residual for the interation control
  // = |res| / |b|
  res_norm = this->Norm(*r);
  b_norm = this->Norm(rhs);
 

  this->iter_ctrl_.InitResidual(b_norm);
  //Apply preconditioning w=M^{-1}r
  L->Apply(*r, LLtx);
  L->Apply(*LLtx, LLtx2);
  LLtx->AddScale(*LLtx2, ValueType(-1.0));
  w->CopyFrom(*r, 0, 0, this->op_->get_nrow());

  // (I-LD^{-1}+(LD^{-1})^{2})r_{i-1}
  w->AddScale(*LLtx, ValueType(-1.0));
  w->PointWiseMult(*Dinv);
  LT->Apply(*w, LLtx);
  LT->Apply(*LLtx, LLtx2);
  w->ScaleAdd2(ValueType(1.0), *LLtx, ValueType(-1.0), *LLtx2, ValueType(1.0));

  // rho = (r,w)
  rho = r->Dot(*w);

  p->CopyFrom(*w, 0, 0, this->op_->get_nrow());

  // q = Ap
  op->Apply(*p, q);
  /// deflation
  // Z^Tq
  ZT->Apply(*q, hat);
  E->Apply(*hat, intmed);
  AZ->Apply(*intmed, w);

  // q = q - w
  q->AddScale(*w, ValueType(-1.0));

  // alpha = rho / (p,q)
  alpha=rho/p->Dot(*q);

  // x = x + alpha * p
  x->AddScale(*p, alpha);

  // x = x + alpha * q
  r->AddScale(*q, alpha*ValueType(-1.0));

  res_norm = this->Norm(*r);
  check_residual = res_norm; 

  while (!this->iter_ctrl_.CheckResidual(check_residual)) {

    //Apply preconditioning w=M^{-1}r
    L->Apply(*r, LLtx);
    L->Apply(*LLtx, LLtx2);
    LLtx->AddScale(*LLtx2, ValueType(-1.0));
    w->CopyFrom(*r, 0, 0, this->op_->get_nrow());

    // (I-LD^{-1}+(LD^{-1})^{2})r_{i-1}
    w->AddScale(*LLtx, ValueType(-1.0));
    w->PointWiseMult(*Dinv);
    LT->Apply(*w, LLtx);
    LT->Apply(*LLtx, LLtx2);
    w->ScaleAdd2(ValueType(1.0), *LLtx, ValueType(-1.0), *LLtx2, ValueType(1.0));

    rho_old = rho;

    // rho = (r,w)
    rho = r->Dot(*w);

    beta = rho / rho_old;

    // p = p + beta * w
    p->ScaleAdd(beta, *w);

    // q = Ap
    op->Apply(*p, q);

    /// deflation
    // Z^Tq
    ZT->Apply(*q, hat);
    E->Apply(*hat, intmed);
    AZ->Apply(*intmed, w);

    // q = q - w
    q->AddScale(*w, ValueType(-1.0));

    // at this point save alpha and beta
//    lanczos<<setprecision(16)<<alpha<<" "<<setprecision(16)<<beta<<" "<<endl;
    
    // alpha = rho / (p,q)
    alpha=rho/p->Dot(*q);

    // x = x + alpha * p
    x->AddScale(*p, alpha);

    // x = x + alpha * q
    r->AddScale(*q, alpha*ValueType(-1.0));

    res_norm = this->Norm(*r);
    check_residual = res_norm; 
  }

//   // correct solution
  // Qb
  //Z^{T}rhs
  ZT->Apply(rhs, hat);
  // E^{-1}hat
  E->Apply(*hat, intmed);
  Z->Apply(*intmed, Qb);

  // Ptx
  // AZ^{T}x
  AZT->Apply(*x, hat);
  // E^{-1}hat
  E->Apply(*hat, intmed);
  Z->Apply(*intmed, Ptx);
  x->AddScale(*Ptx, ValueType(-1.0));
  x->AddScale(*Qb, ValueType(1.0));
//   
  
//   lanczos.close();
}

template <class OperatorType, class VectorType, typename ValueType>
void DPCG<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType &rhs,
                                                            VectorType *x) {

  ofstream lanczos;
//   lanczos.open("ritz_dpcg_meilu_ilu01_32_9bub_1e8_gpu.rec",ios::out);
  assert(x != NULL);
  assert(x != &rhs);
  assert(this->op_  != NULL);
  assert(this->precond_ != NULL);
  assert(this->build_ == true);

  const OperatorType *op = this->op_;

  VectorType *r = &this->r_;
  VectorType *p = &this->p_;
  VectorType *q = &this->q_;
  VectorType *w = &this->w_;


  VectorType *hat = &this->hat_;
  VectorType *intmed = &this->intmed_;
  VectorType *Qb = &this->Qb_;
  VectorType *Ptx = &this->Ptx_;


  OperatorType *ZT = &this->ZT_;
  OperatorType *E = &this->E_;
  OperatorType *AZ = &this->AZ_;

  OperatorType *Z = &this->Z_;
  OperatorType *AZT = &this->AZT_;
 
  ValueType alpha, beta;
  ValueType rho, rho_old;
  ValueType res_norm = 0.0, b_norm = 1.0;
//   ValueType init_residual = 0.0;
  ValueType check_residual = 0.0;
  //std::cout<<"about to start preconditioned solve"<<endl;
  // initial residual = b - Ax
  op->Apply(*x, r); 
  r->ScaleAdd(ValueType(-1.0), rhs);

  // initial residual for the interation control
  // = |res|
  //  init_residual = this->Norm(*r);


  // Z^{T}r
  ZT->Apply(*r, hat);
  E->Apply(*hat, intmed);
  AZ->Apply(*intmed, w);

  // r = r - w
  r->AddScale(*w, ValueType(-1.0));



  // initial residual for the interation control
  // = |res| / |b|
  res_norm = this->Norm(*r);
  b_norm = this->Norm(rhs);
  

  this->iter_ctrl_.InitResidual(b_norm);

  //Apply preconditioning w=M^{-1}r

  this->precond_->SolveZeroSol(*r, w);
  
  // rho = (r,w)
  rho = r->Dot(*w);

  p->CopyFrom(*w, 0, 0, this->op_->get_nrow());

  // q = Ap
  op->Apply(*p, q);

  // Z^Tq
  ZT->Apply(*q, hat);
  E->Apply(*hat, intmed);
  AZ->Apply(*intmed, w);

  // q = q - w
  q->AddScale(*w, ValueType(-1.0));


  // alpha = rho / (p,q)
  alpha=rho/p->Dot(*q);

  // x = x + alpha * p
  x->AddScale(*p, alpha);

  // x = x + alpha * q
  r->AddScale(*q, alpha*ValueType(-1.0));

  res_norm = this->Norm(*r);
  check_residual = res_norm; 

  while (!this->iter_ctrl_.CheckResidual(check_residual)) {

    //Apply preconditioning w=M^{-1}r

    this->precond_->SolveZeroSol(*r, w);
    
    rho_old = rho;

    // rho = (r,w)
    rho = r->Dot(*w);

    beta = rho / rho_old;

    // p = p + beta * w
    p->ScaleAdd(beta, *w);

    // q = Ap
    op->Apply(*p, q);

//     // Z^Tq
    ZT->Apply(*q, hat);
    E->Apply(*hat, intmed);
    AZ->Apply(*intmed, w);

    // q = q - w
    q->AddScale(*w, ValueType(-1.0));
      // at this point save alpha and beta
//   lanczos<<setprecision(16)<<alpha<<" "<<setprecision(16)<<beta<<" "<<endl;
    // alpha = rho / (p,q)
    alpha=rho/p->Dot(*q);

    // x = x + alpha * p
    x->AddScale(*p, alpha);

    // x = x + alpha * q
    r->AddScale(*q, alpha*ValueType(-1.0));

    res_norm = this->Norm(*r);
    check_residual = res_norm; 
  }

  // correct solution
  // Qb
  // Z^{T}rhs
  ZT->Apply(rhs, hat);
  // E^{-1}hat
  E->Apply(*hat, intmed);
  Z->Apply(*intmed, Qb);

  // Ptx
  // AZ^{T}x
  AZT->Apply(*x, hat);
  // E^{-1}hat
  E->Apply(*hat, intmed);
  Z->Apply(*intmed, Ptx);
  x->AddScale(*Ptx, ValueType(-1.0));
  x->AddScale(*Qb, ValueType(1.0));
  
//   lanczos.close();
}





template class DPCG< LocalMatrix<double>, LocalVector<double>, double >;
template class DPCG< LocalMatrix<float>,  LocalVector<float>, float >;

}

