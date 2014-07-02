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


#include "math_functions.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
using namespace std;
namespace paralution {

float paralution_abs(const float val) {

  return fabs(val);

}

double paralution_abs(const double val) {

  return fabs(val);

}

int paralution_abs(const int val) {

  return abs(val);

}

int makew1w2(int *bubmap, int *rowsZ, int *colsZ, int *rowsZ_w1, int *colsZ_w1, int *rowsZ_w2, int *colsZ_w2, 
	     int *nzw1, int *nzw2, int *w1col, int *w2col, int *r_frm_w2, int dim, int xdim, int s, int numdomains,
	     int numvecs, int numbub, int procid)
{
  int domsize, griddiminx, gridarea2d, domarea2d, didx_x, didx_y, didx_z, dom_idx_x, dom_idx_y, dom_idx_z;
  int rowid, col_ctr, new_col_value, w2cols, nnzw1, nnzw2, w1cols=0, rem_frm_w2, i, j, k;
  double dproduct; //FILE *fp;char name[20];
  double *complement_bmap; 
  complement_bmap=(double*)calloc(dim,sizeof(double));
  nnzw2=0;
  /// Now make levelset(bubmap) vector for subdomains(bubble only). Inverse of that is levelset(bubmap) matrix for(no bubbles).
  domsize=s*s*s;  griddiminx=xdim/s; gridarea2d=griddiminx*griddiminx; domarea2d=s*s;
  
  for(j=0;j<numvecs;j++){
    didx_x=(j)%griddiminx;    didx_y=(j%gridarea2d)/griddiminx; didx_z=(j)/gridarea2d;
    for(k=0;k<domsize;k++){
      dom_idx_x=(k)%s;      dom_idx_y=(k%domarea2d)/s;	dom_idx_z=(k)/domarea2d;
      rowid=didx_x*s+didx_y*griddiminx*domarea2d+didx_z*gridarea2d*domsize+ //we are at the first ele of the right block
      //now within the block we have to position
			  dom_idx_x+dom_idx_y*xdim+dom_idx_z*xdim*xdim;
  	if(bubmap[rowid]>0)///encountered a non-zeros
	{
	    colsZ_w2[nnzw2]=bubmap[rowid]-1+j*numbub;// so a new column is assigned if a new value is read
	    rowsZ_w2[nnzw2]=rowid;
	    nnzw2++;
	}
      
    }
  }
 /*sprintf(name,"w2_%d.rec",procid);
 fp=fopen(name,"wt");
for(i=0;i<nnzw2;i++)
  fprintf(fp,"%d %d\n",rowsZ_w2[i], colsZ_w2[i]);
fclose(fp);  */
  // the sorted list coming out of the previous loop makes sure that
  // sequential renumbering works in the following code
  // re-number columns remove one column and then add to it the complement.
  col_ctr=colsZ_w2[0];
  new_col_value=0;
  for (i = 0; i < nnzw2; i++){
        if(col_ctr!=colsZ_w2[i])
	{
	  col_ctr=colsZ_w2[i];
	  new_col_value=new_col_value+1;
	}
	colsZ_w2[i]=new_col_value;
  }
  rem_frm_w2=0;
  for (i = nnzw2-1; i >= 0; i--)
    if(colsZ_w2[i]==new_col_value)
      rem_frm_w2++;
  //printf("\n number of columns in W2 is %d. number of non-zeros to be removed from column %d is %d,",new_col_value+1,new_col_value+1,rem_frm_w2);
  w2cols=new_col_value+1;
  
  
  
  
  /// make (1-union(W_l)) this is just bubmap with 1's in place of zeros and all non-zeros as 0s
  for(i=0;i<dim;i++)
    complement_bmap[i]=1.0f-(bubmap[i]>0?1.0f:0.0f);
 /* fp=fopen("compbmap.txt","wt");
  for(i=0;i<dim;i++)
    fprintf(fp,"%d %f\n",i, complement_bmap[i]);
  fclose(fp);*/
  /// To calculate w1 one just needs to take component-wise AND of complement_bmap with all the vectors in Ws
  
  for(j=0,nnzw1=0;j<dim;j++)
  {
    dproduct=complement_bmap[rowsZ[j]]*(1.0f);
    if(dproduct>0.0)
    {
      colsZ_w1[nnzw1]=colsZ[j]; rowsZ_w1[nnzw1]=rowsZ[j];
      //Z_w1[nnzw1]=dproduct;
      if(!nnzw1)
	w1cols=colsZ[j];
      if(w1cols!=colsZ[j])
	w1cols++;
      nnzw1++;
    }// it is the intersection of 1-levelset with subdomains so its
    //minimum size must be atleast equal to the number of subdomains.
  }
  w1cols=w1cols+1;
  
  *w1col=w1cols; *w2col=w2cols; *nzw1=nnzw1; *nzw2=nnzw2; *r_frm_w2=rem_frm_w2;
  free(complement_bmap);
  return 0;
}

int bubmap_create(const double *phi, int *bubmap, const int xdim, const int ydim, const int zdim, 
		  const int dim, int *maxbmap, const int lvst_offst){
  
  //FILE *fp;  int i;
  makebubmap(xdim, ydim, zdim, dim, phi, bubmap, 0, lvst_offst);
  *maxbmap=fixbubmap(bubmap, xdim, dim);
  printf("\n maxbmap is %d",*maxbmap);
//   fp=fopen("bubmap_par_128_9bub.rec","wt");
//   for(i=0;i<dim;i++)
//     fprintf(fp,"%d \n",bubmap[i]);
//   fclose(fp);
  
  return 1;
}


int makebubmap(const int xdim, const int ydim, const int zdim, const int dim,
	       const double *phi, int *bubmap, const int extension, const int lvst_offst)
{
    int j,k;
    int decide, phin_x, phin_y, phin_z, *submap, *all_nayburs; //char name[20];
    int max_threads, num_procs;
    FILE *fp;
    double *phimap;
    submap	=(int*)calloc(dim,sizeof(int));
    phimap	=(double*)calloc(dim,sizeof(double));
    all_nayburs	=(int*)calloc(dim*3,sizeof(int));
    
    phin_x=xdim+2*lvst_offst;	phin_y=ydim+2*lvst_offst;	phin_z=zdim+2*lvst_offst;
    j=lvst_offst*phin_y*phin_z+lvst_offst*phin_z+lvst_offst;
    ///operate on phi here and generate a Z;
    ///we have to eliminate the surrounding ghost points (2 on each boundary)
    //printf("\n extension chosen is %d", extension);
    k=0;
    
    while(1)
    {    
      if( ((k%zdim)==0) && (k!=0))// we have hit a row boundary in the cude so jump over two points
	j=j+lvst_offst+lvst_offst;
      if( ((k%(ydim*zdim))==0) && (k!=0))//we have hit the top now we need to jump 2+2*n+2*n+2 points
	j=j+lvst_offst*phin_z+lvst_offst*phin_z;
      if(k>(dim-1))
	break;
      phimap[k]=phi[j];
        j++;  k++;      
    }   
//     fp=fopen("phimap_irreg.rec","wt");
//     for(i=0;i<dim;i++);
//       fprintf(fp,"%0.9f\n",phimap[i]);
//     fclose(fp);
    /// phimap has only the phi values for the grid points.
    j=0;decide=0;
//     max_threads=omp_get_max_threads();
//     num_procs= omp_get_num_procs();
//     printf("\n Number of thread for openMP inside makebubmap are %d. max_threads are %d. num_procs %d", omp_get_num_threads(), max_threads, num_procs);
//     omp_set_num_threads(8);
//     printf("\n Number of thread for openMP inside makebubmap are now %d", omp_get_num_threads());
    omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel
    {
      
      #pragma omp for
    for (int i=0;i<dim;i++)
      {
	int Xcord, Ycord, Zcord, tempval, left, bottom, leftface;
	if(phimap[i]>=0.0){
	  //calculate left and bottom for this point
	  Xcord	=	i/(ydim * zdim);	
	  tempval	=	i - Xcord * ydim * zdim;
	  Ycord	=	(tempval>0)?(tempval)/zdim:0;	
	  tempval	=	tempval-Ycord * zdim;
	  Zcord	=	(tempval>0)?tempval:0;

	  if(Zcord==0)	left=-1;	else	left=i-1;
	  if(Ycord==0)	bottom=-1;	else	bottom=i-zdim;
	  if(Xcord==0)	leftface=-1;	else	leftface=i-(ydim*zdim);
	  all_nayburs[i]=left;	all_nayburs[dim+i]=bottom;	all_nayburs[2*dim+i]=leftface;	
	}
      }
    }
    for (int i=0;i<dim;i++)
      {
	if(phimap[i]>=0.0){
	j=calclvlstval(all_nayburs[i],all_nayburs[dim+i],all_nayburs[2*dim+i],
		       j, phimap, bubmap, i, decide, dim);
	}
      }

    decide=1;
    omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel
    {
      
      #pragma omp for
      for (int i=dim-1;i>=0;i--)
      {
	int Xcord, Ycord, Zcord, tempval, right, top, rightface;
	if(phimap[i]>=0.0){
	  //calculate left and bottom for this point
	  Xcord	=	i/(ydim * zdim);	
	  tempval	=	i - Xcord * ydim * zdim;
	  Ycord	=	(tempval>0)?(tempval)/zdim:0;	
	  tempval	=	tempval-Ycord * zdim;
	  Zcord	=	(tempval>0)?tempval:0;

	  if(Zcord==zdim-1)	right=-1;	else	right=i+1;
	  if(Ycord==ydim-1)	top=-1;		else	top=i+zdim;
	  if(Xcord==xdim-1)	rightface=-1;	else	rightface=i+(ydim*zdim);
	  all_nayburs[i]=right;	all_nayburs[dim+i]=top;	all_nayburs[2*dim+i]=rightface;	
	
	}
      }
    }
    for (int i=dim-1;i>=0;i--)
      {
	if(phimap[i]>=0.0){
	  j=calclvlstval(all_nayburs[i],all_nayburs[dim+i],all_nayburs[2*dim+i],
			 j, phimap, bubmap, i, decide, dim);
	}
      }
  ///Now extend boundaries of the bubbles
//    for (k=0;k<extension;k++){
//     memcpy(submap,bubmap,sizeof(int)*dim);
//     for(i=0;i<dim;i++){
//     //Inonzero = find(bubmap>0);
//     //for i=Inonzero'
//       
//       if(submap[i]>0){
//         Xcord=(i%xdim);
//         //if(i>=(xdim*xdim))	Ycord=(i/xdim)-xdim;	else	Ycord=i/xdim;
//         Zcord=i/(xdim*ydim);
// 	Ycord=(i/xdim)%ydim;
// 	if(Xcord==0)		left=-1;	else	left=i-1;
//         if(Xcord==xdim-1)	right=-1;	else	right=i+1;
// 	if(Ycord==0)		bottom=-1;	else	bottom=i-xdim;
// 	if(Ycord==ydim-1)	top=-1;		else	top=i+xdim;
//         if(Zcord==0)		leftface=-1;	else	leftface=i-(xdim*ydim);
// 	if(Zcord==zdim-1)	rightface=-1;	else	rightface=i+(xdim*ydim);
// 	
//         //find all neighbors that are out of phase.
//         if((left>=0) 	&& (phimap[left]<0.0f))		bubmap[left]=bubmap[i];
// 	if((bottom>=0) 	&& (phimap[bottom]<0.0f))	bubmap[bottom]=bubmap[i];
//         if((right>=0) 	&& (phimap[right]<0.0f))	bubmap[right]=bubmap[i];
// 	if((top>=0) 	&& (phimap[top]<0.0f))		bubmap[top]=bubmap[i];
//         if((leftface>=0) && (phimap[leftface]<0.0f))	bubmap[leftface]=bubmap[i];
//         if((rightface>=0) && (phimap[rightface]<0.0f))	bubmap[rightface]=bubmap[i];
// 	
//       }
//     }
//   }
  

  free(phimap);	free(all_nayburs);  free(submap);
  
  return 0;
}

int instore(int *store, int val, int *index, int sizeofstore)//linear search of unique values
{
  int i, found;
  found=0;
// #pragma omp parallel for   
  for(i=0;i<sizeofstore;i++)
    if(val==store[i])
    {
      found=1;
      *index=i;
    }
    return found;
}


int get_vecnum(int rowid, int xdim, int ydim, int zdim, int novecni_x_,
	       int novecni_y_, int novecni_z_)
{
  int x_coord, y_coord, z_coord, d_idx, d_idy, d_idz, column, tempval;
  
  
    x_coord	=	rowid/(ydim * zdim);	
    tempval	=	rowid - x_coord * ydim * zdim;
    y_coord	=	(tempval>0)?(tempval)/zdim:0;	
    tempval	=	tempval-y_coord * zdim;
    z_coord	=	(tempval>0)?tempval:0;
    
    tempval	=	top(z_coord, zdim, novecni_z_) ;
    d_idz	=	(tempval>0)?tempval:0;
    tempval	=	top(y_coord, ydim, novecni_y_);	
    d_idy	=	(tempval>0)?tempval:0;
    tempval	=	top(x_coord, xdim, novecni_x_);
    d_idx	=	(tempval>0)?tempval:0;
    
    
    column=d_idx * novecni_y_ * novecni_z_ + d_idy * novecni_z_ + d_idz;
    
    return column;
}

int top(const int coord, const int dim, const int vec_indim) {
  
  int non_multiple;//, novec_perdirec,
  int domain_id_intg, coord_perdomain_intg;
  non_multiple		=	dim % vec_indim;
  //novec_perdirec	=	dim / vec_indim;
  coord_perdomain_intg	=	dim / vec_indim;
  domain_id_intg	=	coord / coord_perdomain_intg;
  if(non_multiple) // the dimension is not a multiple of no. of vectors
    if(coord+1>coord_perdomain_intg*vec_indim) // check if 
      return (domain_id_intg - 1);
    else
      return domain_id_intg;
  else
    return (domain_id_intg) ;
}
int fixbubmap(int *bubmap, int maxnumbubs, int dim)
{
  int storectr, index, *store; char maxbmap;//name[20], 
  store=(int*)calloc(maxnumbubs,sizeof(int));
  FILE *fp;
  //memset(store,0,sizeof(int)*maxnumbubs);//setting to zeros
  omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel   
  {
    
//     printf("\n no. threads loop 1 of fixbubmap is %d",omp_get_num_threads());
#pragma omp for  
  for(int i=0;i<maxnumbubs;i++)
    store[i]=0;
  }
  storectr=0; index=0;
  // serialize bubmap give it values between 1 and number of bubbles instead of random values 
  // that come out of this code
//  #pragma omp for   
  for(int i=0;i<dim;i++)
   {
     if(bubmap[i]>0 )
       if(!instore(store,bubmap[i], &index, storectr))
	  store[storectr++]=bubmap[i];
   }
 //  printf("\n num bubs on proc %d is %d\n",procid, numbub);
 omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel   
{
  
#pragma omp for  
  for(int i=0;i<dim;i++)
  {
    int index;
    if(bubmap[i]>0)
    {
      instore(store,bubmap[i], &index, storectr);
      
      bubmap[i]=index+1;
    }
  }
}
  
//  fp=fopen("fxdbubmap_.rec","wt");
// for(i=0;i<dim;i++)
//   fprintf(fp,"%d \n",bubmap[i]);
// fclose(fp);  
  free(store);
  //maxbmap=index+1;
  maxbmap=storectr;
  return maxbmap;
}



int calclvlstval(int left, int bottom, int leftface, int inj, double *phimap, int *bubmap, int i, int decide, int dim)
{
  int *valarray;
  int minval, retval=0;  
  valarray= (int*)calloc(3,sizeof(int));
  valarray[0]=9999999; valarray[1]=9999999;valarray[2]=9999999;
  if ( (left>=0) && (left<dim))
    if((phimap[left]>=0.0) ) // valid point and part of bub/intfc
    valarray[0]=bubmap[left];
  
  if ( (bottom>=0) && (bottom<dim))
    if((phimap[bottom]>=0.0) ) // valid point and part of bub/intfc
    valarray[1]=bubmap[bottom];
  
  if ( (leftface>=0) && (leftface<dim))
    if((phimap[leftface]>=0.0) ) // valid point and part of bub/intfc
    valarray[2]=bubmap[leftface];
  
  minval=valarray[0];
  
  if(minval>valarray[1])    minval=valarray[1];
  if(minval>valarray[2])    minval=valarray[2];
  if(minval==9999999){
     if(decide);
     else { retval=inj+1; bubmap[i]=retval;}
  }
  else {bubmap[i]=minval;	retval=inj; }

  free(valarray);
  return retval;
}
void swap_int(int *in1, int *in2, int *tmp)
{
  *tmp=*in1;
  *in1=*in2;
  *in2=*tmp;
}
void swap_dbl(double *in1, double *in2, double *tmp)
{
  *tmp=*in1;
  *in1=*in2;
  *in2=*tmp;
}
void quick_sort(int *cols, int  *rows, double *vals, int low, int high)
{
 int pivot,j,temp,i;
 double temp_double;
 if(low<high)
 {
  pivot = low;
  i = low;
  j = high;
 
  while(i<j)
  {
   while((cols[i]<=cols[pivot])&&(i<high))
   {
    i++;
   }
 
   while(cols[j]>cols[pivot])
   {
    j--;
   }
 
   if(i<j)
   {
//     temp=cols[i];
//     cols[i]=cols[j];
//     cols[j]=temp;
    swap_int(&cols[i], &cols[j], &temp);
    swap_int(&rows[i], &rows[j], &temp);
    swap_dbl(&vals[i], &vals[j], &temp_double);
    // swap rows and vals also
   }
  }
 
//   temp=cols[pivot];
//   cols[pivot]=cols[j];
//   cols[j]=temp;
  swap_int(&cols[pivot], &cols[j], &temp);
  swap_int(&rows[pivot], &rows[j], &temp);
  swap_dbl(&vals[pivot], &vals[j], &temp_double);
  // swap rows and vals also
  quick_sort(cols, rows, vals,low,j-1);
  quick_sort(cols, rows, vals,j+1,high);
 }
}

}

