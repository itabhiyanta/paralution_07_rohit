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

#include "lduMatrix.H"

#include <assert.h>
#include <paralution.hpp>

// TODO
// this should be in a namespace !!!

enum _preconditioner_type {
  paralution_None,
  paralution_Jacobi,
  paralution_MultiColoredGS,
  paralution_MultiColoredSGS,
  paralution_ILU,
  paralution_MultiColoredILU,
  paralution_MultiElimination,
  paralution_FSAI
};

enum _solver_type {
  paralution_CG,
  paralution_BiCGStab,
  paralution_GMRES,
  paralution_FixedPoint,
  paralution_FGMRES
};
  

template <typename ValueType>
void import_openfoam_matrix(const Foam::lduMatrix &foam_mat,
                            paralution::LocalMatrix<ValueType> *mat) {

  // Matrix size
  int n = foam_mat.diag().size();
  int m = foam_mat.diag().size();
  int nnz = foam_mat.diag().size() + foam_mat.lower().size() + foam_mat.upper().size();

  // CSR values
  int *row_offset = NULL;
  int *col = NULL;
  ValueType *val = NULL;

  paralution::allocate_host(n+1, &row_offset);
  paralution::allocate_host(nnz, &col);
  paralution::allocate_host(nnz, &val);

  row_offset[0] = 0;
  // initialize row_offsets with ones from OpenFOAM diagonal part
  for (int i=1; i<n+1; ++i)
    row_offset[i] = 1;

  Foam::UList<int>::const_iterator it_low = foam_mat.lduAddr().lowerAddr().begin();
  Foam::UList<int>::const_iterator it_up  = foam_mat.lduAddr().upperAddr().begin();

  // compute nnz per row in lower part
  for (int i=0; i<foam_mat.lower().size(); ++i) {
    row_offset[*it_low + 1]++;
    row_offset[*it_up  + 1]++;
    it_low++;
    it_up ++;
  }

  int sum = 0;
  for (int i=0; i<n; ++i) {
    int temp = row_offset[i];
    row_offset[i] = sum;
    sum += temp;
  }
  row_offset[n] = sum;

  Foam::UList<double>::const_iterator it_val = foam_mat.lower().begin();
  it_low = foam_mat.lduAddr().lowerAddr().begin();
  it_up  = foam_mat.lduAddr().upperAddr().begin();

  // fill col and val arrays for lower part
  for (int i=0; i<foam_mat.lower().size(); ++i) {
    // row index for lower = upper + 1
    int r_lower = *it_up + 1;
    int dest_lower = row_offset[r_lower];

    col[dest_lower] = *it_low;
    val[dest_lower] = *it_val;

    ++row_offset[r_lower];

    it_low++;
    it_up ++;
    it_val++;
  }

  it_val = foam_mat.diag().begin();

  // fill diagonal part
  for (int i=0; i<foam_mat.diag().size(); ++i) {
    int dest_diag = row_offset[i+1];

    val[dest_diag] = *it_val;
    col[dest_diag] = i;
    ++row_offset[i+1];
    it_val++;
  }

  it_low = foam_mat.lduAddr().lowerAddr().begin();
  it_up  = foam_mat.lduAddr().upperAddr().begin();
  it_val = foam_mat.upper().begin();

  // fill upper part
  for (int i=0; i<foam_mat.upper().size(); ++i) {
    // row index for upper part = lower + 1
    int r_upper = *it_low + 1;
    int dest_upper = row_offset[r_upper];

    col[dest_upper] = *it_up;
    val[dest_upper] = *it_val;

    ++row_offset[r_upper];

    it_low++;
    it_up ++;
    it_val++;
  }

  mat->Clear();
  mat->MoveToHost();
  mat->SetDataPtrCSR(&row_offset, &col, &val, "OpenFOAM Matrix", nnz, n, m);

}

template <typename ValueType>
void import_openfoam_vector(const Foam::scalarField &foam_vec,
                            paralution::LocalVector<ValueType> *vec) {

  vec->Clear();
  vec->MoveToHost();

  Foam::UList<double>::const_iterator it     = foam_vec.begin();
  Foam::UList<double>::const_iterator it_end = foam_vec.end();

  int size = int(foam_vec.size());
  int i = 0;
  ValueType *val = NULL;

  paralution::allocate_host(size, &val);

  for (int i=0; it != it_end; ++it, ++i)
    val[i] = *it;

  vec->SetDataPtr(&val, "OpenFOAM vector", size);

}

template <typename ValueType>
void export_openfoam_vector(paralution::LocalVector<ValueType> &vec,
                            Foam::scalarField *foam_vec) {

  vec.MoveToHost();
  assert(int(vec.get_size()) == int(foam_vec->size()));

  Foam::UList<double>::iterator it = foam_vec->begin();

  int size = vec.get_size();
  ValueType *val = NULL;

  vec.LeaveDataPtr(&val);

  for (int i = 0; i < size; ++i, ++it)
    *it = val[i];

  paralution::free_host(&val);

}

template <typename ValueType>
void import_openfoam_restriction(const Foam::labelField &foam_vec,
                                 paralution::LocalVector<ValueType> *vec) {

  vec->Clear();
  vec->MoveToHost();

  Foam::UList<int>::const_iterator it     = foam_vec.begin();
  Foam::UList<int>::const_iterator it_end = foam_vec.end();

  int size = int(foam_vec.size());
  int i = 0;
  ValueType *val = NULL;

  paralution::allocate_host(size, &val);

  for (; it != it_end; ++it) {
    val[i] = *it;
    ++i;
  }

  vec->SetDataPtr(&val, "OpenFOAM vector", size);

}

template <typename ValueType> paralution::Preconditioner<paralution::LocalMatrix<ValueType>,
                                                         paralution::LocalVector<ValueType>,
                                                         ValueType >*
GetPreconditioner(const std::string precond_name, const std::string lbp_name, const std::string pformat,
                  const int ILUp, const int ILUq, const int MEp) {

  _preconditioner_type pType = paralution_None;
  if      (precond_name == "paralution_Jacobi")           pType = paralution_Jacobi;
  else if (precond_name == "paralution_MultiColoredGS")   pType = paralution_MultiColoredGS;
  else if (precond_name == "paralution_MultiColoredSGS")  pType = paralution_MultiColoredSGS;
  else if (precond_name == "paralution_ILU")              pType = paralution_ILU;
  else if (precond_name == "paralution_MultiColoredILU")  pType = paralution_MultiColoredILU;
  else if (precond_name == "paralution_MultiElimination") pType = paralution_MultiElimination;
  else if (precond_name == "paralution_FSAI")             pType = paralution_FSAI;

  paralution::_matrix_format pf = paralution::CSR;
  if      (pformat == "CSR")   pf = paralution::CSR;
  else if (pformat == "ELL")   pf = paralution::ELL;
  else if (pformat == "HYB")   pf = paralution::HYB;
  else if (pformat == "MCSR")  pf = paralution::MCSR;
  else if (pformat == "BCSR")  pf = paralution::BCSR;
  else if (pformat == "COO")   pf = paralution::COO;
  else if (pformat == "DIA")   pf = paralution::DIA;
  else if (pformat == "DENSE") pf = paralution::DENSE;

  paralution::Preconditioner<paralution::LocalMatrix<ValueType>,
                             paralution::LocalVector<ValueType>,
                             ValueType > *precond = NULL;

  switch(pType) {

    case paralution_None:
      break;

    case paralution_Jacobi:
      precond = new paralution::Jacobi<paralution::LocalMatrix<ValueType>,
                                       paralution::LocalVector<ValueType>,
                                       ValueType >;
      break;

    case paralution_MultiColoredGS:
      paralution::MultiColoredGS<paralution::LocalMatrix<ValueType>,
                                 paralution::LocalVector<ValueType>,
                                 ValueType > *p_gs;
      p_gs = new paralution::MultiColoredGS<paralution::LocalMatrix<ValueType>,
                                            paralution::LocalVector<ValueType>,
                                            ValueType >;
      p_gs->SetPrecondMatrixFormat(pf);
      precond = p_gs;
      break;

    case paralution_MultiColoredSGS:
      paralution::MultiColoredSGS<paralution::LocalMatrix<ValueType>,
                                  paralution::LocalVector<ValueType>,
                                  ValueType > *p_sgs;
      p_sgs = new paralution::MultiColoredSGS<paralution::LocalMatrix<ValueType>,
                                              paralution::LocalVector<ValueType>,
                                              ValueType >;
      p_sgs->SetPrecondMatrixFormat(pf);
      precond = p_sgs;
      break;

    case paralution_ILU:
      paralution::ILU<paralution::LocalMatrix<ValueType>,
                      paralution::LocalVector<ValueType>,
                      ValueType > *p_ilu;
      p_ilu = new paralution::ILU<paralution::LocalMatrix<ValueType>,
                                  paralution::LocalVector<ValueType>,
                                  ValueType >;

      p_ilu->Set(ILUp);
      precond = p_ilu;
      break;

    case paralution_MultiColoredILU:
      paralution::MultiColoredILU<paralution::LocalMatrix<ValueType>,
                                  paralution::LocalVector<ValueType>,
                                  ValueType > *p_mcilu;
      p_mcilu = new paralution::MultiColoredILU<paralution::LocalMatrix<ValueType>,
                                                paralution::LocalVector<ValueType>,
                                                ValueType >;
      p_mcilu->Set(ILUp, ILUq);
      p_mcilu->SetPrecondMatrixFormat(pf);
      precond = p_mcilu;
      break;

    case paralution_FSAI:
      paralution::FSAI<paralution::LocalMatrix<ValueType>,
                       paralution::LocalVector<ValueType>,
                       ValueType > *p_fsai;
      p_fsai = new paralution::FSAI<paralution::LocalMatrix<ValueType>,
                                    paralution::LocalVector<ValueType>,
                                    ValueType >;
      p_fsai->SetPrecondMatrixFormat(pf);
      precond = p_fsai;
      break;

    case paralution_MultiElimination:
      paralution::MultiElimination<paralution::LocalMatrix<ValueType>,
                                   paralution::LocalVector<ValueType>,
                                   ValueType > *p_me;
      p_me = new paralution::MultiElimination<paralution::LocalMatrix<ValueType>,
                                              paralution::LocalVector<ValueType>,
                                              ValueType >;

      _preconditioner_type lbp = paralution_Jacobi;
      if      ( lbp_name == "paralution_Jacobi" )          lbp = paralution_Jacobi;
      else if ( lbp_name == "paralution_MultiColoredGS" )  lbp = paralution_MultiColoredGS;
      else if ( lbp_name == "paralution_MultiColoredSGS" ) lbp = paralution_MultiColoredSGS;
      else if ( lbp_name == "paralution_ILU" )             lbp = paralution_ILU;
      else if ( lbp_name == "paralution_MultiColoredILU" ) lbp = paralution_MultiColoredILU;

      paralution::Preconditioner<paralution::LocalMatrix<ValueType>,
                                 paralution::LocalVector<ValueType>,
                                 ValueType > *LBPrecond = NULL;

      switch(lbp)
      {
        case paralution_None:
          break;

        case paralution_MultiElimination:
        case paralution_Jacobi:
          LBPrecond = new paralution::Jacobi<paralution::LocalMatrix<ValueType>,
                                             paralution::LocalVector<ValueType>,
                                             ValueType >;
          break;

        case paralution_MultiColoredGS:
          LBPrecond = new paralution::MultiColoredGS<paralution::LocalMatrix<ValueType>,
                                                     paralution::LocalVector<ValueType>,
                                                     ValueType >;
          break;

        case paralution_MultiColoredSGS:
          LBPrecond = new paralution::MultiColoredSGS<paralution::LocalMatrix<ValueType>,
                                                      paralution::LocalVector<ValueType>,
                                                      ValueType >;
          break;

        case paralution_ILU:
          paralution::ILU<paralution::LocalMatrix<ValueType>,
                          paralution::LocalVector<ValueType>,
                          ValueType > *p_ilu;
          p_ilu = new paralution::ILU<paralution::LocalMatrix<ValueType>,
                                      paralution::LocalVector<ValueType>,
                                      ValueType >;

          p_ilu->Set(ILUp);
          LBPrecond = p_ilu;
          break;

        case paralution_MultiColoredILU:
          paralution::MultiColoredILU<paralution::LocalMatrix<ValueType>,
                                      paralution::LocalVector<ValueType>,
                                      ValueType > *p_mcilu;
          p_mcilu = new paralution::MultiColoredILU<paralution::LocalMatrix<ValueType>,
                                                    paralution::LocalVector<ValueType>,
                                                    ValueType >;
          p_mcilu->Set(ILUp, ILUq);
          LBPrecond = p_mcilu;
          break;
      }

      p_me->Set(*LBPrecond, MEp);
      precond = p_me;
      break;

  }

  return precond;

}

template <typename ValueType> paralution::IterativeLinearSolver<paralution::LocalMatrix<ValueType>,
                                                                paralution::LocalVector<ValueType>,
                                                                ValueType >*
GetIterativeLinearSolver(const std::string solver_name, const ValueType relax) {

  _solver_type sType = paralution_CG;
  if      ( solver_name == "paralution_CG" )         sType = paralution_CG;
  else if ( solver_name == "paralution_BiCGStab" )   sType = paralution_BiCGStab;
  else if ( solver_name == "paralution_GMRES" )      sType = paralution_GMRES;
  else if ( solver_name == "paralution_FixedPoint" ) sType = paralution_FixedPoint;
  else if ( solver_name == "paralution_FGMRES" )     sType = paralution_FGMRES;

  paralution::IterativeLinearSolver<paralution::LocalMatrix<ValueType>,
                                    paralution::LocalVector<ValueType>,
                                    ValueType > *solver = NULL;

  switch(sType) {

    case paralution_CG:
      solver = new paralution::CG<paralution::LocalMatrix<ValueType>,
                                  paralution::LocalVector<ValueType>,
                                  ValueType >;
      break;

    case paralution_BiCGStab:
      solver = new paralution::BiCGStab<paralution::LocalMatrix<ValueType>,
                                        paralution::LocalVector<ValueType>,
                                        ValueType >;
      break;

    case paralution_GMRES:
      solver = new paralution::GMRES<paralution::LocalMatrix<ValueType>,
                                     paralution::LocalVector<ValueType>,
                                     ValueType >;
      break;

    case paralution_FixedPoint:
      paralution::FixedPoint<paralution::LocalMatrix<ValueType>,
                             paralution::LocalVector<ValueType>,
                             ValueType > *fp;
      fp = new paralution::FixedPoint<paralution::LocalMatrix<ValueType>,
                                      paralution::LocalVector<ValueType>,
                                      ValueType >;
      fp->SetRelaxation(relax);
      solver = fp;
      break;

    case paralution_FGMRES:
      solver = new paralution::FGMRES<paralution::LocalMatrix<ValueType>,
                                      paralution::LocalVector<ValueType>,
                                      ValueType >;
      break;

  }

  return solver;

}

