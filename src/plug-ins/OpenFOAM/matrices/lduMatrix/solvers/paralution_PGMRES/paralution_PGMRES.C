// *************************************************************************
//    Modification of file paralution_PGMRES.C is made by the PARALUTION team,
//    this file is a part of the PARALUTION library
//                                          
//    PARALUTION   www.paralution.com
//
//    Copyright (C) 2012-2013 Dimitar Lukarski
//
//    GPLv3 
//
// *************************************************************************
//
// The origin OpenFOAM license:
//
/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "paralution.hpp"
#include "../paralution/paralution_openfoam.H"
#include "paralution_PGMRES.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(paralution_PGMRES, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<paralution_PGMRES>
        addparalution_PGMRESSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<paralution_PGMRES>
        addparalution_PGMRESAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::paralution_PGMRES::paralution_PGMRES
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::lduMatrix::solverPerformance Foam::paralution_PGMRES::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{

    word precond_name = lduMatrix::preconditioner::getName(controlDict_);
    double div   = controlDict_.lookupOrDefault<double>("div", 1e+08);
    int basis    = controlDict_.lookupOrDefault<int>("BasisSize", 30);
    bool accel   = controlDict_.lookupOrDefault<bool>("useAccelerator", true);
    word mformat = controlDict_.lookupOrDefault<word>("MatrixFormat", "CSR");
    word pformat = controlDict_.lookupOrDefault<word>("PrecondFormat", "CSR");
    int ILUp     = controlDict_.lookupOrDefault<int>("ILUp", 0);
    int ILUq     = controlDict_.lookupOrDefault<int>("ILUq", 1);
    int MEp      = controlDict_.lookupOrDefault<int>("MEp", 1);
    word LBPre   = controlDict_.lookupOrDefault<word>("LastBlockPrecond", "paralution_Jacobi");
    
    lduMatrix::solverPerformance solverPerf(typeName + '(' + precond_name + ')', fieldName_);

    register label nCells = psi.size();

    scalarField pA(nCells);
    scalarField wA(nCells);

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    scalarField rA(source - wA);

    // --- Calculate normalisation factor
    scalar normFactor = this->normFactor(psi, source, wA, pA);

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() = gSumMag(rA)/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // TODO check why we cannot skip 1 iteration when initial residual < relTol_ or why initial residual actually
    // does not drop below relTol_

    if (!solverPerf.checkConvergence(tolerance_, relTol_)) {

      paralution::_matrix_format mf = paralution::CSR;
      if      (mformat == "CSR")   mf = paralution::CSR;
      else if (mformat == "DIA")   mf = paralution::DIA;
      else if (mformat == "HYB")   mf = paralution::HYB;
      else if (mformat == "ELL")   mf = paralution::ELL;
      else if (mformat == "MCSR")  mf = paralution::MCSR;
      else if (mformat == "BCSR")  mf = paralution::BCSR;
      else if (mformat == "COO")   mf = paralution::COO;
      else if (mformat == "DENSE") mf = paralution::DENSE;

      paralution::init_paralution();

      paralution::LocalVector<double> x;
      paralution::LocalVector<double> rhs;
      paralution::LocalMatrix<double> mat;

      paralution::GMRES<paralution::LocalMatrix<double>,
                        paralution::LocalVector<double>,
                        double> ls;

      import_openfoam_matrix(matrix(), &mat);
      import_openfoam_vector(source, &rhs);
      import_openfoam_vector(psi, &x);

      ls.Clear();

      if (accel) {
        mat.MoveToAccelerator();
        rhs.MoveToAccelerator();
        x.MoveToAccelerator();
      }

      paralution::Preconditioner<paralution::LocalMatrix<double>,
                                 paralution::LocalVector<double>,
                                 double > *precond = NULL;

      precond = GetPreconditioner<double>(precond_name, LBPre, pformat, ILUp, ILUq, MEp);
      if (precond != NULL) ls.SetPreconditioner(*precond);

      ls.SetOperator(mat);
      ls.SetBasisSize(basis);
      ls.Verbose(0);

      ls.Init(tolerance_*normFactor, // abs
              relTol_,    // rel
              div,        // div
              maxIter_);  // max iter

      ls.Build();

      switch(mf) {
        case paralution::DENSE:
          mat.ConvertToDENSE();
          break;
        case paralution::CSR:
          mat.ConvertToCSR();
          break;
        case paralution::MCSR:
          mat.ConvertToMCSR();
          break;
        case paralution::BCSR:
          mat.ConvertToBCSR();
          break;
        case paralution::COO:
          mat.ConvertToCOO();
          break;
        case paralution::DIA:
          mat.ConvertToDIA();
          break;
        case paralution::ELL:
          mat.ConvertToELL();
          break;
        case paralution::HYB:
          mat.ConvertToHYB();
          break;
      }

//      mat.info();

      ls.Solve(rhs, &x);

      export_openfoam_vector(x, &psi);

      solverPerf.finalResidual()   = ls.GetCurrentResidual() / normFactor; // divide by normFactor, see lduMatrixSolver.C
      solverPerf.nIterations()     = ls.GetIterationCount();
      solverPerf.checkConvergence(tolerance_, relTol_);

      ls.Clear();
      if (precond != NULL) {
        precond->Clear();
        delete precond;
      }

      paralution::stop_paralution();

    }

    return solverPerf;

}


// ************************************************************************* //
