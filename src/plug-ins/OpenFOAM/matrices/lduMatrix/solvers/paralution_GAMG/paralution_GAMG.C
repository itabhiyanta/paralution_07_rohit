// *************************************************************************
//
//    Modification of file paralution_GAMG.C is made by the PARALUTION team,
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
// The origin OpenFOAM license (file GAMG.C):
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

#include "paralution_GAMG.H"


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(paralution_GAMG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<paralution_GAMG>
        addparalution_GAMGMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<paralution_GAMG>
        addGAMGAsymSolverMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::paralution_GAMG::paralution_GAMG
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
    ),

    // Default values for all controls
    // which may be overridden by those in controlDict
    cacheAgglomeration_(false),
    nPreSweeps_(0),
    nPostSweeps_(2),
    nFinestSweeps_(2),
    scaleCorrection_(matrix.symmetric()),
    directSolveCoarsest_(false),
    agglomeration_(GAMGAgglomeration::New(matrix_, controlDict_)),

    matrixLevels_(agglomeration_.size()),
    interfaceLevels_(agglomeration_.size()),
    interfaceLevelsBouCoeffs_(agglomeration_.size()),
    interfaceLevelsIntCoeffs_(agglomeration_.size())
{
//    double tick, tack;

//    tick = paralution_time();

    readControls();

    forAll(agglomeration_, fineLevelIndex)
    {
        agglomerateMatrix(fineLevelIndex);
    }

    if (matrixLevels_.size())
    {
        const label coarsestLevel = matrixLevels_.size() - 1;

        if (directSolveCoarsest_)
        {
            coarsestLUMatrixPtr_.set
            (
                new LUscalarMatrix
                (
                    matrixLevels_[coarsestLevel],
                    interfaceLevelsBouCoeffs_[coarsestLevel],
                    interfaceLevels_[coarsestLevel]
                )
            );
        }
    }
    else
    {
        FatalErrorIn
        (
            "paralution_GAMG::paralution_GAMG"
            "("
            "const word& fieldName,"
            "const lduMatrix& matrix,"
            "const FieldField<Field, scalar>& interfaceBouCoeffs,"
            "const FieldField<Field, scalar>& interfaceIntCoeffs,"
            "const lduInterfaceFieldPtrsList& interfaces,"
            "const dictionary& solverControls"
            ")"
        )   << "No coarse levels created, either matrix too small for GAMG"
               " or nCellsInCoarsestLevel too large.\n"
               "    Either choose another solver of reduce "
               "nCellsInCoarsestLevel."
            << exit(FatalError);
    }

//    tack = paralution_time();
//    std::cout << "Creating OpenFOAM solver object:" << (tack-tick)/1000000 << " sec" << std::endl;
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::paralution_GAMG::~paralution_GAMG()
{
    // Clear the the lists of pointers to the interfaces
    forAll(interfaceLevels_, leveli)
    {
        lduInterfaceFieldPtrsList& curLevel = interfaceLevels_[leveli];

        forAll(curLevel, i)
        {
            if (curLevel.set(i))
            {
                delete curLevel(i);
            }
        }
    }

    if (!cacheAgglomeration_)
    {
        delete &agglomeration_;
    }
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::paralution_GAMG::readControls()
{
    lduMatrix::solver::readControls();

    // we could also consider supplying defaults here too
    controlDict_.readIfPresent("cacheAgglomeration", cacheAgglomeration_);
    controlDict_.readIfPresent("nPreSweeps", nPreSweeps_);
    controlDict_.readIfPresent("nPostSweeps", nPostSweeps_);
    controlDict_.readIfPresent("nFinestSweeps", nFinestSweeps_);
    controlDict_.readIfPresent("scaleCorrection", scaleCorrection_);
    controlDict_.readIfPresent("directSolveCoarsest", directSolveCoarsest_);
}


const Foam::lduMatrix& Foam::paralution_GAMG::matrixLevel(const label i) const
{
    if (i == 0)
    {
        return matrix_;
    }
    else
    {
        return matrixLevels_[i - 1];
    }
}


const Foam::lduInterfaceFieldPtrsList& Foam::paralution_GAMG::interfaceLevel
(
    const label i
) const
{
    if (i == 0)
    {
        return interfaces_;
    }
    else
    {
        return interfaceLevels_[i - 1];
    }
}


const Foam::FieldField<Foam::Field, Foam::scalar>&
Foam::paralution_GAMG::interfaceBouCoeffsLevel
(
    const label i
) const
{
    if (i == 0)
    {
        return interfaceBouCoeffs_;
    }
    else
    {
        return interfaceLevelsBouCoeffs_[i - 1];
    }
}


const Foam::FieldField<Foam::Field, Foam::scalar>&
Foam::paralution_GAMG::interfaceIntCoeffsLevel
(
    const label i
) const
{
    if (i == 0)
    {
        return interfaceIntCoeffs_;
    }
    else
    {
        return interfaceLevelsIntCoeffs_[i - 1];
    }
}


void Foam::paralution_GAMG::agglomerateMatrix(const label fineLevelIndex)
{

    // Get fine matrix
    const lduMatrix& fineMatrix = matrixLevel(fineLevelIndex);

    // Set the coarse level matrix
    matrixLevels_.set
    (
        fineLevelIndex,
        new lduMatrix(agglomeration_.meshLevel(fineLevelIndex + 1))
    );
    lduMatrix& coarseMatrix = matrixLevels_[fineLevelIndex];

    // Get face restriction map for current level
    const labelList& faceRestrictAddr =
        agglomeration_.faceRestrictAddressing(fineLevelIndex);

    // Coarse matrix diagonal initialised by restricting the finer mesh diagonal
    scalarField& coarseDiag = coarseMatrix.diag();
    agglomeration_.restrictField(coarseDiag, fineMatrix.diag(), fineLevelIndex);

    // Get reference to fine-level interfaces
    const lduInterfaceFieldPtrsList& fineInterfaces =
        interfaceLevel(fineLevelIndex);

    // Get reference to fine-level boundary coefficients
    const FieldField<Field, scalar>& fineInterfaceBouCoeffs =
        interfaceBouCoeffsLevel(fineLevelIndex);

    // Get reference to fine-level internal coefficients
    const FieldField<Field, scalar>& fineInterfaceIntCoeffs =
        interfaceIntCoeffsLevel(fineLevelIndex);


    // Create coarse-level interfaces
    interfaceLevels_.set
    (
        fineLevelIndex,
        new lduInterfaceFieldPtrsList(fineInterfaces.size())
    );

    lduInterfaceFieldPtrsList& coarseInterfaces =
        interfaceLevels_[fineLevelIndex];

    // Set coarse-level boundary coefficients
    interfaceLevelsBouCoeffs_.set
    (
        fineLevelIndex,
        new FieldField<Field, scalar>(fineInterfaces.size())
    );
    FieldField<Field, scalar>& coarseInterfaceBouCoeffs =
        interfaceLevelsBouCoeffs_[fineLevelIndex];

    // Set coarse-level internal coefficients
    interfaceLevelsIntCoeffs_.set
    (
        fineLevelIndex,
        new FieldField<Field, scalar>(fineInterfaces.size())
    );
    FieldField<Field, scalar>& coarseInterfaceIntCoeffs =
        interfaceLevelsIntCoeffs_[fineLevelIndex];

    // Add the coarse level
    forAll(fineInterfaces, inti)
    {
        if (fineInterfaces.set(inti))
        {
            const GAMGInterface& coarseInterface =
                refCast<const GAMGInterface>
                (
                    agglomeration_.interfaceLevel(fineLevelIndex + 1)[inti]
                );

            coarseInterfaces.set
            (
                inti,
                GAMGInterfaceField::New
                (
                    coarseInterface,
                    fineInterfaces[inti]
                ).ptr()
            );

            coarseInterfaceBouCoeffs.set
            (
                inti,
                coarseInterface.agglomerateCoeffs(fineInterfaceBouCoeffs[inti])
            );

            coarseInterfaceIntCoeffs.set
            (
                inti,
                coarseInterface.agglomerateCoeffs(fineInterfaceIntCoeffs[inti])
            );
        }
    }


    // Check if matrix is assymetric and if so agglomerate both upper and lower
    // coefficients ...
    if (fineMatrix.hasLower())
    {
        // Get off-diagonal matrix coefficients
        const scalarField& fineUpper = fineMatrix.upper();
        const scalarField& fineLower = fineMatrix.lower();

        // Coarse matrix upper coefficients
        scalarField& coarseUpper = coarseMatrix.upper();
        scalarField& coarseLower = coarseMatrix.lower();

        const labelList& restrictAddr =
            agglomeration_.restrictAddressing(fineLevelIndex);

        const labelUList& l = fineMatrix.lduAddr().lowerAddr();
        const labelUList& cl = coarseMatrix.lduAddr().lowerAddr();
        const labelUList& cu = coarseMatrix.lduAddr().upperAddr();

        forAll(faceRestrictAddr, fineFacei)
        {
            label cFace = faceRestrictAddr[fineFacei];

            if (cFace >= 0)
            {
                // Check the orientation of the fine-face relative to the
                // coarse face it is being agglomerated into
                if (cl[cFace] == restrictAddr[l[fineFacei]])
                {
                    coarseUpper[cFace] += fineUpper[fineFacei];
                    coarseLower[cFace] += fineLower[fineFacei];
                }
                else if (cu[cFace] == restrictAddr[l[fineFacei]])
                {
                    coarseUpper[cFace] += fineLower[fineFacei];
                    coarseLower[cFace] += fineUpper[fineFacei];
                }
                else
                {
                    FatalErrorIn
                    (
                        "paralution_GAMG::agglomerateMatrix(const label)"
                    )   << "Inconsistent addressing between "
                           "fine and coarse grids"
                        << exit(FatalError);
                }
            }
            else
            {
                // Add the fine face coefficients into the diagonal.
                coarseDiag[-1 - cFace] +=
                    fineUpper[fineFacei] + fineLower[fineFacei];
            }
        }
    }
    else // ... Otherwise it is symmetric so agglomerate just the upper
    {
        // Get off-diagonal matrix coefficients
        const scalarField& fineUpper = fineMatrix.upper();

        // Coarse matrix upper coefficients
        scalarField& coarseUpper = coarseMatrix.upper();

        forAll(faceRestrictAddr, fineFacei)
        {
            label cFace = faceRestrictAddr[fineFacei];

            if (cFace >= 0)
            {
                coarseUpper[cFace] += fineUpper[fineFacei];
            }
            else
            {
                // Add the fine face coefficient into the diagonal.
                coarseDiag[-1 - cFace] += 2*fineUpper[fineFacei];
            }
        }
    }
}


Foam::lduMatrix::solverPerformance Foam::paralution_GAMG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{

  // Time measurement
//    double fullsolver_begin, fullsolver_end;

//    fullsolver_begin = paralution_time();

    // Get all the parameters from input file 'fvSolution'
    word precond_name = lduMatrix::preconditioner::getName(controlDict_);
    double div   = controlDict_.lookupOrDefault<double>("div", 1e+08);
    bool accel   = controlDict_.lookupOrDefault<bool>("useAccelerator", true);
    word mformat = controlDict_.lookupOrDefault<word>("MatrixFormat", "CSR");
    word pformat = controlDict_.lookupOrDefault<word>("PrecondFormat", "CSR");
    word sformat = controlDict_.lookupOrDefault<word>("SmootherFormat", "CSR");
    word solver_name = controlDict_.lookupOrDefault<word>("CoarseGridSolver", "CG");
    word smoother_name = controlDict_.lookupOrDefault<word>("smoother", "MultiColoredILU");
    int iterPreSmooth = controlDict_.lookupOrDefault<int>("nPreSweeps", 1);
    int iterPostSmooth = controlDict_.lookupOrDefault<int>("nPostSweeps", 2);
    int ILUp = controlDict_.lookupOrDefault<int>("ILUp", 0);
    int ILUq = controlDict_.lookupOrDefault<int>("ILUq", 1);
    double relax = controlDict_.lookupOrDefault<double>("Relaxation", 1.0);
    bool scaling = controlDict_.lookupOrDefault<bool>("scaleCorrection", true);

    lduMatrix::solverPerformance solverPerf(typeName + '(' + precond_name + ')', fieldName_);

    // OpenFOAM residual normalization computation
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

    label levels = this->agglomeration_.size();

    // Initialize paralution platform
    paralution::init_paralution();

    // Determine Operator format for each level from input parameter
    paralution::_matrix_format mf = paralution::CSR;
    if      ( mformat == "CSR" )   mf = paralution::CSR;
    else if ( mformat == "HYB" )   mf = paralution::HYB;
    else if ( mformat == "MCSR" )  mf = paralution::MCSR;
    else if ( mformat == "BCSR" )  mf = paralution::BCSR;
    else if ( mformat == "COO" )   mf = paralution::COO;
    else if ( mformat == "DIA" )   mf = paralution::DIA;
    else if ( mformat == "ELL" )   mf = paralution::ELL;
    else if ( mformat == "DENSE" ) mf = paralution::DENSE;

    paralution::LocalVector<double> x;
    paralution::LocalVector<double> rhs;
    paralution::LocalMatrix<double> *mat_fine = NULL;
    paralution::LocalMatrix<double> **mat = NULL;
    paralution::LocalVector<int> **resvec = NULL;
    paralution::LocalMatrix<double> **resOp = NULL;
    paralution::LocalMatrix<double> **proOp = NULL;

    mat_fine = new paralution::LocalMatrix<double>;
    mat = new paralution::LocalMatrix<double>*[levels];
    resvec = new paralution::LocalVector<int>*[levels];
    resOp = new paralution::LocalMatrix<double>*[levels];
    proOp = new paralution::LocalMatrix<double>*[levels];

    for (int i=0; i<levels; ++i) {
      mat[i] = new paralution::LocalMatrix<double>;
      resvec[i] = new paralution::LocalVector<int>;
      resOp[i] = new paralution::LocalMatrix<double>;
      proOp[i] = new paralution::LocalMatrix<double>;
    }

    // MultiGrid solver object
    paralution::MultiGrid<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > multigrid;

    // Coarse Grid Solver and its Preconditioner
    paralution::IterativeLinearSolver<paralution::LocalMatrix<double>,
                                      paralution::LocalVector<double>,
                                      double > *cgs = NULL;
    cgs = GetIterativeLinearSolver<double>(solver_name, relax);
    cgs->Verbose(0);
    paralution::Preconditioner<paralution::LocalMatrix<double>,
                               paralution::LocalVector<double>,
                               double > *cgp = NULL;
    cgp = GetPreconditioner<double>(precond_name, "", pformat, ILUp, ILUq, 1);
    if (cgp != NULL) cgs->SetPreconditioner(*cgp);

    // Smoother via preconditioned FixedPoint iteration
    paralution::IterativeLinearSolver<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > **fp = NULL;
    fp = new paralution::IterativeLinearSolver<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >*[levels];
    paralution::Preconditioner<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > **smoother = NULL;
    smoother = new paralution::Preconditioner<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >*[levels];

    for (int i=0; i<levels; ++i) {
      fp[i] = GetIterativeLinearSolver<double>("paralution_FixedPoint", relax);
      smoother[i] = GetPreconditioner<double>(smoother_name, "", sformat, ILUp, ILUq, 1);
      fp[i]->SetPreconditioner(*smoother[i]);
      fp[i]->Verbose(0);
    }

    // Import OpenFOAM data structures
    import_openfoam_matrix(matrix(), mat_fine);
    import_openfoam_vector(psi, &x);
    import_openfoam_vector(source, &rhs);

    // Move data structures to device
    if (accel) {
      x.MoveToAccelerator();
      rhs.MoveToAccelerator();
      mat_fine->MoveToAccelerator();
    }

    // Initialize MultiGrid with levels, operator type (true = operator, false = mapping vector)
    // and fine level operator
    multigrid.InitLevels(levels+1);
    multigrid.InitOperator(true);
    multigrid.SetScaling(scaling);

    // Setup all other levels and move them to device
    for (int i=0; i<levels; ++i) {
      import_openfoam_matrix(matrixLevels_[i], mat[i]);

      if (accel)
        mat[i]->MoveToAccelerator();
    }

    multigrid.SetOperator(*mat_fine);
    multigrid.SetOperatorHierarchy(mat);

    for (int i=0; i<levels; ++i) {

      // Get OpenFOAM level transfer mapping
      labelField restrictionField = this->agglomeration_.restrictAddressing(i);
      import_openfoam_restriction(restrictionField, resvec[i]);

      // Convert mapping to Operator type
      if (i > 0)
        resOp[i]->CreateFromMap(*resvec[i], mat[i-1]->get_nrow(), mat[i]->get_nrow());
      else
        resOp[i]->CreateFromMap(*resvec[i], mat_fine->get_nrow(), mat[i]->get_nrow());

      if (accel)
        resOp[i]->MoveToAccelerator();

      proOp[i]->CloneFrom(*resOp[i]);

      // Transpose restriction operator to obtain prolongation operator
      proOp[i]->Transpose();

      // OpenFOAM transfer mapping is no longer needed
      resvec[i]->Clear();
      delete resvec[i];
    }
    delete[] resvec;

    multigrid.SetRestrictOperator(resOp);
    multigrid.SetProlongOperator(proOp);

    // Set smoother for each level
    multigrid.SetSmoother(fp);

    // Setup Coarse Grid solver and number of pre/post smoothing steps as given from input file
    multigrid.SetSolver(*cgs);
    multigrid.SetSmootherPreIter(iterPreSmooth);
    multigrid.SetSmootherPostIter(iterPostSmooth);
    multigrid.Verbose(0);

    // Switch to L1 norm to be consistent with OpenFOAM solvers
    multigrid.SetResidualNorm(1);

    // Initialize solver specific tolerances, etc.
    multigrid.Init(tolerance_*normFactor, // abs
                   relTol_,    // rel
                   div,        // div
                   maxIter_);  // max iter

    multigrid.Build();

    // Convert operator matrices to specified format (see input file)
    switch( mf ) {
      case paralution::CSR:
        mat_fine->ConvertToCSR();
        for (int i=0; i<levels; ++i)
          mat[i]->ConvertToCSR();
        break;
      case paralution::HYB:
        mat_fine->ConvertToHYB();
        for (int i=0; i<levels; ++i)
          mat[i]->ConvertToHYB();
        break;
      case paralution::DIA:
        mat_fine->ConvertToDIA();
        for (int i=0; i<levels; ++i)
          mat[i]->ConvertToDIA();
        break;
      case paralution::ELL:
        mat_fine->ConvertToELL();
        for (int i=0; i<levels; ++i)
          mat[i]->ConvertToELL();
        break;
      case paralution::MCSR:
        mat_fine->ConvertToMCSR();
        for (int i=0; i<levels; ++i)
          mat[i]->ConvertToMCSR();
        break;
      case paralution::BCSR:
        mat_fine->ConvertToBCSR();
        for (int i=0; i<levels; ++i)
          mat[i]->ConvertToBCSR();
        break;
      case paralution::COO:
        mat_fine->ConvertToCOO();
        for (int i=0; i<levels; ++i)
          mat[i]->ConvertToCOO();
        break;
      case paralution::DENSE:
        mat_fine->ConvertToDENSE();
        for (int i=0; i<levels; ++i)
          mat[i]->ConvertToDENSE();
        break;
    }

//    parasolve_begin = paralution_time();

    // Solve linear system
    multigrid.Solve(rhs, &x);

//    parasolve_end = paralution_time();
//    std::cout << "multigrid.Solve() execution:" << (parasolve_end-parasolve_begin)/1000000 << " sec" << std::endl;

    // Export solution vector to OpenFOAM data structure
    export_openfoam_vector(x, &psi);

    solverPerf.finalResidual()   = multigrid.GetCurrentResidual() / normFactor; // divide by normFactor, see lduMatrixSolver.C
    solverPerf.nIterations()     = multigrid.GetIterationCount();

    // Clear MultiGrid object
    multigrid.Clear();

    // Free all structures
    for (int i=0; i<levels; ++i) {
      delete fp[i];
      delete smoother[i];
      delete mat[i];
      delete resOp[i];
      delete proOp[i];
    }

    if (cgp != NULL)
      delete cgp;

    delete[] fp;
    delete[] smoother;
    delete[] mat;
    delete[] resOp;
    delete[] proOp;
    delete cgs;
    delete mat_fine;

    // Stop paralution platform
    paralution::stop_paralution();

//    fullsolver_end = paralution_time();
//    std::cout << "OpenFOAM.Solve() execution:" << (fullsolver_end-fullsolver_begin)/1000000 << " sec";

    solverPerf.checkConvergence(tolerance_, relTol_);

    return solverPerf;

}

// ************************************************************************* //
