# *************************************************************************
#
#    PARALUTION   www.paralution.com
#
#    Copyright (C) 2012-2014 Dimitar Lukarski
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *************************************************************************



# PARALUTION version 0.7.0 


set(PLUG-INS_SOURCES)
set(PLUG-INS_PUBLIC_HEADERS
  paralution_dealii.hpp
  OpenFOAM/matrices/lduMatrix/solvers/paralution/paralution_openfoam.H
  OpenFOAM/matrices/lduMatrix/solvers/paralution_PCG/paralution_PCG.H
  OpenFOAM/matrices/lduMatrix/solvers/paralution_PCG/paralution_PCG.C
  OpenFOAM/matrices/lduMatrix/solvers/paralution_PBiCG/paralution_PBiCG.H
  OpenFOAM/matrices/lduMatrix/solvers/paralution_PBiCG/paralution_PBiCG.C
  OpenFOAM/matrices/lduMatrix/solvers/paralution_GAMG/paralution_GAMG.H
  OpenFOAM/matrices/lduMatrix/solvers/paralution_GAMG/paralution_GAMG.C
  OpenFOAM/matrices/lduMatrix/solvers/paralution_AMG/paralution_AMG.H
  OpenFOAM/matrices/lduMatrix/solvers/paralution_AMG/paralution_AMG.C
  OpenFOAM/matrices/lduMatrix/solvers/paralution_PGMRES/paralution_PGMRES.H
  OpenFOAM/matrices/lduMatrix/solvers/paralution_PGMRES/paralution_PGMRES.C
  OpenFOAM/matrices/lduMatrix/solvers/paralution_PFGMRES/paralution_PFGMRES.H
  OpenFOAM/matrices/lduMatrix/solvers/paralution_PFGMRES/paralution_PFGMRES.C
  paralution_fortran.cpp
)
