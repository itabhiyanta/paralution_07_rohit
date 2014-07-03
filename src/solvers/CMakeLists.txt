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


set(SOLVERS_SOURCES
  krylov/cg.cpp
  krylov/cr.cpp
  krylov/bicgstab.cpp
  krylov/gmres.cpp
  krylov/fgmres.cpp
  krylov/minres.cpp
  krylov/idr.cpp
  multigrid/multigrid.cpp
  multigrid/multigrid_amg.cpp
  deflation/dpcg.cpp
  deflation/dpcg_for_dg.cpp
  direct/inversion.cpp
  direct/lu.cpp
  direct/qr.cpp
  solver.cpp
  chebyshev.cpp
  mixed_precision.cpp
  preconditioners/preconditioner.cpp
  preconditioners/preconditioner_ai.cpp
  preconditioners/preconditioner_as.cpp
  preconditioners/preconditioner_multielimination.cpp
  preconditioners/preconditioner_saddlepoint.cpp
  preconditioners/preconditioner_blockprecond.cpp
  preconditioners/preconditioner_multicolored.cpp
  preconditioners/preconditioner_multicolored_gs.cpp
  preconditioners/preconditioner_multicolored_ilu.cpp
  iter_ctrl.cpp
)

set(SOLVERS_PUBLIC_HEADERS
  krylov/cg.hpp
  krylov/cr.hpp
  krylov/bicgstab.hpp
  krylov/gmres.hpp
  krylov/fgmres.hpp
  krylov/minres.hpp
  krylov/idr.hpp
  multigrid/multigrid.hpp
  multigrid/multigrid_amg.hpp
  deflation/dpcg.hpp
  deflation/dpcg_for_dg.hpp
  direct/inversion.hpp
  direct/lu.hpp
  direct/qr.hpp
  solver.hpp
  chebyshev.hpp
  mixed_precision.hpp
  preconditioners/preconditioner.hpp
  preconditioners/preconditioner_ai.hpp
  preconditioners/preconditioner_as.hpp
  preconditioners/preconditioner_multielimination.hpp
  preconditioners/preconditioner_saddlepoint.hpp
  preconditioners/preconditioner_blockprecond.hpp
  preconditioners/preconditioner_multicolored.hpp
  preconditioners/preconditioner_multicolored_gs.hpp
  preconditioners/preconditioner_multicolored_ilu.hpp
  iter_ctrl.hpp
)
