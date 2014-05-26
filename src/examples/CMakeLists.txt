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


include_directories(${INC_BUILD_DIR})

set(EXAMPLES
  amg
  benchmark
  bicgstab
  cg
  cg-amg
  cmk
  dpcg
  dpcg_precon
  idr
  gmres
  sp-precond
  block-precond
  as-precond
  fixed-point
  mixed-precision
  me-preconditioner
  power-method
  simple-spmv
  async
  assembly
  assembly2
  direct
)

foreach(example ${EXAMPLES})
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} ${PARALUTION_LIBRARY})
  install(PROGRAMS ${example} DESTINATION bin)
endforeach()
