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


set(HOST_SOURCES
  host_matrix_csr.cpp
  host_matrix_mcsr.cpp
  host_matrix_bcsr.cpp
  host_matrix_coo.cpp
  host_matrix_dia.cpp
  host_matrix_ell.cpp
  host_matrix_hyb.cpp
  host_matrix_dense.cpp
  host_vector.cpp
  host_conversion.cpp  
  host_affinity.cpp
)

set(HOST_PUBLIC_HEADERS
  host_matrix_csr.hpp
  host_matrix_mcsr.hpp
  host_matrix_bcsr.hpp
  host_matrix_coo.hpp
  host_matrix_dia.hpp
  host_matrix_ell.hpp
  host_matrix_hyb.hpp
  host_matrix_dense.hpp
  host_vector.hpp
  host_conversion.hpp
  host_affinity.hpp
)
