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


include_directories(base/mic)

set(MIC_SOURCES
  mic_allocate_free.cpp
  backend_mic.cpp
  mic_vector.cpp
  mic_matrix_csr.cpp
  mic_matrix_coo.cpp
  mic_matrix_dense.cpp
  mic_matrix_mcsr.cpp
  mic_matrix_bcsr.cpp
  mic_matrix_ell.cpp
  mic_matrix_dia.cpp
  mic_matrix_hyb.cpp
)

set(MIC_PUBLIC_HEADERS
  mic_allocate_free.hpp
  backend_mic.hpp
  mic_vector.hpp
  mic_matrix_csr.hpp
  mic_matrix_coo.hpp
  mic_matrix_dense.hpp
  mic_matrix_mcsr.hpp
  mic_matrix_bcsr.hpp
  mic_matrix_ell.hpp
  mic_matrix_dia.hpp
  mic_matrix_hyb.hpp
)
