#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2009 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart, 
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2009-2010 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2014      NVIDIA Corporation.  All rights reserved.
# $COPYRIGHT$
# 
# Additional copyrights may follow
# 
# $HEADER$
#

dist_ompidata_DATA = help-mpi-btl-nc.txt

libmca_btl_nc_la_sources = \
    btl_nc.c \
    btl_nc.h \
    btl_nc_component.c 

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_btl_nc_DSO
component_noinst =
component_install = mca_btl_nc.la
else
component_noinst = libmca_btl_nc.la
component_install =
endif

# See ompi/mca/common/sm/Makefile.am for an explanation of
# libmca_common_sm.la.

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_btl_nc_la_SOURCES = $(libmca_btl_nc_la_sources)
mca_btl_nc_la_LDFLAGS = -module -avoid-version
mca_btl_nc_la_LIBADD = \
    $(top_ompi_builddir)/ompi/mca/common/sm/libmca_common_sm.la
mca_btl_nc_la_CPPFLAGS = $(btl_nc_CPPFLAGS)

noinst_LTLIBRARIES = $(component_noinst)
libmca_btl_nc_la_SOURCES = $(libmca_btl_nc_la_sources)
libmca_btl_nc_la_LDFLAGS = -module -avoid-version
libmca_btl_nc_la_CPPFLAGS = $(btl_nc_CPPFLAGS)
