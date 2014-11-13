# -*- shell-script -*-
#
# Copyright (c) 2009      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2009-2010 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2010-2012 IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# --------------------------------------------------------
# MCA_btl_nc_CONFIG([action-if-can-compile],
#                   [action-if-cant-compile])
# ------------------------------------------------
AC_DEFUN([MCA_ompi_btl_nc_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/btl/nc/Makefile])

    [$1]
    AC_SUBST([btl_nc_CPPFLAGS])
    OPAL_VAR_SCOPE_POP
])dnl
