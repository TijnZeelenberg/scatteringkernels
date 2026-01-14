#!/bin/bash
#
gfortran -O3 -march=native   -o ../Run/tprops_O2\
     ../Sources/Tprops_O2.f90 \
     ../Sources/Precision.f90 \
	 ../Sources/Particles.f90 \
	 ../Sources/Molprops.f90 \
	 ../Sources/Norm_vars_nn.f90 \
	 ../Sources/Norm_vars_no.f90 \
	 ../Sources/Norm_vars_oo.f90 \
	 ../Sources/Random.f90 \
	 ../Sources/Setup.f90 \
	 ../Sources/Macrovars.f90 \
	 ../Sources/CT_Collider.f90 \
	 ../Sources/Coll_N2_N2.f90 \
	 ../Sources/Coll_N2_O2.f90 \
	 ../Sources/Coll_O2_O2.f90 \
	 ../Sources/VN2N2.f90 \
	 ../Sources/VN2O2.f90 \
	 ../Sources/VO2O2.f90 \
     ../Sources/Geometry.f90 \
	 ../Sources/Scattering.f90	 
