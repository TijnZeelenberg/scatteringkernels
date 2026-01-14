MODULE Norm_vars_nn
  USE Precision
  USE Molprops,ONLY:momin,molmass,Rm
  IMPLICIT NONE
  REAL(dp), PARAMETER::ma=1.0D0,mb=1.0D0,mr=ma*mb/(ma+mb)
  REAL(dp), PARAMETER::Ia=momin(1)/(molmass(1)*Rm(1)**2),Ib=Ia
  REAL(dp), PARAMETER::dnn=SQRT(4.0D0*momin(1)/(molmass(1)*Rm(1)**2))
END MODULE Norm_vars_nn
