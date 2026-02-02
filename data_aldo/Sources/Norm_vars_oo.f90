MODULE Norm_vars_oo
  USE Precision
  USE Molprops,ONLY:momin,molmass,Rm
  IMPLICIT NONE
  REAL(dp), PARAMETER::ma=1.0D0,mb=1.0D0,mr=ma*mb/(ma+mb)
  REAL(dp), PARAMETER::Ia=momin(2)/(molmass(2)*Rm(2)**2),Ib=Ia
  REAL(dp), PARAMETER::doo=SQRT(4.0D0*momin(2)/(molmass(2)*Rm(2)**2))
END MODULE Norm_vars_oo
