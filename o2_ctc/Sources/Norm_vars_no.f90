MODULE Norm_vars_no
  ! Mass is normalized to N2 molecular mass
  ! Length is normalized to the N2_O2 internuclear distance
  !
  USE Precision
  USE Molprops,ONLY:momin,molmass,Rm 
  IMPLICIT NONE
  REAL(dp), PARAMETER::ma=1.0D0,mb=molmass(2)/molmass(1),mr=ma*mb/(ma+mb)
  REAL(dp), PARAMETER::Ia=momin(1)/(molmass(1)*Rm(3)**2),Ib=momin(2)/(molmass(1)*Rm(3)**2)
  REAL(dp), PARAMETER::dnn=SQRT(4.0D0*momin(1)/(molmass(1)*Rm(3)**2)),doo=SQRT(4.0D0*momin(2)/(molmass(2)*Rm(3)**2))
END MODULE Norm_vars_no
