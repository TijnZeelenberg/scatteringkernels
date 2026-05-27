MODULE Molprops
USE Precision
  IMPLICIT NONE
  REAL(dp), PARAMETER::pigreco=4.0D0*ATAN(1.0D0)
  REAL(dp), PARAMETER::kb=1.3806503d-23,amu=1.660539d-27
  REAL(dp), DIMENSION(2),PARAMETER:: molmass=(/2.0d0*14.0064*amu,2.0d0*15.9994*amu/)
  REAL(dp), DIMENSION(2),PARAMETER:: Rgas=(/kb/molmass(1),kb/molmass(2)/)
  REAL(dp), PARAMETER:: mr=molmass(1)*molmass(2)/(molmass(1)+molmass(2))
  REAL(dp), DIMENSION(2),PARAMETER:: momin=(/1.4009d-46,1.9471d-46/)
  REAL(dp), DIMENSION(2),PARAMETER:: d_intnuc=(/1.0977e-10,1.2075e-10/)
  REAL(dp), PARAMETER::bmax=3.0D0
  !
  ! Profondita delle buche del potenziale isotropo V000 [J]
  ! Nell'ordine: N2-N2, O2-O2, N2-O2
  ! 
  REAL(dp), DIMENSION(3),PARAMETER:: eps_pot=1.602177D-19*(/9.85D-03,11.7D-03,10.9D-03/)
  !
  ! Lunghezze di riferimento [m], per i range dei potenziali
  ! N2-N2, O2-O2, N2-O2
  !
  REAL(dp), DIMENSION(3),PARAMETER::Rm=(/4.11D-10,3.90D-10,4.02D-10/)
  !
  ! Diametri molecolari e sezioni d'urto fittizi per la maggiorazione del numero di collisioni
  !
  REAL(dp), DIMENSION(2,2), PARAMETER::diamol=bmax*reshape((/Rm(1),Rm(3),Rm(3),Rm(2)/), shape(diamol))
  REAL(dp), DIMENSION(2,2), PARAMETER::xsect=reshape((/pigreco*diamol(1,1)**2,pigreco*diamol(1,2)**2 &
       ,pigreco*diamol(1,2)**2 ,pigreco*diamol(2,2)**2 /), shape(xsect))
  !
END MODULE Molprops
