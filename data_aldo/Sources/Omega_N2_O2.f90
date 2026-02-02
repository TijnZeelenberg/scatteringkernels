PROGRAM Omega_N2_O2
  USE Precision
  USE Molprops
  USE Random
  USE Setup
  USE CT_Collider
  IMPLICIT NONE
  INTEGER::test,Ntest
  REAL(dp),DIMENSION(3)::v1,v2,omega1,omega2
  REAL(dp),DIMENSION(3)::vr,vrstar,umed
  REAL(dp)::g,gstar,coschi
  REAL(dp)::ttrasl,trot
  REAL(dp)::omega11,omega22,R12,dens0,T0
  REAL(dp)::diffusion_coefficient,viscosity
  !
  OPEN(unit=1,file='Omega.inp')
  OPEN(unit=2,file='Omega_N2_O2.dat')
  !OPEN(unit=11,file='Scattering.dat')
  !
  READ(1,*)T0
  READ(1,*)P0
  READ(1,*)Ntest
  READ(1,*)seme
  !
  dens0=P0/(kb*T0)
  !
  ttrasl=0.0D0;trot=0.0D0;umed=0.0D0
  omega22=0.0D0;omega11=0.0D0;R12=kb/mr
  !
  DO test=1,Ntest
     CALL Maxwell(v1(1),v1(2),v1(3))
     v1=v1*SQRT(Rgas(1)*T0)
     CALL Maxwell(v2(1),v2(2),v2(3))
     v2=v2*SQRT(Rgas(2)*T0)
     CALL Genk(omega1(1),omega1(2),omega1(3))
     erot=-kb*T0*LOG(1.0D0-rf(seme))
     omega1=omega1*SQRT(2.0D0*erot/momin(1))
     CALL Genk(omega2(1),omega2(2),omega2(3))
     erot=-kb*T0*LOG(1.0D0-rf(seme))
     omega2=omega2*SQRT(2.0D0*erot/momin(2))
     umed=umed+v1+v2
     ttrasl=ttrasl+0.5D0*molmass(1)*Dot_PRODUCT(v1,v1)
     ttrasl=ttrasl+0.5D0*molmass(2)*Dot_PRODUCT(v2,v2)
     trot=trot+0.5D0*momin(1)*Dot_PRODUCT(omega1,omega1)
     trot=trot+0.5D0*momin(2)*Dot_PRODUCT(omega2,omega2)
     vr=v2-v1
     CALL Collision(v1,omega1,v2,omega2,1,2)
     vrstar=v2-v1
     g=SQRT(Dot_PRODUCT(vr,vr))
     gstar=SQRT(Dot_PRODUCT(vrstar,vrstar))
     coschi=Dot_PRODUCT(vr,vrstar)/(g*gstar)
     omega11=omega11+(1.0D0-coschi)*g**3
     omega22=omega22+(1.0D0-coschi**2)*g**5
  END DO
  umed=umed/(2.0D0*Ntest)
  ttrasl=ttrasl/(2.0D0*Ntest)-0.5D0*molmass(1)*Dot_PRODUCT(umed,umed)
  ttrasl=2.0D0*ttrasl/(3.0D0*kb)
  trot=trot/(2.0D0*Ntest)
  trot=trot/kb
  Omega11=Omega11/Ntest
  Omega11=Omega11*pigreco/(8.0D0*R12*T0)*diamol(1,2)**2/2.0D0
  Omega22=Omega22/Ntest
  Omega22=Omega22*pi/(16.0D0*(R12*T0)**2)*diamol(1,2)**2/2.0D0
  Diffusion_Coefficient=3.0*kb*T0/(16.0D0*mr*dens0*Omega11)
  Viscosity=5.0*kb*T0/(8.0D0*omega22)
  WRITE(2,100)T0,Ttrasl,Trot,Viscosity,Omega22,Diffusion_Coefficient,Omega11
  100 FORMAT(10(E13.5,2X))
END PROGRAM Omega_N2_O2
