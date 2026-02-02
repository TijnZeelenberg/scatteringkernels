PROGRAM Omega_N2
  USE Precision
  USE Molprops
  USE Random
  USE Setup
  USE CT_Collider
  IMPLICIT NONE
  INTEGER::test,Ntest
  REAL(dp),DIMENSION(3)::v1,v2,omega1,omega2
  REAL(dp),DIMENSION(3)::vr,vrstar,umed
  REAL(dp)::g,gstar,coschi,sinchi,sinchi2
  REAL(dp)::erot1,erot1_star,erot2,erot2_star
  REAL(dp)::ttrasl,trot,T0,dens0
  REAL(dp)::omega22,omega11,viscosity,diffusion_coefficient
  REAL(dp)::omega22_tax,Viscosity_tax
  REAL(dp)::fbulk,omega_bulk_tax,bulk_viscosity_tax
  !
  ! Thermal conductivity
  !
  REAL(dp)::fx,ifx,x_tax
  REAL(dp)::fy,ify,y_tax
  REAL(dp)::fyy,ifyy,yy_tax
  REAL(dp)::fz,ifz,z_tax
  REAL(dp)::therm_cond_tax
  !
  OPEN(unit=1,file='Omega.inp')
  OPEN(unit=2,file='Viscosity_N2.dat')
  OPEN(unit=3,file='Thermal_cond_N2.dat')
  OPEN(Unit=4,file='Diffusion_coeff_N2.dat')
  OPEN(unit=11,file='Scattering.dat')
  !
  READ(1,*)T0
  READ(1,*)P0
  READ(1,*)Ntest
  READ(1,*)seme
  !
  dens0=P0/(kb*T0)
  !
  ttrasl=0.0D0;trot=0.0D0;umed=0.0D0
  omega22=0.0D0;omega11=0.0D0;omega22_tax=0.0D0
  omega_bulk_tax=0.0D0
  ifx=0.0D0;ify=0.0D0;ifyy=0.0D0;ifz=0.0D0
  !
  DO test=1,Ntest
     CALL Maxwell(v1(1),v1(2),v1(3))
     v1=v1*SQRT(Rgas(1)*T0)
     CALL Maxwell(v2(1),v2(2),v2(3))
     v2=v2*SQRT(Rgas(1)*T0)
     CALL Genk(omega1(1),omega1(2),omega1(3))
     erot=-kb*T0*LOG(1.0D0-rf(seme))
     omega1=omega1*SQRT(2.0D0*erot/momin(1))
     erot1=erot
     CALL Genk(omega2(1),omega2(2),omega2(3))
     erot=-kb*T0*LOG(1.0D0-rf(seme))
     omega2=omega2*SQRT(2.0D0*erot/momin(1))
     erot2=erot
     umed=umed+v1+v2
     ttrasl=ttrasl+0.5D0*molmass(1)*Dot_PRODUCT(v1,v1)
     ttrasl=ttrasl+0.5D0*molmass(1)*Dot_PRODUCT(v2,v2)
     trot=trot+0.5D0*momin(1)*Dot_PRODUCT(omega1,omega1)
     trot=trot+0.5D0*momin(1)*Dot_PRODUCT(omega2,omega2)
     vr=v2-v1
     CALL Collision(v1,omega1,v2,omega2,1,1)
     vrstar=v2-v1
     g=SQRT(Dot_PRODUCT(vr,vr))
     gstar=SQRT(Dot_PRODUCT(vrstar,vrstar))
     coschi=Dot_PRODUCT(vr,vrstar)/(g*gstar)
     sinchi2=1.0D0-coschi**2
     erot1_star=0.5D0*momin(1)*Dot_PRODUCT(omega1,omega1)
     erot2_star=0.5D0*momin(1)*Dot_PRODUCT(omega2,omega2)
     omega22=omega22+(1.0D0-coschi**2)*g**5
     omega11=omega11+(1.0D0-coschi)*g**3
     omega22_tax=omega22_tax+(sinchi2*g**4-(2.0D0/3.0D0*gstar**2+sinchi2*g**2)*(g**2-gstar**2))*g
     !
     ! volume viscosity
     !
     fbulk=(3.0D0*erot1_star-3.0D0*kb*T0/2.0D0-molmass(1)*g**2/4.0D0)*(g**2-gstar**2)*molmass(1)/4.0D0
     fbulk=fbulk+((g**2-gstar**2)*molmass(1)/4.0D0)**2
     fbulk=fbulk*g
     omega_bulk_tax=omega_bulk_tax+fbulk
     !
     ! thermal conductivity
     !
     fx=(molmass(1)/4.0D0*g**2)**2*sinchi2
     fx=fx+(molmass(1)/4.0D0*g**2*coschi**2+25.0D0/8.0D0*kb*T0-15.0D0/4.0D0*molmass(1)/4.0D0*g**2)*&
          molmass(1)/4.0D0*(g**2-gstar**2)+11.0D0/4.0D0*(molmass(1)/4.0D0*(g**2-gstar**2))**2
     fx=fx*g
     ifx=ifx+fx
     !
     fy=(3.0D0/2.0D0*kb*T0-molmass(1)/4.0D0*g**2)*molmass(1)/4.0D0*(g**2-gstar**2)
     fy=fy+(molmass(1)/4.0D0*(g**2-gstar**2))**2
     fy=fy*g
     ify=ify+fy
     !
     fyy=(erot1_star-kb*T0)*molmass(1)/4.0D0*(g**2-gstar**2)*g
     ifyy=ifyy+fyy
     !
     fz=3.0D0/2.0D0*kb*T0*(g**2-gstar**2)+gstar**2*(erot1_star-erot2_star)-g*gstar*(erot1-erot2)*coschi
     fz=fz*(erot1_star-kb*T0)*g
     ifz=ifz+fz
  END DO
  umed=umed/(2.0D0*Ntest)
  ttrasl=ttrasl/(2.0D0*Ntest)-0.5D0*molmass(1)*Dot_PRODUCT(umed,umed)
  ttrasl=2.0D0*ttrasl/(3.0D0*kb)
  trot=trot/(2.0D0*Ntest)
  trot=trot/kb
  Omega22=Omega22/Ntest
  Omega22=Omega22*(pi/4.0D0)*(1.0D0/(4.0D0*Rgas(1)*T0)**2)*diamol(1,1)**2/2.0D0
  Viscosity=5.0*kb*T0/(8.0D0*omega22)
  Omega22_tax=Omega22_tax/Ntest
  Omega22_tax=Omega22_tax*(pi/4.0D0)*(1.0D0/(4.0D0*Rgas(1)*T0)**2)*diamol(1,1)**2/2.0D0
  Viscosity_tax=5.0*kb*T0/(8.0D0*omega22_tax)
  !
  omega_bulk_tax=omega_bulk_tax/Ntest
  omega_bulk_tax=omega_bulk_tax*pi*diamol(1,1)**2
  omega_bulk_tax=omega_bulk_tax*SQRT(pi)/(4.0D0*SQRT(4.0D0*Rgas(1)*T0)*(kb*T0)**2) !ok
  bulk_viscosity_tax=SQRT(pi*molmass(1)*kb*T0)/(10.0D0*omega_bulk_tax)
  !
  Omega11=Omega11/Ntest
  Omega11=Omega11*pi/(16.0D0*Rgas(1)*T0)*diamol(1,1)**2/2.0D0
  Diffusion_Coefficient=3.0*kb*T0/(8.0D0*molmass(1)*dens0*Omega11)
  WRITE(4,100)T0,Diffusion_Coefficient,Omega11
  WRITE(2,100)T0,Ttrasl,Trot,Omega22,Viscosity,Omega22_tax,Viscosity_tax,omega_bulk_tax,bulk_viscosity_tax
  !
  ! Thermal conductivity
  !
  ifx=ifx/Ntest;ifx=ifx*pi*diamol(1,1)**2
  x_tax=ifx*SQRT(pi)/(8.0D0*SQRT(Rgas(1)*T0)*(kb*T0)**2)
  x_tax=x_tax*4.0D0*SQRT(Rgas(1)*T0/pi)
  !
  ify=ify/Ntest;ify=ify*pi*diamol(1,1)**2
  y_tax=ify*SQRT(pi)/(8.0D0*SQRT(Rgas(1)*T0)*(kb*T0)**2)
  y_tax=y_tax*(-5.0D0)*SQRT(Rgas(1)*T0/pi)
  !
  ifyy=ifyy/Ntest;ifyy=ifyy*pi*diamol(1,1)**2
  yy_tax=ifyy*SQRT(pi)/(8.0D0*SQRT(Rgas(1)*T0)*(kb*T0)**2)
  yy_tax=yy_tax*(-10.0D0)*SQRT(Rgas(1)*T0/pi)
  !
  ifz=ifz/Ntest;ifz=ifz*pi*diamol(1,1)**2
  z_tax=ifz*SQRT(pi*Rgas(1)*T0)/(32.0D0*(Rgas(1)*T0)**2*(kb*T0)**2)
  z_tax=z_tax*(4.0D0)*SQRT(Rgas(1)*T0/pi)
  !
  therm_cond_tax=3.0D0*kb**2*T0/(2.0D0*molmass(1))*(x_tax-5.0D0*y_tax/2.0D0-5.0D0*yy_tax/2.0D0+25.0D0/4.0D0*z_tax)
  therm_cond_tax=therm_cond_tax/(x_tax*z_tax-y_tax*yy_tax)
  WRITE(3,100)T0,x_tax,y_tax,yy_tax,z_tax,therm_cond_tax
100 FORMAT(10(E13.5,2X))
END PROGRAM Omega_N2
