MODULE Geometry
  USE Precision
  IMPLICIT NONE
CONTAINS
  !
  SUBROUTINE Vec_Product(a,b,c)
  IMPLICIT NONE
  REAL(dp), DIMENSION(3)::a,b,c
  c(1)=a(2)*b(3)-a(3)*b(2)
  c(2)=a(3)*b(1)-a(1)*b(3)
  c(3)=a(1)*b(2)-a(2)*b(1)
END SUBROUTINE Vec_Product
!
SUBROUTINE Derivate_Geometriche(R,ra,rb,rho,cta,sta,ctb,stb,cp,sp,&
     dtheta_a_dR,dtheta_a_dra,dtheta_b_dR,dtheta_b_drb,dphi_dr,&
     dphi_dra,dphi_drb,drho_dR)
  IMPLICIT NONE
  REAL(dp), PARAMETER::pi=4.0D0*ATAN(1.0D0)
  REAL(dp), DIMENSION(3), INTENT(in)::R,ra,rb
  REAL(dp), DIMENSION(3)::ur,ua,ub
  REAL(dp), DIMENSION(3)::na,nb,ua_nb,ub_na,na_ur,nb_ur
  REAL(dp), INTENT(out)::rho,cta,sta,ctb,stb,cp,sp
  REAL(dp)::rho_a,rho_b,phi
  REAL(dp), DIMENSION(3), INTENT(out)::drho_dR
  REAL(dp), DIMENSION(3), INTENT(out)::dtheta_a_dR,dtheta_a_dra
  REAL(dp), DIMENSION(3), INTENT(out)::dtheta_b_dR,dtheta_b_drb
  REAL(dp), DIMENSION(3), INTENT(out)::dphi_dr,dphi_dra,dphi_drb
  REAL(dp), DIMENSION(3,3)::Jacobian
  REAL(dp), DIMENSION(3)::I0,J0,K0
  !
  I0=(/1.0D0,0.0D0,0.0D0/)
  J0=(/0.0D0,1.0D0,0.0D0/)
  K0=(/0.0D0,0.0D0,1.0D0/)
  !
  rho=SQRT(DOT_PRODUCT(R,R))
  rho_a=SQRT(DOT_PRODUCT(ra,ra))
  rho_b=SQRT(DOT_PRODUCT(rb,rb))
  !
  ur=R/rho;ua=ra/rho_a;ub=rb/rho_b
  !
  drho_dR=ur
  !
  cta=DOT_PRODUCT(ur,ua)
  IF(cta.LE.-1.0D0.OR.cta.GE.1.0D0) cta=SIGN(0.9999_dp,cta)
  sta=SQRT(1.0D0-cta**2)
  ctb=DOT_PRODUCT(ur,ub)
  IF(ctb.LE.-1.0D0.OR.ctb.GE.1.0D0) ctb=SIGN(0.9999_dp,ctb)
  stb=SQRT(1.0D0-ctb**2)
  CALL Vec_Product(ur,ua,na)
  CALL Vec_Product(ur,ub,nb)
  na=na/SQRT(DOT_PRODUCT(na,na))
  nb=nb/SQRT(DOT_PRODUCT(nb,nb))
  cp=DOT_PRODUCT(na,nb)
  IF(cp.LE.-1.0D0.OR.cp.GE.1.0D0) cp=SIGN(0.9999_dp,cp)
  sp=SQRT(1.0D0-cp**2)
  phi=ACOS(cp)
  IF(DOT_PRODUCT(na,rb).LT.0.0D0) THEN
     phi=2.0D0*pi-phi
     sp=SIN(phi)
  END IF
  !
  IF(sp.EQ.0.0D0) WRITE(*,*)cp,sp,phi/pi
  !
  !WRITE(3,*)rb(3),phi,DOT_PRODUCT(na,rb)
  !
  dtheta_a_dR=-1.0D0/sta*(ua-cta*ur)/rho
  dtheta_a_dra=-1.0D0/sta*(ur-cta*ua)/rho_a
  !
  dtheta_b_dR=-1.0D0/stb*(ub-ctb*ur)/rho
  dtheta_b_drb=-1.0D0/stb*(ur-ctb*ub)/rho_b
  !
  ! dphi_dr
  !
  Jacobian(1,:)=I0/rho-ur*R(1)/rho**2
  Jacobian(2,:)=J0/rho-ur*R(2)/rho**2
  Jacobian(3,:)=K0/rho-ur*R(3)/rho**2
  !
  CALL Vec_Product(ua,nb,ua_nb)
  CALL Vec_Product(ub,na,ub_na)
  !
  dphi_dr(1)=(DOT_PRODUCT(ua_nb,Jacobian(1,:))-cta*cp*dtheta_a_dR(1))/sta
  dphi_dr(1)=dphi_dr(1)+(DOT_PRODUCT(ub_na,Jacobian(1,:))-ctb*cp*dtheta_b_dR(1))/stb
  dphi_dr(1)=-dphi_dr(1)/sp
  !
  !
  dphi_dr(2)=(DOT_PRODUCT(ua_nb,Jacobian(2,:))-cta*cp*dtheta_a_dR(2))/sta
  dphi_dr(2)=dphi_dr(2)+(DOT_PRODUCT(ub_na,Jacobian(2,:))-ctb*cp*dtheta_b_dR(2))/stb
  dphi_dr(2)=-dphi_dr(2)/sp
  !
  dphi_dr(3)=(DOT_PRODUCT(ua_nb,Jacobian(3,:))-cta*cp*dtheta_a_dR(3))/sta
  dphi_dr(3)=dphi_dr(3)+(DOT_PRODUCT(ub_na,Jacobian(3,:))-ctb*cp*dtheta_b_dR(3))/stb
  dphi_dr(3)=-dphi_dr(3)/sp
  !
  !
  ! dphi_dra
  !
  Jacobian(1,:)=I0/rho_a-ua*ra(1)/rho_a**2
  Jacobian(2,:)=J0/rho_a-ua*ra(2)/rho_a**2
  Jacobian(3,:)=K0/rho_a-ua*ra(3)/rho_a**2
  !
  !
  CALL Vec_Product(nb,ur,nb_ur)
  !
  dphi_dra(1)=(DOT_PRODUCT(nb_ur,Jacobian(1,:))-cta*cp*dtheta_a_dra(1))/sta
  dphi_dra(2)=(DOT_PRODUCT(nb_ur,Jacobian(2,:))-cta*cp*dtheta_a_dra(2))/sta
  dphi_dra(3)=(DOT_PRODUCT(nb_ur,Jacobian(3,:))-cta*cp*dtheta_a_dra(3))/sta
  dphi_dra=-dphi_dra/sp
  !
  !
  ! dphi_drb
  !
  Jacobian(1,:)=I0/rho_b-ub*rb(1)/rho_b**2
  Jacobian(2,:)=J0/rho_b-ub*rb(2)/rho_b**2
  Jacobian(3,:)=K0/rho_b-ub*rb(3)/rho_b**2
  !
  CALL Vec_Product(na,ur,na_ur)
  !
  dphi_drb(1)=(DOT_PRODUCT(na_ur,Jacobian(1,:))-ctb*cp*dtheta_b_drb(1))/stb
  dphi_drb(2)=(DOT_PRODUCT(na_ur,Jacobian(2,:))-ctb*cp*dtheta_b_drb(2))/stb
  dphi_drb(3)=(DOT_PRODUCT(na_ur,Jacobian(3,:))-ctb*cp*dtheta_b_drb(3))/stb
  dphi_drb=-dphi_drb/sp
  !
END SUBROUTINE Derivate_Geometriche
!
END MODULE Geometry
