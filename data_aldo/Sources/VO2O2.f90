MODULE pot_O2_O2
  USE PRECISION
  USE Geometry
  IMPLICIT NONE
  REAL(dp), PARAMETER::x1=1.12d0,x2=1.55d0
  REAL(dp), PARAMETER::beta0=6.7d0
  REAL(dp), PARAMETER::c000=1.2874d0
  REAL(dp), DIMENSION(4)::bs=(/-0.69478d0,1.3999d0,-4.4494d0,4.7200d0/)
  REAL(dp), PARAMETER::a202=6.5812d+05
  REAL(dp), PARAMETER::c202=1.5600d+01,d202=-9.1260d-01
  REAL(dp), PARAMETER::cc202=1.4865d-01
  REAL(dp), PARAMETER::a220=4.2735d+05
  REAL(dp), PARAMETER::c220=1.4937d+01
  REAL(dp), PARAMETER::a222=-3.4188d+05
  REAL(dp), PARAMETER::c222=1.4937d+01
  !
  REAL(dp), PARAMETER::sqt2=SQRT(2.0D0),sqt5=SQRT(5.0D0),sqt35=SQRT(35.0D0),sqt70=SQRT(70.0D0),&
       sqt7=SQRT(7.0D0),sqt14=SQRT(14.0D0)
CONTAINS
  !
  FUNCTION voo(R,ra,rb)
    IMPLICIT NONE
    REAL(dp), PARAMETER::pi=4.0D0*ATAN(1.0D0)
    REAL(dp), DIMENSION(3)::R,ra,rb
    REAL(dp), DIMENSION(3)::Ur,Ua,Ub
    REAL(dp), DIMENSION(3)::na,nb
    REAL(dp)::v000,v202,v220,v222,voo
    REAL(dp)::f202,f220,f222
    REAL(dp)::rho
    REAL(dp)::cta,sta,ctb,stb,cp,sp,phi
    !
    ! Trasforma le variabili
    !
    rho=DOT_PRODUCT(R,R)
    rho=SQRT(rho)
    Ur=R/rho
    !
    Ua=ra/SQRT(DOT_PRODUCT(ra,ra))
    Ub=rb/SQRT(DOT_PRODUCT(rb,rb))
    !
    cta=DOT_PRODUCT(Ua,Ur)
    sta=SQRT(1.0D0-cta**2)
    ctb=DOT_PRODUCT(Ub,Ur)
    stb=SQRT(1.0D0-ctb**2)
    !
    CALL Vec_Product(ur,ua,na)
    CALL Vec_Product(ur,ub,nb)
    na=na/SQRT(DOT_PRODUCT(na,na))
    nb=nb/SQRT(DOT_PRODUCT(nb,nb))
    cp=DOT_PRODUCT(na,nb)
    sp=SQRT(1.0D0-cp**2)
    phi=ACOS(cp)
    IF(DOT_PRODUCT(na,rb).LT.0.0D0) THEN
       phi=2.0D0*pi-phi
       sp=SIN(phi)
    END IF
    !
    !WRITE(2,100)cta,sta,ctb,stb,cp,sp,phi
    !
    ! Funzioni radiali
    !
    V000=0.0D0
    IF(rho.LE.x1) THEN
       v000=EXP(-2.0d0*beta0*(rho-1.0d0))-2.0d0*EXP(-beta0*(rho-1.0d0))
    END IF
    IF(rho.GT.x1.AND.rho.LE.x2) THEN
       v000=bs(1)+(rho-x1)*(bs(2)+(rho-x2)*(bs(3)+(rho-x1)*bs(4)))
    END IF
    IF(rho.GT.x2) THEN
       v000=-c000/rho**6
    END IF
    !
    IF(rho.LT.8.0D0) THEN
       v202=a202*EXP(-c202*rho-d202*rho**2)-cc202/rho**6
    ELSE
       v202=0.0D0
    END IF
    ! 
    v220=a220*EXP(-c220*rho)
    !
    v222=a222*EXP(-c222*rho)
    !
    ! Funzioni angolari
    !
    f202=(3.0d0*sqt5*ctb**2)/2.0d0+&
         (3.0d0*sqt5*cta**2)/2.0d0-sqt5
    !
    f220=((-3.0e+0)*sqt5*sp**2*sta**2*stb**2)/2.0d0&
         +3.0D0*sqt5*sta**2*stb**2+&
         ((-3.0d0)*sqt5*stb**2)/2.0d0+&
         3.0d0*sqt5*cp*cta*sta*ctb*stb&
         +((-3.0d0)*sqt5*sta**2)/2.0d0+sqt5
    !
    f222=(3.0d0*sqt5*sqt35*cp**2*sta**2*stb**2)/(7.0d0*sqt2)+&
         ((-3.0d0)*sqt2*sqt5*sqt35*sta**2*stb**2)/7.0d0+&
         (3.0d0*sqt5*sqt35*stb**2)/(7.0d0*sqt2)+&
         ((-3.0d0)*sqt5*sqt70*cp*cta*sta*ctb*stb)/14.0d0+&
         (3.0d0*sqt5*sqt35*sta**2)/(7.0d0*sqt2)-&
         (sqt2*sqt5*sqt35)/7.0d0
    !
    voo=v000+v202*f202+v220*f220+v222*f222
    !
!!$    WRITE(333,100) x1,x2
!!$    WRITE(333,100) bs
!!$    WRITE(333,100) beta0
!!$    WRITE(333,100) c000
!!$    WRITE(333,100) a202
!!$    WRITE(333,100) c202
!!$    WRITE(333,100) d202
!!$    WRITE(333,100) a220
!!$    WRITE(333,100) c220
!!$    WRITE(333,100) a222
!!$    WRITE(333,100) c222
!!$    !
!!$    STOP
!!$100 FORMAT(14(E12.5,2x))
  END FUNCTION voo
  !
  FUNCTION dvoo_drho(rho,cta,sta,ctb,stb,cp,sp)
    IMPLICIT NONE
    REAL(dp)::dv000,dv202,dv220,dv222,dvoo_drho
    REAL(dp)::f202,f220,f222
    REAL(dp)::rho
    REAL(dp)::cta,sta,ctb,stb,cp,sp
    !
    ! Funzioni radiali
    !
    IF(rho.LE.x1) THEN
       dv000=2.0D0*beta0*EXP(-(beta0*(rho-1.0D0)))-&
            2.0D0*beta0*EXP(-(2.0D0*beta0*(rho-1.0D0)))
    END IF
    IF(rho.GT.x1.AND.rho.LE.x2) THEN
       dv000=(bs(4)*(rho-x1)+bs(3))*(rho-x2)+(rho-x1)*(bs(4)*(rho-x2)+bs(4)*(rho-x1)+bs(3))+bs(2)
    END IF
    IF(rho.GT.x2) THEN
       dv000=(6.0D0*c000)/rho**7
    END IF
    !
    IF(rho.LT.8.0D0) THEN
       dv202=a202*(-(2.0D0*d202*rho)-c202)*EXP(-(d202*rho**2)-c202*rho)+(6.0D0*cc202)/rho**7
    ELSE
       dv202=0.0D0
    END IF
    ! 
    dv220=-(a220*c220*EXP(-(c220*rho)))
    !
    dv222=-(a222*c222*EXP(-(c222*rho)))
    !
    !
    ! Funzioni angolari
    !
    f202=(3.0d0*sqt5*ctb**2)/2.0d0+&
         (3.0d0*sqt5*cta**2)/2.0d0-sqt5
    !
    f220=((-3.0e+0)*sqt5*sp**2*sta**2*stb**2)/2.0d0&
         +3.0D0*sqt5*sta**2*stb**2+&
         ((-3.0d0)*sqt5*stb**2)/2.0d0+&
         3.0d0*sqt5*cp*cta*sta*ctb*stb&
         +((-3.0d0)*sqt5*sta**2)/2.0d0+sqt5
    !
    f222=(3.0d0*sqt5*sqt35*cp**2*sta**2*stb**2)/(7.0d0*sqt2)+&
         ((-3.0d0)*sqt2*sqt5*sqt35*sta**2*stb**2)/7.0d0+&
         (3.0d0*sqt5*sqt35*stb**2)/(7.0d0*sqt2)+&
         ((-3.0d0)*sqt5*sqt70*cp*cta*sta*ctb*stb)/14.0d0+&
         (3.0d0*sqt5*sqt35*sta**2)/(7.0d0*sqt2)-&
         (sqt2*sqt5*sqt35)/7.0d0
    !
    dvoo_drho=dv000+dv202*f202+dv220*f220+dv222*f222
    !
  END FUNCTION dvoo_drho
  !
  FUNCTION dvoo_dtheta_a(rho,cta,sta,ctb,stb,cp,sp)
    IMPLICIT NONE
    REAL(dp)::v000,v202,v220,v222,dvoo_dtheta_a
    REAL(dp)::df202,df220,df222
    REAL(dp)::rho
    REAL(dp)::cta,sta,ctb,stb,cp,sp
    !
    !
    ! Funzioni radiali
    !
    V000=0.0D0
    IF(rho.LE.x1) THEN
       v000=EXP(-2.0d0*beta0*(rho-1.0d0))-2.0d0*EXP(-beta0*(rho-1.0d0))
    END IF
    IF(rho.GT.x1.AND.rho.LE.x2) THEN
       v000=bs(1)+(rho-x1)*(bs(2)+(rho-x2)*(bs(3)+(rho-x1)*bs(4)))
    END IF
    IF(rho.GT.x2) THEN
       v000=-c000/rho**6
    END IF
    !
    IF(rho.LT.8.0D0) THEN
       v202=a202*EXP(-c202*rho-d202*rho**2)-cc202/rho**6
    ELSE
       v202=0.0D0
    END IF
    ! 
    v220=a220*EXP(-c220*rho)
    !
    v222=a222*EXP(-c222*rho)
    !
    df202=-(3.0D0*sqt5*cta*sta)
    !
    df220=-(3.0D0*sqt5*sp**2*cta*sta*stb**2)+&
         6*sqt5*cta*sta*stb**2-3.0D0*sqt5*cp*&
         sta**2*ctb*stb+3.0D0*sqt5*cp*&
         cta**2*ctb*stb-3.0D0*sqt5*cta*sta
    !
    df222=(15.0D0*sqt2*cp**2*cta*sta*stb**2)/sqt7-&
         (15.0D0*2**(3.0d0/2.0d0)*cta*sta*stb**2)/sqt7+&
         (15.0D0*cp*sta**2*ctb*stb)/sqt14-&
         (15.0D0*cp*cta**2*ctb*stb)/sqt14+&
         (15.0D0*sqt2*cta*sta)/sqt7
    !
    dvoo_dtheta_a=v202*df202+v220*df220+v222*df222
    !
  END FUNCTION dvoo_dtheta_a
  !
  FUNCTION dvoo_dtheta_b(rho,cta,sta,ctb,stb,cp,sp)
    IMPLICIT NONE
    REAL(dp)::v000,v202,v220,v222,dvoo_dtheta_b
    REAL(dp)::df202,df220,df222
    REAL(dp)::rho
    REAL(dp)::cta,sta,ctb,stb,cp,sp
    !
    !
    ! Funzioni radiali
    !
    V000=0.0D0
    IF(rho.LE.x1) THEN
       v000=EXP(-2.0d0*beta0*(rho-1.0d0))-2.0d0*EXP(-beta0*(rho-1.0d0))
    END IF
    IF(rho.GT.x1.AND.rho.LE.x2) THEN
       v000=bs(1)+(rho-x1)*(bs(2)+(rho-x2)*(bs(3)+(rho-x1)*bs(4)))
    END IF
    IF(rho.GT.x2) THEN
       v000=-c000/rho**6
    END IF
    !
    IF(rho.LT.8.0D0) THEN
       v202=a202*EXP(-c202*rho-d202*rho**2)-cc202/rho**6
    ELSE
       v202=0.0D0
    END IF
    ! 
    v220=a220*EXP(-c220*rho)
    !
    v222=a222*EXP(-c222*rho)
    !
    df202=-(3.0D0*sqt5*ctb*stb)
    !
    df220=-(3.0D0*sqt5*cp*cta*sta*stb**2)-&
         3.0D0*sqt5*sp**2*sta**2*ctb*stb+&
         6*sqt5*sta**2*ctb*stb-&
         3.0D0*sqt5*ctb*stb+&
         3.0D0*sqt5*cp*cta*sta*ctb**2
    !
    df222=(15.0D0*cp*cta*sta*stb**2)/sqt14+&
         (15.0D0*sqt2*cp**2*sta**2*ctb*stb)/sqt7-&
         (15.0D0*2**(3.0d0/2.0d0)*sta**2*ctb*stb)/sqt7+&
         (15.0D0*sqt2*ctb*stb)/sqt7-&
         (15.0D0*cp*cta*sta*ctb**2)/sqt14
    !
    dvoo_dtheta_b=v202*df202+v220*df220+v222*df222
    !
  END FUNCTION dvoo_dtheta_b
  !
    FUNCTION dvoo_dphi(rho,cta,sta,ctb,stb,cp,sp)
    IMPLICIT NONE
    REAL(dp)::v000,v202,v220,v222,dvoo_dphi
    REAL(dp)::df202,df220,df222
    REAL(dp)::rho
    REAL(dp)::cta,sta,ctb,stb,cp,sp
    !
    !
    ! Funzioni radiali
    !
    V000=0.0D0
    IF(rho.LE.x1) THEN
       v000=EXP(-2.0d0*beta0*(rho-1.0d0))-2.0d0*EXP(-beta0*(rho-1.0d0))
    END IF
    IF(rho.GT.x1.AND.rho.LE.x2) THEN
       v000=bs(1)+(rho-x1)*(bs(2)+(rho-x2)*(bs(3)+(rho-x1)*bs(4)))
    END IF
    IF(rho.GT.x2) THEN
       v000=-c000/rho**6
    END IF
    !
    IF(rho.LT.8.0D0) THEN
       v202=a202*EXP(-c202*rho-d202*rho**2)-cc202/rho**6
    ELSE
       v202=0.0D0
    END IF
    ! 
    v220=a220*EXP(-c220*rho)
    !
    v222=a222*EXP(-c222*rho)
    !
    df202=0.0D0
    !
    df220=-(3.0D0*sqt5*cp*sp*sta**2*stb**2)-&
         3.0D0*sqt5*sp*cta*sta*ctb*stb
    !
    df222=(15.0D0*sp*cta*sta*ctb*stb)/sqt14-&
         (15.0D0*sqt2*cp*sp*sta**2*stb**2)/sqt7
    !
    dvoo_dphi=v202*df202+v220*df220+v222*df222
    !
  END FUNCTION dvoo_dphi
  !
  FUNCTION voo_test(rho,cta,sta,ctb,stb,cp,sp)
    IMPLICIT NONE
    REAL(dp)::v000,v202,v220,v222,voo_test
    REAL(dp)::f202,f220,f222
    REAL(dp)::rho
    REAL(dp)::cta,sta,ctb,stb,cp,sp
    !
    !
    ! Funzioni radiali
    !
    V000=0.0D0
    IF(rho.LE.x1) THEN
       v000=EXP(-2.0d0*beta0*(rho-1.0d0))-2.0d0*EXP(-beta0*(rho-1.0d0))
    END IF
    IF(rho.GT.x1.AND.rho.LE.x2) THEN
       v000=bs(1)+(rho-x1)*(bs(2)+(rho-x2)*(bs(3)+(rho-x1)*bs(4)))
    END IF
    IF(rho.GT.x2) THEN
       v000=-c000/rho**6
    END IF
    !
    IF(rho.LT.8.0D0) THEN
       v202=a202*EXP(-c202*rho-d202*rho**2)-cc202/rho**6
    ELSE
       v202=0.0D0
    END IF
    ! 
    v220=a220*EXP(-c220*rho)
    !
    v222=a222*EXP(-c222*rho)
    !
    ! Funzioni angolari
    !
    f202=(3.0d0*sqt5*ctb**2)/2.0d0+&
         (3.0d0*sqt5*cta**2)/2.0d0-sqt5
    !
    f220=((-3.0e+0)*sqt5*sp**2*sta**2*stb**2)/2.0d0&
         +3.0D0*sqt5*sta**2*stb**2+&
         ((-3.0d0)*sqt5*stb**2)/2.0d0+&
         3.0d0*sqt5*cp*cta*sta*ctb*stb&
         +((-3.0d0)*sqt5*sta**2)/2.0d0+sqt5
    !
    f222=(3.0d0*sqt5*sqt35*cp**2*sta**2*stb**2)/(7.0d0*sqt2)+&
         ((-3.0d0)*sqt2*sqt5*sqt35*sta**2*stb**2)/7.0d0+&
         (3.0d0*sqt5*sqt35*stb**2)/(7.0d0*sqt2)+&
         ((-3.0d0)*sqt5*sqt70*cp*cta*sta*ctb*stb)/14.0d0+&
         (3.0d0*sqt5*sqt35*sta**2)/(7.0d0*sqt2)-&
         (sqt2*sqt5*sqt35)/7.0d0
    !
    voo_test=v000+v202*f202+v220*f220+v222*f222
  END FUNCTION voo_test
  !
  SUBROUTINE voo_derivatives(R,ra,rb,dvoo_dR,dvoo_dra,dvoo_drb)
    IMPLICIT NONE
    REAL(dp), DIMENSION(3), INTENT(in)::R,ra,rb
    REAL(dp), DIMENSION(3), INTENT(out)::dvoo_dR,dvoo_dra,dvoo_drb
    REAL(dp), DIMENSION(3)::dtheta_a_dR,dtheta_a_dra
    REAL(dp), DIMENSION(3)::dtheta_b_dR,dtheta_b_drb
    REAL(dp), DIMENSION(3)::dphi_dr,dphi_dra,dphi_drb
    REAL(dp), DIMENSION(3)::drho_dR
    REAL(dp)::rho,cta,sta,ctb,stb,cp,sp
    INTEGER::i
    !
    CALL Derivate_Geometriche(R,ra,rb,rho,cta,sta,ctb,stb,cp,sp,&
     dtheta_a_dR,dtheta_a_dra,dtheta_b_dR,dtheta_b_drb,dphi_dr,&
     dphi_dra,dphi_drb,drho_dR)
    !
    dvoo_dR=dvoo_drho(rho,cta,sta,ctb,stb,cp,sp)*drho_dR
    dvoo_dR=dvoo_dR+dvoo_dtheta_a(rho,cta,sta,ctb,stb,cp,sp)*dtheta_a_dR
    dvoo_dR=dvoo_dR+dvoo_dtheta_b(rho,cta,sta,ctb,stb,cp,sp)*dtheta_b_dR
    dvoo_dR=dvoo_dR+dvoo_dphi(rho,cta,sta,ctb,stb,cp,sp)*dphi_dR
    !
    dvoo_dra=dvoo_dtheta_a(rho,cta,sta,ctb,stb,cp,sp)*dtheta_a_dra
    dvoo_dra=dvoo_dra+dvoo_dphi(rho,cta,sta,ctb,stb,cp,sp)*dphi_dra
    !
    dvoo_drb=dvoo_dtheta_b(rho,cta,sta,ctb,stb,cp,sp)*dtheta_b_drb
    dvoo_drb=dvoo_drb+dvoo_dphi(rho,cta,sta,ctb,stb,cp,sp)*dphi_drb
    !
  END SUBROUTINE voo_derivatives
  !
END MODULE pot_O2_O2


