MODULE Coll_O2_O2
USE Precision
USE Geometry
USE Scattering
USE Molprops, ONLY:bmax
CONTAINS
  !
  SUBROUTINE Initcoll_O2_O2(vr,omega_a,omega_b,y)
    USE Norm_vars_oo
    USE Random
    IMPLICIT NONE
    INTEGER, PARAMETER::neqmax=18
    REAL(dp), DIMENSION(3),INTENT(IN):: vr,omega_a,omega_b
    REAL(dp), DIMENSION(neqmax),INTENT(OUT)::y
    !
    ! Variabili locali
    !
    REAL(dp), DIMENSION(3)::R
    REAL(dp), DIMENSION(3)::I,J,K
    REAL(dp), DIMENSION(3)::I0,J0,K0
    REAL(dp), DIMENSION(3)::ra,rb
    REAL(dp), DIMENSION(3)::va,vb
    REAL(dp)::b,epsilon
    !
    !Versori Sistema fisso
    !
    I0=(/1.0D0,0.0D0,0.0D0/);J0=(/0.0D0,1.0D0,0.0D0/);K0=(/0.0D0,0.0D0,1.0D0/);
    !
    ! Posizione iniziale del centri di massa molecola 2
    !
    I=-vr/SQRT(DOT_PRODUCT(vr,vr))
    J=-I(2)*I0+I(1)*J0
    J=J/SQRT(DOT_PRODUCT(J,J))
    K=-J(2)*I(3)*I0+J(1)*I(3)*J0+(I(1)*J(2)-I(2)*J(1))*K0
    b=bmax*SQRT(rf(seme))
    !
    !----------------- Dati di scattering---------------------------------------
    !
    impact_parameter=b
    vr_in=vr
    !---------------------------------------------------------------------------
    epsilon=pi2*rf(seme)
    R=SQRT(bmax**2-b**2)*I+b*(COS(epsilon)*J+SIN(epsilon)*K)
    !
    ! Determina gli angoli di Cardano della Molecola A
    !
    I=omega_a/SQRT(DOT_PRODUCT(omega_a,omega_a))
    J=-I(2)*I0+I(1)*J0
    J=J/SQRT(DOT_PRODUCT(J,J))
    K=-J(2)*I(3)*I0+J(1)*I(3)*J0+(I(1)*J(2)-I(2)*J(1))*K0
    !
    epsilon=pi2*rf(seme)
    ra=COS(epsilon)*J+SIN(epsilon)*K;ra=ra*doo
    CALL vec_product(omega_a,ra,va)
    !
    ! Determina gli angoli di Cardano della Molecola B
    !
    I=omega_b/SQRT(DOT_PRODUCT(omega_b,omega_b))
    J=-I(2)*I0+I(1)*J0
    J=J/SQRT(DOT_PRODUCT(J,J))
    K=-J(2)*I(3)*I0+J(1)*I(3)*J0+(I(1)*J(2)-I(2)*J(1))*K0
    !
    epsilon=pi2*rf(seme)
    rb=COS(epsilon)*J+SIN(epsilon)*K;rb=rb*doo
    CALL vec_product(omega_b,rb,vb)
    !
    y(1:3)=R
    y(4:6)=ra
    y(7:9)=rb
    y(10:12)=vr
    y(13:15)=va
    y(16:18)=vb
    !
  END SUBROUTINE Initcoll_O2_O2
  !
  SUBROUTINE CT_O2_O2(vr,omega_a,omega_b,status)
    USE Norm_vars_oo
    USE Pot_O2_O2
    USE Setup, ONLY:h
    IMPLICIT NONE
    INTEGER,INTENT(OUT)::Status
    CHARACTER, PARAMETER:: Oxygen='O'
    REAL(dp), PARAMETER::pi=4.0D0*ATAN(1.0D0)
    INTEGER, PARAMETER::neqmax=18
    INTEGER::nstep,ieq
    REAL(dp), DIMENSION(neqmax)::y
    REAL(dp)::t
    !
    REAL(dp), DIMENSION(3)::I0,J0,K0
    REAL(dp), DIMENSION(3)::R,ra,rb
    REAL(dp), DIMENSION(3)::ra1,ra2,rb1,rb2
    REAL(dp), DIMENSION(3)::va,vb,Vr
    REAL(dp), DIMENSION(3)::omega_a,omega_b,angmom
    REAL(dp)::etrasl,erot_a,erot_b,ekin,epot,etot
    REAL(dp)::ekin_in,epot_in,ekin_fin,epot_fin
    REAL(dp)::etot_in,etot_fin
    REAL(dp)::rho_in,rho,rho_fin
    !
    !
    I0=(/1.0D0,0.0D0,0.0D0/);J0=(/0.0D0,1.0D0,0.0D0/);K0=(/0.0D0,0.0D0,1.0D0/);
    !
    ! Dati Iniziali
    !
    CALL Initcoll_O2_O2(vr,omega_a,omega_b,y)
    !
    R=y(1:3);ra=y(4:6);rb=y(7:9)
    rho_in=SQRT(DOT_PRODUCT(R,R))
    rho=rho_in
    ra1=-ra/2.0D0;ra2=ra/2.0D0
    rb1=R-rb/2.0D0;rb2=R+rb/2.0D0
    vr=y(10:12);va=y(13:15);vb=y(16:18)
    CALL vec_product(ra,va,omega_a)
    CALL vec_product(rb,vb,omega_b)
    omega_a=omega_a/doo**2;omega_b=omega_b/doo**2
    !
    etrasl=0.5*mr*DOT_PRODUCT(Vr,Vr)
    erot_a=0.5*Ia*DOT_PRODUCT(Omega_a,Omega_a)
    erot_b=0.5*Ib*DOT_PRODUCT(Omega_b,Omega_b)
    ekin=etrasl+erot_a+erot_b
    ekin_in=ekin
    epot=voo(R,ra,rb)
    epot_in=epot
    etot_in=ekin+epot
    !
    nstep=0
    t=0.0D0
    !
    status=0
    !
    DO WHILE(rho.LT.1.005D0*rho_in)
       CALL RK4_O2_O2(h,t,y)
       !
       DO ieq=1,18
          IF(ISNAN(y(ieq)).EQV..TRUE.)THEN
             Status=ieq
             WRITE(16,*)'O2-O2','NaN in y(',ieq,')'
!!$             WRITE(16,*)'NaN in y(',ieq,')'
!!$             WRITE(16,200)y(1:3),y(8:10)
!!$             WRITE(16,200)y(4:5),y(11:12)
!!$             WRITE(16,200)y(6:7),y(13:14)
             EXIT
          END IF
       END DO
       !
       IF(status.NE.0) GO TO 111
       !
       nstep=nstep+1
       R=y(1:3);ra=y(4:6);rb=y(7:9)
!!$       ra1=-ra/2.0D0;ra2=ra/2.0D0
!!$       rb1=R-rb/2.0D0;rb2=R+rb/2.0D0
!!$       vr=y(10:12);va=y(13:15);vb=y(16:18)
       rho=SQRT(DOT_PRODUCT(R,R))
!!$       CALL vec_product(ra,va,omega_a)
!!$       CALL vec_product(rb,vb,omega_b)
!!$       omega_a=omega_a/doo**2;omega_b=omega_b/doo**2
!!$       etrasl=0.5*mr*DOT_PRODUCT(Vr,Vr)
!!$       erot_a=0.5*Ia*DOT_PRODUCT(Omega_a,Omega_a)
!!$       erot_b=0.5*Ib*DOT_PRODUCT(Omega_b,Omega_b)
!!$       ekin=etrasl+erot_a+erot_b
!!$       epot=voo(R,ra,rb)
!!$       etot=ekin+epot
!!$       angmom=mr*(R(2)*Vr(3)-R(3)*Vr(2))*I0
!!$       angmom=angmom+mr*(R(3)*Vr(1)-R(1)*Vr(3))*J0
!!$       angmom=angmom+mr*(R(1)*Vr(2)-R(2)*Vr(1))*K0
!!$       angmom=angmom+Ia*Omega_a+Ia*Omega_b
       !
       ! Stampa la traiettoria e le quantita conservate
       !
!!$       IF(MOD(step,50).EQ.0) THEN
!!$          WRITE(10,*)4
!!$          WRITE(10,*)'# Tempo=',t
!!$          WRITE(10,100)Oxygen,ra1
!!$          WRITE(10,100)Oxygen,ra2
!!$          WRITE(10,100)Oxygen,rb1
!!$          WRITE(10,100)Oxygen,rb2
!!$          !
!!$       END IF
!!$       step=step+1
!!$       WRITE(4,200)t,angmom
!!$       WRITE(7,200)t,Omega_a,Omega_b       
!!$       WRITE(8,200)t,etrasl,erot_a,erot_b,ekin,epot,etot
!!$       !
    END DO
    !
    R=y(1:3);ra=y(4:6);rb=y(7:9)
    ra1=-ra/2.0D0;ra2=ra/2.0D0
    rb1=R-rb/2.0D0;rb2=R+rb/2.0D0
    vr=y(10:12);va=y(13:15);vb=y(16:18)
    rho=SQRT(DOT_PRODUCT(R,R))
    CALL vec_product(ra,va,omega_a)
    CALL vec_product(rb,vb,omega_b)
    omega_a=omega_a/doo**2;omega_b=omega_b/doo**2
    etrasl=0.5*mr*DOT_PRODUCT(Vr,Vr)
    erot_a=0.5*Ia*DOT_PRODUCT(Omega_a,Omega_a)
    erot_b=0.5*Ib*DOT_PRODUCT(Omega_b,Omega_b)
    ekin=etrasl+erot_a+erot_b
    epot=voo(R,ra,rb)
    etot=ekin+epot
    angmom=mr*(R(2)*Vr(3)-R(3)*Vr(2))*I0
    angmom=angmom+mr*(R(3)*Vr(1)-R(1)*Vr(3))*J0
    angmom=angmom+mr*(R(1)*Vr(2)-R(2)*Vr(1))*K0
    angmom=angmom+Ia*Omega_a+Ia*Omega_b
    !
    etot_fin=etot
    ekin_fin=ekin
    epot_fin=epot
    rho_fin=SQRT(DOT_PRODUCT(R,R))
    !
    vr_out=vr
    deflection_vr=DOT_PRODUCT(vr_out,vr_in)/(SQRT(DOT_PRODUCT(vr_out,vr_out))*SQRT(DOT_PRODUCT(vr_in,vr_in)))
    deflection_vr=1.0D0-deflection_vr
    !
!!$100 FORMAT(A1,2x,3(e12.5,2x))
!!$200 FORMAT(12(e12.5,2x))
111 CONTINUE
  END SUBROUTINE CT_O2_O2
  !
  SUBROUTINE RK4_O2_O2(h,x,y)
  IMPLICIT NONE
  INTEGER, PARAMETER::neqmax=18
  REAL(dp), DIMENSION(neqmax):: k1,k2,k3,k4
  REAL(dp), DIMENSION(neqmax)::y,dydx,y1
  REAL(dp)::x,x1,h
  !
  CALL ydot_o2_o2(y,dydx)
  !
  k1=dydx*h;y1=y+k1*0.5D0;x1=x+h*0.5D0 
  !
  CALL ydot_o2_o2(y1,dydx)
  !
  k2=dydx*h;y1=y+k2*0.5D0
  !
  CALL ydot_o2_o2(y1,dydx)
  !
  k3=dydx*h; y1=y+k3; x1=x+h
  !
  CALL ydot_o2_o2(y1,dydx)
  !
  k4=dydx*h;y=y+(k1+2.0D0*(k2+k3)+k4)/6.0D0;x=x1
  !
END SUBROUTINE RK4_O2_O2
!
SUBROUTINE Ydot_O2_O2(y,dydt)
  USE Norm_vars_oo
  USE Pot_O2_O2
  IMPLICIT NONE
  INTEGER, PARAMETER::neqmax=18
  REAL(dp), DIMENSION(neqmax)::y,dydt
  !
  REAL(dp), DIMENSION(3)::R,ra,rb
  REAL(dp), DIMENSION(3)::fr,fa,fb
  REAL(dp), DIMENSION(3)::va,vb,vr
  REAL(dp)::lambda_a,lambda_b
  !
  R=y(1:3);ra=y(4:6);rb=y(7:9);
  vr=y(10:12);va=y(13:15);vb=y(16:18);
  !
  lambda_a=-ma*DOT_PRODUCT(va,va)/(4.0D0*doo**2)
  lambda_b=-mb*DOT_PRODUCT(vb,vb)/(4.0D0*doo**2)
  !
  CALL voo_derivatives(R,ra,rb,fr,fa,fb)
  fr=-fr;fa=-fa;fb=-fb;
  dydt(1:9)=y(10:18)
  dydt(10:12)=fr/mr
  dydt(13:15)=4.0D0*(fa+lambda_a*ra)/ma
  dydt(16:18)=4.0D0*(fb+lambda_b*rb)/mb
  !
END SUBROUTINE Ydot_O2_O2
!
END MODULE Coll_O2_O2



