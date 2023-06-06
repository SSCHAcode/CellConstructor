module third_order_bubble

contains

subroutine compute_static_bubble(T,freq,is_gamma,D3,n_mod,bubble)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    !
    complex(kind=DP), dimension(n_mod,n_mod),intent(OUT) :: bubble
    !
    real(kind=DP), intent(IN) :: T
    real(kind=DP), intent(IN) :: freq(n_mod,3)
    logical      , intent(IN) :: is_gamma(3)
    complex(kind=DP), dimension(n_mod,n_mod,n_mod), intent(IN) :: D3
    integer, intent(IN) :: n_mod
    !
    real(kind=DP) :: Lambda_23,q2(n_mod,3),q3(n_mod,3)
    integer :: i, rho2, rho3, nu,mu

    
    ! 
!     do i = 1,n_mod
!         q2(i,1)=freq(i,2)
!         q2(i,2)=freqm1(i,2)
!         q2(i,3)=bose(i,2) 
!         q3(i,1)=freq(i,3)
!         q3(i,2)=freqm1(i,3)        
!         q3(i,3)=bose(i,3)        
!     end do
    !
    
    q2(:,1)=freq(:,2)
    q3(:,1)=freq(:,3) 
 
    q2(:,2)=0.0_dp
    q3(:,2)=0.0_dp
    do i = 1, n_mod
      if (.not. is_gamma(2) .or. i > 3) q2(i,2)=1.0_dp/freq(i,2)
      if (.not. is_gamma(3) .or. i > 3) q3(i,2)=1.0_dp/freq(i,3)
    end do    

    call bose_freq(T, n_mod, freq(:,2), q2(:,3))
    call bose_freq(T, n_mod, freq(:,3), q3(:,3))        

    !
    bubble=(0.0_dp,0.0_dp)
    !
    DO rho3=1,n_mod
    DO rho2=1,n_mod
            !
            call Lambda(T,q2(rho2,:),q3(rho3,:),Lambda_23)
            !
            DO nu = 1,n_mod
            DO mu = 1,n_mod
                   bubble(mu,nu) = bubble(mu,nu) +  & 
                                     CONJG(D3(mu,rho2,rho3))*Lambda_23*D3(nu,rho2,rho3)
            END DO
            END DO 
            !
    END DO
    END DO    
    !
end subroutine compute_static_bubble
!
!
!
subroutine compute_dynamic_bubble(energies,sigma,static_limit,T,freq,is_gamma,D3,diag_approx,ne,nsig,n_mod,bubble)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    !
    complex(kind=DP), dimension(ne,nsig,n_mod,n_mod),intent(OUT) :: bubble
    !
    integer, intent(IN)       :: ne
    real(kind=DP), intent(IN) :: energies(ne)    
    integer, intent(IN)       :: nsig    
    real(kind=DP), intent(IN) :: sigma(nsig)
    real(kind=DP), intent(IN) :: T
    real(kind=DP), intent(IN) :: freq(n_mod,3)
    logical      , intent(IN) :: static_limit
    logical      , intent(IN) :: is_gamma(3)
    logical      , intent(IN) :: diag_approx
    complex(kind=DP), dimension(n_mod,n_mod,n_mod), intent(IN) :: D3
    integer, intent(IN) :: n_mod
    !
    real(kind=DP)    :: q2(n_mod,3),q3(n_mod,3)
    complex(kind=DP) :: Lambda_23(ne,nsig)    
    integer :: i, rho2, rho3, nu,mu

    
    ! 
!     do i = 1,n_mod
!         q2(i,1)=freq(i,2)
!         q2(i,2)=freqm1(i,2)
!         q2(i,3)=bose(i,2) 
!         q3(i,1)=freq(i,3)
!         q3(i,2)=freqm1(i,3)        
!         q3(i,3)=bose(i,3)        
!     end do
    !
    
    q2(:,1)=freq(:,2)
    q3(:,1)=freq(:,3) 
 
    q2(:,2)=0.0_dp
    q3(:,2)=0.0_dp
    do i = 1, n_mod
      if (.not. is_gamma(2) .or. i > 3) q2(i,2)=1.0_dp/freq(i,2)
      if (.not. is_gamma(3) .or. i > 3) q3(i,2)=1.0_dp/freq(i,3)
    end do    

    call bose_freq(T, n_mod, freq(:,2), q2(:,3))
    call bose_freq(T, n_mod, freq(:,3), q3(:,3))        
    !
    bubble=(0.0_dp,0.0_dp)
    !
    DO rho3=1,n_mod
    DO rho2=1,n_mod
            !
            call Lambda_dynamic(ne,energies,nsig,sigma,T,static_limit,q2(rho2,:),q3(rho3,:),Lambda_23)
            !
            if (diag_approx) then
                DO nu = 1,n_mod
                    bubble(:,:,nu,nu) = bubble(:,:,nu,nu) +  & 
                                        CONJG(D3(nu,rho2,rho3))*Lambda_23(:,:)*D3(nu,rho2,rho3)
                END DO
            else
                DO nu = 1,n_mod
                DO mu = 1,n_mod
                    bubble(:,:,mu,nu) = bubble(:,:,mu,nu) +  & 
                                        CONJG(D3(mu,rho2,rho3))*Lambda_23(:,:)*D3(nu,rho2,rho3)
                END DO
                END DO
            end if
            !
    END DO
    END DO    
    !
end subroutine compute_dynamic_bubble
!
!
!                                      
subroutine compute_diag_dynamic_bubble(ne,energies,nsig,sigma,T,freq,is_gamma,D3,n_mod,bubble)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    !
    complex(kind=DP), dimension(ne,nsig,n_mod),intent(OUT) :: bubble
    !
    integer, intent(IN)       :: ne
    real(kind=DP), intent(IN) :: energies(ne)    
    integer, intent(IN)       :: nsig    
    real(kind=DP), intent(IN) :: sigma(nsig)
    real(kind=DP), intent(IN) :: T
    real(kind=DP), intent(IN) :: freq(n_mod,3)
    logical      , intent(IN) :: is_gamma(3)
    complex(kind=DP), dimension(n_mod,n_mod,n_mod), intent(IN) :: D3
    integer, intent(IN) :: n_mod
    !
    real(kind=DP)    :: q2(n_mod,3),q3(n_mod,3)
    complex(kind=DP) :: Lambda_23(ne,nsig)   
    integer :: i, rho2, rho3, nu,mu
    logical, parameter :: static_limit = .false.
    
    ! 
!     do i = 1,n_mod
!         q2(i,1)=freq(i,2)
!         q2(i,2)=freqm1(i,2)
!         q2(i,3)=bose(i,2) 
!         q3(i,1)=freq(i,3)
!         q3(i,2)=freqm1(i,3)        
!         q3(i,3)=bose(i,3)        
!     end do
    !
    
    q2(:,1)=freq(:,2)
    q3(:,1)=freq(:,3) 
 
    q2(:,2)=0.0_dp
    q3(:,2)=0.0_dp
    do i = 1, n_mod
      if (.not. is_gamma(2) .or. i > 3) q2(i,2)=1.0_dp/freq(i,2)
      if (.not. is_gamma(3) .or. i > 3) q3(i,2)=1.0_dp/freq(i,3)
    end do    

    call bose_freq(T, n_mod, freq(:,2), q2(:,3))
    call bose_freq(T, n_mod, freq(:,3), q3(:,3))        

    !
    bubble=(0.0_dp,0.0_dp)
    !
    DO rho3=1,n_mod
    DO rho2=1,n_mod
            !
            call Lambda_dynamic(ne,energies,nsig,sigma,T,static_limit,q2(rho2,:),q3(rho3,:),Lambda_23)           
            !
            DO mu = 1,n_mod
                   ! 
                   bubble(:,:,mu) = bubble(:,:,mu) +  & 
                                     CONJG(D3(mu,rho2,rho3))*Lambda_23(:,:)*D3(mu,rho2,rho3)
                   !
            END DO            
            !
    END DO
    END DO    
    !
end subroutine compute_diag_dynamic_bubble
!
!
subroutine compute_perturb_selfnrg(nsig,sigma,T,freq,is_gamma,D3,n_mod,selfnrg)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    !
    complex(kind=DP), dimension(n_mod,nsig),intent(OUT)     :: selfnrg
    !
    integer, intent(IN)       :: nsig    
    real(kind=DP), intent(IN) :: sigma(nsig)
    real(kind=DP), intent(IN) :: T
    real(kind=DP), intent(IN) :: freq(n_mod,3)
    logical      , intent(IN) :: is_gamma(3)
    complex(kind=DP), dimension(n_mod,n_mod,n_mod), intent(IN) :: D3
    integer, intent(IN) :: n_mod
    !
    real(kind=DP)    :: q2(n_mod,3),q3(n_mod,3)
    complex(kind=DP) :: Lambda_23_freq(n_mod,nsig)    
    integer :: i, rho2, rho3, nu,mu
    ! 
!     do i = 1,n_mod
!         q2(i,1)=freq(i,2)
!         q2(i,2)=freqm1(i,2)
!         q2(i,3)=bose(i,2) 
!         q3(i,1)=freq(i,3)
!         q3(i,2)=freqm1(i,3)        
!         q3(i,3)=bose(i,3)        
!     end do
    !
    
    q2(:,1)=freq(:,2)
    q3(:,1)=freq(:,3) 
 
    q2(:,2)=0.0_dp
    q3(:,2)=0.0_dp
    do i = 1, n_mod
      if (.not. is_gamma(2) .or. i > 3) q2(i,2)=1.0_dp/freq(i,2)
      if (.not. is_gamma(3) .or. i > 3) q3(i,2)=1.0_dp/freq(i,3)
    end do    

    call bose_freq(T, n_mod, freq(:,2), q2(:,3))
    call bose_freq(T, n_mod, freq(:,3), q3(:,3))        

    !
    selfnrg=(0.0_dp,0.0_dp)
    !
    !
    DO rho3=1,n_mod
    DO rho2=1,n_mod
            !
            call Lambda_dynamic_value(n_mod,freq(:,1),nsig,sigma,T,q2(rho2,:),q3(rho3,:),Lambda_23_freq) 
            !
            DO mu = 1,n_mod
            
                   selfnrg(mu,:)  = selfnrg(mu,:) + & 
                                     CONJG(D3(mu,rho2,rho3))*Lambda_23_freq(mu,:)*D3(mu,rho2,rho3)
            END DO            
            !
    END DO
    END DO    
    !
end subroutine compute_perturb_selfnrg
!
! =============================== Spectral function ==========================================
!
subroutine compute_spectralf(smear_id,ener,d2,Pi,notransl,spectralf,mass,nat,ne,nsmear)
    implicit none
    INTEGER, PARAMETER           :: DP = selected_real_kind(14,200)
    real(kind=dp),parameter      :: twopi = 6.283185307179586_dp
    !
    real(kind=dp), intent(in)    :: mass(nat), ener(ne)
    integer, intent(in)          :: ne,nsmear,nat
    real(kind=dp), intent(in)    :: smear_id(nsmear)
    complex(kind=dp), intent(in) :: d2(3*nat,3*nat)
    complex(kind=dp), intent(in) :: Pi(ne,nsmear,3*nat,3*nat)
    logical, intent(in)          :: notransl
    !
    real(kind=dp), intent(out)   :: spectralf(ne,nsmear)
    !
    integer                      :: nat3,n,m,ismear,ie
    complex(kind=dp)             :: G(3*nat,3*nat) 
    complex(kind=dp)             :: fact
    
    nat3=3*nat
        
    spectralf=0.0_dp
    !    
    DO ismear=1,nsmear
    DO ie = 1,ne
         ! GREEN FUNCTION INVERSE
         G=(0.0_dp,0.0_dp)
         FORALL (m=1:nat3, n=1:nat3)
             G(n,m) = -Pi(ie,ismear,n,m)
         END FORALL
         ! SOMMO CONTRIBUTO DEL TERZO ORDINE A MATRICE DINAMICA SSCHA
         G=G-d2
         ! ADD PROPTO IDENTITY PART
         fact=smear_id(ismear)*(0.0_dp,1.0_dp)
         DO n=1,nat3
           G(n,n)=G(n,n)+(ener(ie)+fact)**2
         ENDDO
         ! INVERSE
         CALL invzmat(nat3, G) 
         IF ( notransl ) THEN
           CALL eliminate_transl(G,mass,nat)      
         END IF        
         DO n=1,nat3
           spectralf(ie,ismear)=spectralf(ie,ismear)-2*DIMAG(G(n,n))*ener(ie)/twopi
         END DO         
    ENDDO
    ENDDO    
    !
end subroutine compute_spectralf
!
!
subroutine compute_spectralf_diag(smear_id,ener,d2_freq,selfnrg,nat,ne,nsmear,spectralf)
    implicit none
    INTEGER, PARAMETER           :: DP = selected_real_kind(14,200)
    real(kind=dp),parameter      :: pi = 3.141592653589793_dp
    !
    real(kind=dp), intent(in)    :: ener(ne)
    real(kind=dp), intent(in)    :: smear_id(nsmear)    
    integer, intent(in)          :: ne,nsmear,nat
    real(kind=dp), intent(in)    :: d2_freq(3*nat)
    complex(kind=dp), intent(in) :: selfnrg(ne,nsmear,3*nat)
    !
    real(kind=dp), intent(out)   :: spectralf(ne,3*nat,nsmear)
    !
    integer                      :: nat3,mu,ismear,ie
    real(kind=dp)                :: a,b,denom,num
    !
    nat3=3*nat
    spectralf=0.0_dp
    !
    DO ismear = 1,nsmear
      DO mu = 1,nat3
        DO ie = 1, ne

          a = ener(ie)**2-smear_id(ismear)**2-d2_freq(mu)**2-DBLE(selfnrg(ie,ismear,mu))
          b = 2*smear_id(ismear)*ener(ie)-DIMAG(selfnrg(ie,ismear,mu))          
          
          !
          ! (w/2pi)(-2Im [1/a+ib])= (w/pi) b/(a**2+b**2) 
          !
          num   = ener(ie)*b
          denom = (a**2+b**2)*pi
          !
          IF(ABS(denom)/=0._dp)THEN
            spectralf(ie,mu,ismear) = num / denom
          ELSE
            spectralf(ie,mu,ismear) = 0._dp
          ENDIF
        ENDDO
      ENDDO
    ENDDO
    !
end subroutine compute_spectralf_diag
!
! =================== Lambda calculation ====================================
!
subroutine Lambda(T,w_q2,w_q3,Lambda_out)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    REAL(kind=DP),intent(out) :: lambda_out 
    real(kind=DP), intent(in) :: T,w_q2(3),w_q3(3)
    real(kind=DP) :: w2,w3,n2,n3,w2m1,w3m1
    real(kind=DP) :: bose_P, bose_M, omega_P, omega_M, ctm_P, ctm_M
            !  
            w2=w_q2(1)
            w3=w_q3(1)            
            w2m1=w_q2(2)
            w3m1=w_q3(2)
            n2=w_q2(3)
            n3=w_q3(3)            
            !
            bose_P   = 1 + n2 + n3
            omega_P  = w3+w2
            !
            bose_M   = n3-n2
            omega_M  = w3-w2
            !
            IF(ABS(omega_P)>0._dp)THEN
                ctm_P = bose_P /omega_P
            ELSE
                ctm_P = 0._dp
            ENDIF
              !
            IF(ABS(omega_M)>1.e-5_dp)THEN
                ctm_M =  bose_M /omega_M
            ELSE
                IF(T>0._dp.and.ABS(omega_P)>0._dp)THEN
                  ctm_M = df_bose(0.5_dp * omega_P, T)
                ELSE
                  ctm_M = 0._dp
                ENDIF
            ENDIF
            !
            lambda_out=-(ctm_P - ctm_M) * w2m1*w3m1/4.0_dp
            !
end subroutine Lambda
!
!                           =========
!
subroutine Lambda_dynamic(ne,energies,nsigma,sigma,T,static_limit,w_q2,w_q3,Lambda_out)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    complex(kind=DP), intent(out) :: Lambda_out(ne,nsigma)    
    integer, intent(in)       :: ne,nsigma
    logical, intent(in)       :: static_limit
    real(kind=DP), intent(in) :: energies(ne), sigma(nsigma)    
    real(kind=DP), intent(in) :: T,w_q2(3),w_q3(3)
    real(kind=DP) :: w2,w3,n2,n3,w2m1,w3m1
    real(kind=DP) :: bose_P, bose_M, omega_P, omega_P2 ,&
                     omega_M,omega_M2
    complex(kind=DP) :: reg, ctm_P, ctm_M, ctm(ne,nsigma)
    integer       :: ie, isigma
            !  
            w2=w_q2(1)
            w3=w_q3(1)            
            w2m1=w_q2(2)
            w3m1=w_q3(2)
            n2=w_q2(3)
            n3=w_q3(3)            
            !
            bose_P    = 1 + n2 + n3
            omega_P   = w3+w2
            omega_P2  = (omega_P)**2
            !
            bose_M    = n3-n2
            omega_M   = w3-w2
            omega_M2  = (omega_M)**2
            !
            IF(static_limit) THEN ! sigma and energy do not count
                !
                IF(ABS(omega_P)>0._dp)THEN
                    ctm_P = bose_P /omega_P
                ELSE
                    ctm_P = 0._dp
                ENDIF
                !
                IF(ABS(omega_M)>1.e-5_dp)THEN
                    ctm_M =  bose_M /omega_M
                ELSE
                    IF(T>0._dp.and.ABS(omega_P)>0._dp)THEN
                        ctm_M = df_bose(0.5_dp * omega_P, T)
                    ELSE
                        ctm_M = 0._dp
                    ENDIF
                ENDIF
                !
                ctm = ctm_P - ctm_M
            ELSE
                DO isigma = 1, nsigma
                DO ie = 1,ne
                    reg = CMPLX(energies(ie), sigma(isigma), kind=DP)**2
                    ctm_P = bose_P *omega_P/(omega_P2-reg)
                    ctm_M = bose_M *omega_M/(omega_M2-reg)
                    ctm(ie,isigma) = ctm_P - ctm_M
                END DO              
                END DO
            END IF
            !
            lambda_out=-ctm * w2m1*w3m1/4.0_dp
            !
end subroutine Lambda_dynamic
!
!                           =========
!
subroutine Lambda_dynamic_value(n_mod,value,nsigma,sigma,T,w_q2,w_q3,Lambda_out)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    complex(kind=DP), intent(out) :: Lambda_out(n_mod,nsigma)    
    integer, intent(in)       :: nsigma,n_mod
    real(kind=DP), intent(in) :: sigma(nsigma),value(n_mod)    
    real(kind=DP), intent(in) :: T,w_q2(3),w_q3(3)
    real(kind=DP) :: w2,w3,n2,n3,w2m1,w3m1
    real(kind=DP) :: bose_P, bose_M, omega_P, omega_P2 ,&
                     omega_M,omega_M2
    complex(kind=DP) :: reg, ctm_P, ctm_M, ctm(n_mod,nsigma)
    integer       :: ie, isigma,mu
            !  
            w2=w_q2(1)
            w3=w_q3(1)            
            w2m1=w_q2(2)
            w3m1=w_q3(2)
            n2=w_q2(3)
            n3=w_q3(3)            
            !
            bose_P    = 1 + n2 + n3
            omega_P   = w3+w2
            omega_P2  = (omega_P)**2
            !
            bose_M    = n3-n2
            omega_M   = w3-w2
            omega_M2  = (omega_M)**2
            !
            DO mu=1,n_mod
            DO isigma = 1, nsigma
                    reg = CMPLX(value(mu), sigma(isigma), kind=DP)**2
                    ctm_P = bose_P *omega_P/(omega_P2-reg)
                    ctm_M = bose_M *omega_M/(omega_M2-reg)
                    ctm(mu,isigma) = ctm_P - ctm_M
            END DO       
            END DO
            !
            lambda_out=-ctm * w2m1*w3m1/4.0_dp
            !
end subroutine Lambda_dynamic_value
!
!
! ======================== accessory routines ========================================
!
SUBROUTINE bose_freq(T, n_mod, freq, bose)
    IMPLICIT NONE
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    !
    REAL(DP),INTENT(out) :: bose(n_mod)
    !
    REAL(DP),INTENT(in)  :: T
    INTEGER,INTENT(in)   :: n_mod
    REAL(DP),INTENT(in)  :: freq(n_mod)
    !
    IF(T==0._dp)THEN
     bose = 0._dp
     RETURN
    ENDIF
    !
    WHERE    (freq > 0._dp)
      bose = f_bose(freq, T)
    ELSEWHERE
      bose = 0._dp
    ENDWHERE
    !
END SUBROUTINE bose_freq
!
ELEMENTAL FUNCTION f_bose(freq,T) ! bose (freq,T)
    IMPLICIT NONE
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    !
    REAL(DP) :: f_bose
    !
    REAL(DP),INTENT(in) :: freq,T
    !
    REAL(DP), parameter :: K_BOLTZMANN_RY= 1.3806504E-23_DP /(4.35974394E-18_DP/2) !K_BOLTZMANN_SI / (HARTREE_SI/2)    
    REAL(DP) :: Tm1
    !
    Tm1 = 1/(T*K_BOLTZMANN_RY)
    f_bose = 1 / (EXP(freq*Tm1) - 1)
    !
END FUNCTION f_bose
!
FUNCTION df_bose(freq,T) ! d bose(freq,T)/d freq
    IMPLICIT NONE
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    REAL(DP), parameter :: K_BOLTZMANN_RY= 1.3806504E-23_DP /(4.35974394E-18_DP/2) !K_BOLTZMANN_SI / (HARTREE_SI/2)
    !
    REAL(DP) :: df_bose
    !
    REAL(DP),INTENT(in) :: freq,T
    !
    REAL(KIND=DP) :: expf,Tm1
    !
    Tm1  = 1/(T*K_BOLTZMANN_RY)
    expf = EXP(freq*Tm1)
    df_bose = -Tm1 * expf / (expf-1)**2
    !
END FUNCTION df_bose
!
SUBROUTINE invzmat (n, a)
    !-----------------------------------------------------------------------
    ! computes the inverse "a_inv" of matrix "a", both dimensioned (n,n)
    !
    IMPLICIT NONE
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)    
    INTEGER,INTENT(in) :: n
    COMPLEX(DP),INTENT(inout) :: a(n,n)
    !
    INTEGER :: info, lda, lwork, ipiv(n), nb
    ! info=0: inversion was successful
    ! lda   : leading dimension (the same as n)
    ! ipiv  : work space for pivoting (assumed of length lwork=n)
    COMPLEX(DP),ALLOCATABLE :: work(:) 
    INTEGER,EXTERNAL :: ILAENV
    ! more work space
    !
    lda = n
    !
    nb = ILAENV( 1, 'ZHEEV', 'U', n, -1, -1, -1 )
    lwork=n*nb
    ALLOCATE(work(lwork))
    !
    CALL ZGETRF(n, n, a, lda, ipiv, info)
    CALL ZGETRI(n, a, lda, ipiv, work, lwork, info)
    !
    DEALLOCATE(work)
    !
    RETURN
  END SUBROUTINE invzmat
!
  SUBROUTINE eliminate_transl(A,mass,nat)
     !
     IMPLICIT NONE
     INTEGER, PARAMETER :: DP = selected_real_kind(14,200)         
     INTEGER,     intent(in)     :: nat
     COMPLEX(DP), intent(inout)  :: A(3*nat,3*nat)
     real(kind=dp), intent(in)   :: mass(nat)

     COMPLEX(DP)                 :: QAUX(3,3,nat,nat)
     COMPLEX(DP)                 :: Q(nat*3,nat*3)
     REAL(DP)                    :: Mtot,mi,mj
     INTEGER                     :: i,j,alpha,beta
     !
     ! DEFINE Q=1-P, P is TRANSLATION PROJECTOR
     QAUX=(0.0_DP,0.0_DP)
     ! build -P
     Mtot=SUM(mass) 
     DO i=1,nat
     DO j=1,nat
       mj=mass(j)
       mi=mass(i)
        DO alpha=1,3
          QAUX(alpha,alpha,i,j)=-(1.0_dp,0.0_dp)*SQRT(mi*mj)/Mtot
        END DO
     END DO
     END DO
     ! build Q
     DO i=1,nat
      DO alpha=1,3
        QAUX(alpha,alpha,i,i)=1.0_dp+QAUX(alpha,alpha,i,i)
      END DO
     END DO  
     !
     DO j=1, nat
     DO i=1, nat
        DO alpha=1,3
        DO beta=1,3
          Q(3*(i-1)+alpha,3*(j-1)+beta)=QAUX(alpha,beta,i,j)
        END DO
        END DO
     END DO      
     END DO 
     ! PROJECT   
     A=matmul(A,Q)
     !
  END SUBROUTINE eliminate_transl
!
end module third_order_bubble
