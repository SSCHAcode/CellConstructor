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
    real(kind=DP) :: freqm1(n_mod,3),freqm1_23,bose(n_mod,3)   
    real(kind=DP) :: bose_P, bose_M, omega_P, omega_M, ctm_P, ctm_M
    integer :: i, rho2, rho3, nu,mu
    !
    !     freq(:,1)=w_mq
    !     freq(:,2)=w_k    
    !     freq(:,3)=w_q_mk    
     
     
    freqm1=0.0_dp
    do i = 1, n_mod
      if (.not. is_gamma(1) .or. i > 3) freqm1(i,1)=1.0_dp/freq(i,1)
      if (.not. is_gamma(2) .or. i > 3) freqm1(i,2)=1.0_dp/freq(i,2)
      if (.not. is_gamma(3) .or. i > 3) freqm1(i,3)=1.0_dp/freq(i,3)
    end do    

    call bose_freq(T, n_mod, freq(:,1), bose(:,1))
    call bose_freq(T, n_mod, freq(:,2), bose(:,2))
    call bose_freq(T, n_mod, freq(:,3), bose(:,3))    

    bubble=(0.0_dp,0.0_dp)
    !
    DO rho3=1,n_mod
        DO rho2=1,n_mod
            !
            
            
!             bose_P   = 1 + bose(rho2,2) + bose(rho3,3)
!             omega_P  = freq(rho3,3)+freq(rho2,2)
! !             omega_P2 = omega_P**2
!             bose_M   = bose(rho3,3)-bose(rho2,2)
!             omega_M  = freq(rho3,3)-freq(rho2,2)
! !             omega_M2 = omega_M**2
!             !
! !             IF(sigma<0._dp)THEN
! !               ctm_P =  2 * bose_P *omega_P/(omega_P2+sigma**2)
! !               ctm_M =  2 * bose_M *omega_M/(omega_M2+sigma**2)
! !             ELSE IF (sigma==0._dp)THEN
!             IF(ABS(omega_P)>0._dp)THEN
!                 ctm_P = 2 * bose_P /omega_P
!             ELSE
!                 ctm_P = 0._dp
!             ENDIF
!               !
!             IF(ABS(omega_M)>1.e-5_dp)THEN
!                 ctm_M =  2 * bose_M /omega_M
!             ELSE
!                 IF(T>0._dp.and.ABS(omega_P)>0._dp)THEN
!                   ctm_M =  2* df_bose(0.5_dp * omega_P, T)
!                 ELSE
!                   ctm_M = 0._dp
!                 ENDIF
!             ENDIF
!             print*,bose_P,ctm_M
!             !
!             !
!             freqm1_23 = freqm1(rho2,2)*freqm1(rho3,3)
!             !
            
            DO nu = 1,n_mod
              DO mu = 1,n_mod
!                 bubble(mu,nu) = bubble(mu,nu) + (ctm_P - ctm_M) * freqm1_23 &
!                                      * D3(mu,rho2,rho3) * CONJG(D3(nu,rho2,rho3)) 
                  bubble(mu,nu) =   bubble(mu,nu) +  & 
                                    D3(mu,rho2,rho3) &
                                    *Lambda(T,freq(rho2,2),freq(rho3,3),bose(rho2,3),bose(rho2,3),freqm1(rho2,2),freqm1(rho3,3)) &
                                    *CONJG(D3(nu,rho2,rho3))
              END DO
            END DO 
 
        END DO
    END DO    
    
    
    
    !
end subroutine compute_static_bubble
!
FUNCTION lambda(T,w2,w3,n2,n3,w2m1,w3m1)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    REAL(kind=DP) :: lambda 
    real(kind=DP), intent(in) :: w2,w3,n2,n3,w2m1,w3m1,T
    real(kind=DP) :: bose_P, bose_M, omega_P, omega_M, ctm_P, ctm_M
            !
            bose_P   = 1 + n2 + n3
            omega_P  = w3+w2
            !
            bose_M   = n3-n2
            omega_M  = w3-w2
            !
            IF(ABS(omega_P)>0._dp)THEN
                ctm_P = 2 * bose_P /omega_P
            ELSE
                ctm_P = 0._dp
            ENDIF
              !
            IF(ABS(omega_M)>1.e-5_dp)THEN
                ctm_M =  2 * bose_M /omega_M
            ELSE
                IF(T>0._dp.and.ABS(omega_P)>0._dp)THEN
                  ctm_M =  2* df_bose(0.5_dp * omega_P, T)
                ELSE
                  ctm_M = 0._dp
                ENDIF
            ENDIF
            !
            lambda=-(ctm_P - ctm_M) * w2m1*w3m1/8.0_dp
            !
end function lambda

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


end module third_order_bubble