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
            Lambda_23=Lambda(T,q2(rho2,:),q3(rho3,:))
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
FUNCTION Lambda(T,w_q2,w_q3)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    REAL(kind=DP) :: lambda 
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
            lambda=-(ctm_P - ctm_M) * w2m1*w3m1/4.0_dp
            !
end function Lambda

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
