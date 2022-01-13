module third_order_cond

    contains

    subroutine interpol_v2(fc,R2,R3,q2,q3,fc_interp,n_blocks,nat)
        IMPLICIT NONE
    !
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        INTEGER, intent(IN) :: nat, n_blocks
        REAL(DP), intent(IN) :: R2(3,n_blocks),R3(3,n_blocks)
        REAL(DP),INTENT(in)   :: fc(n_blocks,3*nat,3*nat,3*nat)
        REAL(DP),INTENT(in) :: q2(3), q3(3)
        COMPLEX(DP),INTENT(out) :: fc_interp(3*nat, 3*nat, 3*nat)
    !
        REAL(DP), parameter :: tpi=3.14159265358979323846_DP*2.0_DP
        REAL(DP) :: arg
        COMPLEX(DP) :: phase
        INTEGER :: i_block, a,b,c
    !
        fc_interp = (0._dp, 0._dp)
    !

        DO i_block = 1, n_blocks
            arg = tpi * SUM(q2(:)*R2(:,i_block) + q3(:)*R3(:,i_block))
            phase = CMPLX(Cos(arg),Sin(arg), kind=DP)
      !
            DO c = 1,3*nat
            DO b = 1,3*nat
            DO a = 1,3*nat
                fc_interp(a,b,c) = fc_interp(a,b,c) + phase*fc(i_block,a,b,c)
            ENDDO
            ENDDO
            ENDDO
      ! 
        END DO

    end subroutine interpol_v2

    subroutine compute_diag_dynamic_bubble_single(energies,sigma,T,freq,is_gamma,D3,ne,n_mod,bubble)
        implicit none
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    
        complex(kind=DP), dimension(ne, n_mod),intent(OUT) :: bubble
    
        real(kind=DP), intent(IN) :: energies(ne)    
        real(kind=DP), intent(IN) :: sigma(n_mod)
        real(kind=DP), intent(IN) :: T
        real(kind=DP), intent(IN) :: freq(n_mod,3)
        logical      , intent(IN) :: is_gamma(3)
        complex(kind=DP), dimension(n_mod,n_mod,n_mod), intent(IN) :: D3
        integer, intent(IN) :: n_mod, ne
    
        real(kind=DP)    :: q2(n_mod,3),q3(n_mod,3)
        complex(kind=DP) :: Lambda_23(ne)   
        integer :: i, rho2, rho3, nu,mu
        logical, parameter :: static_limit = .false.
    
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

    
        bubble=(0.0_dp,0.0_dp)
    
        DO rho3=1,n_mod
        DO rho2=1,n_mod
            !
            ! call Lambda_dynamic_single(ne,n_mod,energies,sigma,T,static_limit,q2(rho2,:),q3(rho3,:),Lambda_23)           
            !
            DO mu = 1,n_mod
                   !
                   call Lambda_dynamic_single(ne,energies,sigma(mu),T,static_limit,q2(rho2,:),q3(rho3,:),Lambda_23)           
                   !
                   ! 
                   bubble(:,mu) = bubble(:,mu) +  & 
                                     CONJG(D3(mu,rho2,rho3))*Lambda_23*D3(mu,rho2,rho3)
                   !
            END DO            
            !
        END DO
        END DO    
    !
    end subroutine compute_diag_dynamic_bubble_single

    subroutine compute_perturb_selfnrg_single(sigma,T,freq,is_gamma,D3,n_mod,selfnrg)

        implicit none
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    
        complex(kind=DP), dimension(n_mod),intent(OUT)     :: selfnrg
    
        real(kind=DP), intent(IN) :: sigma(n_mod)
        real(kind=DP), intent(IN) :: T
        real(kind=DP), intent(IN) :: freq(n_mod,3)
        logical      , intent(IN) :: is_gamma(3)
        complex(kind=DP), dimension(n_mod,n_mod,n_mod), intent(IN) :: D3
        integer, intent(IN) :: n_mod
    
        real(kind=DP)    :: q2(n_mod,3),q3(n_mod,3)
        complex(kind=DP) :: Lambda_23_freq
        integer :: i, rho2, rho3, nu,mu
    
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

    
        selfnrg=(0.0_dp,0.0_dp)
    
        DO mu = 1,n_mod
        DO rho3=1,n_mod
        DO rho2=1,n_mod
            !if(abs(freq(mu, 1)**2 - (q2(rho2,1) + q3(rho3,1)**2)) < (3.0_DP*sigma(mu))**2 .or. &
            !        abs(freq(mu, 1)**2 - (q2(rho2,1) - q3(rho3,1)**2)) < (3.0_DP*sigma(mu))**2) then
            
            call Lambda_dynamic_value_single(n_mod,freq(mu,1),sigma(mu),T,q2(rho2,:),q3(rho3,:),Lambda_23_freq) 
            !
            
            selfnrg(mu)  = selfnrg(mu) + CONJG(D3(mu,rho2,rho3))*Lambda_23_freq*D3(mu,rho2,rho3)
            !
            !endif
        END DO
        END DO
        END DO    
    !
    end subroutine compute_perturb_selfnrg_single

    subroutine compute_spectralf_diag_single(sigma,ener,d2_freq,selfnrg,nat,ne,spectralf)
        implicit none
        INTEGER, PARAMETER           :: DP = selected_real_kind(14,200)
        real(kind=dp),parameter      :: pi = 3.141592653589793_dp
    !
        real(kind=dp), intent(in)    :: ener(ne)
        real(kind=dp), intent(in)    :: sigma(3*nat)    
        integer, intent(in)          :: ne,nat
        real(kind=dp), intent(in)    :: d2_freq(3*nat)
        complex(kind=dp), intent(in) :: selfnrg(ne,3*nat)
    !
        real(kind=dp), intent(out)   :: spectralf(ne,3*nat)
    !
        integer                      :: nat3,mu,ie
        real(kind=dp)                :: a,b,denom,num
    !
        nat3=3*nat
        spectralf=0.0_dp
    !
        DO mu = 1,nat3
            DO ie = 1, ne

                a = ener(ie)**2-sigma(mu)**2-d2_freq(mu)**2-DBLE(selfnrg(ie,mu))
                b = 2*sigma(mu)*ener(ie)-DIMAG(selfnrg(ie,mu))          
          
                num   = ener(ie)*b
                denom = (a**2+b**2)*pi
          !
                IF(ABS(denom)/=0._dp)THEN
                    spectralf(ie,mu) = num / denom
                ELSE
                    spectralf(ie,mu) = 0._dp
                ENDIF
            ENDDO
        ENDDO
    !
    end subroutine compute_spectralf_diag_single

    subroutine Lambda_dynamic_single(ne,energies,sigma,T,static_limit,w_q2,w_q3,Lambda_out)
        implicit none
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        complex(kind=DP), intent(out) :: Lambda_out(ne)    
        integer, intent(in)       :: ne
        logical, intent(in)       :: static_limit
        real(kind=DP), intent(in) :: energies(ne), sigma    
        real(kind=DP), intent(in) :: T,w_q2(3),w_q3(3)
        real(kind=DP) :: w2,w3,n2,n3,w2m1,w3m1
        real(kind=DP) :: bose_P, bose_M, omega_P, omega_P2 ,&
                     omega_M,omega_M2
        complex(kind=DP) :: reg, ctm_P, ctm_M, ctm(ne)
        integer       :: ie
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
            DO ie = 1,ne
                reg = CMPLX(energies(ie), sigma, kind=DP)**2
                ctm_P = bose_P *omega_P/(omega_P2-reg)
                ctm_M = bose_M *omega_M/(omega_M2-reg)
                ctm(ie) = ctm_P - ctm_M
            END DO              
        END IF
            !
        lambda_out=-ctm * w2m1*w3m1/4.0_dp
            !
    end subroutine Lambda_dynamic_single

    subroutine Lambda_dynamic_value_single(n_mod,value,sigma,T,w_q2,w_q3,Lambda_out)
        implicit none
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        complex(kind=DP), intent(out) :: Lambda_out 
        integer, intent(in)       :: n_mod
        real(kind=DP), intent(in) :: sigma, value    
        real(kind=DP), intent(in) :: T,w_q2(3),w_q3(3)
        real(kind=DP) :: w2,w3,n2,n3,w2m1,w3m1
        real(kind=DP) :: bose_P, bose_M, omega_P, omega_P2 ,&
                     omega_M,omega_M2
        complex(kind=DP) :: reg, ctm_P, ctm_M, ctm
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
        reg = CMPLX(value, sigma, kind=DP)**2
        ctm_P = bose_P *omega_P/(omega_P2-reg)
        ctm_M = bose_M *omega_M/(omega_M2-reg)
        ctm = ctm_P - ctm_M
            !
        lambda_out=-ctm * w2m1*w3m1/4.0_dp
            !
    end subroutine Lambda_dynamic_value_single
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

end module
