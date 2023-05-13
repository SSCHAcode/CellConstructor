module third_order_cond

    contains

    subroutine interpol_v2(fc,R2,R3,pos,q2,q3,fc_interp,n_blocks,nat)
        IMPLICIT NONE
    !
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        INTEGER, intent(IN) :: nat, n_blocks
        REAL(DP), intent(IN) :: R2(3,n_blocks),R3(3,n_blocks),pos(3,nat)
        REAL(DP),INTENT(in)   :: fc(n_blocks,3*nat,3*nat,3*nat)
        REAL(DP),INTENT(in) :: q2(3), q3(3)
        COMPLEX(DP),INTENT(out) :: fc_interp(3*nat, 3*nat, 3*nat)
    !
        REAL(DP), parameter :: tpi=3.14159265358979323846_DP*2.0_DP
        REAL(DP) :: arg, arg1
        COMPLEX(DP) :: phase, phase1
        INTEGER :: i_block, a,b,c
    !
        fc_interp = cmplx(0._dp, 0._dp, kind=DP)
    !

        DO i_block = 1, n_blocks
            arg = tpi*(dot_product(q2, R2(:,i_block)) + dot_product(q3, R3(:,i_block)))
            phase = exp(cmplx(0.0_DP, arg,kind=DP))

             fc_interp = fc_interp + phase*fc(i_block,:,:,:)
      ! 
        END DO

    end subroutine interpol_v2

    subroutine interpol_v3(fc,pos,R2,R3,q1,q2,q3,fc_interp,n_blocks,nat)
        IMPLICIT NONE
    !
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        INTEGER, intent(IN) :: nat, n_blocks
        REAL(DP), intent(IN) :: R2(3,n_blocks),R3(3,n_blocks), pos(3,nat)
        REAL(DP),INTENT(in)   :: fc(n_blocks,3*nat,3*nat,3*nat)
        REAL(DP),INTENT(in) :: q2(3), q3(3), q1(3)
        COMPLEX(DP),INTENT(out) :: fc_interp(3*nat, 3*nat, 3*nat)
    !
        REAL(DP), parameter :: tpi=3.14159265358979323846_DP*2.0_DP
        REAL(DP) :: arg, arg2
        COMPLEX(DP) :: phase, extra_phase
        INTEGER :: i_block, a,b,c, at1, at2, at3
    !
        fc_interp = (0._dp, 0._dp)
    !

        DO i_block = 1, n_blocks
            arg = tpi * SUM(q2(:)*R2(:,i_block) + q3(:)*R3(:,i_block))
      !
            DO c = 1,3*nat
            DO b = 1,3*nat
            DO a = 1,3*nat
                at1 = ceiling(dble(a)/3.0_DP)
                at2 = ceiling(dble(b)/3.0_DP)
                at3 = ceiling(dble(c)/3.0_DP)
                arg2 = tpi * (dot_product(pos(:,at1), q1) + dot_product(pos(:,at2), q2) + dot_product(pos(:,at3), q3))
                phase = CMPLX(Cos(arg2 + arg),Sin(arg2 + arg), kind=DP)
                fc_interp(a,b,c) = fc_interp(a,b,c) + phase*fc(i_block,a,b,c)
            ENDDO
            ENDDO
            ENDDO
      ! 
        END DO

    end subroutine interpol_v3

    subroutine compute_full_dynamic_bubble_single(energies,sigma,T,freq,is_gamma,D3,ne,n_mod, gaussian, &
                    classical, bubble)

            implicit none
            INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
            !
            complex(kind=DP), dimension(ne,n_mod,n_mod),intent(OUT) :: bubble
            !
            integer, intent(IN)       :: ne
            real(kind=DP), intent(IN) :: energies(ne)    
            real(kind=DP), intent(IN) :: sigma(n_mod)
            real(kind=DP), intent(IN) :: T
            real(kind=DP), intent(IN) :: freq(n_mod,3)
            logical      , intent(IN) :: gaussian
            logical      , intent(IN) :: is_gamma(3)
            logical      , intent(IN) :: classical
            complex(kind=DP), dimension(n_mod,n_mod,n_mod), intent(IN) :: D3
            integer, intent(IN) :: n_mod
            !
            real(kind=DP)    :: q2(n_mod,3),q3(n_mod,3), curr_sigma
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

            if(classical) then
                call eq_freq(T, n_mod, freq(:,2), q2(:,3))
                call eq_freq(T, n_mod, freq(:,3), q3(:,3))
            else
                call bose_freq(T, n_mod, freq(:,2), q2(:,3))
                call bose_freq(T, n_mod, freq(:,3), q3(:,3))
            endif
            !
            bubble=cmplx(0.0_dp,0.0_dp,kind=DP)
            !
            DO rho3=1,n_mod
            DO rho2=1,n_mod
                    !
                    curr_sigma = (sigma(rho2) + sigma(rho3))/2.0_DP
                    call Lambda_dynamic_single(ne,energies,curr_sigma,T,static_limit,q2(rho2,:),q3(rho3,:), gaussian, Lambda_23)           
                    !
                    DO nu = 1,n_mod
                    DO mu = 1,n_mod
                            bubble(:,mu,nu) = bubble(:,mu,nu) +  & 
                                                CONJG(D3(mu,rho2,rho3))*Lambda_23(:)*D3(nu,rho2,rho3)
                    END DO
                    END DO
                    !
            END DO
            END DO   
        end subroutine compute_full_dynamic_bubble_single

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine compute_diag_dynamic_bubble_single(energies,sigma,T,freq,is_gamma,D3,ne,n_mod, gaussian, &
                   classical, bubble)
        implicit none
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    
        complex(kind=DP), dimension(ne, n_mod),intent(OUT) :: bubble
    
        real(kind=DP), intent(IN) :: energies(ne)    
        real(kind=DP), intent(IN) :: sigma(n_mod)
        real(kind=DP), intent(IN) :: T
        real(kind=DP), intent(IN) :: freq(n_mod,3)
        logical      , intent(IN) :: is_gamma(3), gaussian, classical
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

        if(classical) then
                call eq_freq(T, n_mod, freq(:,2), q2(:,3))
                call eq_freq(T, n_mod, freq(:,3), q3(:,3))
        else
                call bose_freq(T, n_mod, freq(:,2), q2(:,3))
                call bose_freq(T, n_mod, freq(:,3), q3(:,3))
        endif
    
        bubble=CMPLX(0.0_dp,0.0_dp, kind=DP)
    
        DO rho3=1,n_mod
        DO rho2=1,n_mod
            !
            ! call Lambda_dynamic_single(ne,n_mod,energies,sigma,T,static_limit,q2(rho2,:),q3(rho3,:),Lambda_23)           
            !
            DO mu = 1,n_mod
                   !
                   call Lambda_dynamic_single(ne,energies,sigma(mu),T,static_limit,q2(rho2,:),q3(rho3,:), gaussian, Lambda_23)           
                   !
                   ! 
                   bubble(:,mu) = bubble(:,mu) +  & 
                                     CONJG(D3(mu,rho2,rho3))*Lambda_23*D3(mu,rho2,rho3)
                   !
                   if(any(bubble(:,mu) .ne. bubble(:,mu))) then
                           if(any(D3 .ne. D3)) then
                                   print*, 'Its D3'
                            else
                                    print*, 'Its Lambda_23'
                            endif
                           STOP
                   endif
            END DO            
            !
        END DO
        END DO    
    !
    end subroutine compute_diag_dynamic_bubble_single

    subroutine compute_perturb_selfnrg_single(sigma,T,freq,is_gamma,D3,n_mod, gaussian, classical, selfnrg)

        implicit none
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    
        complex(kind=DP), dimension(n_mod),intent(OUT)     :: selfnrg
    
        real(kind=DP), intent(IN) :: sigma(n_mod)
        real(kind=DP), intent(IN) :: T
        real(kind=DP), intent(IN) :: freq(n_mod,3)
        logical      , intent(IN) :: is_gamma(3), gaussian, classical
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

        if(classical) then
                call eq_freq(T, n_mod, freq(:,2), q2(:,3))
                call eq_freq(T, n_mod, freq(:,3), q3(:,3))
        else
                call bose_freq(T, n_mod, freq(:,2), q2(:,3))
                call bose_freq(T, n_mod, freq(:,3), q3(:,3))
        endif

    
        selfnrg=CMPLX(0.0_dp,0.0_dp,kind=DP)
    
        DO mu = 1,n_mod
        DO rho3=1,n_mod
        DO rho2=1,n_mod
            !if(abs(freq(mu, 1)**2 - (q2(rho2,1) + q3(rho3,1)**2)) < (3.0_DP*sigma(mu))**2 .or. &
            !        abs(freq(mu, 1)**2 - (q2(rho2,1) - q3(rho3,1)**2)) < (3.0_DP*sigma(mu))**2) then
            
            call Lambda_dynamic_value_single(n_mod,freq(mu,1),sigma(mu),T,q2(rho2,:),q3(rho3,:), gaussian, Lambda_23_freq) 
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

    subroutine Lambda_dynamic_single(ne,energies,sigma,T,static_limit,w_q2,w_q3,gaussian,Lambda_out)
        implicit none
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        complex(kind=DP), intent(out) :: Lambda_out(ne)    
        integer, intent(in)       :: ne
        logical, intent(in)       :: static_limit, gaussian
        real(kind=DP), intent(in) :: energies(ne), sigma    
        real(kind=DP), intent(in) :: T,w_q2(3),w_q3(3)
        real(kind=DP) :: w2,w3,n2,n3,w2m1,w3m1
        real(kind=DP) :: bose_P, bose_M, omega_P, omega_P2 ,&
                     omega_M,omega_M2, re_p, im_p, re_p1, im_p1
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
        bose_P    = 1.0_DP + n2 + n3
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
            IF(gaussian) then
                DO ie = 1,ne
                        im_p = bose_P *gaussian_function(energies(ie) - omega_P, sigma)
                        if(energies(ie) - omega_P .ne. 0.0_DP) then
                                re_p = bose_P/(energies(ie) - omega_P) 

                        else
                                re_p = 0.0_DP
                        endif
                        ctm_P = CMPLX(re_p, im_p, kind=DP)
                        im_p = bose_M *gaussian_function(energies(ie) + omega_M, sigma)
                        im_p1 = bose_M *gaussian_function(energies(ie) - omega_M, sigma)
                        if(energies(ie) + omega_M .ne. 0.0_DP) then
                                re_p = bose_M/(energies(ie) + omega_M)
                        else
                                re_p = 0.0_DP 
                        endif
                        if(energies(ie) - omega_M .ne. 0.0_DP) then
                                re_p1 = bose_M/(energies(ie) - omega_M)
                        else
                                re_p1 = 0.0_DP 
                        endif
                        ctm_M = CMPLX(re_p, im_p, kind=DP) - CMPLX(re_p1, im_p1, kind=DP)
                        ctm(ie) =  ctm_P + ctm_M
                ENDDO
            ELSE
                DO ie = 1,ne
                        reg = CMPLX(energies(ie), sigma, kind=DP)**2
                        ctm_P = bose_P *omega_P/(omega_P2-reg)
                        ctm_M = bose_M *omega_M/(omega_M2-reg)
                        ctm(ie) = ctm_P - ctm_M
                END DO      
            ENDIF        
        END IF
            !
        IF(gaussian) then
                !lambda_out=-ctm/16.0_DP*sqrt(w2m1*w3m1)
                lambda_out=-ctm * w2m1*w3m1/8.0_dp
        ELSE
                lambda_out=-ctm * w2m1*w3m1/4.0_dp
        ENDIF
            !
    end subroutine Lambda_dynamic_single

    subroutine Lambda_dynamic_value_single(n_mod,value,sigma,T,w_q2,w_q3,gaussian,Lambda_out)
        implicit none
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        complex(kind=DP), intent(out) :: Lambda_out 
        integer, intent(in)       :: n_mod
        logical, intent(in)       :: gaussian
        real(kind=DP), intent(in) :: sigma, value    
        real(kind=DP), intent(in) :: T,w_q2(3),w_q3(3)
        real(kind=DP) :: w2,w3,n2,n3,w2m1,w3m1
        real(kind=DP) :: bose_P, bose_M, omega_P, omega_P2 ,&
                     omega_M,omega_M2, re_p, im_p, re_p1, im_p1
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
        bose_P    = 1.0_DP + n2 + n3
        omega_P   = w3+w2
        omega_P2  = (omega_P)**2
            !
        bose_M    = n3-n2
        omega_M   = w3-w2
        omega_M2  = (omega_M)**2
            !
        if(gaussian) then
                im_p = bose_P *gaussian_function(value - omega_P, sigma)
                if(omega_P2-value**2 .ne. 0.0_DP) then
                        re_p = bose_P/(value - omega_P)
                else
                        re_p = 0.0_DP
                endif
                ctm_P = CMPLX(re_p, im_p, kind=DP)

                im_p = bose_M *gaussian_function(value + omega_M, sigma)
                im_p1 = bose_M *gaussian_function(value - omega_M, sigma)
                if(value + omega_M .ne. 0.0_DP) then
                        re_p = bose_M/(value + omega_M)
                else
                        re_p = 0.0_DP
                endif
                if(value - omega_M .ne. 0.0_DP) then
                        re_p1 = bose_M/(value - omega_M)
                else
                        re_p1 = 0.0_DP
                endif 
                ctm_M = CMPLX(re_p, im_p, kind=DP) - CMPLX(re_p1, im_p1, kind=DP)
        else
                reg = CMPLX(value, sigma, kind=DP)**2
                ctm_P = bose_P *omega_P/(omega_P2-reg)
                ctm_M = bose_M *omega_M/(omega_M2-reg)
        endif
            !
        IF(gaussian) then
                ctm = ctm_P + ctm_M
                lambda_out=-ctm/8.0_DP* w2m1*w3m1
                !lambda_out=-ctm * w2m1*w3m1/4.0_dp
        ELSE
                ctm = ctm_P - ctm_M
                lambda_out=-ctm * w2m1*w3m1/4.0_dp
        ENDIF
            !
    end subroutine Lambda_dynamic_value_single

    subroutine calculate_spectral_function(ener, d2_freq, selfnrg, nat, ne, spectralf)

        implicit none
        INTEGER, PARAMETER           :: DP = selected_real_kind(14,200)
        real(kind=dp),parameter      :: pi = 3.141592653589793_dp
    !
        real(kind=dp), intent(in)    :: ener(ne)
        integer, intent(in)          :: ne,nat
        real(kind=dp), intent(in)    :: d2_freq(3*nat)
        complex(kind=dp), intent(in) :: selfnrg(ne,3*nat)
    !
        real(kind=dp), intent(out)   :: spectralf(ne,3*nat)
    !
        integer                      :: nat3,mu,ie, iband
        real(kind=dp)                :: a,b
        complex(kind=dp), dimension(ne, 3*nat) :: zq

        do iband = 1, 3*nat
                zq(:,iband) = sqrt(d2_freq(iband)**2 + selfnrg(:,iband))
                do ie = 1, ne
                        a = 0.0_DP
                        b = 0.0_DP
                        if(((ener(ie) - dble(zq(ie, iband)))**2 + aimag(zq(ie,iband))**2) .ne. 0.0_DP) then
                                a = -1.0_DP*aimag(zq(ie,iband))/((ener(ie) - dble(zq(ie, iband)))**2 + aimag(zq(ie,iband))**2)
                        endif
                        if(((ener(ie) + dble(zq(ie, iband)))**2 + aimag(zq(ie,iband))**2) .ne. 0.0_DP) then
                                b = aimag(zq(ie,iband))/((ener(ie) + dble(zq(ie, iband)))**2 + aimag(zq(ie,iband))**2)
                        endif
                        spectralf(ie, iband) = (a + b)/2.0_DP/pi
                enddo
        enddo

    end subroutine calculate_spectral_function

    subroutine calculate_spectral_function_mode_mixing(ener,smear,wq,Pi,notransl,spectralf,mass,nat,ne)
            implicit none
            INTEGER, PARAMETER           :: DP = selected_real_kind(14,200)
            real(kind=dp),parameter      :: twopi = 6.283185307179586_dp
            !
            real(kind=dp), intent(in)    :: mass(nat), ener(ne), smear(3*nat)
            integer, intent(in)          :: ne,nat
            real(kind=dp), intent(in)    :: wq(3*nat)
            complex(kind=dp), intent(in) :: Pi(ne,3*nat,3*nat)
            logical, intent(in)          :: notransl
            !
            complex(kind=dp), intent(out)   :: spectralf(3*nat, 3*nat, ne)
            !
            integer                      :: nat3,n,m,ie
            complex(kind=dp)             :: G(3*nat,3*nat) 
            complex(kind=dp)             :: fact
            
            nat3=3*nat
                
            spectralf=complex(0.0_DP, 0.0_DP)
            !   
            
            DO ie = 1,ne
                 G=cmplx(0.0_dp,0.0_dp,kind=DP)
                 FORALL (m=1:nat3, n=1:nat3)
                     G(n,m) = -Pi(ie,n,m)
                 END FORALL
                 DO n=1,nat3
                   G(n,n)=G(n,n)+(ener(ie) + complex(0.0_DP,smear(n)))**2 - wq(n)**2
                 ENDDO
                 G = cinv(G) 
                 IF ( notransl ) THEN
                   CALL eliminate_transl(G,mass,nat)      
                 END IF
                 do n = 1, nat3
                 do m = 1, nat3       
                 !spectralf(m,n,ie)=spectralf(m,n,ie)-2.0_DP*DIMAG(G(m,n))*ener(ie)/twopi
                 !spectralf(m,n,ie)=spectralf(m,n,ie)-DIMAG(G(m,n) - conjg(G(n,m)))*ener(ie)/twopi
                 spectralf(m,n,ie)=spectralf(m,n,ie)+complex(0.0_DP, 1.0_DP)*(G(m,n) - conjg(G(n,m)))*ener(ie)/twopi
                 enddo
                 enddo
            ENDDO
            if(all(spectralf == complex(0.0_DP, 0.0_DP))) then
                   print*, 'All of the spectralf is 0!' 
            endif
    end subroutine calculate_spectral_function_mode_mixing

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine calculate_spectral_function_cartesian(ener,smear,d2,Pi,notransl,spectralf,mass,nat,ne)
            implicit none
            INTEGER, PARAMETER           :: DP = selected_real_kind(14,200)
            real(kind=dp),parameter      :: twopi = 6.283185307179586_dp
            !
            real(kind=dp), intent(in)    :: mass(nat), ener(ne), smear(3*nat)
            integer, intent(in)          :: ne,nat
            complex(kind=dp), intent(in) :: d2(3*nat,3*nat)
            complex(kind=dp), intent(in) :: Pi(ne,3*nat,3*nat)
            logical, intent(in)          :: notransl
            !
            complex(kind=dp), intent(out)   :: spectralf(3*nat, 3*nat, ne)
            !
            integer                      :: nat3,n,m,ie
            complex(kind=dp)             :: G(3*nat,3*nat) 
            complex(kind=dp)             :: fact
            
            nat3=3*nat
                
            spectralf=0.0_dp
            !   
            
            DO ie = 1,ne
                 G=cmplx(0.0_dp,0.0_dp,kind=DP)
                 FORALL (m=1:nat3, n=1:nat3)
                     G(n,m) = -Pi(ie,n,m)
                 END FORALL
                 G=G-d2
                 DO n=1,nat3
                   G(n,n)=G(n,n)+(ener(ie) + complex(0.0_DP,smear(n)))**2
                 ENDDO
                 G = cinv(G) 
                 IF ( notransl ) THEN
                   CALL eliminate_transl(G,mass,nat)      
                 END IF
                 do n = 1, nat3
                 do m = 1, nat3       
                 spectralf(m,n,ie)=spectralf(m,n,ie)+complex(0.0_DP, 1.0_DP)*(G(m,n) - conjg(G(n,m)))*ener(ie)/twopi
                 enddo
                 enddo
            ENDDO
            
    end subroutine calculate_spectral_function_cartesian
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
    SUBROUTINE eq_freq(T, n_mod, freq, bose)
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        REAL(DP), parameter :: K_BOLTZMANN_RY= 1.3806504E-23_DP /(4.35974394E-18_DP/2) !K_BOLTZMANN_SI / (HARTREE_SI/2)
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
            bose = (T*K_BOLTZMANN_RY)/freq
        ELSEWHERE
            bose = 0._dp
        ENDWHERE
    !
    END SUBROUTINE eq_freq
!
    ELEMENTAL FUNCTION f_bose(freq,T) ! bose (freq,T)
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        REAL(DP), parameter :: K_BOLTZMANN_RY= 1.3806504E-23_DP /(4.35974394E-18_DP/2) !K_BOLTZMANN_SI / (HARTREE_SI/2)
    !
        REAL(DP) :: f_bose
    !
        REAL(DP),INTENT(in) :: freq,T
    !
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

    FUNCTION gaussian_function(x,sigma)
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        REAL(DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875_DP 
    !
        REAL(DP) :: gaussian_function
        REAL(DP), intent(in) :: x, sigma

        gaussian_function = exp(-0.5_DP*(x/sigma)**2)/sqrt(2.0_DP/PI)/sigma  ! multiplied with pi

    END FUNCTION
  
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
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
    function cinv(A) result(Ainv)
 
        implicit none
        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        complex(kind=DP),intent(in) :: A(:,:)
        complex(kind=DP)            :: Ainv(size(A,1),size(A,2))
        complex(kind=DP), allocatable            :: work(:)            ! work array for LAPACK
        integer         :: n,info,ipiv(size(A,1)), lda, lwork, nb     ! pivot indices
        INTEGER,EXTERNAL :: ILAENV
 
        ! Store A in Ainv to prevent it from being overwritten by LAPACK
        Ainv = A
        n = size(A,1)
        lda = n
    !
        nb = ILAENV( 1, 'ZHEEV', 'U', n, -1, -1, -1 )
        lwork=n*nb
        ALLOCATE(work(lwork))
    !
        ! ZGETRF computes an LU factorization of a general M-by-N matrix A
        ! using partial pivoting with row interchanges.
        CALL ZGETRF(n, n, Ainv, lda, ipiv, info)
        if (info.ne.0) stop 'Matrix is numerically singular!'
        ! ZGETRI computes the inverse of a matrix using the LU factorization
        ! computed by zGETRF.
        CALL ZGETRI(n, Ainv, lda, ipiv, work, lwork, info)
        if (info.ne.0) stop 'Matrix inversion failed!'

        !call ZGETRF(n,n,Ainv,n,ipiv,info)
        !if (info.ne.0) stop 'Matrix is numerically singular!'
        ! ZGETRI computes the inverse of a matrix using the LU factorization
        ! computed by zGETRF.
        !call ZGETRI(n,Ainv,n,ipiv,work,n,info)
        !if (info.ne.0) stop 'Matrix inversion failed!'
    end function cinv

end module
