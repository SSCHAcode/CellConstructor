module get_lf

    contains

    subroutine calculate_lineshapes(irrqgrid, qgrid, weights, scatt_events, fc2, r2_2, &
                    fc3, r3_2, r3_3, rprim, pos, masses, smear, smear_id, T, &
                    energies, ne, nirrqpt, nqpt, nat, nfc2, nfc3, n_events, lineshapes)

        use omp_lib
        use third_order_cond

        implicit none

        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        integer, intent(in) :: nirrqpt, nat, nqpt, nfc2, nfc3, ne, n_events
        integer, intent(in) :: weights(n_events), scatt_events(nirrqpt)
        real(kind=DP), intent(in) :: irrqgrid(3, nirrqpt)
        real(kind=DP), intent(in) :: qgrid(3, n_events)
        real(kind=DP), intent(in) :: fc2(nfc2, 3*nat, 3*nat)
        real(kind=DP), intent(in) :: fc3(nfc3, 3*nat, 3*nat, 3*nat)
        real(kind=DP), intent(in) :: r2_2(3, nfc2)
        real(kind=DP), intent(in) :: r3_2(3, nfc3), r3_3(3, nfc3)
        real(kind=DP), intent(in) :: rprim(3, 3)
        real(kind=DP), intent(in) :: pos(3, nat)
        real(kind=DP), intent(in) :: masses(nat), energies(ne)
        real(kind=DP), intent(in) :: smear(3*nat, nqpt), smear_id(3*nat, nqpt)
        real(kind=DP), intent(in) :: T
        real(kind=DP), dimension(nirrqpt, 3*nat, ne), intent(out) :: lineshapes

        integer :: iqpt, i, jqpt, tot_qpt, prev_events, nthreads
        real(kind=DP), dimension(3) :: qpt
        real(kind=DP), dimension(3,3) :: kprim
        real(kind=DP), dimension(3*nat) :: w2_q, w_q
        real(kind=DP), dimension(ne, 3*nat) :: lineshape
        complex(kind=DP), dimension(ne, 3*nat) :: self_energy
        complex(kind=DP), dimension(3*nat,3*nat) :: pols_q
        real(kind=DP), allocatable, dimension(:,:) :: curr_grid
        integer, allocatable, dimension(:) :: curr_w
        logical :: is_q_gamma, w_neg_freqs, parallelize


        lineshapes(:,:,:) = 0.0_DP
        kprim = transpose(inv(rprim))
        nthreads = omp_get_max_threads()
        print*, 'Maximum number of threads available: ', nthreads

        parallelize = .True.
        if(nirrqpt <= nthreads) then
                parallelize = .False.
        endif

        !$OMP PARALLEL DO IF(parallelize) &
        !$OMP DEFAULT(SHARED) &
        !$OMP PRIVATE(iqpt, w_neg_freqs, qpt, w2_q, pols_q, is_q_gamma, self_energy, &
        !$OMP curr_grid, w_q, curr_w, prev_events, tot_qpt, lineshape)
        do iqpt = 1, nirrqpt

            w_neg_freqs = .False.
            print*, 'Calculating ', iqpt, ' point in the irreducible zone out of ', nirrqpt, '!'
            lineshape(:,:) = 0.0_DP 
            qpt = irrqgrid(:, iqpt) 
            call interpolate_fc2(nfc2, nat, fc2, r2_2, masses, qpt, w2_q, pols_q)
            call check_if_gamma(nat, qpt, kprim, w2_q, is_q_gamma)

            if(any(w2_q < 0.0_DP)) then
                print*, 'Negative eigenvalue of dynamical matrix!'
                w_neg_freqs = .True.
            endif

            if(.not. w_neg_freqs) then
                w_q = sqrt(w2_q)
                self_energy(:,:) = complex(0.0_DP, 0.0_DP)
                allocate(curr_grid(3, scatt_events(iqpt)))
                allocate(curr_w(scatt_events(iqpt)))
                if(iqpt > 1) then
                    prev_events = sum(scatt_events(1:iqpt-1))
                else
                    prev_events = 0
                endif
                do jqpt = 1, scatt_events(iqpt)
                    curr_grid(:,jqpt) = qgrid(:,prev_events + jqpt)
                    curr_w(jqpt) = weights(prev_events + jqpt)
                enddo
                call calculate_self_energy_P(w_q, qpt, pols_q, is_q_gamma, scatt_events(iqpt), nat, nfc2, &
                    nfc3, ne, curr_grid, curr_w, fc2, fc3, &
                    r2_2, r3_2, r3_3, kprim, masses, smear, T, energies, .not. parallelize, self_energy) 
                deallocate(curr_grid)
                tot_qpt = sum(curr_w)
                deallocate(curr_w)
                self_energy = self_energy/dble(tot_qpt)

                call compute_spectralf_diag_single(smear_id(:, iqpt), energies, w_q, self_energy, nat, ne, lineshape)

                do i = 1, 3*nat
                    if(w_q(i) .ne. 0.0_DP) then
                        lineshapes(iqpt, i, :) = lineshape(:,i) 
                    else
                        lineshapes(iqpt, i, :) = 0.0_DP
                    endif
                enddo
            else
                lineshapes(iqpt,:,:) = 0.0_DP
            endif

        enddo
        !$OMP END PARALLEL DO


    end subroutine calculate_lineshapes

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine calculate_lifetimes_perturbative(irrqgrid, qgrid, weights, scatt_events, fc2, r2_2, &
                    fc3, r3_2, r3_3, rprim, pos, masses, smear, T, &
                    nirrqpt, nqpt, nat, nfc2, nfc3, n_events, self_energies)

        use omp_lib
        implicit none

        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        integer, intent(in) :: nirrqpt, nat, nqpt, nfc2, nfc3, n_events
        integer, intent(in) :: weights(n_events), scatt_events(nirrqpt)
        real(kind=DP), intent(in) :: irrqgrid(3, nirrqpt)
        real(kind=DP), intent(in) :: qgrid(3, n_events)
        real(kind=DP), intent(in) :: fc2(nfc2, 3*nat, 3*nat)
        real(kind=DP), intent(in) :: fc3(nfc3, 3*nat, 3*nat, 3*nat)
        real(kind=DP), intent(in) :: r2_2(3, nfc2)
        real(kind=DP), intent(in) :: r3_2(3, nfc3), r3_3(3, nfc3)
        real(kind=DP), intent(in) :: rprim(3, 3)
        real(kind=DP), intent(in) :: pos(3, nat)
        real(kind=DP), intent(in) :: masses(nat)
        real(kind=DP), intent(in) :: smear(3*nat, nqpt)
        real(kind=DP), intent(in) :: T
        complex(kind=DP), dimension(nirrqpt, 3*nat), intent(out) :: self_energies

        integer :: iqpt, i, tot_qpt, jqpt, prev_events, nthreads
        real(kind=DP), dimension(3) :: qpt
        real(kind=DP), dimension(3,3) :: kprim
        real(kind=DP), dimension(3*nat) :: w2_q, w_q
        complex(kind=DP), dimension(3*nat, 3*nat) :: self_energy
        complex(kind=DP), dimension(3*nat,3*nat) :: pols_q
        real(kind=DP), allocatable, dimension(:,:) :: curr_grid
        integer, allocatable, dimension(:) :: curr_w
        logical :: is_q_gamma, w_neg_freqs, parallelize


        self_energies(:,:) = 0.0_DP
        kprim = transpose(inv(rprim))
        print*, 'Calculating phonon lifetimes in perturbative regime!'

        nthreads = omp_get_max_threads()
        print*, 'Maximum number of threads available: ', nthreads

        parallelize = .True.
        if(nirrqpt <= nthreads) then
                parallelize = .False.
        endif
        
        !$OMP PARALLEL DO IF(parallelize) &
        !$OMP DEFAULT(SHARED) &
        !$OMP PRIVATE(iqpt, w_neg_freqs, qpt, w2_q, pols_q, is_q_gamma, self_energy, &
        !$OMP curr_grid, w_q, curr_w, prev_events, tot_qpt)
        do iqpt = 1, nirrqpt

            w_neg_freqs = .False.
            print*, 'Calculating ', iqpt, ' point in the irreducible zone out of ', nirrqpt, '!'
            qpt = irrqgrid(:, iqpt) 
            call interpolate_fc2(nfc2, nat, fc2, r2_2, masses, qpt, w2_q, pols_q)
            call check_if_gamma(nat, qpt, kprim, w2_q, is_q_gamma)
            if(any(w2_q < 0.0_DP)) then
                print*, 'Negative eigenvalue of dynamical matrix! Exit!'
                w_neg_freqs = .True.
            endif
            
            if(.not. w_neg_freqs) then 
                w_q = sqrt(w2_q)
                self_energy(:,:) = complex(0.0_DP, 0.0_DP)
                allocate(curr_grid(3, scatt_events(iqpt)))
                allocate(curr_w(scatt_events(iqpt)))
                if(iqpt > 1) then
                    prev_events = sum(scatt_events(1:iqpt-1))
                else
                    prev_events = 0
                endif
                do jqpt = 1, scatt_events(iqpt)
                    curr_grid(:,jqpt) = qgrid(:,prev_events + jqpt)
                    curr_w(jqpt) = weights(prev_events + jqpt)
                enddo
                call calculate_self_energy_P(w_q, qpt, pols_q, is_q_gamma, scatt_events(iqpt), nat, nfc2, &
                            nfc3, 3*nat, curr_grid, curr_w, fc2, fc3, &
                            r2_2, r3_2, r3_3, kprim, masses, smear, T, w_q, .not. parallelize, self_energy) 

                deallocate(curr_grid)
                tot_qpt = sum(curr_w)
                deallocate(curr_w)
                self_energy = self_energy/dble(tot_qpt)

                do i = 1, 3*nat
                    if(w_q(i) .ne. 0.0_DP) then
                        self_energies(iqpt, i) = self_energy(i,i)/2.0_DP/w_q(i)
                    else
                        self_energies(iqpt, i) = complex(0.0_DP, 0.0_DP)
                    endif
                enddo
            else
                self_energies(iqpt, :) = complex(0.0_DP, 0.0_DP)
            endif

        enddo
        !$OMP END PARALLEL DO


    end subroutine calculate_lifetimes_perturbative

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine calculate_lifetimes(irrqgrid, qgrid, weights, scatt_events, fc2, r2_2, &
                    fc3, r3_2, r3_3, rprim, pos, masses, smear, T, nirrqpt, nqpt, nat, nfc2, nfc3, n_events, self_energies)

        use omp_lib
        implicit none

        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        integer, intent(in) :: nirrqpt, nat, nqpt, nfc2, nfc3, n_events
        integer, intent(in) :: weights(n_events), scatt_events(nirrqpt)
        real(kind=DP), intent(in) :: irrqgrid(3, nirrqpt)
        real(kind=DP), intent(in) :: qgrid(3, n_events)
        real(kind=DP), intent(in) :: fc2(nfc2, 3*nat, 3*nat)
        real(kind=DP), intent(in) :: fc3(nfc3, 3*nat, 3*nat, 3*nat)
        real(kind=DP), intent(in) :: r2_2(3, nfc2)
        real(kind=DP), intent(in) :: r3_2(3, nfc3), r3_3(3, nfc3)
        real(kind=DP), intent(in) :: rprim(3, 3)
        real(kind=DP), intent(in) :: pos(3, nat)
        real(kind=DP), intent(in) :: masses(nat)
        real(kind=DP), intent(in) :: smear(3*nat, nqpt)
        real(kind=DP), intent(in) :: T
        complex(kind=DP), dimension(nirrqpt, 3*nat), intent(out) :: self_energies

        integer :: iqpt, i, prev_events, jqpt, tot_qpt, nthreads
        real(kind=DP) :: start_time, curr_time
        real(kind=DP), dimension(3) :: qpt
        real(kind=DP), dimension(3,3) :: kprim
        real(kind=DP), dimension(3*nat) :: w2_q, w_q
        complex(kind=DP), dimension(3*nat) :: self_energy
        complex(kind=DP), dimension(3*nat,3*nat) :: pols_q
        real(kind=DP), allocatable, dimension(:,:) :: curr_grid
        integer, allocatable, dimension(:) :: curr_w
        logical :: is_q_gamma, w_neg_freqs, parallelize

        call cpu_time(start_time)
        self_energies(:,:) = complex(0.0_DP, 0.0_DP)
        kprim = transpose(inv(rprim))
        print*, 'Calculating phonon lifetimes in Lorentzian regime!'

        nthreads = omp_get_max_threads()
        print*, 'Maximum number of threads available: ', nthreads

        parallelize = .True.
        if(nirrqpt <= nthreads) then
                parallelize = .False.
        endif

        !$OMP PARALLEL DO IF(parallelize) &
        !$OMP DEFAULT(SHARED) &
        !$OMP PRIVATE(iqpt, w_neg_freqs, qpt, w2_q, pols_q, is_q_gamma, self_energy, &
        !$OMP curr_grid, w_q, curr_w, prev_events, tot_qpt)
        do iqpt = 1, nirrqpt
            
            w_neg_freqs = .False.
            print*, 'Calculating ', iqpt, ' point in the irreducible zone out of ', nirrqpt, '!'
            !call cpu_time(curr_time)
            !print*, 'Since start: ', curr_time - start_time, ' seconds!'
            qpt = irrqgrid(:, iqpt) 
            call interpolate_fc2(nfc2, nat, fc2, r2_2, masses, qpt, w2_q, pols_q)
            call check_if_gamma(nat, qpt, kprim, w2_q, is_q_gamma)
            if(any(w2_q < 0.0_DP)) then
                print*, 'Negative eigenvalue of dynamical matrix! Exit!'
                w_neg_freqs = .True.
            endif

            if(.not. w_neg_freqs) then
                w_q = sqrt(w2_q)
                self_energy(:) = complex(0.0_DP, 0.0_DP)
                allocate(curr_grid(3, scatt_events(iqpt)))
                allocate(curr_w(scatt_events(iqpt)))
                if(iqpt > 1) then
                    prev_events = sum(scatt_events(1:iqpt-1))
                else
                    prev_events = 0
                endif
                do jqpt = 1, scatt_events(iqpt)
                    curr_grid(:,jqpt) = qgrid(:,prev_events + jqpt)
                    curr_w(jqpt) = weights(prev_events + jqpt)
                enddo
                call calculate_self_energy_LA(w_q, qpt, pols_q, is_q_gamma, scatt_events(iqpt), nat, nfc2, nfc3, curr_grid, &
                            curr_w, fc2, fc3, &
                            r2_2, r3_2, r3_3, kprim, masses, smear(:,iqpt), T, .not. parallelize, self_energy)

                deallocate(curr_grid)
                tot_qpt = sum(curr_w)
                deallocate(curr_w)
                self_energy = self_energy/dble(tot_qpt)
            else
                self_energy(:) = complex(0.0_DP, 0.0_DP)
            endif

            do i = 1, 3*nat
                if(w_q(i) .ne. 0.0_DP) then
                        self_energies(iqpt, i) = self_energy(i)/2.0_DP/w_q(i)
                else
                        self_energies(iqpt, i) = complex(0.0_DP, 0.0_DP)
                endif
            enddo

        enddo
        !$OMP END PARALLEL DO
                            
                            
    end subroutine calculate_lifetimes

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine calculate_self_energy_LA(w_q, qpt, pols_q, is_q_gamma, nqpt, nat, nfc2, nfc3, &
                    qgrid, weights, fc2, fc3, &
                    r2_2, r3_2, r3_3, kprim, masses, smear, T, parallelize, self_energy)

        use omp_lib
        use third_order_cond

        implicit none
        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        integer, intent(in) :: nqpt, nat, nfc2, nfc3
        integer, intent(in) :: weights(nqpt)
        real(kind=DP), intent(in) :: w_q(3*nat), qpt(3), qgrid(3,nqpt)
        real(kind=DP), intent(in) :: fc2(nfc2, 3*nat, 3*nat), kprim(3,3)
        real(kind=DP), intent(in) :: fc3(nfc3, 3*nat, 3*nat, 3*nat)
        real(kind=DP), intent(in) :: r2_2(3, nfc2)
        real(kind=DP), intent(in) :: r3_2(3, nfc3), r3_3(3, nfc3)
        real(kind=DP), intent(in) :: masses(nat)
        real(kind=DP), intent(in) :: smear(3*nat)
        real(kind=DP), intent(in) :: T
        complex(kind=DP), intent(in) :: pols_q(3*nat,3*nat)
        logical, intent(in) :: is_q_gamma, parallelize
        complex(kind=DP), intent(out) :: self_energy(3*nat)

        integer :: jqpt, iat, jat, kat, i, j, k, i1, j1, k1
        real(kind=DP), dimension(3) :: kpt, mkpt
        real(kind=DP), dimension(3*nat) :: w2_k, w2_mk_mq
        real(kind=DP), dimension(3*nat) :: w_k, w_mk_mq
        real(kind=DP), dimension(3*nat, 3) :: freqs_array
        complex(kind=DP), dimension(3*nat) :: selfnrg
        complex(kind=DP), dimension(3*nat,3*nat) :: pols_k, pols_mk_mq
        complex(kind=DP), dimension(3*nat, 3*nat, 3*nat) :: ifc3, d3, d3_pols
        logical :: is_k_gamma, is_mk_mq_gamma, is_k_neg, is_mk_mq_neg
        logical, dimension(3) :: if_gammas

        self_energy = complex(0.0_DP, 0.0_DP)

        !$OMP PARALLEL DO IF(parallelize) DEFAULT(NONE) &
        !$OMP PRIVATE(i,j,k,i1,j1,k1, jqpt, ifc3, d3, d3_pols, kpt, mkpt, is_k_gamma, is_mk_mq_gamma, w2_k, w2_mk_mq, &
        !$OMP w_k, w_mk_mq, pols_k, pols_mk_mq, iat, jat, kat, freqs_array, if_gammas, is_k_neg, is_mk_mq_neg, selfnrg) &
        !$OMP SHARED(nqpt, nat, fc3, r3_2, r3_3, nfc2, nfc3, masses, fc2, smear, T, weights, qgrid, qpt, kprim, is_q_gamma, &
        !$OMP r2_2, w_q, pols_q) &
        !$OMP REDUCTION(+:self_energy)
        do jqpt = 1, nqpt
            is_k_neg = .False.
            is_mk_mq_neg = .False.
            ifc3(:,:,:) = complex(0.0_DP, 0.0_DP) 
            d3(:,:,:) = complex(0.0_DP, 0.0_DP) 
            d3_pols(:,:,:) = complex(0.0_DP, 0.0_DP) 

            kpt = qgrid(:, jqpt)
            mkpt = -1.0_DP*qpt - kpt
            call interpol_v2(fc3, r3_2, r3_3, kpt, mkpt, ifc3, nfc3, nat)
            call interpolate_fc2(nfc2, nat, fc2, r2_2, masses, kpt, w2_k, pols_k)
            call interpolate_fc2(nfc2, nat, fc2, r2_2, masses, mkpt, w2_mk_mq, pols_mk_mq)
    
            call check_if_gamma(nat, kpt, kprim, w2_k, is_k_gamma)
            call check_if_gamma(nat, mkpt, kprim, w2_mk_mq, is_mk_mq_gamma)
            if(any(w2_k < 0.0_DP)) then
                print*, 'Negative eigenvalue of dynamical matrix! Exit!'
                is_k_neg = .True.
            endif
            if(any(w2_mk_mq < 0.0_DP)) then
                print*, 'Negative eigenvalue of dynamical matrix! Exit!'
                is_mk_mq_neg = .True.
            endif

            if(.not. is_k_neg .and. .not. is_mk_mq_neg) then
            w_k = sqrt(w2_k)
            w_mk_mq = sqrt(w2_mk_mq)

            do iat = 1, nat
            do i = 1, 3
                do jat = 1, nat
                do j = 1, 3
                    do kat = 1, nat
                    do k = 1, 3
                        d3(k + 3*(kat - 1), j + 3*(jat - 1), i + 3*(iat - 1)) = &
                                       ifc3(k + 3*(kat - 1), j + 3*(jat - 1), i + 3*(iat - 1))&
                                        /sqrt(masses(iat)*masses(jat)*masses(kat))
                    enddo
                    enddo
                enddo
                enddo
            enddo
            enddo

            do i = 1, 3*nat
                do j = 1, 3*nat
                    do k = 1, 3*nat
                        do i1 = 1, 3*nat
                        do j1 = 1, 3*nat
                        do k1 = 1, 3*nat
                            d3_pols(k,j,i) = d3_pols(k,j,i) + &
                            d3(k1,j1,i1)*pols_q(k1,k)*pols_k(j1,j)*pols_mk_mq(i1,i) 
                        enddo
                        enddo
                        enddo
                    enddo
                enddo
            enddo

            freqs_array(:, 1) = w_q
            freqs_array(:, 2) = w_k
            freqs_array(:, 3) = w_mk_mq

            if_gammas(1) = is_q_gamma
            if_gammas(2) = is_k_gamma
            if_gammas(3) = is_mk_mq_gamma
      
            selfnrg = complex(0.0_DP,0.0_DP)
            call compute_perturb_selfnrg_single(smear,T, &
                freqs_array, if_gammas, d3_pols, 3*nat, selfnrg)
            self_energy = self_energy + selfnrg*dble(weights(jqpt))
            endif
        enddo
        !$OMP END PARALLEL DO

    end subroutine calculate_self_energy_LA

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine calculate_self_energy_P(w_q, qpt, pols_q, is_q_gamma, nqpt, nat, nfc2, nfc3, ne, qgrid, &
                    weights, fc2, fc3, &
                    r2_2, r3_2, r3_3, kprim, masses, smear, T, energies, parallelize, self_energy)

        use omp_lib
        use third_order_cond

        implicit none
        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        integer, intent(in) :: nqpt, nat, nfc2, nfc3, ne
        integer, intent(in) :: weights(nqpt)
        real(kind=DP), intent(in) :: w_q(3*nat), qpt(3), qgrid(3,nqpt)
        real(kind=DP), intent(in) :: fc2(nfc2, 3*nat, 3*nat), kprim(3,3)
        real(kind=DP), intent(in) :: fc3(nfc3, 3*nat, 3*nat, 3*nat)
        real(kind=DP), intent(in) :: r2_2(3, nfc2)
        real(kind=DP), intent(in) :: r3_2(3, nfc3), r3_3(3, nfc3)
        real(kind=DP), intent(in) :: masses(nat)
        real(kind=DP), intent(in) :: smear(3*nat), energies(ne)
        real(kind=DP), intent(in) :: T
        complex(kind=DP), intent(in) :: pols_q(3*nat,3*nat)
        logical, intent(in) :: is_q_gamma, parallelize
        complex(kind=DP), intent(out) :: self_energy(ne, 3*nat)

        integer :: jqpt, iat, jat, kat, i, j, k, i1, j1, k1
        real(kind=DP), dimension(3) :: kpt, mkpt
        real(kind=DP), dimension(3*nat) :: w2_k, w2_mk_mq
        real(kind=DP), dimension(3*nat) :: w_k, w_mk_mq
        real(kind=DP), dimension(3*nat, 3) :: freqs_array
        complex(kind=DP), dimension(ne, 3*nat) :: selfnrg
        complex(kind=DP), dimension(3*nat,3*nat) :: pols_k, pols_mk_mq
        complex(kind=DP), dimension(3*nat, 3*nat, 3*nat) :: ifc3, d3, d3_pols
        logical :: is_k_gamma, is_mk_mq_gamma, is_k_neg, is_mk_mq_neg
        logical, dimension(3) :: if_gammas

        self_energy = complex(0.0_DP, 0.0_DP)

        !$OMP PARALLEL DO IF(parallelize) &
        !$OMP DEFAULT(NONE) &
        !$OMP PRIVATE(jqpt, ifc3, d3, d3_pols, kpt, mkpt, w2_k, pols_k, w2_mk_mq, pols_mk_mq, is_k_gamma, is_mk_mq_gamma, &
        !$OMP is_k_neg, is_mk_mq_neg, w_k, w_mk_mq, i,j,k,i1,j1,k1,iat,jat,kat, freqs_array, if_gammas, selfnrg) &
        !$OMP SHARED(nqpt, qgrid, fc3, r3_2, r3_3, qpt, nfc3, nat, nfc2, fc2, r2_2, masses, is_q_gamma, smear, &
        !$OMP T, energies, ne, w_q, pols_q,weights, kprim) &
        !$OMP REDUCTION(+:self_energy)
        do jqpt = 1, nqpt
            is_k_neg = .False.
            is_mk_mq_neg = .False.
            ifc3(:,:,:) = complex(0.0_DP, 0.0_DP) 
            d3(:,:,:) = complex(0.0_DP, 0.0_DP) 
            d3_pols(:,:,:) = complex(0.0_DP, 0.0_DP) 

            kpt = qgrid(:, jqpt)
            mkpt = -1.0_DP*qpt - kpt
            call interpol_v2(fc3, r3_2, r3_3, kpt, mkpt, ifc3, nfc3, nat)
            call interpolate_fc2(nfc2, nat, fc2, r2_2, masses, kpt, w2_k, pols_k)
            call interpolate_fc2(nfc2, nat, fc2, r2_2, masses, mkpt, w2_mk_mq, pols_mk_mq)
    
            call check_if_gamma(nat, kpt, kprim, w2_k, is_k_gamma)
            call check_if_gamma(nat, mkpt, kprim, w2_mk_mq, is_mk_mq_gamma)
            if(any(w2_k < 0.0_DP)) then
                print*, 'Negative eigenvalue of dynamical matrix! Exit!'
                is_k_neg = .True.
            endif
            if(any(w2_mk_mq < 0.0_DP)) then
                print*, 'Negative eigenvalue of dynamical matrix! Exit!'
                is_mk_mq_neg = .True.
            endif

            if(.not. is_k_neg .and. .not. is_mk_mq_neg) then
            w_k = sqrt(w2_k)
            w_mk_mq = sqrt(w2_mk_mq)

            do iat = 1, nat
            do i = 1, 3
                do jat = 1, nat
                do j = 1, 3
                    do kat = 1, nat
                    do k = 1, 3
                        d3(k + 3*(kat - 1), j + 3*(jat - 1), i + 3*(iat - 1)) = &
                                       ifc3(k + 3*(kat - 1), j + 3*(jat - 1), i + 3*(iat - 1))&
                                        /sqrt(masses(iat)*masses(jat)*masses(kat))
                    enddo
                    enddo
                enddo
                enddo
            enddo
            enddo

            do i = 1, 3*nat
                do j = 1, 3*nat
                    do k = 1, 3*nat
                        do i1 = 1, 3*nat
                        do j1 = 1, 3*nat
                        do k1 = 1, 3*nat
                            d3_pols(k,j,i) = d3_pols(k,j,i) + &
                            d3(k1,j1,i1)*pols_q(k1,k)*pols_k(j1,j)*pols_mk_mq(i1,i) 
                        enddo
                        enddo
                        enddo
                    enddo
                enddo
            enddo

            freqs_array(:, 1) = w_q
            freqs_array(:, 2) = w_k
            freqs_array(:, 3) = w_mk_mq

            if_gammas(1) = is_q_gamma
            if_gammas(2) = is_k_gamma
            if_gammas(3) = is_mk_mq_gamma
      
            selfnrg = complex(0.0_DP,0.0_DP)
            call compute_diag_dynamic_bubble_single(energies, smear, T, freqs_array, if_gammas, d3_pols, ne, 3*nat, selfnrg)

            self_energy = self_energy + selfnrg*dble(weights(jqpt))
            endif
        enddo
        !$OMP END PARALLEL DO

    end subroutine calculate_self_energy_P

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine check_if_gamma(nat, q, kprim, w2_q, is_gamma)

        implicit none
        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        integer, intent(in) :: nat
        real(kind=DP), dimension(3), intent(in) :: q
        real(kind=DP), intent(in) :: kprim(3,3)

        real(kind=DP), dimension(3*nat), intent(inout) :: w2_q
        logical, intent(out) :: is_gamma

        integer :: iat, i1, j1, k1
        real(kind=DP) :: ikprim(3,3), red_q(3)

        ikprim = inv(kprim)
        is_gamma = .FALSE.
        if(all(abs(q) < 1.0d-6)) then
            is_gamma = .TRUE.
            do iat = 1, 3
                w2_q(iat) = 0.0_DP
            enddo
        else
           red_q(:) = 0.0_DP
           do iat = 1, 3
                red_q = red_q + q(iat)*ikprim(iat, :)
           enddo
           do iat = 1, 3
                red_q(iat) = red_q(iat) - dble(NINT(red_q(iat)))
           enddo
           if(all(abs(red_q) < 1.0d-6)) then
                is_gamma = .TRUE.
                do iat = 1, 3
                        w2_q(iat) = 0.0_DP
                enddo
           endif
        endif

    end subroutine

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
    function inv(A) result(Ainv)
 
        implicit none
        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        real(kind=DP),intent(in) :: A(:,:)
        real(kind=DP)            :: Ainv(size(A,1),size(A,2))
        real(kind=DP)            :: work(size(A,1))            ! work array for LAPACK
        integer         :: n,info,ipiv(size(A,1))     ! pivot indices
 
        ! Store A in Ainv to prevent it from being overwritten by LAPACK
        Ainv = A
        n = size(A,1)
        ! DGETRF computes an LU factorization of a general M-by-N matrix A
        ! using partial pivoting with row interchanges.
        call DGETRF(n,n,Ainv,n,ipiv,info)
        if (info.ne.0) stop 'Matrix is numerically singular!'
        ! DGETRI computes the inverse of a matrix using the LU factorization
        ! computed by DGETRF.
        call DGETRI(n,Ainv,n,ipiv,work,n,info)
        if (info.ne.0) stop 'Matrix inversion failed!'
    end function inv
 
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine interpolate_fc2(nfc2, nat, fc2, r2_2, masses, q, w2_q, pols_q)

        implicit none        
        integer, parameter :: DP = selected_real_kind(14,200)
        real(kind=DP), parameter :: PI = 3.141592653589793115997963468544185161590576171875

        integer, intent(in) :: nfc2, nat
        real(kind=DP), dimension(nfc2, 3*nat, 3*nat), intent(in) :: fc2 
        real(kind=DP), dimension(3, nfc2), intent(in) :: r2_2
        real(kind=DP), dimension(nat), intent(in) :: masses
        real(kind=DP), dimension(3), intent(in) :: q

        real(kind=DP), dimension(3*nat), intent(out) :: w2_q
        complex(kind=DP), dimension(3*nat, 3*nat), intent(out) :: pols_q

        integer :: ir, iat, jat, INFO, LWORK, i, j
        complex(kind=DP), dimension(6*nat + 1) :: WORK
        real(kind=DP), dimension(9*nat - 2) :: RWORK
        complex(kind=DP), dimension(3*nat, 3*nat) :: pols_q1
        real(kind=DP) :: phase 


        pols_q1 = complex(0.0_DP,0.0_DP)

        do ir = 1, nfc2
            phase = dot_product(r2_2(:,ir), q)*2.0_DP*PI
            pols_q1 = pols_q1 + fc2(ir,:,:)*exp(complex(0.0_DP, phase))
        enddo

        do iat = 1, nat
        do i = 1, 3 
            do jat = 1, nat
            do j = 1, 3
                pols_q1(j + 3*(jat - 1),i + 3*(iat - 1)) = pols_q1(j + 3*(jat - 1),i + 3*(iat - 1))/sqrt(masses(iat)*masses(jat))
            enddo
            enddo
        enddo
        enddo

        pols_q = (pols_q1 + conjg(transpose(pols_q1)))/2.0_DP
        LWORK = -1
        call zheev('V', 'L', 3*nat, pols_q, 3*nat, w2_q, WORK, LWORK, RWORK, INFO)
        LWORK = MIN( size(WORK), INT( WORK( 1 ) ) )
        call zheev('V', 'L', 3*nat, pols_q, 3*nat, w2_q, WORK, LWORK, RWORK, INFO)

    end subroutine

end module get_lf
