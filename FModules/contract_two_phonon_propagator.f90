! Author: Lorenzo Monacelli
! 
! This file contracts the given matrix
! over the two phonon propagator.
!
! It computes:
! 
! F(w) =  SUM(a,b) M_ab G_ab(w) M_ab
!
! Where G_ab is the two phonon propagator and M_ab is the given matrix.
! M must be already expressed in the polarization vector basis,
! the modes on which they are contracted must match the w_mu array passed.
!
! n_w is the number of positions inside w_array.
!
! w_array, w_mu and smearing must be provided in [Ry]
! T must be provided in Kelvin
!
! The subroutinine returns a function of w_array
! The output is a double complex type (in fact the smearing adds an imaginary part)

subroutine contract_two_ph_propagator(w_array, w_mu, T, smearing, M, f_output, n_w, n_modes)
    double precision, dimension(n_w), intent(in) :: w_array
    double precision, dimension(n_modes), intent(in) :: w_mu
    double precision, intent(in) :: T, smearing
    double precision, dimension(n_modes, n_modes), intent(in) :: M
    double complex, dimension(n_w), intent(out) :: f_output
    integer :: n_w, n_modes


    integer i1, mu, nu 
    double precision :: w
    double complex, dimension(n_modes, n_modes) :: chi

    !!$OMP DO PRIVATE(i1, mu, nu, w, chi) SHARED(m, f_output, smearing, T, n_modes)
    do i1 = 1, n_w
        w = w_array(i1)

        ! Compute the two phonon propagator
        call get_two_phonon_propagator(w, w_mu, T, smearing, chi, n_modes)

        f_output(i1) = 0
        !!$OMP DO COLLAPSE(2) REDUCTION(+:f_output(i1))
        do mu = 1, n_modes
            do nu = 1, n_modes 
                f_output(i1) = f_output(i1) + M(nu, mu) * M(nu, mu) * chi(nu, mu)
            end do
        end do
        !!$OMP END DO
    enddo   
    !!$OMP END DO

end subroutine contract_two_ph_propagator   

!
! Get the dynamical two phonon propagator
subroutine get_two_phonon_propagator(w_value, ws, T, smearing, chi, n_w)
    double precision, intent(in) :: w_value, T, smearing
    double precision, dimension(n_w), intent(in) :: ws 
    double complex, dimension(n_w, n_w), intent(out) :: chi 
    integer :: n_w

    double precision, parameter :: k_to_ry = 6.336857346553283d-06
    double precision, parameter :: epsil = 1d-6
    integer :: mu, nu
    double precision :: n_mu, n_nu, w_mu, w_nu
    double complex :: chi1, chi2

    double complex, parameter :: i_unit = (0d0, 1d0)

    do mu = 1, n_w
        n_mu = 0
        w_mu = ws(mu)
        if (T .ge. epsil) then
            n_mu = 1 / (dexp(w_mu  / (T * K_to_Ry)) - 1)
        endif
        do nu = 1, n_w 
            n_nu = 0
            w_nu = ws(nu)
            if (T .ge. epsil) then
                n_nu = 1 / (dexp(w_nu / (T * K_to_Ry)) - 1)
            endif

            chi1 = (w_mu +  w_nu) * (n_nu + n_mu + 1)
            chi1 = chi1 /  ( (w_mu + w_nu)**2 - (w_value - i_unit*smearing)**2 )
            chi1 = chi1 /  (2*w_mu*w_nu)

            chi2 = (w_mu - w_nu) * (n_nu - n_mu)
            chi2 = chi2/ ( (w_nu - w_mu)**2 - (w_value - i_unit*smearing)**2 )
            chi2 = chi2/( 2*w_mu*w_nu)

            chi(mu, nu) = chi1 + chi2
        enddo
    enddo

end subroutine get_two_phonon_propagator


