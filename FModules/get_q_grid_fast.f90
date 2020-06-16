
subroutine get_q_grid (bg, supercell_size, q_list, n_size)

    ! Get the q grid of a given cell.
    ! BG: reciprocal lattice vectors
    ! BG(x, y) is the y-th reciprocal vector, x coordinate
    !
    ! SUPERCELL SIZE a vector of int that gives the supercell size
    !

    implicit none
    
    double precision, dimension(3,3), intent(in) :: bg
    integer, dimension(3), intent(in) :: supercell_size
    double precision, dimension(3, n_size), intent(out) :: q_list
    integer :: n_size

    logical, parameter :: debug = .true.
    

    integer :: nr, i1,i2,i3, ka


    ! Check if the n_size is of the correct value
    nr = supercell_size(1) * supercell_size(2) * supercell_size(3)


    if (nr .ne. n_size) then
        print *, "Error, the dimension of q_list must match the supercell"
        call flush()
        STOP "ERROR IN get_q_grid SUBROUTINE"
    endif

    ka = 0
    do i1 = 1, supercell_size(1)
        do i2 = 1, supercell_size(2)
            do i3 = 1, supercell_size(3)
                
                ka = ka + 1
                ! Sum the vector
                q_list(:, ka) = i1 * bg(:, 1) / supercell_size(1) + &
                    i2 * bg(:, 2) / supercell_size(2) + &
                    i3 * bg(:, 3) / supercell_size(3)
                
            end do
        enddo
    enddo
  
  end subroutine get_q_grid