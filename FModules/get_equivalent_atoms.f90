subroutine get_equivalent_atoms(coords1, coords2, unit_cell, ityp1, ityp2, eq_atoms, nat)
    implicit none 

    double precision, dimension(nat, 3), intent(in) :: coords1, coords2
    double precision, dimension(3,3), intent(in) :: unit_cell 
    integer, dimension(nat), intent(in) :: ityp1, ityp2
    integer, dimension(nat), intent(out) :: eq_atoms
    integer :: nat

    integer :: i, j, eq_found, j_aux
    double precision :: min_dist, tmp_dist
    double precision, dimension(3) :: aux_vect
    double precision, dimension(nat) :: total_dist

    ! Variables for get_closest_atoms
    integer :: nx, ny, nz, mynx, myny, mynz
    integer, parameter :: far = 2
    logical :: found

    ! Set all the atoms as not equivalent
    eq_atoms(:) = -1

    ! This subroutine could use get_closest_vector
    ! However, we mix it here to invert the loops
    ! And exploit a most likely scenario that the
    ! Replica of the atoms is on the same cell
    ! This is a huge speedup

    eq_found = 0

    do i = 1, nat 
        total_dist(:) = 100000.d0

        !print *, "ATOM: ", i, " of ", nat
        found = .false.

        ! Decide what cell to probe the distance
        do mynx = -far, far
            ! Reorder the loop to start with 0
            ! which is the most probable value (exit early if found)
            if (mynx .le. 0) then   
                nx = mynx + far
            else 
                nx = -mynx 
            endif
            do myny = -far, far
                if (myny .le. 0) then 
                    ny = myny + far
                else 
                    ny = -myny
                endif
                do mynz = -far, far
                    if (mynz .le. 0) then 
                        nz = mynz + far
                    else 
                        nz = -mynz
                    endif

                    !print *, "Checking cell: ", nx, ny, nz

                    ! Now we have the cell, we can loop over the atoms
                    do j_aux = i, nat + i -1
                        ! Warp the loop over j to start with i
                        ! Which is the most probable value (and the most efficient loop)
                        j = j_aux
                        if (j .gt. nat) then
                            j = j - nat
                        end if

                        ! Jump if the atom is already found
                        if (total_dist(j) .lt. 1.d-6) continue


                        ! Exclude different type of atoms
                        if ( ityp1(i) .ne. ityp2(j) ) continue
                        !call get_closest_vector(unit_cell, coords1(i, :) - coords2(j, :), dist)

                        ! Unwrap the get_closest_vector subroutine here
                        aux_vect(:) = coords1(i, :) - coords2(j, :)
                        if (nx .ne. 0) then
                            aux_vect(:) = aux_vect(:) + nx * unit_cell(1, :)
                        end if 
                        if (ny .ne. 0) then
                            aux_vect(:) = aux_vect(:) + ny * unit_cell(2, :)
                        end if
                        if (nz .ne. 0) then
                            aux_vect(:) = aux_vect(:) + nz * unit_cell(3, :)
                        end if

                        tmp_dist = sum(aux_vect**2)

                        !print *, "Checking atom ", j, " of ", nat, "old_dist: ", &
                        !    total_dist(j), " new_dist: ", tmp_dist, "current: ", eq_atoms(i)
                        
                        if (tmp_dist .lt. total_dist(j)) then
                            total_dist(j) = tmp_dist 
                        end if
                        if (tmp_dist .lt. 1.d-6) then
                            eq_found = eq_found + 1
                            eq_atoms(i) = j - 1 ! - 1 is for Fortan to python correspondance
                            
                            ! Early exit if all the atoms are found
                            if (eq_found .eq. nat) then
                                !print *, "Everything found"
                                !print *, eq_atoms
                                return
                            end if   
                            found = .true.
                            exit
                        end if
                    enddo
                    if (found) exit
                enddo
                if (found) exit
            enddo
            if (found) exit
        enddo

        ! Check if the atom is not found
        if (.not. found .or. eq_atoms(i) .eq. -1) then
            ! Pick the index of the minimum value of total_dist
            eq_atoms(i) = minloc(total_dist, dim=1) - 1
        end if
    enddo

end subroutine get_equivalent_atoms


subroutine get_closest_vector(unit_cell, v_dist, new_v_dist)
    implicit none 

    double precision, dimension(3,3), intent(in) :: unit_cell 
    double precision, dimension(3), intent(in) :: v_dist
    double precision, dimension(3), intent(out) :: new_v_dist

    integer :: nx, ny, nz, mynx, myny, mynz
    integer, parameter :: far = 2
    double precision, dimension(3) :: aux_vect
    double precision :: min_dist, dist
    
    min_dist = sum(v_dist(:) ** 2)
    new_v_dist(:) = v_dist(:)
    do mynx = -far, far
        ! Reorder the loop to start with 0
        ! which is the most probable value (exit early if found)
        if (mynx .le. 0) then   
            nx = mynx + far
        else 
            nx = -mynx 
        endif

        do myny = -far, far
            if (myny .le. 0) then 
                ny = myny + far
            else 
                ny = -myny
            endif
            do mynz = -far, far
                if (mynz .le. 0) then 
                    nz = mynz + far
                else 
                    nz = -mynz
                endif
                ! Define the new vector
                aux_vect(:) = v_dist(:)
                aux_vect(:) = aux_vect(:) + nx * unit_cell(1, :)
                aux_vect(:) = aux_vect(:) + ny * unit_cell(2, :)
                aux_vect(:) = aux_vect(:) + nz * unit_cell(3, :)
                
                dist = sum(aux_vect(:) ** 2)
                if (dist .lt. min_dist) then
                    min_dist = dist 
                    new_v_dist(:) = aux_vect(:)
                end if  

                if (dist .lt. 1.d-6) then
                    return
                end if
            enddo
        enddo
    enddo
end subroutine get_closest_vector


subroutine get_gr_data(cells, coords, ityp, type1, type2, r_min, r_max, n_r, r_value, gr, n_structs, nat)
    implicit none 

    integer :: n_structs, nat
    double precision, dimension(n_structs, 3,3), intent(in) :: cells
    double precision, dimension(n_structs, nat, 3), intent(in) :: coords
    integer, dimension(nat), intent(in) :: ityp 
    integer, intent(in) :: type1, type2 
    double precision, intent(in) :: r_min, r_max
    integer, intent(in) :: n_r
    double precision, dimension(n_r), intent(out) :: r_value, gr


    integer k, i1, i2, index, ntot
    integer other
    double precision, dimension(3) :: r_vec, r_dist
    double precision :: r, dr, v, rho
    double precision, parameter :: M_PI = 3.141592653589793d0

    dr = (r_max - r_min) / n_r

    ntot = 0
    gr(:) = 0.0d0
    v = 4 * M_PI * r_max*r_max*r_max / 3.0d0

    do k = 1, n_structs
        do i1 = 1, nat - 1
            other = 0
            if (ityp(i1) .eq. type1) then
                other = type2
            else if (ityp(i1) .eq. type2) then
                other = type1 
            end if

            if (other .gt. 0) then
                do i2 = i1 +1, nat
                    if (ityp(i2) .eq. other) then
                        ! Get the distance
                        r_dist(:) = coords(k, i1, :) - coords(k, i2, :)
                        call get_closest_vector(cells(k, :, :), r_dist, r_vec)
                        r = dot_product(r_vec, r_vec)
                        r = dsqrt(r)

                        ! Get the index
                        index = int( n_r * (r - r_min) / (r_max - r_min) ) + 1
                        if (index .le. n_r .and. index .ge. 1) then
                            gr(index) = gr(index) + 1
                        endif
                        ntot = ntot + 1
                    endif
                end do
            endif
        end do
    end do

    ! Get the radius variable
    do k = 1, n_r
        r_value(k) = r_min + (k - .5) * dr
    end do

    ! Density
    rho = ntot / v 
    gr(:) = gr(:) / (4 * M_PI * r_value**2 * dr * rho)
end subroutine get_gr_data


subroutine fix_coords_in_unit_cell(coords, unit_cell, new_coords, nat)
    implicit none 

    double precision, dimension(nat, 3), intent(in) :: coords 
    double precision, dimension(3,3), intent(in) :: unit_cell 
    double precision, dimension(nat, 3), intent(out) :: new_coords
    integer :: nat 

    integer :: i, j
    double precision, dimension(3,3) :: mt, inv_mt
    double precision, dimension(3) :: covect, contravect

    do i = 1, 3
        do j = i, 3
            mt(i, j) = dot_product(unit_cell(i, :), unit_cell(j, :))
            mt(j, i) = mt(i, j)
        enddo
    enddo

    ! Invert the metric tnesor
    call matinv3(mt, inv_mt)

    do i = 1, nat 
        contravect(:) = matmul(unit_cell, coords(i, :))
        covect(:) = matmul(inv_mt, contravect)
        
        do j = 1, 3
            covect(j) = covect(j) - floor(covect(j))
            if (covect(j) .lt. 0) then
                covect(j) = 1 + covect(j)
            endif
        enddo

        new_coords(i, :) = matmul(covect, unit_cell)
    enddo
end subroutine fix_coords_in_unit_cell


! subroutine apply_symmetry(coords, unit_cell, symmetry, new_coords, nat)
!     implicit none

!     double precision, dimension(nat, 3), intent(in) :: coords
!     double precision, dimension(3,3), intent(in) :: unit_cell 
!     double precision, dimension(3, 4), intent(in) :: symmetry
!     double precision, dimension(nat, 3), intent(out) :: new_coords
!     integer :: nat 

!     integer i, j
!     double precision :: mt(3,3), inv_mt(3,3), covect(3), contravect(3)


!     do i = 1, 3
!         do j = i, 3
!             mt(i, j) = dot_product(unit_cell(i, :), unit_cell(j, :))
!             mt(j, i) = mt(i, j)
!         enddo
!     enddo

!     ! Invert the metric tnesor
!     call matinv3(mt, inv_mt)

!     do i = 1, nat 
!         contravect(:) = matmul(unit_cell, coords(i, :))
!         covect(:) = matmul(inv_mt, contravect)

! end subroutine apply_symmetry


subroutine matinv3(A, B)
    implicit none

    double precision, intent(in)  :: A(3,3)   !! Matrix
    double precision, intent(out) :: B(3,3)   !! Inverse matrix
    double precision              :: detinv

    ! Calculate the inverse determinant of the matrix
    detinv = 1/(A(1,1)*A(2,2)*A(3,3) - A(1,1)*A(2,3)*A(3,2)&
            - A(1,2)*A(2,1)*A(3,3) + A(1,2)*A(2,3)*A(3,1)&
            + A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1))

    ! Calculate the inverse of the matrix
    B(1,1) = +detinv * (A(2,2)*A(3,3) - A(2,3)*A(3,2))
    B(2,1) = -detinv * (A(2,1)*A(3,3) - A(2,3)*A(3,1))
    B(3,1) = +detinv * (A(2,1)*A(3,2) - A(2,2)*A(3,1))
    B(1,2) = -detinv * (A(1,2)*A(3,3) - A(1,3)*A(3,2))
    B(2,2) = +detinv * (A(1,1)*A(3,3) - A(1,3)*A(3,1))
    B(3,2) = -detinv * (A(1,1)*A(3,2) - A(1,2)*A(3,1))
    B(1,3) = +detinv * (A(1,2)*A(2,3) - A(1,3)*A(2,2))
    B(2,3) = -detinv * (A(1,1)*A(2,3) - A(1,3)*A(2,1))
    B(3,3) = +detinv * (A(1,1)*A(2,2) - A(1,2)*A(2,1))
end subroutine matinv3
