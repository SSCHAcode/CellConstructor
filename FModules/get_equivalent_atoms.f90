subroutine get_equivalent_atoms(coords1, coords2, unit_cell, ityp1, ityp2, eq_atoms, nat)
    implicit none 

    double precision, dimension(nat, 3), intent(in) :: coords1, coords2
    double precision, dimension(3,3), intent(in) :: unit_cell 
    integer, dimension(nat), intent(in) :: ityp1, ityp2
    integer, dimension(nat), intent(out) :: eq_atoms
    integer :: nat

    integer :: i, j
    double precision :: min_dist, tmp_dist
    double precision, dimension(3) :: dist

    do i = 1, nat 
        min_dist = 1000000.d0
        do j = 1, nat 
            ! Exclude different type of atoms
            if ( ityp1(i) .ne. ityp2(j) ) continue
            call get_closest_vector(unit_cell, coords1(i, :) - coords2(j, :), dist)
            tmp_dist = sum(dist**2)
            
            if (tmp_dist .lt. min_dist) then
                min_dist = tmp_dist 
                eq_atoms(i) = j - 1 ! - 1 is for Fortan to python correspondance
            end if
        enddo
    enddo   

end subroutine get_equivalent_atoms


subroutine get_closest_vector(unit_cell, v_dist, new_v_dist)
    implicit none 

    double precision, dimension(3,3), intent(in) :: unit_cell 
    double precision, dimension(3), intent(in) :: v_dist
    double precision, dimension(3), intent(out) :: new_v_dist

    integer :: nx, ny, nz
    integer, parameter :: far = 2
    double precision, dimension(3) :: aux_vect
    double precision :: min_dist, dist
    
    min_dist = sum(v_dist(:) ** 2)
    new_v_dist(:) = v_dist(:)
    do nx = -far, far
        do ny = -far, far
            do nz = -far, far
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
            enddo
        enddo
    enddo
end subroutine get_closest_vector


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
    !! Performs a direct calculation of the inverse of a 3×3 matrix.
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