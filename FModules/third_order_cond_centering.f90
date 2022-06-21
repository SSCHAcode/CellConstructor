module third_order_cond_centering

    contains

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine check_permutation_symmetry(tensor, r_vector2, r_vector3, n_R, natom, perm) 
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)

        integer, intent(in) :: n_R, natom
        real(kind=DP), intent(in) :: tensor(n_R, 3*natom, 3*natom, 3*natom)
        real(kind=DP), intent(in) :: r_vector2(n_R, 3), r_vector3(n_R, 3) 
        logical, intent(out) :: perm

        integer :: i, j, k, ipair, jpair, npairs, unfound
        integer :: pairs(n_R, 2)
        real(kind=DP) :: rvec12(3), rvec13(3), rvec22(3), rvec23(3)
        real(kind=DP) :: tol, tscale
        logical :: got(n_R)

        tol = 1.0_DP/10.0_DP**6
        perm = .True.
        got = .False.
        npairs = 0

        tscale = 0.0_DP
        do i = 1, n_R
                do j = 1, 3*natom
                        if(norma2(tensor(i,j,:,:)) > tscale) then
                                tscale = norma2(tensor(i,j,:,:))
                        endif
                enddo
        enddo

        do i = 1, n_R
                if(.not. got(i)) then
                        rvec12 = r_vector2(i,:)
                        rvec13 = r_vector3(i,:)
                        if(norm(rvec12 - rvec13) < tol) then
                                npairs = npairs + 1
                                got(i) = .True.
                                pairs(npairs, :) = i
                        else
                                do j = 1, n_R
                                        if(.not. got(j)) then
                                                rvec22 = r_vector2(j,:)
                                                rvec23 = r_vector3(j,:)
                                                if(norm(rvec12 - rvec23) < tol .and. norm(rvec13 - rvec22) < tol) then
                                                        npairs = npairs + 1
                                                        got(i) = .True.
                                                        got(j) = .True.
                                                        pairs(npairs, 1) = i
                                                        pairs(npairs, 2) = j
                                                endif
                                        endif
                                enddo
                        endif
                endif
        enddo

        if(.not. all(got)) then
                print*, 'Could not find all the pairs!'
                perm = .False.
                unfound = 0
                do i = 1, n_R
                        if(.not. got(i)) then
                                unfound = unfound + 1
                                !print*, r_vector2(i,:), r_vector3(i,:)
                        endif
                enddo
                print*, dble(unfound)/dble(n_R)*100.0, ' percentage of triplets without a pair!'
        else
                do i = 1, npairs
                        if(perm) then
                                ipair = pairs(i, 1)
                                jpair = pairs(i, 2)
                                do j = 1, 3*natom
                                        if(norma2(tensor(ipair, j,:,:) - transpose(tensor(jpair, j,:,:))) > &
                                                tol*tol*tscale) then
                                                print*, 'Permutation symmetry not satisfied!'
                                                print*, norma2(tensor(ipair, j,:,:) - transpose(tensor(jpair, j,:,:)))
                                                print*, norma2(tensor(ipair,j,:,:))
                                                print*, tensor(ipair,j,:,:)
                                                print*, tensor(jpair,j,:,:)
                                                perm = .False.
                                                EXIT
                                         endif
                                enddo
                        endif
                enddo
        endif

    end subroutine

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    subroutine find_triplets(tensor, r_vector2, r_vector3, rsup, irsup, pos, help_tensor, help_rvec2, help_rvec3, &
                    tot_trip, n_R, natom)
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)

        integer, intent(in) :: n_R, natom
        real(kind=DP), intent(in) :: tensor(n_R, 3*natom, 3*natom, 3*natom)
        real(kind=DP), intent(in) :: r_vector2(n_R, 3), r_vector3(n_R, 3)
        real(kind=DP), intent(in) :: rsup(3,3), irsup(3,3), pos(natom, 3)

        real(kind=DP), dimension(natom*natom*natom, n_R*81, 3, 3, 3), intent(out) :: help_tensor 
        real(kind=DP), dimension(natom*natom*natom, n_R*81, 3), intent(out) :: help_rvec2, help_rvec3
        integer, dimension(natom*natom*natom), intent(out) :: tot_trip

        integer :: i, iat, jat, kat, index1, i1, j1, k1, curr_trip(2)
        integer :: itrip, ntrip, index2
        real(kind=DP) :: rvec1(3), rvec2(3), xvec1(3), xvec2(3)
        real(kind=DP) :: rvec21(3), rvec22(3), xvec21(3), xvec22(3)
        real(kind=DP) :: new_rvec2(27,3), new_rvec3(27,3)
        real(kind=DP) :: size1, size2, tol

        tol = 1.0_DP/10.0_DP**6

        print*, 'Finding triplets ...'
        do iat = 1, natom
        do jat = 1, natom
        do kat = 1, natom
                index1 = kat + natom*(jat-1) + natom**2*(iat-1)
                tot_trip(index1) = 0
                do i = 1, n_R
                        curr_trip = 0
                        rvec1 = r_vector2(i,:) + pos(jat,:) - pos(iat,:)
                        rvec2 = r_vector3(i,:) + pos(kat,:) - pos(iat,:)
                        size1 = norm(rvec1)
                        size2 = norm(rvec2)
                        xvec1 = dot(rvec1, irsup)
                        xvec2 = dot(rvec2, irsup)
                        new_rvec2 = 0.0_DP
                        new_rvec3 = 0.0_DP
                        !curr_trip = curr_trip + 1
                        !new_rvec2(curr_trip(1),:) = r_vector2(i,:)
                        !new_rvec3(curr_trip(2),:) = r_vector3(i,:)
                        do i1 = -1, 1
                        do j1 = -1, 1
                        do k1 = -1, 1
                                xvec21 = xvec1 + (/i1, j1, k1/)
                                rvec21 = dot(xvec21, rsup)
                                if(abs(norm(rvec21) - size1) < tol) then
                                        curr_trip(1) = curr_trip(1) + 1
                                        rvec21 = r_vector2(i,:)
                                        xvec21 = dot(rvec21, irsup)
                                        xvec21 = xvec21 + (/i1, j1, k1/)
                                        rvec21 = dot(xvec21, rsup)
                                        new_rvec2(curr_trip(1),:) = rvec21
                                else if(norm(rvec21) < size1) then
                                        !rvec1 = rvec21
                                        size1 = norm(rvec21)
                                        !xvec1 = dot(rvec1, irsup)
                                        new_rvec2 = 0.0_DP
                                        curr_trip(1) = 1
                                        rvec21 = r_vector2(i,:)
                                        xvec21 = dot(rvec21, irsup)
                                        xvec21 = xvec21 + (/i1, j1, k1/)
                                        rvec21 = dot(xvec21, rsup)
                                        new_rvec2(curr_trip(1),:) = rvec21
                                endif
                                xvec22 = xvec2 + (/i1, j1, k1/)
                                rvec22 = dot(xvec22, rsup)
                                if(abs(norm(rvec22) - size2) < tol) then
                                        curr_trip(2) = curr_trip(2) + 1
                                        rvec22 = r_vector3(i,:)
                                        xvec22 = dot(rvec22, irsup)
                                        xvec22 = xvec22 + (/i1, j1, k1/)
                                        rvec22 = dot(xvec22, rsup)
                                        new_rvec3(curr_trip(2),:) = rvec22
                                 else if(norm(rvec22) < size2) then
                                        rvec2 = rvec22
                                        size2 = norm(rvec22)
                                        !xvec2 = dot(rvec2, irsup)
                                        new_rvec3 = 0.0_DP
                                        curr_trip(2) = 1
                                        rvec22 = r_vector3(i,:)
                                        xvec22 = dot(rvec22, irsup)
                                        xvec22 = xvec22 + (/i1, j1, k1/)
                                        rvec22 = dot(xvec22, rsup)
                                        new_rvec3(curr_trip(2),:) = rvec22
                                endif
                        enddo
                        enddo
                        enddo
                        do i1 = 1, curr_trip(1)
                                do j1 = 1, curr_trip(2)
                                        tot_trip(index1) = tot_trip(index1) + 1
                                        !index2 = j1 + curr_trip(2)*(i1-1)
                                        help_rvec2(index1, tot_trip(index1),:) = new_rvec2(i1,:)
                                        help_rvec3(index1, tot_trip(index1),:) = new_rvec3(j1,:)
                                        help_tensor(index1, tot_trip(index1), :, :, :) = &
                                        tensor(i, 3*(iat-1)+1:3*iat, 3*(jat-1)+1:3*jat, 3*(kat-1)+1:3*kat)&
                                        /dble(curr_trip(1)*curr_trip(2))
                                enddo
                        enddo
                        !tot_trip(index1) = tot_trip(index1) + curr_trip(1)*curr_trip(2)
                        !print*, index1, dble(i)/dble(n_R), tot_trip(index1)
                enddo
        enddo
        enddo
        enddo

        !print*, shape(help_tensor)

    end subroutine

    subroutine distribute_fc3(tensor, rvec2, rvec3, tot_trip, f_tensor, f_rvec2, f_rvec3, itrip, ntrip, natom, n_R)
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)

        integer, intent(in) :: ntrip, n_R, natom
        real(kind=DP), dimension(natom*natom*natom, n_R*81, 3, 3, 3), intent(in) :: tensor
        real(kind=DP), dimension(natom*natom*natom, n_R*81, 3), intent(in) :: rvec2,rvec3
        integer, dimension(natom*natom*natom), intent(in) :: tot_trip

        real(kind=DP), dimension(ntrip, 3*natom, 3*natom, 3*natom), intent(out) :: f_tensor
        real(kind=DP), dimension(ntrip, 3), intent(out) :: f_rvec2, f_rvec3
        integer, intent(out) :: itrip

        integer :: iat, jat, kat, index1, i, j, index0
        real(kind=DP) :: tol
        logical :: found
        
        tol = 1.0_DP/10.0_DP**6

        print*, 'Distributing force constants ... '

        index0 = 1
        do i = 2, natom*natom*natom
                if(tot_trip(i) > tot_trip(index0)) then
                        index0 = i
                endif
        enddo 

        f_rvec2 = 0.0_DP
        f_rvec3 = 0.0_DP
        f_tensor = 0.0_DP
        itrip = 0
        do iat = 1, natom
        do jat = 1, natom
        do kat = 1, natom
                index1 = kat + natom*(jat-1) + natom**2*(iat-1)
                if(index1 .eq. index0) then
                do i = 1, tot_trip(index0)
                        itrip = i
                        f_rvec2(itrip, :) = rvec2(index0, i, :)
                        f_rvec3(itrip, :) = rvec3(index0, i, :)
                        f_tensor(itrip, 3*(iat-1)+1:3*iat, 3*(jat-1)+1:3*jat, 3*(kat-1)+1:3*kat) = &
                                tensor(index0, i,:,:,:)
                enddo
                endif
        enddo
        enddo
        enddo
        do iat = 1, natom
        do jat = 1, natom
        do kat = 1, natom
                index1 = kat + natom*(jat-1) + natom**2*(iat-1)
                if(index0 .ne. index1) then
                do i = 1, tot_trip(index1)
                        !if(itrip .eq. 0) then
                        !        itrip = itrip + 1
                        !        f_rvec2(itrip, :) = rvec2(index1, i, :)
                        !        f_rvec3(itrip, :) = rvec3(index1, i, :)
                        !        f_tensor(itrip, 3*(kat-1)+1:3*kat, 3*(jat-1)+1:3*jat, 3*(iat-1)+1:3*iat) = &
                        !                tensor(index1, i,:,:,:)
                        !else
                                found = .False.
                                do j = 1, itrip
                                        if(norm(rvec2(index1, i, :) - f_rvec2(j, :)) < tol .and. &
                                         norm(rvec3(index1, i, :) - f_rvec3(j, :)) < tol) then
                                                !print*, rvec2(index1, i, :), rvec3(index1, i, :)
                                                !print*, f_rvec2(j, :), f_rvec3(j, :)
                                                !print*, f_tensor(j, 3*(kat-1)+1:3*kat, 3*(jat-1)+1:3*jat, 3*(iat-1)+1:3*iat)
                                                !print*, tensor(index1, i,:,:,:)
                                                found = .True.
                                                f_tensor(j, 3*(iat-1)+1:3*iat, 3*(jat-1)+1:3*jat, 3*(kat-1)+1:3*kat) = &
                                                !f_tensor(j, 3*(kat-1)+1:3*kat, 3*(jat-1)+1:3*jat, 3*(iat-1)+1:3*iat) + &
                                                tensor(index1, i,:,:,:)
                                                EXIT
                                        endif
                                enddo
                                if(.not. found) then
                                        itrip = itrip + 1
                                        f_rvec2(itrip, :) = rvec2(index1, i, :)
                                        f_rvec3(itrip, :) = rvec3(index1, i, :)
                                        f_tensor(itrip, 3*(iat-1)+1:3*iat, 3*(jat-1)+1:3*jat, 3*(kat-1)+1:3*kat) = &
                                                tensor(index1, i,:,:,:)
                                endif
                        !endif
                enddo
                endif
        enddo
        enddo
        enddo

        print*, 'Expect ', itrip, ' triplets!'

    end subroutine

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    function norm(r) result (d)
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
        
        real(kind = DP), intent(in) :: r(3)
        real(kind = DP) :: d

        d = sqrt(dot_product(r,r))    

    end function norm 

    function norma2(m) result (d)
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)

        real(kind=DP), intent(in) :: m(:,:)

        real(kind=DP) :: d
        integer :: i, j, shapem(2)

        shapem = shape(m)
        d = 0.0_DP
        do i = 1, shapem(2)
                do j = 1, shapem(1)
                        d = d + m(j,i)**2
                enddo
        enddo
        d = sqrt(d)

    end function

    function dot(v,m) result (v1)
        IMPLICIT NONE
        INTEGER, PARAMETER :: DP = selected_real_kind(14,200)

        real(kind=DP), intent(in) :: v(3), m(3,3)

        real(kind=DP) :: v1(3)
        integer :: i

        v1 = 0.0_DP
        do i = 1, 3
                v1(:) = v1(:) + v(i)*m(i,:)
        enddo

    end function

end module
