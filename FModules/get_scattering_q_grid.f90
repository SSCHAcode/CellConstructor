
 module scattering_grids

    contains

    subroutine get_scattering_q_grid(rotations, irrqgrid, qgrid, scatt_qgrid, &
                    nirrqpt, nqpt, scatt_nqpt, nsym, scattering_grid, scattering_weights)

        implicit none

        integer, parameter :: DP = selected_real_kind(14,200)

        integer, intent(in) :: nirrqpt, nqpt, nsym, scatt_nqpt 
        real(kind=DP), intent(in) :: rotations(nsym,3,3)
        real(kind=DP), intent(in) :: irrqgrid(nirrqpt, 3), qgrid(nqpt, 3), scatt_qgrid(scatt_nqpt, 3)

        real(kind=DP), intent(out) :: scattering_grid(nirrqpt, scatt_nqpt, 3)
        integer, intent(out) :: scattering_weights(nirrqpt, scatt_nqpt)

        integer :: iqpt, jqpt, isym, jsym, isg, nsg, ilist, lenlist, tot_events
        integer :: sg_ind(nsym)
        real(kind=DP) :: q1(3), q2(3), q3(3), q21(3), q31(3)
        logical :: in_list

        scattering_grid(:,:,:) = 0.0_DP
        scattering_weights(:,:) = 0

        do iqpt = 1, nirrqpt
            q1 = irrqgrid(iqpt, :)

            !sg_ind(:) = 0
            !isg = 0
            !do isym = 1, nsym
            !    if(same_vector(q1, matmul(rotations(isym,:,:), q1))) then
            !            isg = isg + 1
            !            sg_ind(isg) = isym
            !     endif
            !enddo
            !nsg = isg
            !print*, 'Size of the small group of ', iqpt, ' q point is ', nsg

            lenlist = 0
            do jqpt = 1, scatt_nqpt
                q2 = scatt_qgrid(jqpt, :)
                q3 = -1.0_DP*q1 - q2
                in_list = .False.

                do ilist = 1, lenlist
                    if(.not. in_list) then
                        q21 = scattering_grid(iqpt, ilist, :)
                        q31 = -1.0_DP*q1 - q21
                        if(same_vector(q21, q3) .and. same_vector(q31, q2)) then
                            in_list = .True.
                            scattering_weights(iqpt, ilist) = scattering_weights(iqpt, ilist) + 1
                            EXIT
                       ! else
                           ! do isym = 1, nsg
                           !     jsym = sg_ind(isym)
                           !     if(same_vector(matmul(rotations(jsym,:,:), q2), q21) .and. &
                           !     same_vector(matmul(rotations(jsym,:,:), q3), q31)) then
                           !         in_list = .True.
                           !         scattering_weights(iqpt, ilist) = scattering_weights(iqpt, ilist) + 1
                           !         EXIT
                                !else if(same_vector(q2, matmul(rotations(jsym,:,:), q31)) .and. &
                                !same_vector(q3,  matmul(rotations(jsym,:,:), q21))) then
                                !    in_list = .True.
                                !    scattering_weights(iqpt, ilist) = scattering_weights(iqpt, ilist) + 1
                                !    EXIT
                           !     endif
                           ! enddo
                       endif 
                    endif
                enddo

                if(.not. in_list) then
                    lenlist = lenlist + 1
                    scattering_grid(iqpt, lenlist, :) = q2
                    scattering_weights(iqpt, lenlist) = scattering_weights(iqpt, lenlist) + 1
                endif

            enddo
            !print*, 'Final number of scattering events: ', lenlist
            !tot_events = 0 
            !do ilist = 1, lenlist
            !    tot_events = tot_events + scattering_weights(iqpt,ilist)
            !enddo
            !print*, 'Total number of scattering events: ', tot_events, scatt_nqpt
        enddo

    end subroutine get_scattering_q_grid

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    logical function same_vector(v1, v2)

        implicit none
        integer, parameter :: DP = selected_real_kind(14,200)

        real(kind=DP), intent(in) :: v1(3), v2(3)

        integer :: i, j, k
        real(kind=DP) :: v3(3)
        logical :: same 
        same = .False.
        if(all(abs(v1 - v2) < 1.0d-6)) then
            same = .True.
        else
            do i = -1, 1
                if(.not. same) then
                do j = -1, 1
                    if(.not. same) then
                    do k = -1, 1
                        v3 = v2 + (/dble(i), dble(j), dble(k)/)
                        if(all(abs(v1 - v3) < 1.0d-6)) then
                            same = .True.
                            EXIT
                         endif
                    enddo
                    endif
                enddo
                endif
            enddo
        endif
        same_vector = same
        return

    end function

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    logical function same_vector_nopbc(v1, v2)

        implicit none
        integer, parameter :: DP = selected_real_kind(14,200)

        real(kind=DP), intent(in) :: v1(3), v2(3)

        integer :: i, j, k
        real(kind=DP) :: v3(3)
        logical :: same 

        same = .False.
        if(all(abs(v1 - v2) < 1.0d-6)) then
            same = .True.
        endif

        same_vector_nopbc = same
        return

    end function

end module 
