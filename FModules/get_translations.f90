subroutine get_translations(pols, masses, nmod, nat, trans)
    implicit none

    integer, intent(in) :: nat, nmod
    double precision, intent(in) :: pols(nat * 3, nmod), masses(nat)
    integer, intent(out) :: trans(nmod)

    integer i, j
    double precision thr, move

    trans(:) = 0

    do i = 1, nmod
        thr = 0
        do j = 1, nat 
            move = sum(abs(pols(3 * (j-1) + 1 : 3 * (j-1) + 3, i) / dsqrt(masses(j)) - &
                pols(:3, i) / dsqrt(masses(1))))
            thr = thr + move
        enddo

        if (thr .lt. 1d-6) then    
            trans(i) = 1
        end if
    enddo
end subroutine get_translations