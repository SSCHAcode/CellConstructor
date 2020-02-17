! Memory optimized centering
module second_order_centering
    contains

    subroutine analysis(Far, nat, nq1, nq2, nq3, tol, alat, tau, tensor, weight, xR2)
        implicit none
        integer, intent(in) :: nat, nq1, nq2, nq3
        double precision, intent(in), dimension(3,3) :: alat 
        double precision, intent(in), dimension(nat,3) :: tau 
        integer, intent(in) :: Far 
        double precision, intent(in) :: tol 
        double precision, intent(in), dimension(nq1*nq2*nq3, 3*nat, 3*nat, 3*nat) :: tensor 

        integer, intent(out) :: weight(nat, nat, nq1*nq2*nq3) 
        integer, intent(out) :: xR2(3, (2*Far + 1)*(2*Far + 1)*(2*Far + 1), nat,nat,nq1*nq2*nq3)

        ! Here the variable we need for the computations
        integer :: s, t, t_lat, xt_cell_orig(3), xt(3), alpha, beta, RRt(3, (2*Far+1)**3)
        integer :: LLt, MMt, NNt, ww, h 
        double precision, dimension(3) :: s_vec, t_vec, r_vect 
        double precision :: summa, dist_min, dist
        

        ! Ok, lets start by cycling over the atomic indices
        do s = 1, nat
            ! Get the vector to the first atom
            ! in the unit cell
            s_vec = tau(s, :)

            do t = 1, nat 
                ! Get the second vector on the unit cell
                t_vec = tau(t, :)

                ! Get the crystal coordinate of the lattice vector
                ! that represent the t atom
                xt_cell_orig = one_to_three

                ! Cycle on the lattice for the second atom
                do t_lat = 1, nq1*nq2*nq3

                    summa = 0.0d0
                    do alpha = 1, 3
                        do beta = 1, 3
                            summa = summa + dabs(tensor(t_lat, alpha + (s-1)*3, beta +(t-1)*3))
                        enddo
                    enddo
                    
                    ! Discard if the block of the tensor is empty
                    if (summa < 1.0d-8) cycle 

                    ! Lets start creating the replica in the supercell
                    dist_min = 1d10 
                    ww = 0 ! The temporany weight

                    ! Cycle over the replicas
                    do LLt = -Far, Far 
                        do MMt = -Far, Far
                            do NNt = -Far, Far 
                                ! Generate the position in the supercell
                                ! Of the t atom with respect to s
                                xt = xt_cell_orig(:)
                                xt(1) = xt(1) + LLt*nq1
                                xt(2) = xt(2) + MMt*nq2
                                xt(3) = xt(3) + NNt*nq3

                                r_vect = xt(1) * alat(1,:)
                                r_vect = r_vect + xt(2) * alat(2, :)
                                r_vect = r_vect + xt(3) * alat(3, :)
                                
                                ! Now add and subtract the atomic coordinates
                                r_vect = r_vect + t_vec - s_vec
                                
                                ! Get the distance
                                dist = sum(r_vect(:)*r_vect(:))
                                dist = dsqrt(dist)

                                ! Check if it must be updated
                                if (dist < (dist_min - tol)) then
                                    dist_min = dist 
                                    ww = 1
                                    
                                    ! Update the correct value
                                    RRt(:, ww) = xt
                                else if (abs(dist_min - dist) <= tol) then 
                                    ! Add a new weight and vector
                                    ww = ww+1
                                    RRt(:, ww) = xt
                                end if
                            enddo
                        enddo
                    enddo

                    ! Now we found for this couple of vectors
                    ! The list of the correct vectors
                    weight(s, t, t_lat) = ww 
                    do h = 1, ww 
                        xR2(:, h, s, t, t_lat) = RRt(:, h)
                    enddo
                enddo
            enddo
        enddo
    end subroutine analysis


    
end module second_order_centering

                                



