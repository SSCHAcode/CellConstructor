module second_order_centering


contains 




! Memory optimized centering
!=================================================================================
!
subroutine analysis(Far,tol, dmax,sc_size,xR2_list,alat, tau, tensor,weight,xR2,nat,n_blocks)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    integer, intent(in) :: Far, nat
    real(kind=DP), intent(in) :: dmax(nat)
    real(kind=DP), intent(in), dimension(3,3) :: alat 
    real(kind=DP), intent(in), dimension(nat,3) :: tau
    integer, intent(in), dimension(3,n_blocks) :: xR2_list 
    real(kind=DP), intent(in) :: tol
    real(kind=DP), intent(in), &
     dimension(3*nat,3*nat,n_blocks) :: tensor
    integer, intent(in) :: n_blocks 
    integer, intent(in) :: sc_size(3)
    !
    integer, intent(out) :: weight(nat,nat,n_blocks)
    integer, intent(out) :: xR2(3,(2*Far+1)*(2*Far+1),nat,nat,n_blocks)      
    !
    integer :: s,t,u,t_lat, u_lat, xt_cell_orig(3),xt(3)
    integer :: RRt(3,(2*Far+1)**2),ww,i_block
    real(kind=DP) :: s_vec(3), t_vec(3), SC_t_vec(3)
    integer :: LLt,MMt,NNt, alpha, beta, jn1 ,jn2,h
    real(kind=DP) :: perimeter, perim_min,summa
    logical :: Found
    
    Found=.False.
    weight=0
    !
    do s = 1, nat
        s_vec=tau(s,:)
    do t = 1, nat        
        t_vec=tau(t,:)
    do i_block=1,n_blocks
        xt_cell_orig=xR2_list(:,i_block)
           ! 
           ! Check total value  ==================================================================
           summa=0.0_dp
           !
           do alpha=1,3
           do beta=1,3
               summa=summa+abs(tensor( alpha+(s-1)*3,beta+(t-1)*3,i_block))
           end do
           end do
           !
           if (summa < 1.0d-8) cycle
           ! =====================================================================================
           !                
           !========================= Supercell replicas =========================================
           perim_min=1.0e10_dp 
           ww=0            
           !
           do LLt = -Far, Far
           do MMt = -Far, Far
           do NNt = -Far, Far
               xt=xt_cell_orig+(/LLt*sc_size(1),MMt*sc_size(2),NNt*sc_size(3)/)
               SC_t_vec=cryst_to_cart(xt,alat)+t_vec
                                   !
                                   ! Go on only if distances are within the maximum allowed
                                   ! 
                                   if ( within_dmax(s_vec,SC_t_vec ,dmax(s),dmax(t),tol) ) then
                                        !
                                        perimeter=compute_perimeter(s_vec,SC_t_vec)
                                        !                                    
                                        if (perimeter < (perim_min - tol)) then
                                            ! update minimum perimeter/triangle
                                            perim_min = perimeter
                                            ww = 1
                                            !
                                            RRt(:,ww)=xt        
                                            !
                                        else if (abs(perimeter - perim_min) <= tol) then
                                            ! add weight/triangle for this perimeter
                                            ww = ww+1
                                            !
                                            RRt(:,ww)=xt        
                                            !
                                        end if 
                                        !
                                        !
                                        !
                                   end if
                                   !
                                   !
                                   !
           end do
           end do
           end do
           !========================= End supercell replicas ========================================
           ! Assign
           weight(s,t,i_block)=ww
           if ( ww > 0 ) then
           !
           Found=.True. ! at least one couple found
           !
           do h = 1, ww
               xR2(:,h,s,t,i_block)=RRt(:,h)
           end do
           !
           end if
           !
    end do
    !
    end do
    !
    end do
    !
    if ( .not. Found ) then
            write(*,*) " "
            write(*,*) " ERROR: no nonzero couple found during centering,        "
            write(*,*) "        the execution stops here.                        "
            write(*,*) "        Relax the constraint imposed                     "
            write(*,*) "        on the maximum distance allowed (nneigh)         " 
            write(*,*) " "
            stop    
    end if    
    !
end subroutine analysis
!=================================================================================


!=================================================================================
subroutine center(original,weight,xR2_list,xR2,nat,centered,Far,n_blocks,n_blocks_old)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    integer, intent(in) :: nat,n_blocks,Far
    integer, intent(in), dimension(3,n_blocks) :: xR2_list
    integer, intent(in), &
                         dimension(3,(2*Far+1)*(2*Far+1), &
                         nat,nat,n_blocks_old) :: xR2   
    integer, intent(in) :: weight(nat,nat,n_blocks_old)                     
    integer, intent(in) :: n_blocks_old
    real(kind=DP), intent(in), &
     dimension(3*nat,3*nat,n_blocks_old) :: original
    !
    real(kind=DP), intent(out), &
     dimension(3*nat,3*nat,n_blocks) :: centered
    !
    integer :: s,t,t_lat,h,i_block,j_block,alpha,beta,jn1,jn2

    centered=0.0_dp
    !
    do s = 1, nat
    do t = 1, nat
    do j_block=1, n_blocks_old
        !
        if ( weight(s,t,j_block) > 0 ) then
        !
        do h=1,weight(s,t,j_block)
           ! 
           do i_block=1, n_blocks
           !
           if ( sum( abs(xR2_list(:,i_block)-xR2(:,h,s,t,j_block)) ) < 1.0d-8 ) then
                    !
                    do jn1 = 1 + (s-1)*3,3 + (s-1)*3
                    do jn2 = 1 + (t-1)*3,3 + (t-1)*3
                    !
                    centered(jn1,jn2,i_block)=original(jn1,jn2,j_block)/weight(s,t,j_block)
                    !
                    end do
                    end do
                    !
                    cycle     
           end if
           !
           end do
           !
         end do  
         !  
         end if
         !
    end do
    end do
    end do
    !       
end subroutine center
!
!=================================================================================


!=================================================================================
function within_dmax(v1,v2,d1,d2,tol)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    logical  :: within_dmax    
    real(kind=DP), intent(in), dimension(3) :: v1,v2
    real(kind=DP), intent(in) :: d1,d2,tol
    !
    within_dmax =  ( norm2(v1-v2) <= min(d1,d2)+tol ) 
    !               
end function within_dmax
!=================================================================================
function compute_perimeter(v1,v2)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP) :: compute_perimeter
    real(kind=DP), intent(in), dimension(3) :: v1,v2
    !
    compute_perimeter=norm2(v1-v2)
    !
end function compute_perimeter
!=================================================================================                     
function cryst_to_cart(v,alat)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP), dimension(3)    :: cryst_to_cart
    integer, dimension(3), intent(in)          :: v
    real(kind=DP), dimension(3,3), intent(in) :: alat
    !
    integer :: i
    !  
    do i=1,3
        cryst_to_cart(i)=alat(1,i)*v(1)+alat(2,i)*v(2)+alat(3,i)*v(3)
    end do
    !   
end function cryst_to_cart

end module second_order_centering