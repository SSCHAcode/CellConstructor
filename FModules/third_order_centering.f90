module third_order_centering


contains 




! Memory optimized centering
!=================================================================================
!
subroutine analysis(Far,tol, dmax,sc_size,xR2_list,xR3_list,alat, tau, tensor,weight,xR2,xR3,nat,n_blocks)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    integer, intent(in) :: Far, nat
    real(kind=DP), intent(in) :: dmax(nat)
    real(kind=DP), intent(in), dimension(3,3) :: alat 
    real(kind=DP), intent(in), dimension(nat,3) :: tau
    integer, intent(in), dimension(3,n_blocks) :: xR2_list,xR3_list    
    real(kind=DP), intent(in) :: tol
    real(kind=DP), intent(in), &
     dimension(3*nat,3*nat,3*nat,n_blocks) :: tensor
    integer, intent(in) :: n_blocks 
    integer, intent(in) :: sc_size(3)
    !
    integer, intent(out) :: weight(nat,nat,nat,n_blocks)
    integer, intent(out) :: xR2(3,(2*Far+1)*(2*Far+1)*(2*Far+1),nat,nat,nat,n_blocks)    
    integer, intent(out) :: xR3(3,(2*Far+1)*(2*Far+1)*(2*Far+1),nat,nat,nat,n_blocks)    
    !
    integer :: s,t,u,t_lat, u_lat, xt_cell_orig(3),xt(3),xu_cell_orig(3),xu(3)
    integer :: RRt(3,(2*Far+1)**3),RRu(3,(2*Far+1)**3),ww,i_block
    real(kind=DP) :: s_vec(3), t_vec(3), u_vec(3), SC_t_vec(3), SC_u_vec(3)
    integer :: LLt,MMt,NNt,LLu,MMu,NNu, alpha, beta, gamma, jn1 ,jn2, jn3,h
    real(kind=DP) :: perimeter, perim_min,summa
    logical :: Found
    
    Found=.False.
    weight=0
    !
    do s = 1, nat
        s_vec=tau(s,:)
    do t = 1, nat        
        t_vec=tau(t,:)
    do u = 1,nat
        u_vec=tau(u,:)
    
    do i_block=1,n_blocks
        xt_cell_orig=xR2_list(:,i_block)
        xu_cell_orig=xR3_list(:,i_block)
           ! 
           ! Check total value  ==================================================================
           summa=0.0_dp
           !
           do alpha=1,3
           do beta=1,3
           do gamma=1,3
               summa=summa+abs(tensor( alpha+(s-1)*3,beta+(t-1)*3,gamma+(u-1)*3,i_block))
           end do
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
           do LLu = -Far, Far
           do MMu = -Far, Far
           do NNu = -Far, Far
               xu=xu_cell_orig+(/LLu*sc_size(1),MMu*sc_size(2),NNu*sc_size(3)/)
               SC_u_vec=cryst_to_cart(xu,alat)+u_vec
                                   !
                                   ! Go on only if distances are within the maximum allowed
                                   ! 
                                   if ( within_dmax(s_vec,SC_t_vec ,SC_u_vec,dmax(s),dmax(t),dmax(u),tol) ) then
                                        !
                                        perimeter=compute_perimeter(s_vec,SC_t_vec ,SC_u_vec)
                                        !                                    
                                        if (perimeter < (perim_min - tol)) then
                                            ! update minimum perimeter/triangle
                                            perim_min = perimeter
                                            ww = 1
                                            !
                                            RRt(:,ww)=xt        
                                            RRu(:,ww)=xu
                                            !
                                        else if (abs(perimeter - perim_min) <= tol) then
                                            ! add weight/triangle for this perimeter
                                            ww = ww+1
                                            !
                                            RRt(:,ww)=xt        
                                            RRu(:,ww)=xu      
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
           !
           !
           end do
           end do
           end do    
           !========================= End supercell replicas ========================================
           ! Assign
           weight(s,t,u,i_block)=ww
           if ( ww > 0 ) then
           !
           Found=.True. ! at least one triplet found
           !
           do h = 1, ww
               xR2(:,h,s,t,u,i_block)=RRt(:,h)
               xR3(:,h,s,t,u,i_block)=RRu(:,h)
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
    end do
    !
    if ( .not. Found ) then
            write(*,*) " "
            write(*,*) " ERROR: no nonzero triplets found during centering,      "
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
subroutine center(original,weight,xR2_list,xR2,xR3_list,xR3,nat,centered,Far,n_blocks,n_blocks_old)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    integer, intent(in) :: nat,n_blocks,Far
    integer, intent(in), dimension(3,n_blocks) :: xR2_list,xR3_list
    integer, intent(in), &
                         dimension(3,(2*Far+1)*(2*Far+1)*(2*Far+1), &
                         nat,nat,nat,n_blocks_old) :: xR2,xR3   
    integer, intent(in) :: weight(nat,nat,nat,n_blocks_old)                     
    integer, intent(in) :: n_blocks_old
    real(kind=DP), intent(in), &
     dimension(3*nat,3*nat,3*nat,n_blocks_old) :: original
    !
    real(kind=DP), intent(out), &
     dimension(3*nat,3*nat,3*nat,n_blocks) :: centered
    !
    integer :: s,t,u,t_lat,u_lat,h,i_block,j_block,alpha,beta,gamma,jn1,jn2,jn3

    centered=0.0_dp
    !
    do s = 1, nat
    do t = 1, nat
    do u = 1, nat    
    do j_block=1, n_blocks_old
        !
        if ( weight(s,t,u,j_block) > 0 ) then
        
        do h=1,weight(s,t,u,j_block)
           ! 
           do i_block=1, n_blocks
           !
           if ( sum( abs(xR2_list(:,i_block)-xR2(:,h,s,t,u,j_block)) +  &
                     abs(xR3_list(:,i_block)-xR3(:,h,s,t,u,j_block))   ) < 1.0d-8 ) then
                    !
                    do jn1 = 1 + (s-1)*3,3 + (s-1)*3
                    do jn2 = 1 + (t-1)*3,3 + (t-1)*3
                    do jn3 = 1 + (u-1)*3,3 + (u-1)*3                    
                    !
                    centered(jn1,jn2,jn3,i_block)=original(jn1,jn2,jn3,j_block)/weight(s,t,u,j_block)
                    !
                    end do
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
    end do
    !       
end subroutine center
!
!=================================================================================


!=================================================================================
subroutine center_sparse(original,weight,xR2_list,xR2,xR3_list,xR3,centered,n_sparse_blocks, &
                        xR2_sparse_list,xR3_sparse_list,atom_sparse_list, &
                        r_blocks_sparse_list,Far, nat,n_sup,n_blocks)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    integer, intent(in) :: nat,n_sup,n_blocks,Far
    integer, intent(in), dimension(3,n_blocks) :: xR2_list,xR3_list
    integer, intent(in), &
                         dimension(3,(2*Far+1)*(2*Far+1)*(2*Far+1)*n_sup, &
                         nat,nat,n_sup,nat,n_sup) :: xR2,xR3   
    integer, intent(in) :: weight(nat,nat,n_sup,nat,n_sup)                     
    real(kind=DP), intent(in), &
     dimension(3*nat,3*nat,3*nat,n_sup,n_sup) :: original
    !
    real(kind=DP), intent(out), &
     dimension(3*nat,3*nat,3*nat,n_blocks) :: centered
    integer, intent(out) :: n_sparse_blocks,xR2_sparse_list(3,n_blocks*nat*nat*nat), &
                            xR3_sparse_list(3,n_blocks*nat*nat*nat),atom_sparse_list(3,n_blocks*nat*nat*nat), &
                            r_blocks_sparse_list(n_blocks*nat*nat*nat)
    !
    integer :: s,t,u,t_lat,u_lat,h,i_block,alpha,beta,gamma,jn1,jn2,jn3
    logical :: takeit
    
    n_sparse_blocks=0
    centered=0.0_dp
    !
    do s = 1, nat
    do t = 1, nat        
    do t_lat = 1, n_sup
    do u = 1,nat
    do u_lat = 1, n_sup
        !   
        do h=1,weight(s,t,t_lat,u,u_lat)
           ! 
           do i_block=1, n_blocks
           !
           if ( sum( abs(xR2_list(:,i_block)-xR2(:,h,s,t,t_lat,u,u_lat)) +  &
                     abs(xR3_list(:,i_block)-xR3(:,h,s,t,t_lat,u,u_lat))   ) < 1.0d-8 ) then
                    !        
                    takeit=.false.
                    do jn1 = 1 + (s-1)*3,3 + (s-1)*3
                    do jn2 = 1 + (t-1)*3,3 + (t-1)*3
                    do jn3 = 1 + (u-1)*3,3 + (u-1)*3                    
                    !
                    centered(jn1,jn2,jn3,i_block)=original(jn1,jn2,jn3,t_lat,u_lat)/weight(s,t,t_lat,u,u_lat)
                    if ( centered(jn1,jn2,jn3,i_block) > 1.0d-7 ) takeit=.true.
                    !
                    end do
                    end do
                    end do
                    
                    if (takeit) then
                       n_sparse_blocks=n_sparse_blocks+1
                       atom_sparse_list(:,n_sparse_blocks)=(/s-1,t-1,u-1/) 
                       xR2_sparse_list(:,n_sparse_blocks)=xR2_list(:,i_block)
                       xR3_sparse_list(:,n_sparse_blocks)=xR3_list(:,i_block)
                       r_blocks_sparse_list(n_sparse_blocks)=i_block-1
                    end if
                    
                    !
                    cycle     
           end if
           !
           end do
           !
         end do  
         !  
    end do
    end do
    end do
    end do
    end do
    !
    !       
end subroutine center_sparse

! !=================================================================================
! 
! Fast centering (memory expensive)
!
subroutine pre_center(Far,nat,nq1,nq2,nq3,tol, alat, tau, original,centered,lat_min,lat_max)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    integer, intent(in) :: nat,nq1,nq2,nq3
    real(kind=DP), intent(in), dimension(3,3) :: alat 
    real(kind=DP), intent(in), dimension(nat,3) :: tau
    integer, intent(in) :: Far
    real(kind=DP), intent(in) :: tol
    real(kind=DP), intent(in), &
     dimension(3*nat,3*nat,3*nat,nq1*nq2*nq3,nq1*nq2*nq3) :: original
    ! 
    real(kind=DP), intent(out), &
     dimension(3*nat,3*nat,3*nat,(2*Far+1)*(2*Far+1)*(2*Far+1)*nq1*nq2*nq3,&
               (2*Far+1)*(2*Far+1)*(2*Far+1)*nq1*nq2*nq3) :: centered
    integer, intent(out), dimension(3) :: lat_min, lat_max 
    !
    integer :: s,t,u,t_lat, u_lat, xt_prov(3),xt(3),xu_prov(3),xu(3),RRt(3,(2*Far+1)**3),RRu(3,(2*Far+1)**3)
    real(kind=DP) :: s_vec(3), t_vec(3), u_vec(3), SC_t_vec(3), SC_u_vec(3)
    integer :: LLt,MMt,NNt,LLu,MMu,NNu, I,J, weight, alpha, beta, gamma, jn1 ,jn2, jn3,h
    real(kind=DP) :: perimeter, perim_min
    integer :: lat_min_tmp(3), lat_max_tmp(3), lat_min_iniz(3), lat_max_iniz(3)

    
    
    lat_min_iniz=(/-Far*nq1,-Far*nq2,-Far*nq3/)
    lat_max_iniz=(/(nq1-1)+Far*nq1,(nq2-1)+Far*nq2,(nq3-1)+Far*nq3/)
    
    
    
    lat_min=(/100,100,100/)
    lat_max=(/-100,-100,-100/)
    centered=0.0_dp
    !
    do s = 1, nat
        s_vec=tau(s,:)
        do t = 1, nat        
        do t_lat = 1, nq1*nq2*nq3
            t_vec=tau(t,:)
            xt_prov=one_to_three( t_lat,(/0,0,0/),(/nq1-1,nq2-1,nq3-1/) )
            do u = 1,nat
            do u_lat = 1, nq1*nq2*nq3
                u_vec=tau(u,:)
                xu_prov=one_to_three( u_lat,(/0,0,0/),(/nq1-1,nq2-1,nq3-1/) ) 
                !
                !
                perim_min=1.0e10_dp
                weight=0            
                !
                do LLt = -Far,Far
                    do MMt = -Far, Far
                        do NNt = -Far, Far
                            xt=xt_prov+(/LLt*nq1,MMt*nq2,NNt*nq3/)
                            SC_t_vec=cryst_to_cart(xt,alat)+t_vec
                            do LLu = -Far,Far
                                do MMu = -Far,Far
                                    do NNu = -Far, Far
                                        xu=xu_prov+(/LLu*nq1,MMu*nq2,NNu*nq3/)
                                        SC_u_vec=cryst_to_cart(xu,alat)+u_vec
                                        !
                                        !
                                        perimeter=compute_perimeter(s_vec,SC_t_vec ,SC_u_vec)
                                        !
                                        !
                                        if (perimeter < (perim_min - tol)) then
                                            perim_min = perimeter
                                            weight = 1
!                                           !
                                            RRt(:,weight)=xt        
                                            RRu(:,weight)=xu        
                                            lat_min_tmp=min_el_wise_2(xt, xu) 
                                            lat_max_tmp=max_el_wise_2(xt, xu)  
                                        else if (abs(perimeter - perim_min) <= tol) then
                                            weight = weight+1
!                                           !
                                            RRt(:,weight)=xt        
                                            RRu(:,weight)=xu        
                                            lat_min_tmp=min_el_wise_3(lat_min_tmp, xt , xu )
                                            lat_max_tmp=max_el_wise_3(lat_max_tmp, xt , xu )  
                                        end if
                                        !
                                    end do
                                end do
                            end do
                        end do
                    end do
                end do    
                ! Assign
                do h = 1, weight
                    I=three_to_one(RRt(:,h),lat_min_iniz,lat_max_iniz)
                    J=three_to_one(RRu(:,h),lat_min_iniz,lat_max_iniz)    
                    !
                    do alpha = 1,3
                        jn1 = alpha + (s-1)*3
                    do beta = 1,3
                        jn2 = beta  + (t-1)*3
                    do gamma = 1,3
                        jn3 = gamma + (u-1)*3                        
                        !
                        centered(jn1,jn2,jn3,I,J)=original(jn1,jn2,jn3,t_lat,u_lat)/weight
                        !
                    end do
                    end do    
                    end do
                end do    
                !
                !
                lat_min=min_el_wise_2(lat_min, lat_min_tmp)
                lat_max=max_el_wise_2(lat_max, lat_max_tmp)
                !
                !
            end do
            end do
        end do
        end do
    end do
    !
    !
end subroutine pre_center
!=================================================================================
subroutine assign(alat,lat_min_prev,lat_max_prev,centered_prev,lat_min,lat_max,n_sup_WS,nat,n_sup_WS_prev,centered,x2,x3,R2,R3)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    integer, intent(in) :: nat,n_sup_WS,n_sup_WS_prev
    real(kind=DP), intent(in), dimension(3,3) :: alat 
    integer, intent(in), dimension(3) :: lat_min_prev,lat_max_prev,lat_min,lat_max
    real(kind=DP), intent(in), &
     dimension(3*nat,3*nat,3*nat,n_sup_WS_prev,n_sup_WS_prev) :: centered_prev
    !
    real(kind=DP), intent(out), &
     dimension(3*nat,3*nat,3*nat,n_sup_WS*n_sup_WS) :: centered
    integer, intent(out), dimension(3,n_sup_WS*n_sup_WS) :: x2,x3
    real(kind=DP), intent(out), dimension(3,n_sup_WS*n_sup_WS) :: R2,R3
    !
    integer :: I,J,II,JJ,counter
    !
    centered=0.0_dp
    counter=0 
     
    do I=1,n_sup_WS
        do J=1, n_sup_WS
            ! 
            counter=counter+1
            !
            x2(:,counter)=one_to_three(I,lat_min,lat_max)
            x3(:,counter)=one_to_three(J,lat_min,lat_max)

            !          
            R2(:,counter)=cryst_to_cart(x2(:,counter),alat)          
            R3(:,counter)=cryst_to_cart(x3(:,counter),alat)                    
            !
            II=three_to_one(x2(:,counter),lat_min_prev,lat_max_prev)
            JJ=three_to_one(x3(:,counter),lat_min_prev,lat_max_prev)  
            !
            centered(:,:,:,counter)=centered_prev(:,:,:,II,JJ)
            !
        end do
    end do    
    !
    !
end subroutine assign
!=================================================================================

!=================================================================================
function within_dmax(v1,v2,v3,d1,d2,d3,tol)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    logical  :: within_dmax    
    real(kind=DP), intent(in), dimension(3) :: v1,v2,v3
    real(kind=DP), intent(in) :: d1,d2,d3,tol
    
    within_dmax =  ( norm2(v1-v2) <= min(d1,d2)+tol ) .AND. &
                   ( norm2(v1-v3) <= min(d1,d3)+tol ) .AND. &
                   ( norm2(v2-v3) <= min(d2,d3)+tol )   
                   
end function within_dmax
!=================================================================================
function compute_perimeter(v1,v2,v3)
    implicit none
    INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
    real(kind=DP) :: compute_perimeter
    real(kind=DP), intent(in), dimension(3) :: v1,v2,v3
    !
    compute_perimeter=norm2(v1-v2)+norm2(v2-v3)+norm2(v3-v1)
end function compute_perimeter
!=================================================================================    
function three_to_one_len(v,v_min,v_len)
   implicit none
   INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
   integer :: three_to_one_len
   integer, intent(in), dimension(3) :: v, v_min, v_len 
   !
   three_to_one_len=(v(1)-v_min(1))*v_len(2)*v_len(3)+(v(2)-v_min(2))*v_len(3)+(v(3)-v_min(3))+1
end function three_to_one_len   
!=================================================================================    
function three_to_one(v,v_min,v_max)
   implicit none
   INTEGER, PARAMETER :: DP = selected_real_kind(14,200)
   integer :: three_to_one
   integer, intent(in), dimension(3) :: v, v_min, v_max 
   integer             :: v_len(3)
   !
   v_len=v_max-v_min+(/1,1,1/)
   three_to_one=(v(1)-v_min(1))*v_len(2)*v_len(3)+(v(2)-v_min(2))*v_len(3)+(v(3)-v_min(3))+1
end function three_to_one
!=================================================================================    
function one_to_three_len(J,v_min,v_len)
    implicit none
    integer, dimension(3) :: one_to_three_len
    integer, intent(in) :: J 
    integer, intent(in) :: v_min(3), v_len(3)
    !
    one_to_three_len(1)              =   (J-1) / (v_len(3) * v_len(2))                    + v_min(1) 
    one_to_three_len(2)              =   MOD( J-1 ,  (v_len(3) * v_len(2)) ) / v_len(3)   + v_min(2) 
    one_to_three_len(3)              =   MOD( J-1 ,   v_len(3) )                          + v_min(3) 
end function one_to_three_len
!=================================================================================    
function one_to_three(J,v_min,v_max)
    implicit none
    integer, dimension(3) :: one_to_three
    integer, intent(in) :: J 
    integer, intent(in) :: v_min(3), v_max(3)
    integer             :: v_len(3)
    !
    v_len=v_max-v_min+(/1,1,1/)
    one_to_three(1)              =   (J-1) / (v_len(3) * v_len(2))                    + v_min(1) 
    one_to_three(2)              =   MOD( J-1 ,  (v_len(3) * v_len(2)) ) / v_len(3)   + v_min(2) 
    one_to_three(3)              =   MOD( J-1 ,   v_len(3) )                          + v_min(3)   
end function one_to_three
!=================================================================================    
function min_el_wise_2(a,b)
    implicit none        
    integer, dimension(3)             :: min_el_wise_2
    integer, dimension(3), intent(in) :: a, b
    !
    min_el_wise_2=(/min(a(1),b(1)),min(a(2),b(2)),min(a(3),b(3))/)
    !
end function min_el_wise_2   
!=================================================================================                     
function max_el_wise_2(a,b)
    implicit none        
    integer, dimension(3)             :: max_el_wise_2
    integer, dimension(3), intent(in) :: a, b
    !
    max_el_wise_2=(/max(a(1),b(1)),max(a(2),b(2)),max(a(3),b(3))/)
    !
end function max_el_wise_2   
!=================================================================================                     
function min_el_wise_3(a,b,c)
    implicit none        
    integer, dimension(3)             :: min_el_wise_3
    integer, dimension(3), intent(in) :: a, b, c
    !
    min_el_wise_3=(/min(a(1),b(1),c(1)),min(a(2),b(2),c(2)),min(a(3),b(3),c(3))/)
    !
end function min_el_wise_3   
!=================================================================================                     
function max_el_wise_3(a,b,c)
    implicit none        
    integer, dimension(3)             :: max_el_wise_3
    integer, dimension(3), intent(in) :: a, b, c
    !
    max_el_wise_3=(/max(a(1),b(1),c(1)),max(a(2),b(2),c(2)),max(a(3),b(3),c(3))/)
    !
end function max_el_wise_3   
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

end module third_order_centering