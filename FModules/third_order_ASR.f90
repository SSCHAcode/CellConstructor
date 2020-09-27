module third_order_ASR



logical :: perm_initialized = .false.
integer, allocatable :: P(:,:)

logical :: R2index_initialized = .false.
integer, allocatable :: index_blocks_R2(:,:)
integer, allocatable :: num_blocks_R2(:)

contains 

! ============================== INITIALIZATIONS ============================================

subroutine initialize_R2index(xR2,xR2list,totnum_R2, n_blocks,SCLat,PBC)
implicit none
integer, intent(in)  :: xR2(3,n_blocks),n_blocks,totnum_R2,xR2list(3,totnum_R2)
integer, intent(in), optional :: SCLat(3)
logical, intent(in)           :: PBC
!
integer :: i_R2, i_block, counter,vec(3)


    write(*,*) " "
    write(*,*) "Initialize indices for sum on third index..."

    !
    ! Find blocks with a fixed R2 value
    !          
    allocate(num_blocks_R2(totnum_R2))
    allocate(index_blocks_R2(n_blocks,totnum_R2))    
    !
    !
    do i_R2=1,totnum_R2 ! loop on (different) R2 
        !
        vec(:)=xR2list(:,i_R2)
        ! Look for blocks having R2_list(:,i_R2) as R2
        counter=0
        do i_block=1, n_blocks  ! loop on blocks            
            if ( Geq( vec, xR2(:,i_block),SCLat,PBC) ) then
             counter=counter+1
             index_blocks_R2(counter,i_R2)=i_block 
            end if 
        end do
        num_blocks_R2(i_R2)=counter ! total number of blocks with this R2
        !
    end do
    !
    R2index_initialized = .true.
    !
    write(*,*) "Initialize indices for sum on third index...done"    
    write(*,*) " "
end subroutine initialize_R2index

! ============================================================================================

subroutine initialize_perm(R23, n_blocks,SClat,PBC)
    implicit none
    integer, intent(in)           :: R23(6,n_blocks),n_blocks
    integer, intent(in)           :: SCLat(3)
    logical, intent(in)           :: PBC
    !
    integer :: x(6), y(6), xF(6,2:6),i_block,j_block
    logical :: found(2:6)
    integer :: SCLat2(6)
   
    write(*,*) " "
    write(*,*) "Initialize indices for permutation..."
   
    SCLat2(1:3) = SCLat(1:3)
    SCLat2(4:6) = SCLat(1:3)
    
    allocate(P(n_blocks,2:6))

    do i_block = 1, n_blocks
                
        x=R23(:,i_block)

        xF=F(x)

        found=.False. 

        do j_block = 1, n_blocks
        !    
            y=R23(:,j_block)
            
            if ( Geq( y, xF(:,2) ,SCLat2, PBC) ) then
                P(i_block,2)=j_block
                found(2)=.True.
            end if

            if ( Geq( y, xF(:,3) ,SCLat2, PBC) ) then
                P(i_block,3)=j_block
                found(3)=.True.
            end if
            
            if ( Geq( y, xF(:,4) ,SCLat2, PBC) ) then
                P(i_block,4)=j_block
                found(4)=.True.
            end if        

            if ( Geq( y, xF(:,5) ,SCLat2, PBC) ) then
                P(i_block,5)=j_block
                found(5)=.True.
            end if        
            
            if ( Geq( y, xF(:,6) ,SCLat2, PBC) ) then
                P(i_block,6)=j_block
                found(6)=.True.
            end if
            
            if ( ALL (found)  ) cycle
        !    
        end do
        
        if ( .not. ALL (found) ) then
        
            write(*,*) " ERROR: new triplets found during the permutation symmetry initialization "
            write(*,*) "        the execution stops here. "
            write(*,*) "        If the 3rdFC is centered, try to repeat the centering with higher Far "            
            write(*,*) "        If the 3rdFC is not centered, try to repeat the ASR imposition with PBC=True "            
            stop
            
        end if
    
end do

perm_initialized =.true.

    write(*,*) "Initialize indices for permutation...done"
    write(*,*) " "
    
end subroutine initialize_perm

!==========================================================================================

function Geq(v1,v2,Lat,PBC)
    implicit none
    logical    :: Geq
    integer, dimension(:), intent(in)           :: v1,v2
    integer, dimension(:), intent(in) :: Lat
    logical, intent(in) :: PBC
    !    
    if (PBC) then
     Geq= ( ALL (mod(v1-v2,Lat)==0) )
    else
     Geq= ( ALL (v1-v2 ==0) )
    end if
    !
    !   
end function Geq

!==============================================================================================

function F(x)
    implicit none
    integer :: F(6,2:6)
    integer, intent(in) :: x(6)
    
    
    ! Permutations of 0 R2 R3 a b c, exploiting the translation
    ! symmetry to keep the first lattice vector equal to 0

    ! 2.  0  R3 R2  a c b      0     R3     R2  a c b        
    ! 3.  R3  0 R2  c a b  ->  0    -R3  R2-R3  c a b              
    ! 4.  R3 R2  0  c b a  ->  0  R2-R3    -R3  c b a             
    ! 5.  R2 R3  0  b c a  ->  0  R3-R2    -R2  b c a 
    ! 6.  R2  0 R3  b a c  ->  0    -R2  R3-R2  b a c 
    
    !
    F(:,2)=(/      x(4),      x(5),      x(6),      x(1),      x(2),      x(3) /)
    F(:,3)=(/     -x(4),     -x(5),     -x(6), x(1)-x(4), x(2)-x(5), x(3)-x(6) /)
    F(:,4)=(/ x(1)-x(4), x(2)-x(5), x(3)-x(6),     -x(4),     -x(5),     -x(6) /)
    F(:,5)=(/ x(4)-x(1), x(5)-x(2), x(6)-x(3),     -x(1),     -x(2),     -x(3) /)
    F(:,6)=(/     -x(1),     -x(2),     -x(3), x(4)-x(1), x(5)-x(2), x(6)-x(3) /)
    !
end function F

!=============================== PERM SYM =========================================

subroutine impose_perm_sym(FC,R23,SClat,PBC,verbose,FCvar,FC_sym,nat,n_blocks)
    implicit none
    integer, parameter :: DP = selected_real_kind(14,200)
    real(kind=DP), intent(in)     :: FC(3*nat,3*nat,3*nat,n_blocks)
    integer, intent(in)           :: nat, n_blocks
    integer, intent(in)           :: R23(6,n_blocks)
    integer, intent(in)           :: SClat(3)
    logical, intent(in)           :: PBC, verbose
    !
    real(kind=DP), intent(out)    :: FC_sym(3*nat,3*nat,3*nat,n_blocks), FCvar
    !
    integer                       :: jn1, jn2, jn3, i_block,nat3
    !

    
    if ( .not. perm_initialized ) call initialize_perm(R23,n_blocks,SClat,PBC)
          
    nat3=3*nat  
      
    do i_block = 1,n_blocks

    do jn1 = 1, nat3
    do jn2 = 1, nat3
    do jn3 = 1, nat3
    
    FC_sym(jn1,jn2,jn3,i_block)=      FC(jn1,jn2,jn3,  i_block)    &
                                    + FC(jn1,jn3,jn2,P(i_block,2)) &
                                    + FC(jn3,jn1,jn2,P(i_block,3)) &
                                    + FC(jn3,jn2,jn1,P(i_block,4)) &
                                    + FC(jn2,jn3,jn1,P(i_block,5)) &
                                    + FC(jn2,jn1,jn3,P(i_block,6))                                
    end do
    end do
    end do
                                    
    end do

    FC_sym=FC_sym/6.0_dp


    FCvar=SUM(ABS(FC-FC_sym))/ SUM(ABS(FC)) 

    if (verbose) then
    write(*, * ) ""
    write(*,  "(' FC variation due to permut. symm.= 'e20.6)") FCvar
    write(*, * ) ""
    end if
    !
end subroutine impose_perm_sym

!================================ ASR ON 3rd INDEX ============================================

subroutine impose_ASR_3rd(FC,xR2,xR2list,pow,SClat,PBC,verbose, &
                          FCvar,sum3rd,FC_asr,totnum_R2,nat,n_blocks)
    implicit none
    integer, parameter :: DP = selected_real_kind(14,200)
    real(kind=DP), intent(in) :: FC(3*nat,3*nat,3*nat,n_blocks)
    real(kind=DP), intent(in)    :: pow
    integer, intent(in) :: totnum_R2,nat,n_blocks,xR2(3,n_blocks),xR2list(3,totnum_R2)
    integer, intent(in) :: SCLat(3)
    logical, intent(in) :: PBC,verbose
    !
    real(kind=DP), intent(out) :: FCvar,sum3rd,FC_asr(3*nat,3*nat,3*nat,n_blocks)
    !
    integer :: nat3,jn1,jn2,j3,n3,i_block,i_R2
    real(kind=DP) :: d1,num,den,ratio,invpow

    if ( .not. R2index_initialized ) call initialize_R2index(xR2, xR2list, totnum_R2, n_blocks,SCLat,PBC)
    !    
    nat3=3*nat
    
    FC_asr=0.0_dp
    
    d1=0.0_dp
    sum3rd=0.0_dp
    !
    do i_R2=1,totnum_R2
    do jn1=1,nat3
    do jn2=1,nat3
    do j3=1,3
            !
            num=0.0_dp
            den=0.0_dp
            !
            if ( pow > 0.01_dp ) then
                !
                !
                do i_block=1,num_blocks_R2(i_R2)
                do n3=1,nat
                    num=num+ FC(jn1 , jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2))
                    den=den+ ABS(FC(jn1 , jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2)))**pow     
                end do
                end do
                !
                if ( den > 0.0_dp ) then
                    ratio=num/den
                    !
                    do i_block=1,num_blocks_R2(i_R2)
                    do n3=1,nat
                        FC_asr(jn1, jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2))= &
                            FC(jn1, jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2))- &
                            ratio*ABS(FC(jn1, jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2)))**pow
                    end do
                    end do
                    !  
                else ! no need to modify the FC
                    !
                    do i_block=1,num_blocks_R2(i_R2)
                    do n3=1,nat
                        FC_asr(jn1, jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2))= &
                            FC(jn1, jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2))
                    end do
                    end do                
                    !           
                end if
                !
                !
            else
                !
                !
                do i_block=1,num_blocks_R2(i_R2)
                do n3=1,nat
                    num=num+ FC(jn1 , jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2))
                    den=den+1      
                end do
                end do            
                !
                ratio=num/den
                !
                do i_block=1,num_blocks_R2(i_R2)
                do n3=1,nat
                    FC_asr(jn1, jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2))= &
                        FC(jn1, jn2, j3+3*(n3-1), index_blocks_R2(i_block,i_R2))- ratio
                end do
                end do
                !  
            end if
            !            
            !
            d1 = d1 + den
            sum3rd = sum3rd + ABS(num) ! should be zero if the ASR were fulfilled
            !
            !
    end do
    end do
    end do
    end do
    !    
    FCvar=SUM(ABS(FC-FC_ASR))/ SUM(ABS(FC)) 
    sum3rd=sum3rd/SUM(ABS(FC))
    !
    if (verbose) then
    if ( pow > 0.01_dp ) then
    
        invpow=1.0_dp/pow
        
        write(*, * ) ""   
        write(*, "(' ASR imposition on 3rd index with pow= 'f5.3)") pow
        write(*, "(' Previous values: sum(|sum_3rd phi|)/sum(|phi|)=       'e20.6)" ) sum3rd
        write(*, "('                  sum(|phi|**pow)**(1/pow)/sum(|phi|)= 'e20.6)" ) d1**invpow/SUM(ABS(FC))
        write(*, "(' FC relative variation= 'e20.6)" ) FCvar    
        write(*, * ) "" 
    
    else

        write(*, * ) ""   
        write(*, "(' ASR imposition on 3rd index with pow= 0' )") 
        write(*, "(' Previous value: sum(|sum_3rd phi|)/sum(|phi|)= 'e20.6 )" ) sum3rd
        write(*, "(' FC relative variation= 'e20.6)" ) FCvar    
        write(*, * ) ""     
    
    
    end if
    end if
    !
    !
end subroutine impose_ASR_3rd


! ==================================  MAIN ==========================================


subroutine impose_ASR(FC,R23,xR2,xR2list,pow,SClat,PBC,threshold,maxite,FC_out,verbose,totnum_R2,nat,n_blocks)
implicit none
integer, parameter :: DP = selected_real_kind(14,200)
real(kind=DP), intent(in) :: FC(3*nat,3*nat,3*nat,n_blocks)
real(kind=DP), intent(in) :: pow,threshold
integer, intent(in)       :: R23(6,n_blocks), maxite
integer, intent(in) :: totnum_R2,nat,n_blocks,xR2(3,n_blocks),xR2list(3,totnum_R2)
integer, intent(in) :: SCLat(3)
logical, intent(in) :: PBC,verbose
!
real(kind=DP), intent(out) :: FC_out(3*nat,3*nat,3*nat,n_blocks)
!
real(kind=DP)   :: FC_tmp(3*nat,3*nat,3*nat,n_blocks)
integer         :: ite, contr, iter, ios
real(kind=DP)   :: FCvar, sum3rd
logical :: converged

if (verbose) then
write(*,*) " "
write(*, "(' Iterative ASR imposition with pow= 'f5.3)") pow
write(*,*) " "
end if

call initialize_perm(R23, n_blocks,SClat,PBC)
call initialize_R2index(xR2,xR2list,totnum_R2, n_blocks,SCLat,PBC)


 converged = .false.
 FC_tmp=FC

ite=1
if ( maxite == 0 ) then
 contr=-1
else
 contr=1
end if
iter=ite*contr

do while (iter < maxite)

    if (verbose) write(*,"(' Iter #' I5 '  ====')") ite
    if (verbose) write(*,*) ""
        call impose_ASR_3rd(FC_tmp,xR2,xR2list,pow,SClat,PBC,.false.,FCvar,sum3rd,FC_out,totnum_R2,nat,n_blocks)
    if (verbose) write(*,"('         Sum on 3rd='  e20.6  '    Imp. ASR on 3rd:  => delta FC=' e20.6)") sum3rd,FCvar
        call impose_perm_sym(FC_out,R23,SClat,PBC,.false.,FCvar,FC_tmp,nat,n_blocks)
    if (verbose) write(*,"('                    '  20X    '    Imp. permut sym:  => delta FC=' e20.6)") FCvar
    if (verbose) write(*,*) ""

   !  check converg
   if ( sum3rd < threshold  .and. FCvar < threshold ) then
        write(*,*) " "
        write(*,"( ' * Convergence reached within threshold:' e20.6 )") threshold
        write(*,*) " "
        write(*,"( ' * Total FC relative variation:' e20.6 )") SUM(ABS(FC-FC_out))/ SUM(ABS(FC)) 
        converged = .True.
        EXIT
   end if
   
   !  check if there is STOP file
   OPEN(unit=100, file="STOP", status='OLD', iostat=ios)
   IF(ios==0) THEN
        CLOSE(100,status="DELETE")
        write(*,*) " File STOP found, the ASR execution terminates here"
        EXIT 
   ENDIF
   
   ite=ite+1
   iter=ite*contr
end do


if (.not. converged ) then
   write(*,*) " "
   write(*,"( ' Max number of iteration reached ('I6')' )") maxite
   write(*,"( ' Convergence not reached within threshold:' e20.6 )") threshold
   write(*,*) " "
end if   


end subroutine impose_ASR


end module third_order_ASR
