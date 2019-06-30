
! This subroutine writes a (3*nat,3*nat,3*nat) real matrix in the
! (nat,nat,nat,3,3,3) 

subroutine threetosix_real (mat3,mat6, nat)

    implicit none
  
    double precision, dimension(nat*3,nat*3,nat*3), intent(in) :: mat3
    double precision, dimension(nat,nat,nat,3,3,3), intent(out) :: mat6
  
    integer :: nat
    integer :: i, j, k, alpha, beta, gamm
  
  
    do k = 1, nat
      do j = 1, nat
        do i = 1, nat
          do gamm = 1, 3
            do beta = 1, 3
              do alpha = 1, 3
                mat6(i,j,k,alpha,beta,gamm) = mat3(3*(i-1)+alpha,3*(j-1)+beta,3*(k-1)+gamm)
              end do
            end do
          end do
        end do
      end do
    end do
  
  end subroutine threetosix_real
  
  ! This subroutine writes a (3*nat,3*nat,3*nat) real matrix in the
  ! (nat,nat,nat,3,3,3) 
  
  subroutine sixtothree_real (mat6,mat3, nat)
  
    implicit none
  
    double precision, dimension(nat*3,nat*3,nat*3), intent(out) :: mat3
    double precision, dimension(nat,nat,nat,3,3,3), intent(in) :: mat6
  
    integer :: nat
    integer :: i, j, k, alpha, beta, gamm
  
  
    do k = 1, nat
      do j = 1, nat
        do i = 1, nat
          do gamm = 1, 3
            do beta = 1, 3
              do alpha = 1, 3
                mat3(3*(i-1)+alpha,3*(j-1)+beta,3*(k-1)+gamm) = mat6(i,j,k,alpha,beta,gamm)
              end do
            end do
          end do
        end do
      end do
    end do
  
  end subroutine sixtothree_real