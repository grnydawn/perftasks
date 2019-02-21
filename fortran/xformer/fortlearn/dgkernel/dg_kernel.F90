!!------------------------------------------------
 !R.Nair NCAR/scd 08/03
 !Qudrature of the RHS evaluation 
!!------------------------------------------------

      PROGRAM Grad_Term_GPU
      
      IMPLICIT NONE

      INTEGER, PARAMETER :: DOUBLE=SELECTED_REAL_KIND(p=14,r=100)

      INTEGER, PARAMETER :: nx=4      ! element order
      INTEGER, PARAMETER :: npts=nx*nx
      INTEGER, PARAMETER :: nit=100   ! iteration count
      INTEGER, PARAMETER :: nelem=6*120*120

      REAL(KIND=DOUBLE), PARAMETER :: dt=.005D0 ! fake timestep

      REAL(KIND=DOUBLE) :: der(nx,nx)   ! Derivative matrix
      REAL(KIND=DOUBLE) :: delta(nx,nx) ! Kronecker delta function
      REAL(KIND=DOUBLE) :: gw(nx)       ! Gaussian wts
      REAL(KIND=DOUBLE), DIMENSION(nx*nx,nelem) :: flx,fly
      REAL(KIND=DOUBLE), DIMENSION(nx*nx,nelem) :: grad     

      REAL(KIND=DOUBLE) :: s1, s2
      REAL(KIND=DOUBLE) :: start_time, stop_time, elapsed_time

      INTEGER :: i, j, k, l, ii, ie, it

      ! Init static matrices

      der(:,:)=1.0_8
      gw(:) = 0.5_8

      delta(:,:)=0.0_8
      delta(1,1)=1.0_8
      delta(2,2)=1.0_8

      ! Load up some initial values

      flx(:,:) = 1.0_8
      fly(:,:) = -1.0_8

      start_time = second()

      DO it=1,nit
      DO ie=1,nelem
         DO ii=1,npts
            k=MODULO(ii-1,nx)+1
            l=(ii-1)/nx+1
            s2 = 0.0_8
            DO j = 1, nx
               s1 = 0.0_8
               DO i = 1, nx
                  s1 = s1 + (delta(l,j)*flx(i+(j-1)*nx,ie)*der(i,k) + &
                             delta(i,k)*fly(i+(j-1)*nx,ie)*der(j,l))*gw(i)
               END DO  ! i loop
               s2 = s2 + s1*gw(j) 
            END DO ! j loop
            grad(ii,ie) = s2
         END DO ! i1 loop
      END DO ! ie

     !write(*,*) "Done with gradient"

      DO ie=1,nelem
         DO ii=1,npts
            flx(ii,ie) = flx(ii,ie)+ dt*grad(ii,ie)
            fly(ii,ie) = fly(ii,ie)+ dt*grad(ii,ie)
         END DO
      END DO
      
      END DO ! iteration count, it

      stop_time = second()

      elapsed_time = stop_time - start_time

      WRITE(*, *) "****************** RESULT ********************"
      WRITE(*, *)
      WRITE(*, "(A,E15.7)") "MAX(flx) = ", MAXVAL(flx)
      WRITE(*, "(A,E15.7)") "MIN(fly) = ", MINVAL(fly)
      WRITE(*, "(A,F7.2)") "Gflops   = ",(1.0d-9*nit*nelem*npts*(nx*nx*7.D0+2.D0*nx+4.0D0))/elapsed_time
      WRITE(*, "(A,F10.3,A)") 'completed in ', elapsed_time, ' seconds'

      contains

      real function second()
        integer :: cnt, count_rate, count_max
        call system_clock ( cnt, count_rate, count_max )
        second = real(cnt) / real(count_rate)
      return
      end

      END PROGRAM Grad_Term_GPU

