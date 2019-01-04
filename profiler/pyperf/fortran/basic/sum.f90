program summation
    integer, parameter :: N=1024*32
    real, dimension(N) :: a, b, c
    integer i, j

    do i=1,N
        do j=1,N
            c(i) = a(i) + b(i) + real(j)
        end do
    end do

    print *, "sum = ", sum(c)

end program
