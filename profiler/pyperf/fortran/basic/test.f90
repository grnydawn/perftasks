program summation
    integer, parameter :: N=1024*32
    real, dimension(N) :: a, b, c
    integer i

    do i=1,N
        c(i) = a(i) + b(i)
    end do

    print *, "sum = ", sum(c)

end program
