#include <math.h>
#include <stdio.h>

#define printd(s) printf("%-50s = %.18lf\n", #s, (s));

double array_sum(void)
{
    double result = 0;
    for (double v = 0; v < 1000; v += 0.1)
        result += v;
    return result;
}

int main(void)
{
    printd(1.125 + 1.5);
    printd(1.1 + 1.2);
    printd(1.2 - 1.1);
    printd(1.23254332482E-12 + 3.9382498452394832094E+9)
    printd(sin(0.237238493));
    printd(cos(3.9382498452394832094E+9));
    printd(log10(3.9382498452394832094E+9))
    printd(array_sum());
    printd(NAN);
}
