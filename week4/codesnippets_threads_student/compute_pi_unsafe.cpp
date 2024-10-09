#include <thread>
#include <iostream>
#include <iomanip>

void thread_sum( long rank, int num_threads, long long n, double& sum) {
    long my_rank = (long ) rank;
    double factor;
    long long i;
    long long my_n = n / num_threads;
    long long my_first_i = my_n * my_rank;
    long long my_last_i = my_first_i + my_n;
    
    if ( my_first_i % 2 == 0)  /* my_first_i is even */
        factor = 1.0;
    else /* my_first_i is odd */
        factor = -1.0;
        

    for ( i = my_first_i ; i < my_last_i ; i++ , factor = -factor)
    {
        sum += factor / ( 2 * i + 1 );
    }
    
}




int main (int argc, char *argv[]) { 
    
    int num_threads =20;
    long long n = 10e7;
    
    std::thread t[num_threads];

    double sum = 0.0;
    for (int i=0; i<num_threads; i++) {
        t[i] = std::thread(thread_sum, (long) i, num_threads, n, std::ref(sum));
    }
    

    for (int i=0; i<num_threads; i++) {
        t[i].join(); 
    }
    
    std::cout << "Approximation of pi: " << std::setprecision(10) << 4.0 * sum << std::endl;

}