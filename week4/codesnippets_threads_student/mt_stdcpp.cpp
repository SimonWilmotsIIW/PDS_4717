#include <iostream>
#include <thread>

static const int num_threads =10;

void threadmain(int id)
{
    std::cerr << "Thread " << id << " loopt.\n";    
}

int main(int argc, char **argv)
{
  std::thread t[num_threads];

  for (int i=0; i<num_threads; i++) {
    t[i] = std::thread(threadmain, i);
  }

  std::cerr << "Main loopt.\n";

  for (int i=0; i<num_threads; i++) {
    t[i].join(); 
    std::cerr << "Thread " << i << "  is afgelopen.\n";
  }

  return 0;
}
