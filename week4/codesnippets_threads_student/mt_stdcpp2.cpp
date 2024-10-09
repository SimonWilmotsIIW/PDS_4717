#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <vector>

static const int num_threads = 10;

class threadmain {
public:
  threadmain() 
    : id(-1), return_value(-1)
  {
  }

  void set_id(int i)
  {
    id = i;
  }

  int get_result() const
  {
    return return_value;
  }

  void operator() (void)
  {
    std::cerr << "Thread " << id << " loopt.\n";    
    return_value = id;
  }

private:
  int id;
  int return_value;
};

int main(int argc, char **argv)
{
  std::vector<std::thread> t(num_threads);
  std::vector<threadmain> objects(num_threads);

  for (int i = 0 ; i < num_threads ; i++)
    objects[i].set_id(i);

  for (int i = 0; i < num_threads; i++)
    t[i] = std::thread(std::ref(objects[i]));

  std::cerr << "Main loopt.\n";

  for (int i = 0; i < num_threads; i++) {
    t[i].join(); 
    std::cerr << "Thread " << i << "  is afgelopen met resultaat " << objects[i].get_result() << ".\n";
  }

  return 0;
}
