
#name "main"
#include "stdlib.h"
#runtime thread 8

// 线程1的函数
export void thread1()
{
  while (1)
  {
    print("I'm thread 1");
    sleep(1000);
  }
}

// 线程2的函数
export void thread2()
{
  while (1)
  {
    print("I'm thread 2");
    sleep(1000);
  }
}

// 线程3的函数
export void thread3()
{
  while (1)
  {
    print("I'm thread 3");
    sleep(1000);
  }
}

// 主线程
export int main()
{
  CreateThread("thread1"); // 开始线程1
  CreateThread("thread2"); // 开始线程2
  CreateThread("thread3"); // 开始线程3
  while (1)
    sleep(1000);
  return 0;
}