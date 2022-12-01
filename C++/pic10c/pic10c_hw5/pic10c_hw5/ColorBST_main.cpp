#include<vector>
#include<iostream>
#include"ColorBST.h"
using namespace std;
int main()
{
    
    // red-black tree
    ColorBST<int> all_true;
    all_true.insert(11,'b');
    all_true.insert(2);
    all_true.insert(14,'b');
    all_true.insert(15,'r');
    all_true.insert(7,'b');
    all_true.insert(1,'b');
    all_true.insert(5);
    all_true.insert(8);
    cout << "Black root rule preserved: " << boolalpha << all_true.BlackRoot()
         << "\nNo double reds rule preserved: " << boolalpha << all_true.NoDoubleRed()
         << "\nBlack depth rule preserved: " << boolalpha << all_true.BlackDepth() << endl;
    
    std::cout << "Largest value: " << all_true.largest() << std::endl;
    std::cout << "********** print in descending order ****************" << std::endl;
    all_true.descending_print();
    std::cout << "********** print in asending order ****************" << std::endl;
    all_true.ascending_print();
    
    std:: cout << std::endl;
    
    // fake red-black tree
    ColorBST<int> all_false;
    all_false.insert(11);
    all_false.insert(2);
    all_false.insert(14,'b');
    all_false.insert(15,'r');
    all_false.insert(7,'b');
    all_false.insert(1);
    all_false.insert(5);
    all_false.insert(8);
    all_false.insert(16);
    all_false.insert(0,'b');
    cout << "Black root rule preserved: " << boolalpha <<  all_false.BlackRoot()
         << "\nNo double reds rule preserved: " << boolalpha << all_false.NoDoubleRed()
         << "\nBlack depth rule preserved: " << boolalpha <<  all_false.BlackDepth() << endl;
    std::cout << "Largest value: " << all_false.largest() << std::endl;
    std::cout << "********** print in descending order ****************" << std::endl;
    all_false.descending_print();
    std::cout << "********** print in asending order ****************" << std::endl;
    all_false.ascending_print();
    
    
    return 0;
} 
