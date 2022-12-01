/*
    PIC 10C Homework 2, Heap.h
    Purpose: Define a template heap class
    Author: Penggao GU
    Date: 10/18/2022
*/

#ifndef Heap_h
#define Heap_h

template<typename T, typename CMP=std::less<T>>
//  comparison operator, CMP, is std::less<T> by default
class Heap
{
public:
    Heap() {
    }
//  default constructor that initialize the heap as an empty heap
    
    Heap(std::vector<T> vec) {
        for (int i = 0; i < vec.size(); i++){
            T newnum = vec[i];
            push(newnum);
        }
    }
//  a constructor with a single argument of type std::vector<T>, constructing the
//  heap with all elements in the vector;
    

    Heap(std::vector<T> vec, CMP cmp) {
        for (int i = 0; i < vec.size(); i++){
            T newnum = vec[i];
            push(newnum);
        }
    }
//  store a std::vector<T> called data and a CMP called comparator as membe variables;
//  all elements of the heap should be saved in a vector
    
    
    
    template<typename ... Ts>
    void push(T new_element, Ts ... elements)
    {
        int index = data.size();
        data.push_back(new_element);
        while (index > 0 && !CMP()(get_parent(index), new_element))
        {
        data[index] = get_parent(index);
        index = get_parent_index(index);
        }
        data[index] = new_element;
        push(elements...);
    }
    
    void push(){}
//  have a push function that, accepts a variadic list of arguments of type T and
//  places the objects into the data structure;
    

    
    void pop(){
        if (data.size()){
            data[0]=data[data.size()-1];
            data.pop_back();
            fix_heap();
        }
    }
//  have a pop function that, when called, removes the ”maximum” value from the heap
//  (as defined by comparator), making no effect to the heap if the heap is empty at
//  the call.
    
    void fix_heap(){
        int index = 0;
        int index_max = data.size();
        while (2*index+2 < index_max){
            int new_index = (CMP()(data[2*index+1], data[2*index+2]))? (2*index+1) : (2*index + 2);
            if (!CMP()(data[index], data[new_index])){
                T temp = data[index];
                data[index]  = data[new_index];
                data[new_index] = temp;
                
            }
            index = new_index ;
        }
        int new_index = 2*index + 1;
        if (2*index+1 < index_max){
            if (!CMP()(data[index], data[new_index])){
                T temp = data[index];
                data[index]  = data[new_index];
                data[new_index] = temp;
                
            }
        }
    }
//  use to fix the order of the data[]
    
    int get_parent_index(int index) const
    {
        return index / 2;
    }
//  get the index of parent
    
    T get_parent(int index) const
    {
        return data[index / 2];
    }
//  get the data of parent

    T top() const
    {
        return data[0];
    }
//  have a top function that returns the ”maximum” value from the heap (as defined by
//  comparator), assuming top won’t be called for an empty heap;
    
    int size() const
    {
        return data.size();
    }
//  get the size of the data
    
    void display(){
        for (int i = 0; i < data.size(); i++){
            std::cout<<data[i]<<'\t';
        }
        std::cout << std::endl;
    }
//  have a display function that prints all elements in the order stored in the vector
    
    private:
    std::vector<T> data;
};

void print(){}

template<typename T, typename ... Ts>
void print( T heap,  Ts... heaps)
{
    while (heap.size()){
        std::cout << heap.top() << '\t';
        heap.pop();
    }
    std::cout << std::endl;
    print(heaps...);
}
// a print function that takes a variadic list of Heap<T,CMP> type containers and print elements in the order defined by CMP (such as print(heap1, heap2, heap3);.) The printed elements of the same heap are separated by horizontal tabs (i.e., std::cout << [element you want to print] << ’\t’;) and elements of different heaps are printed in different lines.
#endif
