/*
    PIC 10C Homework 3, Safedeque.h
    Author: Penggao Gu
    Date: 10/24/2022
*/

#ifndef Safedeque_h
#define Safedeque_h

using namespace std;

template<typename T>
class Iterator;

template<typename T>
class Safedeque{
public:
    
    //    a default constructor to begin storing nothing
    Safedeque(){
        data = nullptr;
        cap = 0;
        sz = 0;
        front = nullptr;
        back = nullptr;
    }
    
//    get front and get back functions to access values at the front/back of the deque; throw assert exceptions if the container is empty;
    void get_front(){
        assert(cap != 0);
        return *front;
    }

    void get_back(){
        assert(cap != 0);
        return *back;    }
    
    
//    push front and push back functions to append values at the front/back of the deque; if the capacity is zero, add one spot to the deque, and print ‘‘One spot added."; if all spots are taken, double the deque capacity and print ‘‘Capacity doubled.’’ to the console;
    void push_back(T new_element){
        if (cap == 0){
            cout << "One spot added. ";
            cap++;
            sz++;
            data = new T[cap];
            data[0] = new_element;
            cout << "Current capacity is " << cap << "\n";
            front = &data[cap - 1];
            back = &data[sz - 2];
        }
        else if (sz == cap){
            cout << "Capicity doubled. ";
            sz++;
            cap = cap*2;
            T* data2 = new T[cap];
            for (int i = 0; i < cap/2; i++){
                data2[i] = data[i];
            }
            data = new T[cap];
            for (int i = 0; i < cap; i++){
                data[i] = data2[i];
            }
            data[cap-1] = data[sz-2];
            data[sz-2] = new_element;
            cout << "Current capacity is " << cap << "\n";
            front = &data[cap - 1];
            back = &data[sz - 2];
        }
        else{
            data[sz-1] = new_element;
            sz++;
            front = &data[cap - 1];
            back = &data[sz - 2];
        }

    }
    
    void push_front(T new_element){
        if (cap == 0){
            cout << "One spot added. ";
            cap++;
            sz++;
            data = new T[cap];
            data[0] = new_element;
            cout << "Current capacity is " << cap << "\n";
            front = &data[cap - 1];
            back = &data[sz - 2];
        }
        else if (sz == cap){
            cout << "Capicity doubled. ";
            T temp = data[cap - 1];
            for (int i = cap - 1; i > 0; i--){
                data[i] = data[i-1];
            }
            data[0] = temp;
            sz++;
            cap = cap*2;
            T* data2 = new T[cap];
            for (int i = 0; i < cap/2; i++){
                data2[i] = data[i];
            }
            data = new T[cap];
            for (int i = 0; i < cap; i++){
                data[i] = data2[i];
            }
            data[cap-1] = (new_element);
            cout << "Current capacity is " << cap << "\n";
            front = &data[cap - 1];
            back = &data[sz - 2];
        }
        else{
            T temp = data[cap - 1];
            for (int i = cap - 1; i > 0; i--){
                data[i] = data[i-1];
            }
            data[0] = temp;
            data[cap - 1] = new_element;
            sz++;
            front = &data[cap - 1];
            back = &data[sz - 2];
        }
    }

//    pop front and pop back functions to remove the frontmost/backmost element; use assert to throw exceptions if the container is already empty; you don’t need to account for shrinking the deque capacity;
    void pop_front(){
        sz--;
        data[*front] = {};
        T temp = data[0];
        for (int i = 0; i < cap - 1; i++){
            data[i] = data[i+1];
        }
        data[*front] = temp;
        front = &data[cap - 1];
        back = &data[sz - 2];
        }

    void pop_back(){
        if(cap == 2){
            sz--;
            data[0] = {};
            front = &data[1];
            back = &data[1];
        }
        else{
            sz--;
            data[*back] = {};
            front = &data[cap - 1];
            back = &data[sz - 2];
        }
    }


//    size and capacity functions to return the size and capacity
    int capacity(){
        assert(cap >= 0);
        return cap;
    }

    int size(){
        assert(sz >= 0);
        return sz;
    }
    
//    print function to print all elements from front to back;
    void print(){
        if(this->size() == 2 && this->capacity() == 2){
            cout << data[1] << " " << data[0];
        }
        else{
            for (Iterator<T> i = this->begin(); i != this->end(); i++){
                cout << *i << ' ';
            }
        }
        cout << endl;
    }
    
//    real print function to print all elements in data in their actual array positions from the first to the last. (This function helps us to determine if you implement the class correctly.) Use * to replace spots that are not taken.
    void real_print(){
        for (int i = 0; i < cap; i++){
            T* ptr = &data[i];
            if (ptr > back && ptr < front) {
                cout<< "*" <<" ";
            }
            else{
                cout<<data[i] << " ";
            }
        }
        cout << endl;
    }
    
//    begin and end functions, overloaded on const, to return Iterator objects;
    Iterator<T> begin(){
        Iterator<T> iter;
        iter.position = this->front;
        iter.container = this;
        return iter;
    }
    
    Iterator<T> end(){
        Iterator<T> iter;
        iter.position = this->back;
        iter.container = this;
        if (back){
            iter++;
        }
        return iter;
    }
    
//    a random access operator, overloaded on const, to retrieve the element at a given index;
    T& operator[](int a){
        return data[a-1];
    }

    
    friend class Iterator<T>;

    
private:
    T* data;
    int cap = 0;
//    track capacity
    int sz = 0;
//    track size
    T* front;
//    track the "frontmost" indice
    T* back;
//    track the "backmost" indice
    
};






template<typename T>
class Iterator
{
public:
    Iterator<T>(){
        position = nullptr;
        container = nullptr;
    }
    
//    operator!=
    bool operator!=(Iterator<T> it2){
        if ((this->position != it2.position) || (this->container != it2.container)){
            return true;
        }
        else{
            return false;
        }
    }
    
//    operator==;
    bool operator==(Iterator<T> it2){
        if ((this->position == it2.position) && (this->container == it2.container)){
            return true;
        }
        else{
            return false;
        }
    }
    
//    prefix/postfix operator++;
    Iterator<T>& operator++(){
        Iterator<T> iter;
        iter.container = this->container;
        iter.position = (this->container->back)+1;
        assert(*this != iter);
        if (this->position == &this->container->data[this->container->cap - 1]){
            this->position = &this->container->data[0];
        }
        else{
            this->position++;
        }
        return *this;
    }
    
    Iterator<T> operator++(int a){
        Iterator<T> iter;
        iter.container = this->container;
        iter.position = this->position;
        ++(*this);
        return iter;
    }
    
    
//    prefix/postfix operator--;
    Iterator<T>& operator--(){
//        Iterator<T> iter;
//        iter.container = this->container;
//        iter.position = (this->container->front);
//        assert(*this != iter);
        if (this->position == this->container->data){
            this->position = &this->container->data[this->container->cap - 1];
        }
        else{
            this->position--;
        }
        return *this;
    }
    
    Iterator<T> operator--(int a){
        Iterator iter;
        iter.container = this->container;
        iter.position = this->position;
        --(*this);
        return iter;
    }
    
    
//    dereferencing operator operator*;
    T& operator*() {
        return *position;
    };


    Iterator<T>& operator=(int other){
        *position = other;
    }
    
    friend class Safedeque<T>;

private:
    T* position;
    Safedeque<T>* container;
};

#endif /* Safedeque_h */




