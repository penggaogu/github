/*
    PIC 10C Homework 1, list.cpp
    Author: Penggao Gu
    Date: 10/10/2022
*/

#include "list.h"

using namespace std;

Node::Node(string s){
    data = s;
    previous = nullptr;
    next = nullptr;
}

List::List(){
    first = nullptr;
    last = nullptr;
}

void List::push_front(string data)
{
    Node* new_node = new Node(data);
    if (first == nullptr)
    {
        first = new_node;
        last = new_node; }
    else
    {
        new_node -> next = first;
        first -> previous = new_node;
        first = new_node;
    }
}


void List::push_back(string data)
{
    Node* new_node = new Node(data);
    if (last == nullptr)
    {
        first = new_node;
        last = new_node;
    }
    else
    {
        new_node -> previous = last;
        last -> next = new_node;
        last = new_node;
    }
}


void List::reverse()
{
    Node* prev = nullptr;
    Node* current = first;
    Node* Next = nullptr;
    last = current;
    while (current != nullptr)
    {
        Next = current->next;
        current->next = prev;
        current->previous = Next;
        prev = current;
        current = Next;
    }
    first = prev;
    
}

void List::swap(Iterator iter1,Iterator iter2)
{
    if(iter1.position != NULL && iter2.position != NULL && iter1.position != iter2.position)
    {
        if(iter1.position->next == iter2.position || iter2.position->next == iter1.position)
        {
            if(iter2.position->next == iter1.position){
                iter2.next();
                iter1.previous();
            }
            Node *i1 = iter1.position;
            Node *i2 = iter2.position;
            Node *i1before = i1->previous;
            Node *i2after = i2->next;

            i1->next = i2after;
            i2->previous = i1before;
            i1->previous = i2;
            i2->next = i1;
            if(i1before ==  NULL)
            {
                first = i2;
            }
            else
            {
                i1before->next = i2;
            }

            if(i2after ==  NULL){
                last = i1;
            }
            else
            {
                i2after->previous = i1;
            }

        }
        else
        {
            Node *it1 = iter1.position;
            Node *it2 = iter2.position;
            Node *it1before = iter1.position->previous;
            Node *it1after = iter1.position->next;
            Node *it2before = iter2.position->previous;
            Node *it2after = iter2.position->next;

            it1->next = it2after;
            it1->previous = it2before;
            it2->next = it1after;
            it2->previous = it1before;

            if (it1before == NULL)
            {
                first = it2;
            }
            else
            {
                it1before->next = it2;
            }

            if (it2before == NULL)
            {
                first = it1;
            }
            else
            {
                it2before->next = it1;
            }

            if (it1after == NULL)
            {
                last = it2;
            }
            else
            {
                it1after->previous = it2;
            }

            if (it2after == NULL)
            {
                last = it1;
            }
            else
            {
                it2after->previous = it1;
            }
        }

    }

}



Iterator List::erase(Iterator iter)
{
    Node* remove = iter.position;
    Node* before = remove -> previous;
    Node* after = remove -> next;
    if (remove == first)
    {
        first = after;
    }
    else
    {
        
        before -> next = after;
    }
    
    if (remove == last)
    {
        last = before;
    }
    else
    {
        after -> previous = before;
    }
    delete remove;
    Iterator r;
    r.position = after;
    r.container = this;
    return r;
}

Iterator List::begin()
{
    Iterator iter;
    iter.position = first;
    iter.container = this;
    return iter;
}

Iterator List::end()
{
    Iterator iter;
    iter.position = nullptr;
    iter.container = this;
    return iter;
}

Iterator::Iterator(){
    position=nullptr;
    container=nullptr;
}

string Iterator::get() const
{
    return position -> data;
}

void Iterator::next()
{
    position = position -> next;
}

void Iterator::previous()
{
    if (position == nullptr)
    {
        position = container -> last;
    }
    else
    {
        position = position -> previous;
    }
}

bool Iterator::equals(Iterator b) const
{
    return position == b.position;
}

