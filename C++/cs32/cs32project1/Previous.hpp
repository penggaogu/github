//
//  Previous.hpp
//  cs32project1
//
//  Created by Penggao Gu on 6/22/22.
//

#ifndef Previous_hpp
#define Previous_hpp

#include <iostream>
#include <string>

#include "Player.hpp"
#include "globals.hpp"

class Arena;

class Previous
{
public:
    Previous(int nRows, int nCols);
    bool record(int r, int c);
    void display() const;
private:
    char arr[MAXROWS][MAXCOLS];
    int m_rows;
    int m_cols;
};

#endif /* Previous_hpp */
