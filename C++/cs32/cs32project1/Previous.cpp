#include "Previous.hpp"
#include <iostream>
using namespace std;

Previous::Previous(int nRows, int nCols){
    m_rows = nRows;
    m_cols = nCols;
    for (int i = 0; i < m_rows; i++){
        for (int j = 0; j < m_cols; j++){
            arr[i][j] = 0;
        }
    }
}

bool Previous::record(int r, int c)
{
    if (r > m_rows || c > m_cols || r < 1 || c < 1)
    {
        return false;
    }

    else
    {
        arr[r-1][c-1]++;
        return true;
    }
}

void Previous::display() const
{
    for (int r = 0; r < m_rows; r++)
    {
        for (int c = 0; c < m_cols; c++){
            if (arr[r][c] == 0)
            {
                cout << '.';
            }
            else if (1 <= arr[r][c] && arr[r][c] < 26)
            {
                char x = 'A' + arr[r][c] - 1;
                cout << x;
            }
            else if (arr[r][c] >= 26)
            {
                cout << 'Z';
            }
        }
        cout << endl;
    }
    cout << endl;
}

