//
//  Arena.hpp
//  cs32project1
//
//  Created by Penggao Gu on 6/21/22.
//

#ifndef Arena_hpp
#define Arena_hpp

#include <iostream>
#include <string>

#include "Player.hpp"
#include "Robot.hpp"
#include "Previous.hpp"

//Arena.h
class Arena
{
  public:
        // Constructor/destructor
    Arena(int nRows, int nCols);
    ~Arena();

        // Accessors
    int     rows() const;
    int     cols() const;
    Player* player() const;
    int     robotCount() const;
    int     nRobotsAt(int r, int c) const;
    void    display(string msg) const;

        // Mutators
    bool   addRobot(int r, int c);
    bool   addPlayer(int r, int c);
    void   damageRobotAt(int r, int c);
    bool   moveRobots();
    Previous& previous();

  private:
    int     m_rows;
    int     m_cols;
    Player* m_player;
    Robot*  m_robots[MAXROBOTS];
    int     m_nRobots;
    Previous m_previous;
};

#endif /* Arena_hpp */
