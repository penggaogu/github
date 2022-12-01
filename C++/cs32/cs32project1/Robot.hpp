//
//  Robot.hpp
//  cs32project1
//
//  Created by Penggao Gu on 6/21/22.
//

#ifndef Robot_hpp
#define Robot_hpp

class Arena;

class Robot
{
  public:
        // Constructor
    Robot(Arena* ap, int r, int c);

        // Accessors
    int  row() const;
    int  col() const;

        // Mutators
    void move();
    bool takeDamageAndLive();

private:
    Arena* m_arena;
    int    m_row;
    int    m_col;
    int    m_health;
};

#endif /* Robot_hpp */
