//
//  Game.hpp
//  cs32project1
//
//  Created by Penggao Gu on 6/21/22.
//

#ifndef Game_hpp
#define Game_hpp

#include "Player.hpp"
#include "Arena.hpp"
#include "globals.hpp"

#include <stdio.h>

//Game.h
class Game
{
  public:
        // Constructor/destructor
    Game(int rows, int cols, int nRobots);
    ~Game();

        // Mutators
    void play();

  private:
    Arena* m_arena;
};



#endif /* Game_hpp */
