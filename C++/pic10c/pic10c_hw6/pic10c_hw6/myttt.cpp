/*
    PIC 10C Homework 6, myttt.cpp
    Purpose: Tic Tac Toe game
    Author: Penggao Gu
    Date: 11/15/2022
*/
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <iostream>


using namespace std;


// the same functions and global variable as the given ttt.cpp file
char square[10] = { 'o','1','2','3','4','5','6','7','8','9' };
int checkwin();
void print_board();
void mark_square(char choice, char mark);

// the extra one which use to initialize the square for the option of restarting the game.
void initial_square();


int main()
{
    // prompts users to choose one-player or two-player game
    cout << "Do you want to play a 1-player or 2-player game? Please enter 1 or 2?\n";
    // print information to let users know they can exit by entering Q and restart by entering N

//    firstly store the player number in string.
    string string_player_number;
    cin >> string_player_number;
    char player_number;
    //    use player_number_check to check whether to stop the loop of checking exceptions.
    bool player_number_check = true;
    while (player_number_check) {
        try {
            //            store the string_player_number from string into char_array sothat we can check the size of the sting
            //            let the array[0] be the player_number.
            int n = string_player_number.length();

            char char_array[1];
            char_array[0] = string_player_number.c_str()[0];

            player_number = char_array[0];

            //            since only valid input are 1 and 2, any length non-one input should be wrong.
            if (n != 1) {
                logic_error description("illegal player_number parameter, input should only be 1 or 2.");
                throw description;
            }

            //            after the last exception check, if the input pass the last one, their size mast be 1.
            //            then we need to check if the input is 1 or 2.
            if (player_number != '1' && player_number != '2') {
                logic_error description("illegal player_number parameter, input should only be 1 or 2.");
                throw description;
            }
            //            if it is, then stop the while loop.
            else {
                player_number_check = false;
            }
        }

        //        if there is a description which is thrown, then let the user input the value again.
        catch (logic_error& e) {
            cout << "logica error " << e.what() << "\n";
            cout << "Do you want to play a 1-player or 2-player game? Please enter 1 or 2?\n";
            cin >> string_player_number;
            int n = string_player_number.length();
            // declaring character array
            char char_array[1];
            // copying the contents of the
            // string to char array
            char_array[0] = string_player_number.c_str()[0];

            player_number = char_array[0];

            if (n == 1 && (player_number == '1' || player_number == '2')) {
                player_number_check = false;
            }
        }
    }

    //    player -1 means the fisrt player, payer 1 means the second player.
    int player = -1; // player = 1 or -1
    int if_win;
    char choice; // player's move
    char mark;


    do
    {
        //        the auto play code for single play mode.
        //        the auto play work when the play mode is chosen to be single, which is player == 1, and when the second player is on charge.
        if (player_number == '1' && player == 1) {
            //            use bool add to check if the auto player makes any move.
            bool add = false;
            //            if there are any move which could let the auto player win the game, make that move.
            for (int i = 1; i < 11; i++) {
                if (square[i] != 'O' && square[i] != 'X') {
                    mark_square(i + 48, 'O');
                    if (checkwin() == true) {
                        mark_square(i + 48, 'O');
                        i = 10;
                        add = true;
                    }
                    else {
                        mark_square(i + 48, i + 48);
                    }
                }
            }

            //            if there are any move which could stop the first player win the game, make that move.
            if (!add) {
                for (int i = 1; i < 11; i++) {
                    if (square[i] != 'O' && square[i] != 'X') {
                        mark_square(i + 48, 'X');
                        if (checkwin() == true) {
                            mark_square(i + 48, 'O');
                            i = 10;
                            add = true;
                        }
                        else {
                            mark_square(i + 48, i + 48);
                        }
                    }
                }
            }

            //            if the middle block empty, choose that block for the move.
            if (!add) {
                if (square[5] == '5') {
                    mark_square('5', 'O');
                    add = true;
                }
            }

            //            if all the above situations have not happened, just choose the first unoccqupied block as the move.
            if (!add) {
                for (int i = 1; i < 11; i++) {
                    if (square[i] != 'O' && square[i] != 'X') {
                        mark_square(i + 48, 'O');
                        i = 11;
                        add = true;
                    }
                }
            }
            //            check win
            if_win = checkwin();
            //            chenge the player
            player *= -1;
        }


        else {
            print_board();
            //            let the user know that there are q and n options which can exit or restart the game.
            cout << "Player " << (player + 3) / 2 << ", enter a number('Q' to exit, 'N' to restart):  ";
            //            store the input as string first.
            string string_choice;
            cin >> string_choice;


            bool entering_position_check = true;
            while (entering_position_check) {
                try {

                    //                    store the string_choice from string into char_array sothat we can check the size of the sting
                    //                    let the array[0] be the choice.
                    int n = string_choice.length();
                    char char_array[1];
                    char_array[0] = string_choice.c_str()[0];
                    choice = char_array[0];

                    //                    if size is not one, and any element in the array are not number
                    //                    then it's not an integers.
                    if (n > 1) {
                        for (int j = 0; j < n; j++) {
                            if (string_choice.c_str()[j] < 48 || string_choice.c_str()[j] > 57) {
                                logic_error description("entering inputs that are not integers.");
                                throw description;
                            }
                        }
                    }

                    //                    if the size is not one and is a integer
                    //                    then it's size cannot be from 1 to 9
                    if (n > 1) {
                        logic_error description("entering integers that are not from 1 to 9.");
                        throw description;
                    }

                    //                    if the size is one and it's not Q, N, or 1 to 9
                    //                    then it's not integers
                    if (n == 1) {
                        if (choice != 'Q' && choice != 'N' && (choice < 48 || choice > 57)) {
                            logic_error description("entering inputs that are not integers.");
                            throw description;
                        }
                    }

                    //                    if it's 0, even it's integer, it's not from 1 to 9
                    if (n == 1) {
                        if ((choice == 48)) {
                            logic_error description("entering integers that are not from 1 to 9.");
                            throw description;
                        }
                    }

                    //                    if the position it try to make a move has already equal to "O" or "X", then it's illegal.
                    //                    make Q and N as exception for this situation so that user could use N for as many time as they want.
                    if (choice != 'Q' && choice != 'N' && (square[choice - 48] == 'O' || square[choice - 48] == 'X')) {
                        logic_error description("illegal choice parameter, input should only be a position which has not been played.");
                        throw description;
                    }
                    else {
                        entering_position_check = false;
                    }

                }
                //                catch the exceptions
                catch (logic_error& e) {
                    cout << "logica error" << e.what() << "\n";
                    cout << "Player " << (player + 3) / 2 << ", enter a number('Q' to exit, 'N' to restart):  ";
                    cin >> string_choice;
                    int n = string_choice.length();

                    // declaring character array
                    char char_array[1];

                    // copying the contents of the
                    // string to char array
                    char_array[0] = string_choice.c_str()[0];

                    choice = char_array[0];
                    if (square[choice - 48] != 'O' && square[choice - 48] != 'X') {
                        player_number_check = false;
                    }
                }
            }

            //            stop the code when player inter Q.
            if (choice == 'Q') {
                cout << "The player exits the game. " << endl;
                return 0;
            }

            //            reuse the previous code which let the user choose the mode of game between one player or two players.
            else if (choice == 'N') {
                cout << "The player restarts the game. " << endl;
                initial_square();
                cout << "Do you want to play a 1-player or 2-player game? Please enter 1 or 2.\n";
                cin >> string_player_number;

                player_number_check = true;
                while (player_number_check) {
                    try {
                        int n = string_player_number.length();

                        // declaring character array
                        char char_array[1];

                        // copying the contents of the
                        // string to char array
                        char_array[0] = string_player_number.c_str()[0];

                        player_number = char_array[0];

                        if (n != 1) {
                            logic_error description("illegal player_number parameter, input should only be 1 or 2.");
                            throw description;
                        }

                        if (player_number != '1' && player_number != '2') {
                            logic_error description("illegal player_number parameter, input should only be 1 or 2.");
                            throw description;
                        }
                        else {
                            player_number_check = false;
                        }
                    }
                    catch (logic_error& e) {
                        cout << "logica error " << e.what() << "\n";
                        cout << "Do you want to play a 1-player or 2-player game? Please enter 1 or 2?\n";
                        cin >> string_player_number;
                        int n = string_player_number.length();
                        // declaring character array
                        char char_array[1];
                        // copying the contents of the
                        // string to char array
                        char_array[0] = string_player_number.c_str()[0];

                        player_number = char_array[0];

                        if (n == 1 && (player_number == '1' || player_number == '2')) {
                            player_number_check = false;
                        }
                    }
                }

                player = 1;
            }

            // select player's mark symbol
            if (player == -1)
                mark = 'X';
            else if (player == 1)
                mark = 'O';

            // update square array according to player's move
            mark_square(choice, mark);

            // check if game stops
            if_win = checkwin();

            // change to next player's move
            player *= -1;
        }

    } while (if_win == -1);

    print_board();
    if (if_win == 1)
        cout << "\nPlayer " << (-player + 3) / 2 << " win.\n ";
    else
        cout << "\nTie Game.\n";
    return 0;
}

/*********************************************
    FUNCTION TO RETURN GAME STATUS
    1 FOR GAME IS OVER WITH RESULT
    -1 FOR GAME IS IN PROGRESS
    O GAME IS OVER AND NO RESULT
**********************************************/

int checkwin()
{
    if (square[1] == square[2] && square[2] == square[3])
        return 1;
    else if (square[4] == square[5] && square[5] == square[6])
        return 1;
    else if (square[7] == square[8] && square[8] == square[9])
        return 1;
    else if (square[1] == square[4] && square[4] == square[7])
        return 1;
    else if (square[2] == square[5] && square[5] == square[8])
        return 1;
    else if (square[3] == square[6] && square[6] == square[9])
        return 1;
    else if (square[1] == square[5] && square[5] == square[9])
        return 1;
    else if (square[3] == square[5] && square[5] == square[7])
        return 1;
    // the board is full
    else if (square[1] != '1' && square[2] != '2' && square[3] != '3'
        && square[4] != '4' && square[5] != '5' && square[6] != '6'
        && square[7] != '7' && square[8] != '8' && square[9] != '9')
        return 0; // tie situation
    else
        return -1; // game in progress
}


/*******************************************************************
     FUNCTION TO DRAW BOARD OF TIC TAC TOE WITH PLAYER MARKS
********************************************************************/
void print_board()
{
    cout << "\n \t Tic Tac Toe \n";
    cout << "Player 1 (X)  -  Player 2 (O)" << endl << endl;

    cout << "     |     |     " << endl;
    cout << "  " << square[1] << "  |  " << square[2] << "  |  " << square[3] << endl;

    cout << "_____|_____|_____" << endl;
    cout << "     |     |     " << endl;

    cout << "  " << square[4] << "  |  " << square[5] << "  |  " << square[6] << endl;

    cout << "_____|_____|_____" << endl;
    cout << "     |     |     " << endl;

    cout << "  " << square[7] << "  |  " << square[8] << "  |  " << square[9] << endl;

    cout << "     |     |     " << endl << endl;
}

// update the square array accoring to player's move
void mark_square(char choice, char mark)
{
    // if choice = 1, c = '1', ...
    char c = choice;
    square[c - 48] = mark;
}

void initial_square()
{
    for (int i = 1; i < 11; i++) {
        square[i] = '0' + i;
    }
}

