
//
//  dijkstra.h
//  test
//
//  Created by Penggao Gu on 10/31/22.
//

#ifndef dijkstra_h
#define dijkstra_h
#include <vector>
#include <string>
#include <numeric>



std::vector<std::vector<size_t>> dijkstra(const std::vector<std::vector<double>> & map,
                                          const std::vector<std::string> & names,
                                          std::string origin,
                                          std::string target,
                                          double & cost){
    int origin_iter = 0;
    while(names[origin_iter] != origin){
        origin_iter ++;
    }
    int target_iter = 0;
    while(names[target_iter] != target){
        target_iter ++;
    }
    int size = map.size();
    
    
    std::vector<int> shortestDistances(size);
    std::vector<bool> visited(size);
    
    // Initialize all distances as
    // INFINITE and added[] as false
    for (int vertexIndex = 0; vertexIndex < size;
         vertexIndex++) {
        shortestDistances[vertexIndex] = 10000*size;
        visited[vertexIndex] = false;
    }
    
    shortestDistances[origin_iter] = 0;
    
    auto find_visited = find(visited.begin(),visited.end(),0);
    bool if_visited = true;
    if (find_visited != visited.end()){
        if_visited = false;
    }
    
    int check = 0;
    while(!if_visited && check < size+2 ){
            for(int i = 0; i < size; i++){
                if (visited[i] == 0 && shortestDistances[i] != 10000*size){
                    int row = i;
                    for(int j = 0; j < size; j++){
                        if(shortestDistances[j] > shortestDistances[row] + map[row][j] &&
                           map[row][j] != 0){
                            shortestDistances[j] = shortestDistances[row] + map[row][j];
                        }
                    }
                    visited[row] = true;
                    auto find_visited = find(visited.begin(),visited.end(),0);
                    if (find_visited == visited.end()){
                        if_visited = true;
                    }
                }
            }
        check++;
    }
    for(int i = 0; i < size; i++){
    }
    
    if(check > size){
        cost = -1;
    }
    else{
        cost = shortestDistances[target_iter];
    }
    
    
    std::vector<std::vector<std::vector<int>>> paths_all;        //record all paths

    
    
    int row  = origin_iter;
    int column = 1;
    double paths_size_sum;                         //record the total displacement
    std::vector<int> paths_size;                //record the size of displacement between each visited nodes
    std::vector<int> paths_visited;             //record the visited path in form of []
    std::vector<std::vector<int>> paths;        //record the path in from of [][]
    std::vector<bool> path_visited_bool(size);        //record if the row nodes are visited

    //initialized path_visited_bool all as false
    for (int i = 0; i < size; i++) {
        path_visited_bool[i] = false;
    }
    
    paths_visited.push_back(origin_iter);
    path_visited_bool[origin_iter] = true;
    
    
    for (int a = 0; a < abs(origin_iter - target_iter)-1; a++){
        if(map[0][a] != 0){
            row  = origin_iter;
            column = a;
            paths_size_sum = 0;                         //record the total displacement
            paths_size.clear();                //record the size of displacement between each visited nodes
            paths_visited.clear();             //record the visited path in form of []
            paths.clear();        //record the path in from of [][]
            //initialized path_visited_bool all as false
            for (int i = 0; i < size; i++) {
                path_visited_bool[i] = false;
            }
            
            paths_visited.push_back(origin_iter);
            path_visited_bool[origin_iter] = true;
            //        int breaker = 0;
            //        while(breaker > 30){
            if(map[row][column] != 0){
                std::vector<int> temp;
                temp.push_back(row);
                temp.push_back(column);
                paths.push_back(temp);
                temp.clear();
                paths_size.push_back(map[row][column]);
                paths_size_sum = accumulate(paths_size.begin(), paths_size.end(),0);
                row = column;
                
                while(row != origin_iter){
                    row = column;
                    for (int j = 0; j < size; j++){
                        column = j;
                        if(path_visited_bool[column] == 0 && map[row][column] != 0 && paths_size_sum + map[row][column] <= cost){
                            temp.push_back(row);
                            temp.push_back(column);
                            paths.push_back(temp);
                            temp.clear();
                            paths_size.push_back(map[row][column]);
                            paths_size_sum = accumulate(paths_size.begin(), paths_size.end(),0);
                            paths_visited.push_back(row);
                            path_visited_bool[column] = true;
                            break;
                            
                        }
                        else if(paths_size_sum == cost && paths[paths.size()-1][1] == target_iter){
                            paths_all.push_back(paths);
                            int last_row = paths[paths_visited.size()-1][0];
                            int last_column = paths[paths_visited.size()-1][1];
                            row = last_row;
                            column = last_column;
                            j = column;
                            paths.pop_back();
                            paths_size.pop_back();
                            paths_size_sum = accumulate(paths_size.begin(), paths_size.end(),0);
                            paths_visited.pop_back();
                            path_visited_bool[last_column] = false;
                        }
                        else if (column == size-1 && paths_size_sum != cost && (path_visited_bool[row] == 0)){
                            int last_row = paths[paths_visited.size()-1][0];
                            int last_column = paths[paths_visited.size()-1][1];
                            row = last_row;
                            column = last_column;
                            j = column;
                            paths.pop_back();
                            paths_size.pop_back();
                            paths_size_sum = accumulate(paths_size.begin(), paths_size.end(),0);
                            paths_visited.pop_back();
                            path_visited_bool[last_column] = false;
                        }
                        
                    }
                }
            }
            //            breaker ++;
            //        }
            column = a;
        }
    }

//
//
//
//    row  = origin_iter;
//    column = 2;
//    paths_size_sum = 0;                         //record the total displacement
//    paths_size.clear();                //record the size of displacement between each visited nodes
//    paths_visited.clear();             //record the visited path in form of []
//    paths.clear();        //record the path in from of [][]
//    //initialized path_visited_bool all as false
//    for (int i = 0; i < size; i++) {
//        path_visited_bool[i] = false;
//    }
//
//    paths_visited.push_back(origin_iter);
//    path_visited_bool[origin_iter] = true;
//
//    time_t endwait;
//    double seconds = 5;
//
//    endwait = time (NULL) + seconds ;
//    while (time (NULL) < endwait)
//    {
//        if(map[row][column] != 0){
//            std::vector<int> temp;
//            temp.push_back(row);
//            temp.push_back(column);
//            paths.push_back(temp);
//            temp.clear();
//            paths_size.push_back(map[row][column]);
//            paths_size_sum = accumulate(paths_size.begin(), paths_size.end(),0);
//            row = column;
//
//            while(row != origin_iter){
//                row = column;
//                for (int j = 0; j < size; j++){
//                    column = j;
//                    if(path_visited_bool[column] == 0 && map[row][column] != 0 && paths_size_sum + map[row][column] <= cost){
//                        temp.push_back(row);
//                        temp.push_back(column);
//                        paths.push_back(temp);
//                        temp.clear();
//                        paths_size.push_back(map[row][column]);
//                        paths_size_sum = accumulate(paths_size.begin(), paths_size.end(),0);
//                        paths_visited.push_back(row);
//                        path_visited_bool[column] = true;
//                        break;
//                    }
//                    else if(paths_size_sum == cost && paths[paths.size()-1][1] == target_iter){
//                        paths_all.push_back(paths);
//                        int last_row = paths[paths_visited.size()-1][0];
//                        int last_column = paths[paths_visited.size()-1][1];
//                        row = last_row;
//                        column = last_column;
//                        j = column;
//                        paths.pop_back();
//                        paths_size.pop_back();
//                        paths_size_sum = accumulate(paths_size.begin(), paths_size.end(),0);
//                        paths_visited.pop_back();
//                        path_visited_bool[last_column] = false;
//                        break;
//                    }
//                    else if (column == size-1 && paths_size_sum != cost && (path_visited_bool[row] == 0)){
//                        int last_row = paths[paths_visited.size()-1][0];
//                        int last_column = paths[paths_visited.size()-1][1];
//                        row = last_row;
//                        column = last_column;
//                        j = column;
//                        paths.pop_back();
//                        paths_size.pop_back();
//                        paths_size_sum = accumulate(paths_size.begin(), paths_size.end(),0);
//                        paths_visited.pop_back();
//                        path_visited_bool[last_column] = false;
//                        break;
//                    }
//                }
//            }
//        }
//    }
//
    
    
    
    
//    paths_visited.push_back(origin_iter);
//    std::vector<std::vector<size_t>> true_path;
//    for (int i = 0; i < paths_all.size();i++){
//        true_path[i].push_back(paths_all[i][0][0]);
//        for (int j = 0; j < paths_all[i].size(); j++){
//            true_path[i].push_back(paths_all[i][j][1]);
//        }
//    }

//    std::vector<std::vector<size_t>> true_path =
//         { { 0, 5, 6, 0, 10, 0},
//           { 3, 0, 0, 1, 3, 0},
//           { 0, 0, 0, 0, 2, 4},
//           { 0, 0, 1, 0, 2, 0},
//           { 0, 0, 0, 0, 0, 2},
//           { 0, 0, 3, 9, 0, 0}};
    
    std::vector<std::vector<size_t>> true_path;
    std::vector<size_t> temp;
    for (int i = 0; i < paths_all.size(); i++){
        for (int j = 0; j < paths_all[i].size(); j++){
            if (j ==0){
                temp.push_back(paths_all[i][j][0]);
                temp.push_back(paths_all[i][j][1]);
                true_path.push_back(temp);
                temp.clear();
            }
            else{
                true_path[i].push_back(paths_all[i][j][1]);
            }
        }
    }
    return true_path;
}
                                    
void print_paths(const std::vector<std::vector<size_t>>& paths, const std::vector<std::string>& names)
{
    if (paths.empty() == true){
        std::cout << "No path found!"<<std::endl;
    }
    else{
        for (size_t i = 0; i < paths.size(); i++){
            std::cout << names[paths[i][0]];
            for (size_t j = 1; j < paths[i].size(); ++j){
                std::cout << " -- " << names[paths[i][j]];
            }
            std::cout << std::endl;
        }
    }
}

#endif /* dijkstra_h */
