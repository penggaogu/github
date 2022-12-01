//
//PIC 10C Homework 5, ColorBST.h
//Purpose: Define a color binary search tree class
//Author: Penggao GU
//Date: 11/9/2022
//

#ifndef ColorBST_h
#define ColorBST_h

#include <iostream>
#include <string>

using namespace std;



template <typename T>
class TreeNode;



template <typename T>
class ColorBST
{
public:
//    initialize the ColoBST, initialize the root as nullptr
    ColorBST(){
        root = nullptr;
    }
    
    
//    similar to the given example of BST. the difference is that we also need to store the char color.
//    the defalt color is red so if there are only one input, T data, then set color = 'r'
    void insert(T data){
        TreeNode<T>* new_node = new TreeNode<T>;
        new_node->data = data;
        new_node-> left = nullptr;
        new_node->right = nullptr;
        new_node -> color = 'r';
        if (root == nullptr) root = new_node;
        else root -> insert_node(new_node);
    }
    
    
//    similar to the given example of BST. the difference is that we also need to store the char color.
    void insert(T data, char c){
        TreeNode<T>* new_node = new TreeNode<T>;
        new_node->data = data;
        new_node-> left = nullptr;
        new_node->right = nullptr;
        new_node -> color = c;
        if (root == nullptr) root = new_node;
        else root -> insert_node(new_node);
    }
    
    
//    use this function for the print subnode of the root node recursively.
//    From left to mid to right, which meaning ascendingly.
    void node_print_ascending(TreeNode<T>* node){
        if (node == nullptr){
            return;
        }
        node_print_ascending(node->left);
        cout << node -> data << " ";
        node_print_ascending(node->right);
    }
    
//    use the above function on the root of the tree to print all the node of the colorBST
    void ascending_print(){
        TreeNode<T>* node = this -> root;
        node_print_ascending(node);
        cout << endl;
    }
    
    
//    use this function for the print subnode of the root node recursively.
//    From right to mid to left, which meaning descendingly.
    void node_print_descending(TreeNode<T>* node){
        if (node == nullptr){
            return;
        }
        node_print_descending(node->right);
        cout << node -> data << " ";
        node_print_descending(node->left);
    }
    
//    use the above function on the root of the tree to print all the node of the colorBST
    void descending_print(){
        TreeNode<T>* node = this -> root;
        node_print_descending(node);
        cout << endl;
    }
    
    
//    initialize the large as root data
//    then use while loop find the most right node, which should be the largest node in the colorBST
    T largest() const{
        TreeNode<T>* node = this -> root;
        T large = this -> root -> data;
        while(node -> right != nullptr){
            large = node -> right -> data;
            node = node -> right;
        }
        return large;
    }
    
    
//    check root color
    bool BlackRoot(){
        TreeNode<T>* node = this -> root;
        if (node->color == 'b'){
            return true;
        }
        else{
            return false;
        }
    }
    
    
//    for ever subnode, if the color is red, check if the nexted left or right node has the same color.
//    recursively to check all the possilbe nodes.
    bool NoDoubleRed_nodecheck(TreeNode<T>* node){
        if (node == nullptr){
            return true;
        }
        if (node -> color == 'r'){
            if ((node -> right != nullptr && node -> right -> color == 'r')
                || (node -> left != nullptr && node -> left -> color == 'r')){
                return false;
            }
        }
        if ((NoDoubleRed_nodecheck(node->left))
            && (NoDoubleRed_nodecheck(node->right))){
            return true;
        }
        else{
            return false;
        }
    }

//    use the above function on the root of the tree to check all the node of the colorBST
    bool NoDoubleRed(){
        TreeNode<T>* node = this -> root;
        return NoDoubleRed_nodecheck(node);
    }
    
    
    int BlackDepth_nodecheck(TreeNode<T>* node) {
        if (node == nullptr){
            return 1;
        }
//        compute the depth recursively
        int leftdepth = BlackDepth_nodecheck(node->left);
        int rightdepth = BlackDepth_nodecheck(node->right);
//        if the leftdepth not equal rightdepth or there substring obey the blackdepth
        if ((leftdepth != rightdepth) || (leftdepth == -1) || (rightdepth == -1)){
            return -1;
        }
//        else follow up the depth
        else
            if (node -> color == 'b'){
                leftdepth ++;
            }
        return leftdepth;
    }
    
//    appply the BlackDepth_nodecheck to root to check all possible path
    bool BlackDepth(){
        TreeNode<T>* node = this -> root;
        if (BlackDepth_nodecheck(node) == -1){
            return false;
        }
        else{
            return true;
        }
    };
    
    
private:
    TreeNode<T>* root;
};





template <typename T>
class TreeNode
{
public:
//    similar to the example of BST's treenode class. insert the node in the same way.
    void insert_node(TreeNode* new_node){
        if (new_node->data < data){
            if (left == nullptr) left = new_node;
            else left -> insert_node(new_node);
        }
        else if (data < new_node -> data){
            if (right == nullptr) right = new_node;
            else right -> insert_node(new_node);
        }
    }
    
    
private:
//    similar to the example of BST's treenode class. has the char color to store the color of node. 
    T data;
    char color;
    TreeNode* left;
    TreeNode* right;
    friend class ColorBST<T>;
};


#endif /* ColorBST_h */
