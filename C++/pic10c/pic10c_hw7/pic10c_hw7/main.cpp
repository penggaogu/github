/*
    PIC 10C Homework 7, main.cpp
    Purpose: employee fstream & makefile
    Author: Penggao Gu
    Date: 11/28/2022
*/

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include "Employee.h"


int main(){
//    infile the employee.txt
    ifstream in_file;
    in_file.open("/Users/gupenggao/Downloads/employee.txt");
    if (in_file.fail()) { return 0; }
//    create the string str for getline in the next while loop
    string str;
    vector<Employee> employee_list;
    
//    in the loop, get each line as string
    while (getline(in_file, str)) {
//        convert the str into vector char
        vector<char> char_line(str.begin(), str.end());
//        with the vector, we could use the first 30 elements in the vector as the name, and the next 10 elements as the initial salary.
        string name(char_line.begin(), char_line.begin()+30);
        string salary_string(char_line.begin()+31, char_line.end());
//        convert the string salary into double for the employee class
        double salary = stod(salary_string);
//        use employee to initialize each employee base on the information from each line
        Employee employee(name,salary);
//        use raise to increase the salary of the employee
        employee.raise(5.0);
//        store the employee into the vector<employee> employee_list
        employee_list.push_back(employee);
    }

//    sort the vector by Lambda Expression.
    sort(employee_list.begin(), employee_list.end(), [](const Employee &a, const Employee &b)-> bool {return a.get_salary()<b.get_salary();} );
//    use ofstream to create a txt file for output
    ofstream output_data;
    output_data.open("/Users/gupenggao/Downloads/SalaryRanking.txt");
    for (int i = 0; i < employee_list.size(); i++){
//        adjust the format to mach the screenshot in the hw7 instruction.
        output_data << right << setw(30) << employee_list[i].get_name() << setfill(' ') << right << setw(10) << fixed << setprecision(2) <<  employee_list[i].get_salary() << endl;
    }

}
