#ifndef TRANSFORMATIONS_H
#define TRANSFORMATIONS_H

#include <Eigen/Dense>
#include <math.h> // For cosine function
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>


// transformation class that stores all transformations as matrices in a vector list
class transformations{
public:
    // Storing all linear transformations in a vector list
    std::vector<std::shared_ptr<Eigen::Matrix4d>> list;
public:

    // Constructor initializes identity
    transformations() {}

     // transformations;
     Eigen::Matrix4d getTransformations();
     void translate(double vx, double vy, double vz);
     void rotate(double ux, double uy, double uz, double radian);
     void scale(double sx, double sy, double sz);
     void applyOperationsFromFile(const char* filename);

};

Eigen::Matrix4d transformations::getTransformations(){
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    for(auto& item : list){
        matrix = *item * matrix;
    }
    return matrix;
}



void transformations::translate(double vx, double vy, double vz){
    // First construct the translation matrix
    Eigen::Matrix4d matrix;
    matrix << 1, 0, 0, vx,
    0, 1, 0, vy,
    0, 0, 1, vz,
    0, 0, 0, 1;

    list.push_back(std::make_shared<Eigen::Matrix4d>(matrix));
}

void transformations::rotate(double ux, double uy, double uz, double rad){
    // First construct the rotation matrix
    Eigen::Matrix4d matrix;
    matrix << ux*ux + (1-ux*ux)*cos(rad), ux*uy*(1-cos(rad))-uz*sin(rad), ux*uz*(1-cos(rad)) + uy*sin(rad), 0,
    uy*ux*(1-cos(rad)) + uz*sin(rad), uy*uy+(1-uy*uy)*cos(rad), uy*uz*(1-cos(rad))-ux*sin(rad), 0,
    uz*ux*(1-cos(rad)) - uy*sin(rad), uz*uy*(1-cos(rad))+ux*sin(rad), uz*uz+(1-uz*uz)*cos(rad), 0,
    0, 0, 0, 1;

    list.push_back(std::make_shared<Eigen::Matrix4d>(matrix));
}

void transformations::scale(double sx, double sy, double sz){
    // First construct the scale matrix
    Eigen::Matrix4d matrix;
    matrix << sx, 0, 0, 0,
    0, sy, 0, 0,
    0, 0, sz, 0,
    0, 0, 0, 1;

    list.push_back(std::make_shared<Eigen::Matrix4d>(matrix));
}

void transformations::applyOperationsFromFile(const char* filename){
    std::ifstream file(filename);

    if(!file.is_open()){
        std::cout << "error: " << filename << " does not exist." << std::endl;
        exit(1);
    }

    std::string line;
    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string type;

        // Parse string
        ss >> type;
        if(ss.fail()){
            std::cout << "error: unable to parse line." << std::endl;
            exit(1);
        }

        if(type=="t"){
            double x0, x1, x2;
            ss >> x0 >> x1 >> x2;
            translate(x0, x1, x2);
        }
        else if(type=="r"){
            double x0, x1, x2, x3;
            ss >> x0 >> x1 >> x2 >> x3;
            rotate(x0, x1, x2, x3);
        }
        else{
            double x0, x1, x2;
            ss >> x0 >> x1 >> x2;
            scale(x0, x1, x2);
        }
    }
    file.close();
}

#endif