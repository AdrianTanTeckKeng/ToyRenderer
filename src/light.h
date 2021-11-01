#ifndef _LIGHT_H
#define _LIGHT_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

class light{
public:
    Eigen::Vector3d position;
    Eigen::Vector3d rgb;
    double attenuation;
public:
    light() {}
    light(double x, double y, double z, double r, double g, double b, double k){
        setPosition(x, y, z);
        setRGB(r, g, b);
        setAttenuation(k);
    }

    // functions to set position, RGB and attenuation
    void setPosition(double x, double y, double z);
    void setRGB(double r, double g, double b);
    void setAttenuation(double k);

    // functions to return position, RGB and attenuation
    Eigen::Vector3d returnPosition(){return position;}
    Eigen::Vector3d returnRGB(){return rgb;}
    double returnAttenuation(){return attenuation;}


    // functions to print data of lighting
    void printLight();
    void printPosition();
    void printRGB();
    void printAttenuation();
};

void light::setPosition(double x, double y, double z){
    position[0] = x;
    position[1] = y;
    position[2] = z;
}

void light::setRGB(double r, double g, double b){
    rgb[0] = r;
    rgb[1] = g;
    rgb[2] = b;
}

void light::setAttenuation(double k){
    attenuation = k;
}

void light::printPosition(){
    std::cout << "x: " << position[0] << " y: " << position[1] << " z: " << position[2] << std::endl;
}

void light::printRGB(){
    std::cout << "r: " << rgb[0] << " g: " << rgb[1] << " b: " << rgb[2] << std::endl;
}

void light::printAttenuation(){
    std::cout << "attenuation: " << attenuation << std::endl;
}

void light::printLight(){
    printPosition();
    printRGB();
    printAttenuation();
}
#endif