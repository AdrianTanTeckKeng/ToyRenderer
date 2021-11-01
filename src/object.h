#ifndef OBJECT_H
#define OBJECT_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

#include "utilities.h"

struct Material{
    Eigen::Vector3d diffuse;
    Eigen::Vector3d ambient;
    Eigen::Vector3d specular;
    double shininess;
};

class object{
public:
    // Variables that determine objects(vertices, normals, faces, and indexes to determine which normals)
    std::vector<double> vertices;
    std::vector<double> normals;
    std::vector<int> faces;
    std::vector<int> normalIdx; //Store index for normals
    int numOfVertices, numOfFaces, numOfNorms;

    // Variables that determine lighting properties
    Material properties;

    // Variable to tag names
    std::string objectName;
    int copyIdx;
public:
    // Constructors
    object() {}
    object(std::string name, const char* filename) {
        properties.ambient << 0., 0., 0.;
        properties.diffuse << 0., 0., 0.;
        properties.specular << 0., 0., 0.;
        properties.shininess = 0.;
        load_data(name, filename);}

    // Copy constructor
    object(const object &o1) {vertices = o1.vertices;
                              normals = o1.normals; 
                              faces = o1.faces;
                              normalIdx = o1.normalIdx;
                              numOfVertices = o1.numOfVertices;
                              numOfFaces = o1.numOfFaces;
                              numOfNorms = o1.numOfNorms;
                              properties = o1.properties;
                              objectName = o1.objectName;
                              copyIdx = o1.copyIdx;}

    // Function to load data
    void load_data(std::string name, const char* filename);
    
    // Function to write out data
    void writeData();
    void writeVertices();
    void writeFaces();
    void writeNormal();
    void writeFaceNormal();
    void writeMaterial();

    // Function to apply linear operation on the object
    void applyOperationsOnObject(Eigen::Matrix4d operations);
    void applyOperationsOnNormal(Eigen::Matrix4d operations);

    // Function to change lighting properties
    void changeAmbient(double x0, double x1, double x2){
        properties.ambient[0] = x0;
        properties.ambient[1] = x1;
        properties.ambient[2] = x2;
    }

    void changeDiffuse(double x0, double x1, double x2){
        properties.diffuse[0] = x0;
        properties.diffuse[1] = x1;
        properties.diffuse[2] = x2;
    }

    void changeSpecular(double x0, double x1, double x2){
        properties.specular[0] = x0;
        properties.specular[1] = x1;
        properties.specular[2] = x2;
    }

    void changeShininess(double x0){
        properties.shininess = x0;
    }

    // Return the name of the object
    std::string getName(){
        return objectName;
    }

    // Return face
    int getFace(int idx){
        return faces[idx];
    }

    // Return vertices
    int getVertices(int idx){
        return vertices[idx];
    }

    // Return component of vertices
    Eigen::Vector3d getVertexComponent(int idx){
        Eigen::Vector3d tmp;
        tmp << vertices[3*(idx-1)], vertices[3*(idx-1) + 1], vertices[3*(idx-1) + 2];
        return tmp;
    }

    // Return component of normal
    Eigen::Vector3d getNormalComponent(int idx){
        Eigen::Vector3d tmp;
        tmp << normals[3*(idx-1)], normals[3*(idx-1) + 1], normals[3*(idx-1) + 2];
        return tmp;
    }

    // Change copy index
    void changeCopyIdx(int idx){
        copyIdx = idx;
    }
};

void object::load_data(std::string name, const char* filename){
    objectName = name;
    copyIdx = 0;
    std::string directory = "data/";
    directory += filename;
    const char *fileLocation = directory.c_str();

    // Opening file
    std::ifstream file(fileLocation);
    if (!file.is_open()){
        std::cout << "error: " << filename << " does not exist." << std::endl;
        exit(1);
    }


    // Parsing data
    std::string line;
    numOfVertices = 0;
    numOfFaces = 0;
    numOfNorms = 0;
    while(std::getline(file, line)){
        // Initialize a stringstream to parse the contents of the line.
        std::stringstream ss(line);

        // Declare variable to store the string, int, and double values.
        std::string type;

        // Parse string
        ss >> type;
        if(ss.fail()){
            std::cout << "Error: unable to parse line." << std::endl;
        }

        if (type == "v"){
            double x0, x1, x2;
            ss >> x0 >> x1 >> x2;
            numOfVertices += 1;
            vertices.push_back(x0);
            vertices.push_back(x1);
            vertices.push_back(x2);
        }
        else if (type == "vn"){
            double x0, x1, x2;
            ss >> x0 >> x1 >> x2;
            numOfNorms += 1;
            normals.push_back(x0);
            normals.push_back(x1);
            normals.push_back(x2);
        }
        else{
            std::string tmp1, tmp2, tmp3;
            ss >> tmp1 >> tmp2 >> tmp3;
            std::string regex_str = "//"; 
            auto tokens_tmp1 = split(tmp1, regex_str);
            auto tokens_tmp2 = split(tmp2, regex_str);
            auto tokens_tmp3 = split(tmp3, regex_str);
            
            
            faces.push_back(stoi(tokens_tmp1[0]));
            faces.push_back(stoi(tokens_tmp2[0]));
            faces.push_back(stoi(tokens_tmp3[0]));
            normalIdx.push_back(stoi(tokens_tmp1[1]));
            normalIdx.push_back(stoi(tokens_tmp2[1]));
            normalIdx.push_back(stoi(tokens_tmp3[1]));
            numOfFaces += 1;
        }

    }
    file.close();
}



void object::writeData(){
    if (copyIdx==0)
        std::cout << objectName << std::endl;
    else{
        std::cout << objectName << "_copy" << copyIdx << std::endl;
    }

    writeVertices();
    writeFaces();
}

void object::writeMaterial(){
    std::cout << "ambient: " << properties.ambient[0] << " " << properties.ambient[1] << " " << properties.ambient[2] << std::endl;
    std::cout << "diffuse: " << properties.diffuse[0] << " " << properties.diffuse[1] << " " << properties.diffuse[2] << std::endl;
    std::cout << "specular: " << properties.specular[0] << " " << properties.specular[1] << " " << properties.specular[2] << std::endl;
    std::cout << "shininess: " << properties.shininess << std::endl;
}

void object::writeVertices(){
    for (int i=0; i<numOfVertices; i++){
        std::cout << "v " << vertices[3*i] << " " << vertices[3*i + 1] << " " << vertices[3*i +2] << std::endl;
    }
    std::cout << std::endl;
}

void object::writeFaces(){
    for (int i=0; i<numOfFaces; i++){
        std::cout << "f " << faces[3*i] << " " << faces[3*i+1] << " " << faces[3*i+2] << std::endl;
    }
    std::cout << std::endl;
}

void object::writeNormal(){
    for (int i=0; i<numOfNorms; i++){
        std::cout << normals[3*i] << " " << normals[3*i+1] << " " << normals[3*i+2] << std::endl;
    }
    std::cout << std::endl;
}

void object::writeFaceNormal(){
    for (int i=0; i<numOfFaces; i++){
        std::cout << "f " << faces[3*i] << "//" << normalIdx[3*i] 
        << " " << faces[3*i+1] << "//" << normalIdx[3*i+1]
        << " " << faces[3*i+2] << "//" << normalIdx[3*i+2] << std::endl;
    }
    std::cout << std::endl;
}

void object::applyOperationsOnObject(Eigen::Matrix4d operations){
    // Obtain w of the operations
    // Loop through all the vertices
    for(int i=0; i<numOfVertices; i++){
        //std::cout << "Checking multiplication " << std:endl;
        // Construct Homogeneous coordinates
        Eigen::Vector4d v(vertices[3*i], vertices[3*i+1], vertices[3*i+2], 1);

        // Apply operations
        Eigen::Vector4d new_v = operations * v;

        // Update vertices value
        vertices[3*i  ] = new_v[0] / new_v[3];
        vertices[3*i+1] = new_v[1] / new_v[3];
        vertices[3*i+2] = new_v[2] / new_v[3];
    }
}

void object::applyOperationsOnNormal(Eigen::Matrix4d operations){
    // First construct the correct operations by taking inverse and then transpose
    Eigen::Matrix4d tmp1 = operations.inverse();
    Eigen::Matrix4d tmp2 = tmp1.transpose();
    
    // Loop through all the normals
    for(int i=0; i<numOfNorms; i++){
        // Construct coordinates
        Eigen::Vector4d n(normals[3*i], normals[3*i+1], normals[3*i+2], 1);

        // Apply operations
        Eigen::Vector4d new_n = tmp2 * n;

        // Find the norm of the normsl to normalize
        double norm = sqrt(new_n[0] * new_n[0] + new_n[1] * new_n[1] + new_n[2] * new_n[2]);

        // Update normal value
        normals[3*i  ] = new_n[0] / norm;
        normals[3*i+1] = new_n[1] / norm;
        normals[3*i+2] = new_n[2] / norm;
    }
}

#endif