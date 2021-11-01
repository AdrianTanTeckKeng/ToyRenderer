#include <iostream>
#include <fstream>
#include <sstream>
#include "scene.h"

int main(int argc, const char* argv[]){
    if(argc != 5){
        std::cout << "Wrong number of input!" << std::endl;
        exit(1);
    }

    // Storing inputs
    int xres = atoi(argv[2]);
    int yres = atoi(argv[3]);
    int mode = atoi(argv[4]);
    scene world(argv[1], xres, yres, mode);
    world.renderScene();
    return 0;
}