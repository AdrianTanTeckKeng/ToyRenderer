#ifndef _SCENE_H
#define _SCENE_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <algorithm> // For max
#include <math.h> // For pow

#include "object.h"
#include "light.h"
#include "transformations.h"

struct camera{
    double position[3];
    double orientation[4];
    double perspective[6];
    Eigen::Matrix4d cameraTransform;
    Eigen::Matrix4d invCameraTransform;
    Eigen::Matrix4d perspectiveProjection;
};

struct transformedObjectData{
    int objectIdx;
    object transformedObject;
    transformations actionsOnObject;
    transformations actionsOnNormal;
};

class scene{
public:
    std::vector<std::shared_ptr<object>> listOriginalObjects;
    std::vector<std::shared_ptr<transformedObjectData>> listTransformedObjects;
    std::vector<std::shared_ptr<light>> listLights;
    camera scene_camera;
    int xres, yres, mode;
    double stepx, stepy;
    int* image;

    // Color channels for the grid
    double* gridR;
    double* gridG;
    double* gridB;
    double* zbuffer;
public:
    scene() {}
    scene(const char* filename, int x, int y, int m) {
        xres = x, yres = y, mode = m;
        stepx = 2.0 / (xres - 1);
        stepy = 2.0 / (yres - 1);
        image = new int[xres*yres];
        loadScene(filename);}

    void loadScene(const char* filename);
    void printCamera();
    void writeTransformed();

    // Functions for wireframe rendering
    void rasterization();
    void draw(int x0, int y0, int x1, int y1, int* image);

    // Functions for triangle rasterization
    void renderScene();
    double fij(double x, double y, double xi, double yi, double xj, double yj);
    Eigen::Vector3d SceneLighting(Eigen::Vector3d P, Eigen::Vector3d n, Material properties);
    Eigen::Vector3d WorldToNDC(Eigen::Vector3d v);

    void GouraudShading(std::shared_ptr<transformedObjectData> item);
    void RasterColoredTriangle(Eigen::Vector3d colorA, Eigen::Vector3d NDCA,
        Eigen::Vector3d colorB, Eigen::Vector3d NDCB,
        Eigen::Vector3d colorC, Eigen::Vector3d NDCC);

    void PhongShading(std::shared_ptr<transformedObjectData> item);
    void RasterPhong(Eigen::Vector3d va, Eigen::Vector3d vb, Eigen::Vector3d vc,
        Eigen::Vector3d na, Eigen::Vector3d nb, Eigen::Vector3d nc, Material properties);
    
    


    // Functions to construct camera transform and perspective transform
    Eigen::Matrix4d getRotationMatrix(double ux, double uy, double uz, double rd);
    Eigen::Matrix4d getTranslationMatrix(double vx, double vy, double vz);
    Eigen::Matrix4d getPerspectiveMatrix(double n, double f, double l, double r, double t, double b);
};

Eigen::Matrix4d scene::getRotationMatrix(double ux, double uy, double uz, double rad){
    // First construct the rotation matrix
    Eigen::Matrix4d matrix;
    matrix << ux*ux + (1-ux*ux)*cos(rad), ux*uy*(1-cos(rad))-uz*sin(rad), ux*uz*(1-cos(rad)) + uy*sin(rad), 0,
    uy*ux*(1-cos(rad)) + uz*sin(rad), uy*uy+(1-uy*uy)*cos(rad), uy*uz*(1-cos(rad))-ux*sin(rad), 0,
    uz*ux*(1-cos(rad)) - uy*sin(rad), uz*uy*(1-cos(rad))+ux*sin(rad), uz*uz+(1-uz*uz)*cos(rad), 0,
    0, 0, 0, 1;

    return matrix;
}

Eigen::Matrix4d scene::getTranslationMatrix(double vx, double vy, double vz){
    // First construct the translation matrix
    Eigen::Matrix4d matrix;
    matrix << 1, 0, 0, vx,
    0, 1, 0, vy,
    0, 0, 1, vz,
    0, 0, 0, 1;

    return matrix;
}

Eigen::Matrix4d scene::getPerspectiveMatrix(double n, double f, double l, double r, double t, double b){
    Eigen::Matrix4d matrix;
    matrix << 2*n/(r-l), 0, (r+l)/(r-l), 0,
    0, 2*n/(t-b), (t+b)/(t-b), 0,
    0, 0, -(f+n)/(f-n), -2*f*n/(f-n),
    0, 0, -1, 0;

    return matrix;
}

void scene::printCamera(){
    std::cout << "Position: " << scene_camera.position[0] << " " << scene_camera.position[1] << " " << scene_camera.position[2] << std::endl;
    std::cout << "orientation: " << scene_camera.orientation[0] << " " << scene_camera.orientation[1] << " " << scene_camera.orientation[2] <<  " " << scene_camera.orientation[3] << std::endl;
    std::cout << "perspective: " << scene_camera.perspective[0] << " "
    << scene_camera.perspective[1] << " " 
    << scene_camera.perspective[2] << " "
    << scene_camera.perspective[3] << " "
    << scene_camera.perspective[4] << " "
    << scene_camera.perspective[5] << " " << std::endl;
}

void scene::writeTransformed(){
    for(auto& item : listTransformedObjects){
        item->transformedObject.writeData();
    }
}

void scene::draw(int x0, int y0, int x1, int y1, int* image){
    

    // First check if the line is purely horizontal or vertical
    int dx = x1 - x0;
    int dy = y1 - y0;

    
    
    // Draw horizontal lines
    if(dy==0){
        if (x0>x1){
            std::swap(x0, x1);
            std::swap(y0, y1);
        }
        for(int x=x0; x<=x1; x++){
        // Only draw if pixel is within the screen
        if(0<=x && x<xres && 0<=y0 && y0<yres){
            image[y0*xres + x] = 1;
            }
        }
        return;
    }

    // Draw vertical lines
    if(dx==0){
        if(y0>y1){
            std::swap(y0, y1);
        }
        
        for(int j=y0; j<=y1; j++){
        // Only draw if pixel is within the screen
        if(0<=x0 && x0<xres && 0<=j && j<yres){
            image[j*xres + x0] = 1;
            }
        }
        return;
    }
    

    
    // Now draw general lines
   
    // First check if 0 < |m| < 1, if no we swap x and y and keep track of this when writing the values
    
    bool swap = false;
    if(std::abs(dx) < std::abs(dy)){
        std::swap(x0, y0);
        std::swap(x1, y1);
        swap = true;
    }

    
    // Check direction from x0 to x1. We always want to go from left to right
    if (x0>x1){
        std::swap(x0, x1);
        std::swap(y0, y1);
    }

    // Calculate new dx and dy in case we did swap the coordinates
    dx = x1 - x0;
    dy = y1 - y0;
    

    // line drawing algorithm
    int epsilon = 0;
    int y = y0;

    // Positive gradient
    if(dy>0){
        for(int x=x0; x<=x1; x++){
            // Only draw if pixel is within the screen
            if(swap){
                // If we swap our x and y, we need to take note when rendering
                if(0<=x && x<yres && 0<=y && y<xres){
                    image[x*xres + y] = 1;
                }
            }
            else{
                if(0<=x && x<xres && 0<=y && y<yres){
                    image[y*xres + x] = 1;
                }
            }
            if (2*(epsilon + dy) < dx){
                epsilon += dy;
            }
            else{
                epsilon += dy - dx;
                y += 1;
            }
        }
    }

    else{
        // Line drawing for negative values of m
        for(int x=x0; x<=x1; x++){
            // Only draw if pixel is within the screen
            if(swap){
                // If we swap our x and y, we need to take note when rendering
                if(0<=x && x<yres && 0<=y && y<xres){
                    image[x*xres + y] = 1;
                }
            }
            else{
                if(0<=x && x<xres && 0<=y && y<yres){
                    image[y*xres + x] = 1;
                }
            }

            if (2*(epsilon + dy) > -dx){
                epsilon += dy;
            }
            else{
                // -dx instead of + dx
                epsilon += dy + dx;
                //Substract from y instead of adding to y since the slope is negative
                y -= 1;
            }
        }
    }
}


// Function to perform rasterization
void scene::rasterization(){
    
    double dx = 2.0 / (xres - 1);
    double dy = 2.0 / (yres - 1);
    object copiedData;
        for(auto& item : listTransformedObjects){
            copiedData = item->transformedObject;

            // Perform the transformations to get the NDC coordinates
            copiedData.applyOperationsOnObject(scene_camera.invCameraTransform);
            copiedData.applyOperationsOnObject(scene_camera.perspectiveProjection);

            
            // Loop through all the faces and render them
            for(int i=0; i<copiedData.numOfFaces; i++){
                int f0, f1, f2;
                f0 = copiedData.faces[3*i]; 
                f1 = copiedData.faces[3*i+1]; 
                f2 = copiedData.faces[3*i+2];

                double x0, y0, x1, y1, x2, y2;
                x0 = copiedData.vertices[3*(f0-1)];
                y0 = copiedData.vertices[3*(f0-1) + 1];

                x1 = copiedData.vertices[3*(f1-1)];
                y1 = copiedData.vertices[3*(f1-1) + 1];

                x2 = copiedData.vertices[3*(f2-1)];
                y2 = copiedData.vertices[3*(f2-1) + 1];

                // Convert data to integer. Map from [-1, 1] to the grid
                int ix0, iy0, ix1, iy1, ix2, iy2;
                ix0 = static_cast<int> ((x0 - -1) / dx);
                ix1 = static_cast<int> ((x1 - -1) / dx);
                ix2 = static_cast<int> ((x2 - -1) / dx);


                iy0 = static_cast<int> ((y0 - -1) / dy);
                iy1 = static_cast<int> ((y1 - -1) / dy);
                iy2 = static_cast<int> ((y2 - -1) / dy);
                
                // Start drawing lines
                draw(ix0, iy0, ix1, iy1, image);
                draw(ix1, iy1, ix2, iy2, image);
                draw(ix2, iy2, ix0, iy0, image);
            }
        }

    
    // First part of PPM file
    std::cout << "P3" << std::endl;
    std::cout << xres << " " << yres << std::endl;
    std::cout << 255 << std::endl;
    
    
    for(int j=0; j<yres; j++){
        for(int i=0; i<xres; i++){
            if(image[(yres - j - 1)*xres + i] > 0)
                std::cout << 255 << ' ' << 0 << ' ' << 0 << std::endl;
            else
                std::cout << 0 << ' ' << 0 << ' ' << 0 << std::endl;
        }
    }
}


// Helper function to compute fij for barycentric coordinates
double scene::fij(double x, double y, double xi, double yi, double xj, double yj){
        return (yi - yj) * x + (xj - xi) * y + xi * yj - xj * yi;
}

// Helper function to convert world coordinates to Cartesian NDC coordinates
Eigen::Vector3d scene::WorldToNDC(Eigen::Vector3d _v){
        Eigen::Vector4d tmp1(_v[0], _v[1], _v[2], 1);
        tmp1 = scene_camera.invCameraTransform * tmp1;
        Eigen::Vector4d tmp2(tmp1[0] / tmp1[3], tmp1[1] / tmp1[3], tmp1[2] / tmp1[3], 1);
        tmp1 = scene_camera.perspectiveProjection * tmp2;
        Eigen::Vector3d NDC;
        NDC << tmp1[0] / tmp1[3], tmp1[1] / tmp1[3], tmp1[2] / tmp1[3];
        return NDC;
}

// Function to compute lighting at a particular point
Eigen::Vector3d scene::SceneLighting(Eigen::Vector3d P, Eigen::Vector3d n, Material properties){

    // Unpack material data
    Eigen::Vector3d cd = properties.diffuse;
    Eigen::Vector3d ca = properties.ambient;
    Eigen::Vector3d cs = properties.specular;
    double phong = properties.shininess;

    // Compute e
    Eigen::Vector3d cameraPosition;
    cameraPosition << scene_camera.position[0], scene_camera.position[1], scene_camera.position[2];
    Eigen::Vector3d e = cameraPosition - P;
    e = e / e.norm();

    // Initialize diffuse sum and specular sum
    Eigen::Vector3d diffuse_sum;
    diffuse_sum << 0.0, 0.0, 0.0;
    Eigen::Vector3d specular_sum;
    specular_sum << 0.0, 0.0, 0.0;
    
    for(auto& light : listLights){
        Eigen::Vector3d lp = light->returnPosition();
        Eigen::Vector3d lc = light->returnRGB();
        double k = light->returnAttenuation();
        Eigen::Vector3d ldirection = lp - P;
        double d = ldirection.norm();

        // Attenuation lc with 1/(1+kd^2)
        lc = lc * (1.0 / (1.0 + k * pow(d, 2)));
        ldirection = ldirection / d;

        Eigen::Vector3d ldiffuse = lc * std::max(0.0, n.dot(ldirection));
        diffuse_sum += ldiffuse;

        Eigen::Vector3d ePlusL = e + ldirection;
        ePlusL = ePlusL / ePlusL.norm();
        Eigen::Vector3d lspecular = lc * pow(std::max(0.0, n.dot(ePlusL)), phong);
        specular_sum += lspecular;
    }
    
    Eigen::Vector3d c;
    double components[3];
    for(int i=0; i<3; i++){
        components[i] = std::min(1.0, ca[i] + diffuse_sum[i] * cd[i] + specular_sum[i] * cs[i]);
    }
    c << components[0], components[1], components[2];

    return c;
}

// Function to perform Gouraud shading with backface culling and depth buffering
void scene::GouraudShading(std::shared_ptr<transformedObjectData> item){

    // Obtain color of each face in the object
    
    for(int i=0; i<item->transformedObject.numOfFaces; i++){
                int f0, f1, f2;

                // Obtain va, vb, vc
                f0 = item->transformedObject.faces[3*i]; 
                f1 = item->transformedObject.faces[3*i+1]; 
                f2 = item->transformedObject.faces[3*i+2];

                
                Eigen::Vector3d va = item->transformedObject.getVertexComponent(f0);
                Eigen::Vector3d vb = item->transformedObject.getVertexComponent(f1);
                Eigen::Vector3d vc = item->transformedObject.getVertexComponent(f2);

                // Obtain na, nb, nc
                f0 = item->transformedObject.normalIdx[3*i]; 
                f1 = item->transformedObject.normalIdx[3*i+1]; 
                f2 = item->transformedObject.normalIdx[3*i+2]; 

                Eigen::Vector3d na = item->transformedObject.getNormalComponent(f0);
                Eigen::Vector3d nb = item->transformedObject.getNormalComponent(f1);
                Eigen::Vector3d nc = item->transformedObject.getNormalComponent(f2);

                // Get color due to lighting 
                Eigen::Vector3d colorA = SceneLighting(va, na, item->transformedObject.properties);
                Eigen::Vector3d colorB = SceneLighting(vb, nb, item->transformedObject.properties);
                Eigen::Vector3d colorC = SceneLighting(vc, nc, item->transformedObject.properties);

                // Convert coordinates to NDC
                Eigen::Vector3d NDCA = WorldToNDC(va);
                Eigen::Vector3d NDCB = WorldToNDC(vb);
                Eigen::Vector3d NDCC = WorldToNDC(vc);

                RasterColoredTriangle(colorA, NDCA, colorB, NDCB, colorC, NDCC);
    }    
}


void scene::RasterColoredTriangle(Eigen::Vector3d colorA, Eigen::Vector3d NDCA,
    Eigen::Vector3d colorB, Eigen::Vector3d NDCB,
    Eigen::Vector3d colorC, Eigen::Vector3d NDCC){

    // Check for backface culling
    Eigen::Vector3d cross = (NDCC - NDCB).cross(NDCA - NDCB);
    if (cross[2] < 0){
        return;
    }

    // Convert to screen coordinates(integer form)
    int xa, ya, xb, yb, xc, yc;
    xa = static_cast<int> ((NDCA[0] - -1) / stepx);
    xb = static_cast<int> ((NDCB[0] - -1) / stepx);
    xc = static_cast<int> ((NDCC[0] - -1) / stepx);

    ya = static_cast<int> ((NDCA[1] - -1) / stepy);
    yb = static_cast<int> ((NDCB[1] - -1) / stepy);
    yc = static_cast<int> ((NDCC[1] - -1) / stepy);

    int xmin = std::min(xa, std::min(xb, xc));
    int xmax = std::max(xa, std::max(xb, xc));
    int ymin = std::min(ya, std::min(yb, yc));
    int ymax = std::max(ya, std::max(yb, yc));

    for(int x=xmin; x<=xmax; x++){
        for(int y=ymin; y<=ymax; y++){
            //  Compute alpha, beta and gamma for barycentric coordinates

            double alpha = fij(x, y, xb, yb, xc, yc) / fij(xa, ya, xb, yb, xc, yc);
            double beta = fij(x, y, xa, ya, xc, yc) / fij(xb, yb, xa, ya, xc, yc);
            double gamma = fij(x, y, xa, ya, xb, yb) / fij(xc, yc, xa, ya, xb, yb);

            // Check if this is a valid value, if yes, perform interpolation
            if(0<=alpha && alpha<=1 && 0<=beta && beta<=1 && 0<=gamma && gamma<=1){
                Eigen::Vector3d NDC = alpha*NDCA + beta*NDCB + gamma*NDCC;

                // Now check NDC coordinates lie within screen
                if(-1<=NDC[0] && NDC[0]<=1 && -1<=NDC[1] && NDC[1]<=1 && -1<=NDC[2] && NDC[2]<=1 && NDC[2] <= zbuffer[y*xres + x]){

                    // Set depth buffer to nearest location
                    zbuffer[y*xres + x] = NDC[2]; // Negative because of the definition of z
                    double R = alpha * colorA[0] + beta * colorB[0] + gamma * colorC[0];
                    double G = alpha * colorA[1] + beta * colorB[1] + gamma * colorC[1];
                    double B = alpha * colorA[2] + beta * colorB[2] + gamma * colorC[2];

                    gridR[y*xres + x] = R;
                    gridG[y*xres + x] = G;
                    gridB[y*xres + x] = B;
                }
            }

        }
    }
}

// Function to perform Phong shading with backface culling and depth buffering
void scene::PhongShading(std::shared_ptr<transformedObjectData> item){

    // Obtain color of each face in the object
    
    for(int i=0; i<item->transformedObject.numOfFaces; i++){
                int f0, f1, f2;

                // Obtain va, vb, vc
                f0 = item->transformedObject.faces[3*i]; 
                f1 = item->transformedObject.faces[3*i+1]; 
                f2 = item->transformedObject.faces[3*i+2];

                
                Eigen::Vector3d va = item->transformedObject.getVertexComponent(f0);
                Eigen::Vector3d vb = item->transformedObject.getVertexComponent(f1);
                Eigen::Vector3d vc = item->transformedObject.getVertexComponent(f2);

                // Obtain na, nb, nc
                f0 = item->transformedObject.normalIdx[3*i]; 
                f1 = item->transformedObject.normalIdx[3*i+1]; 
                f2 = item->transformedObject.normalIdx[3*i+2]; 

                Eigen::Vector3d na = item->transformedObject.getNormalComponent(f0);
                Eigen::Vector3d nb = item->transformedObject.getNormalComponent(f1);
                Eigen::Vector3d nc = item->transformedObject.getNormalComponent(f2);

                // Get color due to lighting 

                RasterPhong(va, vb, vc, na, nb, nc, item->transformedObject.properties);
    }    
}

void scene::RasterPhong(Eigen::Vector3d va, Eigen::Vector3d vb, Eigen::Vector3d vc,
    Eigen::Vector3d na, Eigen::Vector3d nb, Eigen::Vector3d nc,
    Material properties){

    // First construct normalized device coordinates
    Eigen::Vector3d NDCA = WorldToNDC(va);
    Eigen::Vector3d NDCB = WorldToNDC(vb);
    Eigen::Vector3d NDCC = WorldToNDC(vc);


    // Check for backface culling
    Eigen::Vector3d cross = (NDCC - NDCB).cross(NDCA - NDCB);
    if (cross[2] < 0){
        return;
    }

    // Convert to screen coordinates(integer form)
    int xa, ya, xb, yb, xc, yc;
    xa = static_cast<int> ((NDCA[0] - -1) / stepx);
    xb = static_cast<int> ((NDCB[0] - -1) / stepx);
    xc = static_cast<int> ((NDCC[0] - -1) / stepx);

    ya = static_cast<int> ((NDCA[1] - -1) / stepy);
    yb = static_cast<int> ((NDCB[1] - -1) / stepy);
    yc = static_cast<int> ((NDCC[1] - -1) / stepy);

    int xmin = std::min(xa, std::min(xb, xc));
    int xmax = std::max(xa, std::max(xb, xc));
    int ymin = std::min(ya, std::min(yb, yc));
    int ymax = std::max(ya, std::max(yb, yc));

    for(int x=xmin; x<=xmax; x++){
        for(int y=ymin; y<=ymax; y++){
            //  Compute alpha, beta and gamma for barycentric coordinates

            double alpha = fij(x, y, xb, yb, xc, yc) / fij(xa, ya, xb, yb, xc, yc);
            double beta = fij(x, y, xa, ya, xc, yc) / fij(xb, yb, xa, ya, xc, yc);
            double gamma = fij(x, y, xa, ya, xb, yb) / fij(xc, yc, xa, ya, xb, yb);

            // Check if this is a valid value, if yes, perform interpolation
            if(0<=alpha && alpha<=1 && 0<=beta && beta<=1 && 0<=gamma && gamma<=1){
                Eigen::Vector3d NDC = alpha*NDCA + beta*NDCB + gamma*NDCC;

                // Now check NDC coordinates lie within screen
                if(-1<=NDC[0] && NDC[0]<=1 && -1<=NDC[1] && NDC[1]<=1 && -1<=NDC[2] && NDC[2]<=1 && NDC[2] <= zbuffer[y*xres + x]){

                    // Set depth buffer to nearest location
                    zbuffer[y*xres + x] = NDC[2];

                    // Perform interpolation 
                    Eigen::Vector3d interpolated_n = alpha * na + beta * nb + gamma * nc;
                    // Normalize the interpolated normal vector
                    interpolated_n = interpolated_n / interpolated_n.norm();
                    Eigen::Vector3d interpolated_coordinates = alpha * va + beta * vb + gamma * vc;

                    // Obtain color using
                    Eigen::Vector3d color = SceneLighting(interpolated_coordinates, interpolated_n, properties);
                    gridR[y*xres + x] = color[0];
                    gridG[y*xres + x] = color[1];
                    gridB[y*xres + x] = color[2];
                }
            }

        }
    }
}

// Function to render scene
void scene::renderScene(){

    // First, allocate memory to the color channels for the world and zbuffer
    gridR = new double[xres*yres];
    gridG = new double[xres*yres];
    gridB = new double[xres*yres];
    zbuffer = new double[xres*yres];
    for(int i=0; i<xres*yres; i++){
        zbuffer[i] = 9999;
    }

    // We determine whether to perform Gouraud Shading or Phong shading
    
    if(mode==0){
        // For each object, we will perform Gouraud shading
        for(auto& item : listTransformedObjects){
            GouraudShading(item);
        }
    }
    else if(mode==1){
        // For each object, we will perform Phong shading
        for(auto& item : listTransformedObjects){
            PhongShading(item);
        }  
    }
    else{
        std::cout << "No such shading mode " << std::endl;
            exit(1);
    }

    // Writing out to PPM file
    // First part of PPM file
    std::cout << "P3" << std::endl;
    std::cout << xres << " " << yres << std::endl;
    std::cout << 255 << std::endl;
    
    
    
    for(int j=0; j<yres; j++){
        for(int i=0; i<xres; i++){
            std::cout << static_cast<int>(255.999 * gridR[(yres - j - 1)*xres + i]) << ' '
            << static_cast<int>(255.999 * gridG[(yres - j - 1)*xres + i]) << ' '
            << static_cast<int>(255.999 * gridB[(yres - j - 1)*xres + i]) << std::endl;
        }
    }
}

void scene::loadScene(const char* filename){
    // Opening file
    std::ifstream file(filename);

    if(!file.is_open()){
        std::cout << "error: " << filename << " does not exist." << std::endl;
        exit(1);
    }

    // 1. Reading in camera configurations. Pretty much have to hardcore these lines
    std::string line;
    std::string dummy;

    // This reads camera:
    std::getline(file, line);

    // Get position
    std::getline(file, line);
    std::stringstream ss(line);
    ss >> dummy >> scene_camera.position[0] >> scene_camera.position[1] >> scene_camera.position[2];
    Eigen::Matrix4d translation = getTranslationMatrix(scene_camera.position[0], scene_camera.position[1], scene_camera.position[2]);

    
    // Get orientation
    std::getline(file, line);
    ss.str("");
    ss.clear();
    ss << line;
    
    ss >> dummy >> scene_camera.orientation[0] >> scene_camera.orientation[1] >> scene_camera.orientation[2] >> scene_camera.orientation[3];
    Eigen::Matrix4d rotation = getRotationMatrix(scene_camera.orientation[0], 
        scene_camera.orientation[1], 
        scene_camera.orientation[2], 
        scene_camera.orientation[3]);

    // Camera transform C = TR
    scene_camera.cameraTransform = translation * rotation;
    scene_camera.invCameraTransform = scene_camera.cameraTransform.inverse(); 
    
    
    // Get near, far, left, right, top, bottom
    for(int i=0; i<6; i++){
        std::getline(file, line);
        ss.str("");
        ss.clear();
        ss << line;
        ss >> dummy >> scene_camera.perspective[i];
    }

    scene_camera.perspectiveProjection = getPerspectiveMatrix(scene_camera.perspective[0], 
        scene_camera.perspective[1], 
        scene_camera.perspective[2],
        scene_camera.perspective[3],
        scene_camera.perspective[4],
        scene_camera.perspective[5]);


    std::getline(file, line); // Clear the white space

    // 2. Storing lightings
    while(std::getline(file, line)){
        std::stringstream tmp_ss(line);
        std::string dummy;
        tmp_ss >> dummy;
        if(dummy!="light"){
            break;
        }
        double x0, x1, x2, r0, r1, r2, k;
        tmp_ss >> x0 >> x1 >> x2 >> dummy >> r0 >> r1 >> r2 >> dummy >> k;
        listLights.push_back(std::make_shared<light>(x0, x1, x2, r0, r1, r2, k));
    }

    
    // 3. Loading objects
    std::getline(file, line); // Clear the objects: line

    while(std::getline(file, line)){
        std::stringstream tmp_ss(line);
        std::string type, objectFileName;
        tmp_ss >> type;

        if(type == ""){
            break;
        }
        
        tmp_ss >> objectFileName;
        //std::cout << "Getting object " << type << " with file name: " << objectFileName << std::endl;       
        listOriginalObjects.push_back(std::make_shared<object>(type, objectFileName.c_str()));
    }

    
    // 4. Reading materials and transformations
    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string name;
        ss >> name;
        transformations operationsOnObject;
        transformations operationsOnNormal;

        double _ca[3];
        double _cd[3];
        double _cs[3];
        double _shininess;
        while(std::getline(file, line)){
            std::stringstream ss(line);
            std::string type;
            ss >> type;
            if(type == ""){
                break;
            }

            // Handling materials property, I will hardcore this since it always follows as
            // ambient
            // diffuse
            // specular
            // shininess


            // ambient
            if(type == "ambient"){
                ss >> _ca[0] >> _ca[1] >> _ca[2];
                continue;
            }

            // diffuse
            if(type == "diffuse"){
                ss >> _cd[0] >> _cd[1] >> _cd[2];
                continue;
            }

            // specular
            if(type == "specular"){
                ss >> _cs[0] >> _cs[1] >> _cs[2];
                continue;
            }

            // shininess
            if(type == "shininess"){
                ss >> _shininess;
                continue;
            }


            if(type=="t"){
            double x0, x1, x2;
            ss >> x0 >> x1 >> x2;
            operationsOnObject.translate(x0, x1, x2);
            }
            else if(type=="r"){
            double x0, x1, x2, x3;
            ss >> x0 >> x1 >> x2 >> x3;
            operationsOnObject.rotate(x0, x1, x2, x3);
            operationsOnNormal.rotate(x0, x1, x2, x3);
            }
            else if(type=="s"){
            double x0, x1, x2;
            ss >> x0 >> x1 >> x2;
            operationsOnObject.scale(x0, x1, x2);
            operationsOnNormal.scale(x0, x1, x2);
            }

        }
        bool flag = false;
        object copiedData;
        int tmpIdx = 0;
        for(auto& item : listOriginalObjects){
            if (item->getName() == name){
                flag = true;
                copiedData = *item;
                break;
            }
            tmpIdx += 1;
        }
        if(!flag){
            std::cout << "Couldnt find the object to copy data " << std::endl;
            exit(1);
        }

        copiedData.applyOperationsOnObject(operationsOnObject.getTransformations());
        copiedData.applyOperationsOnNormal(operationsOnNormal.getTransformations());
        copiedData.changeAmbient(_ca[0], _ca[1], _ca[2]);
        copiedData.changeDiffuse(_cd[0], _cd[1], _cd[2]);
        copiedData.changeSpecular(_cs[0], _cs[1], _cs[2]);
        copiedData.changeShininess(_shininess);

        int idx = 1;
        for(auto& item : listTransformedObjects){
            if(item->transformedObject.getName() == copiedData.getName())
                idx += 1;
        }
        copiedData.changeCopyIdx(idx);
        transformedObjectData tmpStruct;
        tmpStruct.objectIdx = tmpIdx;
        tmpStruct.transformedObject = copiedData;
        tmpStruct.actionsOnObject = operationsOnObject;
        listTransformedObjects.push_back(std::make_shared<transformedObjectData>(tmpStruct));
    }

    /*
    for(auto& item : listTransformedObjects){
        std::cout << item->objectIdx << std::endl;
        item->transformedObject.writeMaterial();
        item->transformedObject.writeNormal();
    } 
    */
    
    // 5. Closing file
    
    file.close();
}

#endif 