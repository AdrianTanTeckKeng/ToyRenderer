# ToyRenderer

<img src="images/bunny2.jpg?raw=true"/>

A program written for a class in C++ that reads in a description of a scene and rasterizes 3D shaded surfaces based on the objects in the scene as a PPM image output.

Scene files were provided for the class, which contains information on setting up the camera, on light sources, on the objects to be loaded, the linear transformations to set up the scene and the materials properties needed for shading.

The graphics pipeline work by applying the coordinate transformations on the objects to map it from world spadce to camera space and then, to normalized device coordinates. Pixel colors at vertices were computed using a lighting model that includes diffuse reflections, specular reflections and attenuation from distances and interpolation was done using barycentric coordinates. Both Gouraud and Phong shading methods were implemented. Finally, backface culling and depth buffering were also implemented to correctly color and shade all the surfaces.
