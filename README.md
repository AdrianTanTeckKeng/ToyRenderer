# ToyRenderer

<img src="images/bunny2.jpg?raw=true"/>

A program written for a class in C++ that reads in a description of a scene and rasterizes 3D shaded surfaces based on the objects in the scene as a PPM image output.

Scene files were provided for the class, which contains information on setting up the camera, on light sources, on the objects to be loaded, the linear transformations to set up the scene and the materials properties needed for shading.

The graphics pipeline work by applying the coordinate transformations on the objects to map it from world spadce to camera space and then, to normalized device coordinates. Pixel colors at vertices were computed using a lighting model that includes diffuse reflections, specular reflections and attenuation from distances and interpolation was done using barycentric coordinates. Both Gouraud and Phong shading methods were implemented. Finally, backface culling and depth buffering were also implemented to correctly color and shade all the surfaces.
___

### References

[1] Jekyll theme "Minimal" for GitHub Pages: https://github.com/pages-themes/minimal (CC0 1.0 Universal License)
<br>[2] Dummy photo via: https://pixabay.com/photos/man-male-adult-person-caucasian-1209494/ (Pixabay License)
<br>[3] Dummy thumbnail image created by rawpixel.com: https://www.freepik.com/free-vector/set-elements-infographic_2807573.htm (Standard Freepik License)

