# skastic
Visual programming language: SKetches of Abstract Syntax Trees. IC.

Skastic is a visual programming language where programs are contained in image files (png). Skastic program files contain nodes and edges representing an abstract syntax tree, or parse tree.

## Requirements and installation

Skastic requires python 3.4 or later. With a relatively small amount of effort it could be made to work on earlier versions of python, but the source code uses the python ast module, which has changed somewhat since 2.7. I've installed python 3.6.2 from the official python site. On MacOS, you should be able to install python 3.6 using brew. For best results, create a virtualenv for running skastic.

Skastic uses opencv for image analysis. On MacOS, these instructions should help get things working. http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

Skastic also requires tesseract 3.x for optical character recognition. The command line tool must be installed on the system and accessible from the user's PATH. On MacOS, tesseract can be installed using brew. For other platforms, see https://github.com/tesseract-ocr/tesseract for detailed instructions for installing this.

I have yet to create a setup.py file and it's not yet on pypi. For now, you just copy the python source files into a directory, put your image files there, and run "./skastic.py <filename>". Filenames should be of the form <name>.ska.png.

## How it works

A valid skastic program is an image with nodes and edges.

### Nodes

Nodes must be box-like shapes of mid-brightness (in HSB terms, something like the color of a highlighter) colors filled with dark-colored text representing the node's value. By "box-like" it is meant that the shape is roughly oval or rectangular, with significant width and height.

### Edges

Edges must be relatively thin line segments in a dark color. Line segments in this context are thin shapes that have 2 distinct ends. They may curve as desired to connect nodes. The lines *must not touch any nodes*, including the nodes it connects. Each endpoint of an edge is associated with either the closest node (based on distance to the node's center) or the top edge of the image (if that is closer than any node's center). In a valid program, only one endpoint of one edge can be associated with top edge of the program, and this edge is used to indicated the root node.

### Image analysis

Image analysis is relatively simple. OpenCV thresholding is used to create an inverted, greyscale image where the nodes and lines are full white on a black background. OpenCV contours are then found in this image, which are the outlines of the continuous white spaces. Contours that are very small are considered "specks" and discarded. Contours that are "thin" are considered edges, and "thick" contours are considered nodes.

- Specks: Contour perimeter is smaller than 40 pixels
- Thin: Ratio of contour area / perimeter < 5
- Thick: Ration of contour area / perimeter >= 5
