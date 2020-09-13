# OOP_VRP
Object Oriented approach to a VRP solver
version 1: has issues with keeping track of distance and time for vehicles, test 1 and First() are not updated for multivehicle

version 2: still has (worse?!?) issues with keeping track of distance and time for vehicles, test 1 and First() are updated for multivehicle (but have not been verified to be sure they are working correctly).

use the given .csv file to test VRP code

9/13/20:
updates1: 
1) I merged the progress on the VRP versions of First() and Test1 from version 2 with the code in version 1.  
2) I updated the data tracking and it appears that vehicle distance and time is working (at least working better than either of the previous versions)
To do:
1) I'm still getting errors when running the VRP first() and as a result I cannot check test 1
2) I still need to update the dominance test (with data sorting)
3) I need to figure out a way to not do duplicate extensions, i.e. if I have created the vehicle assignment (0,0,0) -> (1,0,0), I do not want to also do (0,0,0) -> (0,1,0).
