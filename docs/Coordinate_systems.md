## Standard CARLA coordinate system
Right hand side. \
X is pointing forward \
Y is pointing towards the right \
Z is pointing upward

![Alt text](../assets/CARLA_coordinateSystem.jpg?raw=true "CARLA coordinate system")

Positive steering leads to a right turn.\
Negative steering leads to a left turn.

The design philosophy of this repo is to convert all data into this unified coordinate system.\
The only exceptions are bounding boxes (and voxelized LiDAR) that is in are in the image (pixels) coordinate system detailed below.

## Bicycle Model coordinate system
Uses the same coordinate system as the standard one above.

## LiDAR coordinate system
Uses the standard CARLA coordinate system but is special in that we rotate it by 90°.
(We want the front half and the LiDAR sweep happens clockwise.)
The y axis points towards the front of the vehicle.
Since the LiDAR is mounted on the car the points are also shifted by it's position.

![Alt text](../assets/LiDAR_coordinate_system.png?raw=true "LiDAR coordinate system")

We directly convert the LiDAR into the ego vehciles reference frame so that is uses the CARLA coordinate system
This is done using the function: lidar_to_ego_coordinate from transfuser_utils.
The stored LiDAR points in the dataset are also in ego vehicle frame using the CARLA coordinate system.

When voxelizing the LiDAR we convert it from the CARLA coordinate system into an image coordinate system needed for bounding box detection.
Width (Y axis) is forward. Height (X axis) is right. The vehicle is at the center of the image and units are in pixels.
See bounding boxes.

# Rotation min, max
CARLA's rotation system uses rotations in degree. \
Their range goes from [-180°, 180°]. 0 is forward facing. \
180 degree is backward facing. \
We save the yaw of bounding boxes in radians in the dataset.


## Bounding box prediction coordinate system
We save the bounding boxes in the standard CARLA coordinate system.
They are saved in the ego vehicles reference frame.

For predictions the bounding boxes are converted into image space of the BEV projected LiDAR. 
The units are pixels not meters.
Currently, 1 pixel represents 0.25x0.25 meters.
The conversion is done using the function bb_vehicle_to_image_system in transfuser_utils.py and can be reverted with bb_image_to_vehicle_system.
In the image coordinate system the vehicle is at the center of the image.
The coordinate system is increasing width is going forward into the direction of the car (front), increasing height is going towards the right side of the car.
Note, that pytorch stores image in (height, width) format.


# IMU compass
According to the CARLA docs north is (0.0, -1.0, 0.0) w.r.t. the default CARLA coordinate system: \
![Alt text](../assets/Compass.png?raw=true "CARLA coordinate system")

Moving left increases x, moving up increases y etc. \
Nort 0°\
W -90° \
S -180° \
E 90°


# GPS 
The GPS uses the coordinate system, similar to the compass:\
![Alt text](../assets/LiDAR_coordinate_system.png?raw=true "GPS coordinate system")


We convert the GPS into the default CARLA coordinate system in the tick function.
The GPS saved in the hidden biases dataset is saved in the CARLA coordinate system.

# Target Point and Route in the Dataset
They are saved in the local coordinate frame of the ego vehicle in the dataset. \
Uses the default CARLA coordinate system.