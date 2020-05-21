import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

custom_data = np.load('Output/plotPoints_custom.npy')
opencv_data = np.load('Output/plotPoints_org.npy')

fig = plt.figure()
gs = plt.GridSpec(2, 3)
plt.legend(['Our Code', 'Built-In'])
plt.title('Comparison of Our Code with OpenCV based code')

# Comment in case a complete movie is desired
drift = 0
for i in range(len(custom_data)):
    print("FRAME {}".format(i+20))
    frame = cv2.imread('../Visual-Odometry/FRAMES/{}.jpg'.format(i))
    cv2.imshow('frame', frame)
    plt.plot(-custom_data[i][1], -custom_data[i][3], 'ro')
    plt.plot(opencv_data[i][1], -opencv_data[i][3], 'bo')
    drift += (0.5)*np.sum(np.sqrt((opencv_data[i, 1] + custom_data[i, 1])**2 + (opencv_data[i, 3] - custom_data[i, 3])**2))
    plt.pause(0.01)
    cv2.waitKey(1)

# Uncomment in case only final output is desired

#plt.plot(-custom_data[:,1], -custom_data[:,3], 'o', color='red')
#plt.plot(opencv_data[:, 1], -opencv_data[:, 3], 'o', color='blue')
#plt.legend(['Our Code', 'Built-In'])
#plt.title('Comparison of Our Code with OpenCV based code')
#plt.savefig('comparison.png')
#plt.show()
