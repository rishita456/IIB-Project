"""
A class to hold position and quarternion data: The code for manipulating
quaternions is derived from: https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/
"""

import matplotlib.pyplot as plt

class quaternionVector:
    def __init__(self, loc, quaternion):
        self.pos = loc
        self.quaternion = quaternion
    

    def q_conjugate(self,q):
        w, x, y, z = q
        return [w, -x, -y, -z]

    
    def qv_mult(self,q1, v1):
        q2 = [0.0,v1[0],v1[1],v1[2]]
        return self.q_mult(self.q_mult(q1, q2), self.q_conjugate(q1))[1:]
    
    def q_mult(self,q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return w, x, y, z
    

if __name__ == "__main__":
    quat1 = quaternionVector([0,0,0],[0.653,0.270,0.653,0.271])
    v2 = quat1.qv_mult(quat1.quaternion,[1,0,0])
    print(v2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0,0,0,1,0,0,color = 'b')
    ax.quiver(0, 0, 0,v2[0],v2[1],v2[2] ,color='r')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    plt.show()