from math import sqrt, acos, sin, cos, asin, atan2
from util.Vector3 import Vector3


class Quaternion:
    def __init__(self, vec=None):
        self._vec = vec

    def __getitem__(self, ind):
        return self._vec[ind]

    def __str__(self):
        return(
            "(" + str(self._vec[0]) + "," +
            str(self._vec[1]) + "," +
            str(self._vec[2]) + "," +
            str(self._vec[3]) + ")"
        )

    def magnitude(self):
        return sqrt(
            self._vec[0]**2 +
            self._vec[1]**2 +
            self._vec[2]**2 +
            self._vec[3]**2
        )

    def __len__(self):
        return len(self._vec)

    def __mul__(self, vec):
        if (isinstance(vec, Quaternion)):
            if (len(vec) < 3):
                return "Quaternion required for multiplication"
            return Quaternion(
                vec=[
                    self._vec[0] * vec[3] + self._vec[1] * vec[2] - self._vec[2] * vec[1] + self._vec[3] * vec[0],
                    -self._vec[0] * vec[2] + self._vec[1] * vec[3] + self._vec[2] * vec[0] + self._vec[3] * vec[1],
                    self._vec[0] * vec[1] - self._vec[1] * vec[0] + self._vec[2] * vec[3] + self._vec[3] * vec[2],
                    -self._vec[0] * vec[0] - self._vec[1] * vec[1] - self._vec[2] * vec[2] + self._vec[3] * vec[3]
                ]
            )
        elif (isinstance(vec, Vector3)):
            if (len(vec) < 2):
                return "Vector3 required for multiplication"
            return Quaternion(
                vec=[
                    self._vec[3]*vec[0] - self._vec[2]*vec[1] + self._vec[1]*vec[2],
                    self._vec[2]*vec[0] + self._vec[3]*vec[1] - self._vec[0]*vec[2],
                    -self._vec[2]*vec[0] + self._vec[0]*vec[1] + self._vec[3]*vec[2],
                    -self._vec[0]*vec[0] - self._vec[1]*vec[1] - self._vec[2]*vec[2]
                ]
            )
        else:
            return Quaternion(
                vec=[
                    self._vec[0]*vec,
                    self._vec[1]*vec,
                    self._vec[2]*vec,
                    self._vec[3]*vec
                ]
            )

    def normalized(self):
        mag = self.magnitude()
        return Quaternion(
            vec=[
                self._vec[0]/mag,
                self._vec[1]/mag,
                self._vec[2]/mag,
                self._vec[3]/mag
            ]
        )

    def conjugate(self):
        return Quaternion(
            vec=[
                -self._vec[0],
                -self._vec[1],
                -self._vec[2],
                self._vec[3]
            ]
        )

    def inverse(self):
        mag_squared = (self.magnitude())**2
        return Quaternion(
            vec=[
                -self._vec[0]/mag_squared,
                -self._vec[1]/mag_squared,
                -self._vec[2]/mag_squared,
                self._vec[3]/mag_squared
            ]
        )

    def rotate(self, vec):
        qt = Quaternion([vec[0], vec[1], vec[2], 0])
        rotated_qt = Quaternion(vec=self._vec) * qt*Quaternion(vec=self._vec).conjugate()

        return Vector3(
            vec=[
                rotated_qt[0],
                rotated_qt[1],
                rotated_qt[2]
            ]
        )

    def angle(self, q):
        #        inv = self.inverse()
        #        res = q * inv
        #        return acos(res[0])
        q12 = self.conjugate() * q
        return 2 * atan2(Vector3(q12[0:3]).magnitude(), q12[3])

    def euler_angle(self, axis):
        if axis == 'y':
            return asin(2*(self._vec[0]*self._vec[2]-self._vec[3]*self._vec[1]))
        else:
            return 0

    def rotate_by_euler(self, x, y, z, angle):
        c = cos(angle/2)
        s = sin(angle/2)
        result = Quaternion(
            vec=[
                c,
                x*s,
                y*s,
                z*s
            ])
        return self*result.normalized()
