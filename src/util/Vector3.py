from math import sqrt, sin, cos, acos, isclose

class Vector3(object):
    def __init__(self, vec=None):
        self._vec = vec

    def __getitem__(self, ind):
        return self._vec[ind]

    def __sub__(self, vec):
        return Vector3(
        vec = [
        self._vec[0] - vec[0],
        self._vec[1] - vec[1],
        self._vec[2] - vec[2]
        ]
        )

    def __str__(self):
        return (
            "(" + str(self._vec[0]) + "," +
             str(self._vec[1]) + "," +
             str(self._vec[2]) + ")"
             )

    def magnitude(self):
        return sqrt(
            self._vec[0]**2 +
            self._vec[1]**2 +
            self._vec[2]**2
            )

    def __add__(self, vec):
        return Vector3(
            vec = [
            self._vec[0] + vec[0],
            self._vec[1] + vec[1],
            self._vec[2] + vec[2]
            ]
        )

    def dot(self, vec):
        return (self._vec[0] * vec[0]) +\
            (self._vec[1] * vec[1]) +\
            (self._vec[2] * vec[2])

    def __len__(self):
        return len(self._vec)

    def __rmul__(self, vec):
        if (isinstance(vec, Vector3)):
            if (len(vec) < 2):
                return "Vector3 required for multiplication"
            return Vector3(
            vec =
                [
                    self._vec[1]*vec[2] - self._vec[2]*vec[1],
                    self._vec[2]*vec[0] - self._vec[0]*vec[2],
                    self._vec[0]*vec[1] - self._vec[1]*vec[0],
                ]
            )
        else:
            return Vector3(
                vec =
                    [
                        self._vec[0]*vec,
                        self._vec[1]*vec,
                        self._vec[2]*vec
                    ]
            )

    def __mul__(self, vec):
        if (isinstance(vec, Vector3)):
            if (len(vec) < 2):
                return "Vector3 required for multiplication"
            return Vector3(
            vec =
                [
                    self._vec[2]*vec[1] - self._vec[1]*vec[2],
                    self._vec[0]*vec[2] - self._vec[2]*vec[0],
                    self._vec[1]*vec[0] - self._vec[0]*vec[1],
                ]
            )
        else:
            return Vector3(
                vec =
                    [
                        self._vec[0]*vec,
                        self._vec[1]*vec,
                        self._vec[2]*vec
                    ]
            )
    
    def __truediv__(self, vec):
        return Vector3(
            vec =
                [
                    self._vec[0]*1.0/vec,
                    self._vec[1]*1.0/vec,
                    self._vec[2]*1.0/vec
                ]
        )

    def normalized(self):
        mag = self.magnitude()
        if mag == 0.0:
            return Vector3(
            vec = [
                0.0,
                0.0,
                0.0,
            ]
            )
        else:
            return Vector3(
            vec = [
                self._vec[0]/mag,
                self._vec[1]/mag,
                self._vec[2]/mag
            ]
        )
    
    def rotation_with_angle(self, angle, axis):
        if axis == "x":
            return Vector3(
                vec = 
                [
                    self._vec[0],
                    self._vec[1]*cos(angle) - self._vec[2]*sin(angle),
                    self._vec[1]*sin(angle) + self._vec[2]*cos(angle)
                ]
            )        
        if axis == "y":
            return Vector3(
                vec = 
                [
                    self._vec[0]*cos(angle) + self._vec[2]*sin(angle),
                    self._vec[1],
                    -self._vec[0]*sin(angle) + self._vec[2]*cos(angle)
                ]
            )
        if axis == "z":
            return Vector3(
                vec = 
                [
                    self._vec[0]*cos(angle) - self._vec[1]*sin(angle),
                    self._vec[0]*sin(angle) + self._vec[1]*cos(angle),
                    self._vec[2]
                    
                ]
            )
            
    def angle_about_axis(self, vec, axis):
        if axis == "y":
            v1 = Vector3(vec = [self._vec[0], 0, self._vec[2]])
            v2 = Vector3(vec = [vec[0], 0, vec[2]])
            if isclose(v1.magnitude(),0.0) or isclose(v2.magnitude(),0.0):
                return 0.0
            else:
                return acos(max(0, min(1,(v1.dot(v2))/(v1.magnitude()*v2.magnitude()))))
            
    
    def clamp(self, minval, maxval):
        if self.magnitude() < minval:
            return self.normalized()*minval
        elif self.magnitude() > maxval:
            return self.normalized()*maxval
        else:
            return self