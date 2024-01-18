import numpy as np

class GeometryCalculator:
    def __init__(self, scale):
        self.scale = scale

    def findTheta(self, center, posX, posY):
        disX = center[0] - posX
        disY = center[1] - posY
        theta = np.arctan2(disY, disX)
        return np.mod(theta + np.pi, np.pi)

    def findPos(self, r, theta, center):
        r = r * 1.164
        x = r * np.cos(theta) + center[0]
        y = r * np.sin(theta) + center[1]
        return int(x), int(y)

    def pixelConvert(self, mid_pixel, pixel):
        x = pixel[0] - mid_pixel[0]
        y = pixel[1] - mid_pixel[1]
        return x * self.scale, y * self.scale

# Example Usage
scale_value = 1.5  # replace with your desired scale value
geometry_calculator = GeometryCalculator(scale_value)

center = (100, 100)
posX, posY = 120, 80
theta = geometry_calculator.findTheta(center, posX, posY)
x, y = geometry_calculator.findPos(2, theta, center)
mid_pixel = (50, 50)
pixel = (30, 20)
converted_pixel = geometry_calculator.pixelConvert(mid_pixel, pixel)

print("Theta:", theta)
print("Position:", (x, y))
print("Converted Pixel:", converted_pixel)