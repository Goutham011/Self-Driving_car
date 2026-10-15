import cv2


class Vehicle:
    """Axis-aligned bounding box describing one detected vehicle."""

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)

    @property
    def coords(self):
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def center(self):
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    @property
    def area(self):
        return max(0, self.x_max - self.x_min) * max(0, self.y_max - self.y_min)

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def iou(self, other):
        """Intersection-over-union with another box (Vehicle or x/y coords)."""
        ox_min, oy_min, ox_max, oy_max = (other.coords if isinstance(other, Vehicle)
                                          else other)
        inter_w = max(0, min(self.x_max, ox_max) - max(self.x_min, ox_min))
        inter_h = max(0, min(self.y_max, oy_max) - max(self.y_min, oy_min))
        inter = inter_w * inter_h
        union = self.area + max(0, ox_max - ox_min) * max(0, oy_max - oy_min) - inter
        return inter / union if union > 0 else 0.0

    def merged_with(self, other):
        """Union of this box and another box."""
        ox_min, oy_min, ox_max, oy_max = (other.coords if isinstance(other, Vehicle)
                                          else other)
        return Vehicle(min(self.x_min, ox_min), min(self.y_min, oy_min),
                       max(self.x_max, ox_max), max(self.y_max, oy_max))

    def draw(self, frame, color=(0, 255, 255), thickness=2):
        cv2.rectangle(frame, (self.x_min, self.y_min), (self.x_max, self.y_max),
                      color, thickness)
        return frame
