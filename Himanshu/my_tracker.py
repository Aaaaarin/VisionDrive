import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

class Tracker:
    def __init__(self, max_disappeared=8, max_distance=35):
        self.next_id = 0
        self.objects = OrderedDict()
        self.rects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, rect):
        x1, y1,x2, y2 = rect
        width, height = x2-x1, y2-y1
        if width > 20 and height > 30 and width < 400:
            self.objects[self.next_id] = centroid
            self.rects[self.next_id] = rect
            self.disappeared[self.next_id] = 0
            self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.rects[object_id]
        del self.disappeared[object_id]

    def update(self, rects_in):
        if len(rects_in) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_output_format()

        valid_rects = []
        input_centroids = np.zeros((len(rects_in), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects_in):
            width, height = endX-startX, endY-startY
            if width > 20 and height > 30 and width < 400:
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                input_centroids[i] = (cX, cY)
                valid_rects.append([startX, startY, endX, endY])

        if len(valid_rects) == 0:
            return self.get_output_format()

        input_centroids = input_centroids[:len(valid_rects)]
        rects_in = valid_rects

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], rects_in[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                if D[row, col] > self.max_distance: continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.rects[object_id] = rects_in[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], rects_in[col])

        return self.get_output_format()

    def get_output_format(self):
        output = []
        for obj_id, box in self.rects.items():
            x1, y1, x2, y2 = box
            output.append([x1, y1, x2, y2, obj_id])
        return output
