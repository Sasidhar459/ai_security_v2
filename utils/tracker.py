import time
import math


class FaceTracker:
    def __init__(self, timeout=10, distance_threshold=80):
        self.timeout = timeout
        self.distance_threshold = distance_threshold

        self.tracked = {}  # id -> face data
        self.next_id = 0

    # ----------------------------
    # Calculate center of bbox
    # ----------------------------
    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    # ----------------------------
    # Distance between centers
    # ----------------------------
    def _distance(self, b1, b2):
        c1 = self._center(b1)
        c2 = self._center(b2)
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    # ----------------------------
    # MAIN UPDATE METHOD
    # ----------------------------
    def update_with_ids(self, detections):

        now = time.time()
        events = []
        updated_ids = set()

        tracked_faces_output = []

        # ----------------------------
        # Match detections with tracked faces
        # ----------------------------
        for det in detections:

            bbox = det["bbox"]
            name = det["name"]

            matched_id = None

            for face_id, data in self.tracked.items():
                dist = self._distance(bbox, data["bbox"])

                if dist < self.distance_threshold:
                    matched_id = face_id
                    break

            if matched_id is None:
                # New face
                face_id = self.next_id
                self.next_id += 1

                self.tracked[face_id] = {
                    "bbox": bbox,
                    "name": name,
                    "last_seen": now,
                    "counted": False
                }

                if name != "Unknown":
                    events.append(("ARRIVAL", name))
                    self.tracked[face_id]["counted"] = True

            else:
                # Existing face
                self.tracked[matched_id]["bbox"] = bbox
                self.tracked[matched_id]["last_seen"] = now

                if name != "Unknown" and not self.tracked[matched_id]["counted"]:
                    events.append(("ARRIVAL", name))
                    self.tracked[matched_id]["counted"] = True

                self.tracked[matched_id]["name"] = name

                face_id = matched_id

            updated_ids.add(face_id)

        # ----------------------------
        # Remove old faces (Departure)
        # ----------------------------
        remove_ids = []

        for face_id, data in self.tracked.items():
            if face_id not in updated_ids:
                if now - data["last_seen"] > self.timeout:

                    if data["name"] != "Unknown":
                        events.append(("DEPARTURE", data["name"]))

                    remove_ids.append(face_id)

        for face_id in remove_ids:
            del self.tracked[face_id]

        # ----------------------------
        # Prepare output
        # ----------------------------
        for face_id, data in self.tracked.items():
            tracked_faces_output.append({
                "id": face_id,
                "bbox": data["bbox"],
                "name": data["name"]
            })

        return events, tracked_faces_output