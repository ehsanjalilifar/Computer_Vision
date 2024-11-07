import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon, MultiPolygon
import os, shutil
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict

def plot(masks, boxes, height):
    for mask, box in zip(masks, boxes):
        x = mask[:, 0]
        y = mask[:, 1]
        # Make the boundries close loop
        x = np.append(x, mask[0,0])
        y = np.append(y, mask[0,1])

        # Pixel origins is top left corner of screeen.
        y = height - y

        plt.plot(x,y)
        plt.scatter(box[0], height-box[1], s=10)
    plt.show()

xml_directory = os.path.join(os.getcwd(), 'input_files/XMLs/')

# LANES
tree = ET.parse(f'{xml_directory}/ntta_lanes.xml')
root = tree.getroot()
polygons = root.findall('.//polygon')

poly_dict = dict()

for polygon in polygons:
    label = polygon.get('label')
    id = polygon.find('attribute')
    if id != None:
        label = label+id.text
    points_list = (polygon.get('points')).split(';')
    points = [list(map(float, point.split(','))) for point in points_list]
    _polygon = Polygon(points)
    poly_dict[label] = _polygon

# poly_dict contains the lanes' geometries. Key is the lane name and value is the polygon geometry.

# ORANGE MARKINGS
tree = ET.parse(f'{xml_directory}/orange_markings.xml')
root = tree.getroot()
markings = root.findall('.//polygon')

markings_dict = dict()

for marking in markings:
    label = marking.get('label')
    points_list = (marking.get('points')).split(';')
    points = [list(map(float, point.split(','))) for point in points_list]
    _polygon = Polygon(points)
    markings_dict[label] = _polygon

# markings_dict contains the orange markings' geometries. Key is the marking associated with the lane and value is the polygon geometry.
print(f"*** {len(markings_dict)} markings defined to check the tires hits. ***")
for key, value in markings_dict.items():
    print(f"\n{key}: {value}")

model_name = "yolov8m-seg.pt"
model = YOLO(model_name)   # segmentation model
print(f"\nModel: {model_name}")

model.to('cuda')
video_directory = os.path.join(os.getcwd(), 'input_files/Videos/Stream_1')
source_filename = "20240611_081842_8C7D.mkv"
cap = cv2.VideoCapture(f"{video_directory}/{source_filename}")
print(f"Source file: {source_filename}")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(f"Video Specification:\nwidth: {w}, height:{h}, fps: {fps}")

results_directory = os.path.join(os.getcwd(), f'result_files/{source_filename[:-4]}')
if os.path.exists(results_directory):
    shutil.rmtree(results_directory)
    print(f"Directory '{results_directory}' removed.")

os.makedirs(os.path.join(results_directory, 'annotated'))
os.makedirs(os.path.join(results_directory, 'clean'))
print(f"Directory '{results_directory}' created.")

annotated_dir = os.path.join(results_directory, 'annotated')
clean_dir = os.path.join(results_directory, 'clean')

frame_count = 0
while True:
    # Frame timestamp
    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    # print(f"Frame timestamp: {timestamp_ms} ms")

    hit_detected_in_the_frame = False # If any hit detected, the screenshot with annotation must be saved to a file.

    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=1)

    results = model.track(im0, persist=True, imgsz=(1088, 1920), retina_masks=True)
    # Result length will be >1  if you submit a batch for prediction.

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for mask, track_id, box in zip(masks, track_ids,boxes):
            # Get the zone for each object
            for zone_id, polygon in poly_dict.items():
                # Get the middle point on the lower edge (on bounding box) as the tracking point. The center is buggy with the current camera view point.
                # Remember the coordinate system's origin is top-left of the screen.
                track_point = Point(box[0], box[1] + box[3]/2.0)
                if  polygon.contains(track_point):
                    # print(f"Object with ID={track_id} is in {zone_id}")
                    # Check if the object hits the line (note: each zone has its own line)
                    # We are only detecting the hit to one side of the vehicle.

                    # Now, we know the tracked object is in one of the zones.
                    # Create a geometry out of the detected mask and check whether the tracked object hit the orange marking or not?
                    if (zone_id in ['lane1', 'lane2', 'lane3']):
                        hit_polygon = markings_dict[zone_id].intersection(Polygon(mask))

                        # if(markings_dict[zone_id].intersects(Polygon(mask))):
                        if not (hit_polygon.is_empty):
                            _,_,_,ymax = hit_polygon.bounds # Note that the origin is top-left. Thus, the ymax shows the lowest point on the screen
                            bounding_box_height = box[3]
                            bounding_box_ymax = box[1] + bounding_box_height/2.0
                            ratio = abs(bounding_box_ymax - ymax) / bounding_box_height
                            threshold = 0.15 # This parameter can be adjusted to get more accurate hits. It is the ratio of the hit height to the bounding box height.
                            if(ratio < threshold):
                                print(f"##################################################\n\nA hit detected betwen ID={track_id} and {zone_id}\n\n##################################################")
                                hit_detected_in_the_frame = True

                                # Save the unannotated frame.
                                object_clean_dir = os.path.join(clean_dir, f'objectID_{track_id}')
                                if not os.path.exists(object_clean_dir):
                                    os.makedirs(object_clean_dir)
                                frame_filename = os.path.join(object_clean_dir, f"frame_{frame_count:04d}_objectID_{track_id}.jpg")  # Generates filenames like frame_0001.jpg
                                cv2.imwrite(frame_filename, im0)

                                # Draw the marking for debugging:
                                polygon_points = np.array(list(markings_dict['lane1'].exterior.coords), dtype=np.int32)
                                cv2.polylines(im0, [polygon_points], isClosed=True, color=(51, 51, 51), thickness=2) # illustrated the hitted area.
                                polygon_points = np.array(list(markings_dict['lane2'].exterior.coords), dtype=np.int32)
                                cv2.polylines(im0, [polygon_points], isClosed=True, color=(240, 128, 128), thickness=2) # illustrated the hitted area.

                                # Save the annotated frame.
                                annotator.seg_bbox(mask=mask,
                                    mask_color=colors(track_id, True),
                                    label=f"{zone_id}_id{track_id}",
                                    txt_color=(0,0,0))
                                
                                if isinstance(hit_polygon, Polygon):
                                    polygon_points = np.array(list(hit_polygon.exterior.coords), dtype=np.int32)
                                    cv2.polylines(im0, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2) # illustrated the hitted area.
                                elif isinstance(hit_polygon, MultiPolygon):
                                    for _poly in hit_polygon.geoms:
                                        polygon_points = np.array(list(_poly.exterior.coords), dtype=np.int32)
                                        cv2.polylines(im0, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2) # illustrated the hitted area.

                                object_annotated_dir = os.path.join(annotated_dir, f'objectID_{track_id}')
                                if not os.path.exists(object_annotated_dir):
                                    os.makedirs(object_annotated_dir)
                                frame_filename = os.path.join(object_annotated_dir, f"frame_{frame_count:04d}_objectID_{track_id}_annotated.jpg")  # Generates filenames like frame_0001.jpg
                                cv2.imwrite(frame_filename, im0)
            
    cv2.imshow("instance-segmentation-object-tracking", im0)

    frame_count+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()