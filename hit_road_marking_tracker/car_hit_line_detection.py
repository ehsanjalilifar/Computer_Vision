# print("importing libraries")
import os, shutil, cv2, time, argparse, csv, json
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon, MultiPolygon
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
from datetime import datetime
import pandas as pd
# print("all libraries are imported")


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

def loadPolygon(relative_path, image_name):

    ### LABEL STUDIO INPUT
    file_path = os.path.join(os.getcwd(), relative_path)
    with open(file_path, 'r') as f:
        data = json.load(f)

    poly_dict = dict()
    for record in data:
        _image_name = record['image'].split('/')[-1].split('-')[-1][:-4] # extract name of the image used for the drawings
        if (_image_name == image_name):
            print(f'image name is {_image_name}')
            for annotation in record['label']:
                label = annotation['polygonlabels'][0] # assigned label to the polygon one of ['lane1', 'lane2', 'shoulder1', 'shoulder2']
                # The Label Studio output is the between 0 to 100. So, we scale them to the original heights and width in pixel coordinates.
                widthScaleFactor = annotation['original_width'] / 100.0
                heightScaleFactor = annotation['original_height'] / 100.0
                original_coord_points = [[w * widthScaleFactor, h * heightScaleFactor] for w, h in annotation['points']]
                poly_dict[label] = {
                    'polygon': Polygon(original_coord_points),
                    'coords': np.array(original_coord_points)
                }
            return poly_dict
    print('Error: could not the load polygons!')
    return None


def createROIMask(relative_path, base_frame):
    ### Label Studio Input
    file_path = os.path.join(os.getcwd(), relative_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    cnt = 0
    rois_dict = dict()
    roi_masks_list = []
    for record in data:
        for annotation in record['label']:
            cnt += 1
            label = f"{annotation['polygonlabels'][0]}_{cnt}" # assigned label to the polygon one of ['ROI1', 'ROI2', ...]
            # The Label Studio output is the between 0 to 100. So, we scale them to the original heights and width in pixel coordinates.
            widthScaleFactor = annotation['original_width'] / 100.0
            heightScaleFactor = annotation['original_height'] / 100.0
            original_coord_points = [[w * widthScaleFactor, h * heightScaleFactor] for w, h in annotation['points']]
            roi_vertices = np.array(original_coord_points, dtype=np.int32).reshape(-1, 1, 2)
            roi_mask = np.zeros_like(base_frame, dtype=np.uint8)
            cv2.fillPoly(roi_mask, [roi_vertices], 255) # The mask is a black and white filter for bitwise operation.
            rois_dict[label] = {
                'vertices': roi_vertices,
                'mask': roi_mask
            }
            roi_masks_list.append(roi_mask)

    roi_compostite_mask = roi_masks_list[0]
    for mask in roi_masks_list[1:]:
        roi_compostite_mask = cv2.bitwise_or(roi_compostite_mask, mask)

    return rois_dict, roi_compostite_mask

def displayKeyPoints(frame, key_points, mask):
    frame_with_keypoints = cv2.drawKeypoints(frame, key_points, mask, color=(0, 255, 0), flags=0)
    cv2.imshow("Keypoints in Multiple ROIs", frame_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getHomographyMatrix(orb, ref_gray, ref_kp, ref_des, frame, roi_compostite_mask, smoothed_homography):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect and match key points in the new frame
    kp, des = orb.detectAndCompute(frame_gray, roi_compostite_mask)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(ref_des, des)
    matches = sorted(matches, key=lambda x: x.distance)
    matchedPointsLimit = 200
    good_matches = matches
    if (len(good_matches) > matchedPointsLimit):
        good_matches = matches[:matchedPointsLimit] # Use the X top matches. we can adjust the number as needed.

    # Extract matched key points
    src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # reshapes are necessary.
    dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    ### DEBUG ###
    if args.debug:
        matches_img = cv2.drawMatches(ref_gray, ref_kp, frame_gray, kp, good_matches, None)
        cv2.imshow("Matches", matches_img)

    # Compute homography if enough matches found
    if len(good_matches) >= 10:
        ransacReprojThreshold = 1.0
        homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)

        # Reject unstable homographies
        if homography_matrix is not None:
            det = cv2.determinant(homography_matrix)
            # print(f"--> NOTE: det = {det} <--")
            # if det < 0.9 or det > 1.1: # These threshold are assigned based on the observed determinant per frame
            if det < 0.1 or det > 10: # These threshold are assigned based on the observed determinant per frame
                print("Warning: Skipping frame with unstable homography!!!")
            else:
                # Stabilize the jittery homography
                if smoothed_homography is None:
                    smoothed_homography = homography_matrix
                else:
                    smoothed_homography = alpha * smoothed_homography + (1.0 - alpha) * homography_matrix
    else:
        print("Not enough matches to compute homography")
    
    return smoothed_homography

def adjustPolygons(poly_dict, homography_matrix):
    adjusted_polygons = dict()
    for label, poly in poly_dict.items():
        _coords = cv2.perspectiveTransform(poly['coords'].astype(np.float32).reshape(-1, 1, 2), homography_matrix).reshape(-1, 2)
        adjusted_polygons[label] = {
            'coords': _coords,
            'polygon': Polygon(_coords)
        }
    return adjusted_polygons

def saveFrame(dir, track_id, frame_count, frame):
    dir = os.path.join(dir, f'objectID_{track_id}')
    if not os.path.exists(dir):
        os.makedirs(dir)
    frame_filename = os.path.join(dir, f"frame_{frame_count:04d}_objectID_{track_id}.jpg")  # Generates filenames like frame_0001.jpg
    cv2.imwrite(frame_filename, frame)

def drawPolygonsOnFrame(poly, frame, color):
    if isinstance(poly, Polygon):
        polygon_points = np.array(list(poly.exterior.coords), dtype=np.int32)
        cv2.polylines(frame, [polygon_points], isClosed=True, color=color, thickness=1) # illustrated the hitted area.
    elif isinstance(poly, MultiPolygon):
        for _poly in poly.geoms:
            polygon_points = np.array(list(_poly.exterior.coords), dtype=np.int32)
            cv2.polylines(frame, [polygon_points], isClosed=True, color=color, thickness=1) # illustrated the hitted area.
    return frame

def addToVehicleHistory(history, track_id, zone_id):
    history[track_id][zone_id] =  history[track_id][zone_id] + 1

def isHitAtTheBottom(hitYMax, screenHeight):
    return (hitYMax > 0.93 * screenHeight)

def logProgress(frame_count, total_frames, start_time):
    current_time = datetime.now()
    print(f'\n#### {int(frame_count/total_frames*100)}% COMPLETED ####')
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print(f'Processed {frame_count} frames out of {total_frames} by {current_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"Process time per frame in frame {frame_count}:")

###########################################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation and Tracking with YOLO")
    parser.add_argument("--model", type=str, required=True, help="YOLO model name")
    parser.add_argument("--source", type=str, required=True, help="input video filename")
    parser.add_argument("--lanes", type=str, required=True, help="road's lanes polygon boundaries")
    parser.add_argument("--markings", type=str, required=True, help="road's orange markinging polygon boundaries")
    parser.add_argument("--rois", type=str, required=True, help="regions of interest polygon boundaries")
    parser.add_argument("--zoom", type=str, required=True, help="regions defining the boundaries for object detection")
    parser.add_argument("--skipframes", type=int, required=False, default=0, help="number of frames to be skipped between analyzed frames")
    parser.add_argument("--logstep", type=int, required=False, default=1000, help="log the progress every logstep frames")
    parser.add_argument("--debug", type=bool, required=False, default=False, help="run in debug mode for more info")
    args = parser.parse_args()

    print("------ INPUT FILES -----")
    video_name = args.source[:-4]
    # lanes_dic contains the lanes' geometries. Key is the lane name and value is the polygon geometry.
    # lanes_dict = loadPolygon(f'input_files/XMLs/{args.lanes}.xml', f'frame_zero_{video_name}.png')
    lanes_dict = loadPolygon(f'input_files/JSON/{args.lanes}.json', f'frame_zero_{video_name}')
    print(f"{len(lanes_dict)} polygons defining the lanes were loaded.")

    # markings_dict contains the orange markings' geometries. Key is the marking associated with the lane and value is the polygon geometry.
    # markings_dict = loadPolygon(f'input_files/XMLs/{args.markings}.xml', f'frame_zero_{video_name}.png')
    markings_dict = loadPolygon(f'input_files/JSON/{args.markings}.json', f'frame_zero_{video_name}')
    print(f"{len(markings_dict)} polygons defining the markings were loaded.")

    print("----- MODEL DEFINITION -----")
    model_name = args.model
    model = YOLO(f"model/{model_name}")   # segmentation model
    print(f"Model: {model_name}")
    model.to('cuda')

    # Read the input video.
    video_directory = os.path.join(os.getcwd(), 'input_files/Videos/Stream_2')
    source_filename = args.source
    cap = cv2.VideoCapture(f"{video_directory}/{source_filename}")
    print(f"Source file: {source_filename}")
    screenWidth, screenHeights, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    print(f"Video Specification: width = {screenWidth}, height ={screenHeights}, fps = {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create the necessary directories to save the results.
    results_directory = os.path.join(os.getcwd(), f'result_files/{source_filename[:-4]}')
    print(f"Results will be saved in {results_directory}")
    if os.path.exists(results_directory):
        shutil.rmtree(results_directory)
        print(f"Directory '{results_directory}' removed.")

    os.makedirs(os.path.join(results_directory, 'annotated'))
    os.makedirs(os.path.join(results_directory, 'clean'))
    os.makedirs(os.path.join(results_directory, 'summary'))
    print(f"Directory '{results_directory}' created.")

    annotated_dir = os.path.join(results_directory, 'annotated')
    clean_dir = os.path.join(results_directory, 'clean')

    # Read the first frame and detect key points. Key points are matched between frames to estimate the camera movements.
    ret, ref_frame = cap.read()
    # ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY) # Read from the first frame
    ref_path = os.path.join(os.getcwd(), f'input_files/Photos/frame_zero_{video_name}.png')
    ref_gray = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    # ROIs
    # Region of Interests (ROIs) are defined to limit the area used for key point detection. 
    # Only the static part of the frame which won't be occluded should be used for key point detection.
    # Create the composite mask which contains of a couple of polygons.
    rois_dict, roi_compostite_mask  = createROIMask(f'input_files/JSON/{args.rois}.json', ref_gray)
    # Define the the boundaries of object detection in the screen.
    zoom_dict, zoom_compostite_mask = createROIMask(f'input_files/JSON/{args.zoom}.json', ref_gray)
    zoom_compostite_mask = cv2.cvtColor(zoom_compostite_mask, cv2.COLOR_GRAY2BGR) # It has to be applied in all 3 channels.
    # Initialize ORB detector. 
    # This algorithm is used to estimate the camera movements (translation, rotation, and scale) to stabilize the camera movements.
    # Camera movements cause the polygons location instability. The lane and marking polygons location must always match with their physical location on the road.
    # orb = cv2.ORB_create(scaleFactor=1.8)
    orb = cv2.SIFT_create()

    # Find the key points and descriptors in the reference frame (frame_zero)
    ref_kp, ref_des = orb.detectAndCompute(ref_gray, roi_compostite_mask)

    ### DEBUG ###
    if args.debug:
        displayKeyPoints(ref_gray, ref_kp, roi_compostite_mask)

    smoothed_homography = None
    alpha = 0.8 # Smoothing factor for jittery frame stabilization (higher = more smoothing)
    vehicles_history = defaultdict(lambda: {
        'shoulder1': 0,
        'shoulder2': 0,
        'lane1': 0,
        'lane2': 0,
        # 'lane3': 0
    }) # Stores the number of frames each vehicle was in while passing the recorded road section.
    # It's a dictionaty with object/vehicle ID as key. The value itself is a dictionary with names of the lanes as key and the number of the frames that the specific vehicle was present in a specific lane as value.

    # Stores ID's of the vehicles that entered to each of the specific zone (keys)
    # In each frame ID's of the vehicle inside the zone will be added to the zone. We use Set datatype to prevent duplication.
    # If a vehicle ID is present in two sets, it indicates a "lane changing" event.
    lanes_history = {
        'shoulder1': set(),
        'shoulder2': set(),
        'lane1': set(),
        'lane2': set()
    }
    hit_set = set()
    last_reported_frame = -1
    frame_count = 0
    error_counter = 0
    start_time = time.time()
    while True:
        hit_detected_in_the_frame = False # If any hit detected, the screenshot with annotation must be saved to a file.

        ret, curr_frame = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        if(frame_count % (args.skipframes + 1) == 0): # skip frames to speed-up. It comes at the cost of reduced accuracy (possibly missing a barely hit situation).
            # Adjust the polygons locations w.r.t the camera movements.
            smoothed_homography = getHomographyMatrix(orb, ref_gray, ref_kp, ref_des, curr_frame, roi_compostite_mask, smoothed_homography)
            adjusted_lanes = adjustPolygons(lanes_dict, smoothed_homography)
            adjusted_markings = adjustPolygons(markings_dict, smoothed_homography)
            
            # Mask the input frame to limit the detected vehicle to the area of interest. 
            # It has to be after adjusting the lanes to make sure there is enough features in the screen to estimate the camera movements.
            zoomed_frame = cv2.bitwise_and(curr_frame, zoom_compostite_mask)


            if (frame_count // args.logstep > last_reported_frame // args.logstep):
                last_reported_frame += args.logstep # Print a log every 1000 frames
                logProgress(frame_count, total_frames, start_time)
                results = model.track(zoomed_frame, persist=True, retina_masks=True, verbose=True)
            else:
                results = model.track(zoomed_frame, persist=True, retina_masks=True, verbose=False)
            # Result length will be >1  if you submit a batch for prediction.

            ### DEBUG ###
            if args.debug:
                # print(f"MARKINGS --> {adjusted_markings['lane1']['polygon']}")
                curr_frame = drawPolygonsOnFrame(adjusted_markings['lane1']['polygon'], curr_frame, (255, 0, 0)) 
                curr_frame = drawPolygonsOnFrame(adjusted_markings['lane2']['polygon'], curr_frame, (255, 0, 0)) 
                # label the hitted object.
                annotator = Annotator(curr_frame, line_width=2)
            ### DEBUG ###

            if results[0].boxes.id is not None and results[0].masks is not None:
                masks = results[0].masks.xy
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for mask, track_id, box in zip(masks, track_ids,boxes):
                    # Get the zone for each object
                    for zone_id, lane in adjusted_lanes.items():
                        polygon = lane['polygon']
                        # Get the middle point on the lower edge (on bounding box) as the tracking point. The center is buggy with the current camera view point.
                        # Remember the coordinate system's origin is top-left of the screen.
                        track_point = Point(box[0], box[1] + box[3]/2.0) # track the vehicles at the bottom-middle of the bounding box for higher eaccuracy.
                        if  polygon.contains(track_point):
                            lanes_history[zone_id].add(track_id) # Add the vehicle ID to the history of the lane
                            addToVehicleHistory(vehicles_history, track_id, zone_id)
                            # Check if the object hits the line (note: each zone has its own line)
                            # We are only detecting the hit to one side of the vehicle.
                            # Now, we know the tracked object is in one of the zones.
                            # Create a geometry out of the detected mask and check whether the tracked object hit the orange marking or not?
                            if (zone_id in ['lane1', 'lane2']):
                                try:
                                    hit_polygon = adjusted_markings[zone_id]['polygon'].intersection(Polygon(mask))
                                    if not (hit_polygon.is_empty): # A hit is detected
                                        # We define a true hit as hit that occurred in the lower 0.15 of the bounding box heigt. This lowers the falsely detected hits (e.g, top left of the vehicle hits the line. We only check the tires.)
                                        _,_,_,ymax = hit_polygon.bounds # Note that the origin is top-left. Thus, the ymax shows the lowest point on the screen
                                        bounding_box_height = box[3]
                                        bounding_box_ymax = box[1] + bounding_box_height/2.0
                                        ratio = abs(bounding_box_ymax - ymax) / bounding_box_height
                                        threshold = 0.15 # This parameter can be adjusted to get more accurate hits. It is the ratio of the hit's height to the bounding box's height.
                                        # Exclude partially detected vehicles. When a vehicle is on edges of the screen, it is partially visible to the camera and the segmentation does not represent the whole vehicle. Additionally, the view is occluded.
                                        if((ratio < threshold) & (not isHitAtTheBottom(ymax, screenHeights))):
                                            if track_id not in hit_set: # Only print the the first hit of the vehicle
                                                print(f"\nVehicle with Object_id {track_id} hitted {zone_id} in frame {frame_count}!")
                                            hit_detected_in_the_frame = True
                                            hit_set.add(track_id)
                                            # Save the unannotated frame.
                                            saveFrame(clean_dir, track_id, frame_count, curr_frame)
                                            # Draw the marking for debugging.
                                            curr_frame = drawPolygonsOnFrame(adjusted_markings[zone_id]['polygon'], curr_frame, (51, 51, 51)) 
                                            # label the hitted object.
                                            annotator = Annotator(curr_frame, line_width=1)
                                            annotator.seg_bbox(mask=mask,
                                                mask_color=colors(track_id, True),
                                                label=f"{zone_id}_id{track_id}",
                                                txt_color=(0,0,0))
                                            # Highlight the hitted area.
                                            curr_frame = drawPolygonsOnFrame(hit_polygon, curr_frame, (255, 0, 0))
                                            # Save the annotated frame.
                                            saveFrame(annotated_dir, track_id, frame_count, curr_frame)
                                except Exception as e:
                                    error_counter += 1
                                    print(f"Error: {e}")
                    
            if args.debug:
                cv2.imshow("instance-segmentation-object-tracking", curr_frame)
        frame_count+=1
        if args.debug:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if args.debug:
        cv2.destroyAllWindows()
    end_time = time.time()
    hours, remainder = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print()
    print(f"Analysis succesfully completed.")
    print(f"{error_counter} exception occurred during analysis!")
    print(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    history_df = pd.DataFrame.from_dict(vehicles_history, orient='index', columns=['lane1', 'lane2'])
    history_df.to_csv('lane_changes_history.csv')

    # Traffic Volume and Lane Changing Events
    print("\n### Lane Changes Summary ###")
    lane_change_file_path = os.path.join(os.getcwd(), f'result_files/{source_filename[:-4]}/summary/lane_changes.csv')
    pairs = [['shoulder1', 'lane1'], ['lane1', 'lane2'], ['lane2', 'shoulder2']]
    sum_lane_changes = 0
    with open(lane_change_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['lane #1', 'lane #2', 'number_of_lane_changes'])
        for pair in pairs:
            lane1_history = lanes_history[pair[0]]
            lane2_history = lanes_history[pair[1]]
            n_lane_changes = len(lane1_history & lane2_history)
            sum_lane_changes += n_lane_changes
            print(f"Number of lane changes between {pair[0]} and {pair[1]} = {n_lane_changes}")
            writer.writerow([pair[0], pair[1], n_lane_changes])

    print("\n### Traffic Volume ### ")
    volume_file_path = os.path.join(os.getcwd(), f'result_files/{source_filename[:-4]}/summary/traffic_volumes.csv')
    total_set = set()
    with open(volume_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['lane #1', 'vehicle_counts'])
        for lane, vehicles_set in lanes_history.items():
            print(f"Vehicles count in {lane} = {len(vehicles_set)}")
            writer.writerow([lane, len(vehicles_set)])
            total_set = total_set | vehicles_set
    
        print(f"Total road volume = {len(total_set)}")
        writer.writerow(['All', len(total_set)])
        writer.writerow(['TOTAL HITS', len(hit_set)])
        writer.writerow(['TOTAL LANE CHANGES', sum_lane_changes])
        writer.writerow(['TOTAL HITS EXCLUDING LANE CHANGES', len(hit_set) - sum_lane_changes])

    print(f'\nTOTAL NUMBER OF HITS = {len(hit_set)}')
    print(f'TOTAL NUMBER OF LANE CHANGES = {sum_lane_changes}')
    print(f'TOTAL NUMBER OF HITS EXCLUDING LANE CHANGES = {len(hit_set) - sum_lane_changes}')