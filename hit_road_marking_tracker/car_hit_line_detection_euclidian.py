# print("importing libraries")
import os, shutil, cv2, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon, MultiPolygon
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
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

def loadPolygon(relative_path):
    xml_directory = os.path.join(os.getcwd(), relative_path)
    tree = ET.parse(f'{xml_directory}')
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
        _polygon = Polygon(points) # For geometry operations.
        _coords = [tuple(map(float, point.split(','))) for point in points_list]
        _coords = np.array([_coords], dtype=np.int32) # For frame/pixel operations.
        poly_dict[label] = {
            'polygon': _polygon,
            'coords': _coords
        }
    return poly_dict

def createROIMask(relative_path, base_frame):
    file_path = os.path.join(os.getcwd(), relative_path)
    tree = ET.parse(f'{file_path}')
    root = tree.getroot()
    rois = root.findall('.//polygon') # Regions Of Interest are locations in the image where we are looking for key points to adjust the camera rotation/shift.
    
    rois_dict = dict()
    roi_masks_list = []
    cnt = 0

    for polygon in rois:
        cnt += 1 # An enumerator to distinguish between different polygons. Can also be addressed in the input file.
        label = f"{polygon.get('label')}_{cnt}"
        # Re-format the points coordinate to create an image mask. The mask confines the location of the key points.
        points_list = (polygon.get('points')).split(';')
        points = [tuple(map(float, point.split(','))) for point in points_list]
        roi_vertices = np.array([points], dtype=np.int32)
        roi_mask = np.zeros_like(base_frame)
        cv2.fillPoly(roi_mask, roi_vertices, 255) # The mask is a black and white filter for bitwise operation.
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
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
            if det < 0.9 or det > 1.1: # These threshold are assigned based on the observed determinant per frame
                print("ALARM: Skipping frame with unstable homography!!!")
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
        cv2.polylines(frame, [polygon_points], isClosed=True, color=color, thickness=2) # illustrated the hitted area.
    elif isinstance(poly, MultiPolygon):
        for _poly in poly.geoms:
            polygon_points = np.array(list(_poly.exterior.coords), dtype=np.int32)
            cv2.polylines(frame, [polygon_points], isClosed=True, color=color, thickness=2) # illustrated the hitted area.
    return frame

def addToLaneHistory(history, track_id, zone_id):
    history[track_id][zone_id] =  history[track_id][zone_id] + 1

def isHitAtTheBottom(hitYMax, screenHeight):
    return (hitYMax > 0.93 * screenHeight)

def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  # Define the filter
  f = np.ones(window_size)/window_size
  # Add padding to the boundaries
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  # Apply convolution
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  # Remove padding
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed

def smooth(trajectory):
  smoothed_trajectory = np.copy(trajectory)
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=2)
 
  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

###########################################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation and Tracking with YOLO")
    parser.add_argument("--model", type=str, required=True, help="YOLO model name")
    parser.add_argument("--source", type=str, required=True, help="input video filename")
    args = parser.parse_args()

    print("------ INPUT FILES -----")

    # lanes_dic contains the lanes' geometries. Key is the lane name and value is the polygon geometry.
    lanes_dict = loadPolygon('input_files/XMLs/ntta_lanes.xml')
    print(f"{len(lanes_dict)} polygons defining the lanes were loaded.")

    # markings_dict contains the orange markings' geometries. Key is the marking associated with the lane and value is the polygon geometry.
    markings_dict = loadPolygon('input_files/XMLs/orange_markings.xml')
    print(f"{len(markings_dict)} polygons defining the markings were loaded.")

    print("----- MODEL DEFINITION -----")
    model_name = args.model
    model = YOLO(f"model/{model_name}")   # segmentation model
    print(f"Model: {model_name}")
    model.to('cuda')

    # Read the input video.
    video_directory = os.path.join(os.getcwd(), 'input_files/Videos/Stream_1')
    source_filename = args.source
    cap = cv2.VideoCapture(f"{video_directory}/{source_filename}")
    # Get frame count
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Source file: {source_filename}")
    screenWidth, screenHeights, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    print(f"Video Specification: width = {screenWidth}, height ={screenHeights}, fps = {fps}")

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
    # Set up output video
    out = cv2.VideoWriter(f'video_out_{source_filename}.mp4', fourcc, fps, (screenWidth, screenHeights)) 

    # Create the necessary directories to save the results.
    results_directory = os.path.join(os.getcwd(), f'result_files/{source_filename[:-4]}')
    print(f"Results will be saved in {results_directory}")
    if os.path.exists(results_directory):
        shutil.rmtree(results_directory)
        print(f"Directory '{results_directory}' removed.")

    os.makedirs(os.path.join(results_directory, 'annotated'))
    os.makedirs(os.path.join(results_directory, 'clean'))
    print(f"Directory '{results_directory}' created.")

    annotated_dir = os.path.join(results_directory, 'annotated')
    clean_dir = os.path.join(results_directory, 'clean')

    # Read the first frame and detect key points. Key points are matched between frames to estimate the camera movements.
    # ret, ref_frame = cap.read()
    _, prev = cap.read() 
    # ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY) # Read from the first frame
    # ref_path = os.path.join(os.getcwd(), f'input_files/Photos/frame_zero_{source_filename[:-4]}.png')
    # ref_gray = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # ROIs
    # Region of Interests (ROIs) are defined to limit the area used for key point detection. 
    # Only the static part of the frame which won't be occluded should be used for key point detection.
    # Create the composite mask which contains of a couple of polygons.
    rois_dict, roi_compostite_mask = createROIMask('input_files/XMLs/ROIs.xml', prev_gray)

    # Initialize ORB detector. 
    # This algorithm is used to estimate the camera movements (translation, rotation, and scale) to stabilize the camera movements.
    # Camera movements cause the polygons location instability. The lane and marking polygons location must always match with their physical location on the road.
    # orb = cv2.ORB_create()

    # Find the key points and descriptors in the reference frame (frame_zero)
    # ref_kp, ref_des = orb.detectAndCompute(ref_gray, roi_compostite_mask)
    # displayKeyPoints(ref_gray, ref_kp, roi_compostite_mask)

    # smoothed_homography = None
    # alpha = 0.8 # Smoothing factor for jittery frame stabilization (higher = more smoothing)
    lane_history = defaultdict(lambda: {
        'shoulder1': 0,
        'shoulder2': 0,
        'lane1': 0,
        'lane2': 0,
        'lane3': 0
    }) # Stores the number of frames each vehicle was in while passing the recorded road section.
    frame_count = 0
    error_counter = 0
    start_time = time.time()

    transforms = np.zeros((n_frames-1, 3), np.float32)

    for i in range(n_frames-2):
        # Detect feature points in previous frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                            maxCorners=200,
                                            qualityLevel=0.01,
                                            minDistance=30,
                                            blockSize=3,
                                            mask=roi_compostite_mask)
        
        # Read next frame
        success, curr = cap.read()
        if not success:
            break
        
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
        
        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
        
        # Sanity check
        assert prev_pts.shape == curr_pts.shape 
        
        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        #Find transformation matrix
        # m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
        m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
        
        # Extract traslation
        dx = m[0,2]
        dy = m[1,2]
        
        # Extract rotation angle
        da = np.arctan2(m[1,0], m[0,0])
        
        # Store transformation
        transforms[i] = [dx,dy,da]
        
        # Move to next frame
        prev_gray = curr_gray
        
        print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    	
# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth(trajectory)

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
 
# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
 
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
  # Read next frame
  success, frame = cap.read()
  if not success:
    break
 
  # Extract transformations from the new transformation array
  dx = trajectory[i,0]
  dy = trajectory[i,1]
  da = trajectory[i,2]

  dx =-dx
  dy = -dy
  da = -da
 
  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy

  print(m)
 
  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (screenWidth,screenHeights))
 
  # Fix border artifacts
  frame_stabilized = fixBorder(frame_stabilized) 

  frame_stabilized = drawPolygonsOnFrame(markings_dict['lane1']['polygon'], frame_stabilized, (255, 0, 0)) 
  frame_stabilized = drawPolygonsOnFrame(markings_dict['lane2']['polygon'], frame_stabilized, (255, 0, 0))

  frame = drawPolygonsOnFrame(markings_dict['lane1']['polygon'], frame, (255, 0, 0)) 
  frame = drawPolygonsOnFrame(markings_dict['lane2']['polygon'], frame, (255, 0, 0))
 
  # Write the frame to the file
  frame_out = cv2.hconcat([frame, frame_stabilized])
 
  # If the image is too big, resize it.
#   if(frame_out.shape[1] &gt; 1920):
#     frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2))
 
  cv2.imshow("Before and After", frame_out)
  cv2.waitKey(10)
  out.write(frame_out)



    # while True:
    #     hit_detected_in_the_frame = False # If any hit detected, the screenshot with annotation must be saved to a file.

    #     ret, curr_frame = cap.read()
    #     if not ret:
    #         print("Video frame is empty or video processing has been successfully completed.")
    #         break
    #     if(frame_count % 1 == 0): # skip frames to speed-up. It comes at the cost of reduced accuracy (possibly missing a barely hit situation).
    #         # Adjust the polygons locations w.r.t the camera movements.
    #         smoothed_homography = getHomographyMatrix(orb, ref_gray, ref_kp, ref_des, curr_frame, roi_compostite_mask, smoothed_homography)
    #         adjusted_lanes = adjustPolygons(lanes_dict, smoothed_homography)
    #         adjusted_markings = adjustPolygons(markings_dict, smoothed_homography)

    #         results = model.track(curr_frame, persist=True, imgsz=(1088, 1920), retina_masks=True)
    #         # Result length will be >1  if you submit a batch for prediction.

    #         ### DEBUG ###
    #         curr_frame = drawPolygonsOnFrame(adjusted_markings['lane1']['polygon'], curr_frame, (255, 0, 0)) 
    #         curr_frame = drawPolygonsOnFrame(adjusted_markings['lane2']['polygon'], curr_frame, (255, 0, 0)) 
    #         # label the hitted object.
    #         annotator = Annotator(curr_frame, line_width=1)
    #         ### DEBUG ###

    #         if results[0].boxes.id is not None and results[0].masks is not None:
    #             masks = results[0].masks.xy
    #             boxes = results[0].boxes.xywh.cpu().numpy()
    #             track_ids = results[0].boxes.id.int().cpu().tolist()
                
    #             for mask, track_id, box in zip(masks, track_ids,boxes):
    #                 # Get the zone for each object
    #                 for zone_id, lane in adjusted_lanes.items():
    #                     polygon = lane['polygon']
    #                     # Get the middle point on the lower edge (on bounding box) as the tracking point. The center is buggy with the current camera view point.
    #                     # Remember the coordinate system's origin is top-left of the screen.
    #                     track_point = Point(box[0], box[1] + box[3]/2.0)
    #                     if  polygon.contains(track_point):
    #                         addToLaneHistory(lane_history, track_id, zone_id)
    #                         # print(f"Object with ID={track_id} is in {zone_id}")
    #                         # Check if the object hits the line (note: each zone has its own line)
    #                         # We are only detecting the hit to one side of the vehicle.
    #                         # Now, we know the tracked object is in one of the zones.
    #                         # Create a geometry out of the detected mask and check whether the tracked object hit the orange marking or not?
    #                         if (zone_id in ['lane1', 'lane2', 'lane3']):
    #                             try:
    #                                 hit_polygon = adjusted_markings[zone_id]['polygon'].intersection(Polygon(mask))
    #                                 if not (hit_polygon.is_empty): # A hit is detected
    #                                     # We define a true hit as hit that occurred in the lower 0.15 of the bounding box heigt. This lowers the falsely detected hits (e.g, top left of the vehicle hits the line. We only check the tires.)
    #                                     _,_,_,ymax = hit_polygon.bounds # Note that the origin is top-left. Thus, the ymax shows the lowest point on the screen
    #                                     bounding_box_height = box[3]
    #                                     bounding_box_ymax = box[1] + bounding_box_height/2.0
    #                                     ratio = abs(bounding_box_ymax - ymax) / bounding_box_height
    #                                     threshold = 0.15 # This parameter can be adjusted to get more accurate hits. It is the ratio of the hit's height to the bounding box's height.
    #                                     # Exclude partially detected vehicles. When a vehicle is on edges of the screen, it is partially visible to the camera and the segmentation does not represent the whole vehicle. Additionally, the view is occluded.
    #                                     if((ratio < threshold) & (not isHitAtTheBottom(ymax, screenHeights))):
    #                                         print(f"##################################################\n\nA hit detected between ID={track_id} and {zone_id}\n\n##################################################")
    #                                         hit_detected_in_the_frame = True
    #                                         # Save the unannotated frame.
    #                                         saveFrame(clean_dir, track_id, frame_count, curr_frame)
    #                                         # Draw the marking for debugging.
    #                                         curr_frame = drawPolygonsOnFrame(adjusted_markings[zone_id]['polygon'], curr_frame, (51, 51, 51)) 
    #                                         # label the hitted object.
    #                                         annotator = Annotator(curr_frame, line_width=1)
    #                                         annotator.seg_bbox(mask=mask,
    #                                             mask_color=colors(track_id, True),
    #                                             label=f"{zone_id}_id{track_id}",
    #                                             txt_color=(0,0,0))
    #                                         # Highlight the hitted area.
    #                                         curr_frame = drawPolygonsOnFrame(hit_polygon, curr_frame, (255, 0, 0))
    #                                         # Save the annotated frame.
    #                                         saveFrame(annotated_dir, track_id, frame_count, curr_frame)
    #                             except Exception as e:
    #                                 error_counter += 1
    #                                 print(f"Error: {e}")
                    
    #         cv2.imshow("instance-segmentation-object-tracking", curr_frame)
    #         # if frame_count == 10:
    #         #     break
    #     frame_count+=1

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
    # end_time = time.time()
    # hours, remainder = divmod(end_time - start_time, 3600)
    # minutes, seconds = divmod(remainder, 60)
    # print(f"Analysis succesfully completed.")
    # print(f"{error_counter} exception occurred during analysis!")
    # print(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    # history_df = pd.DataFrame.from_dict(lane_history, orient='index', columns=['lane1', 'lane2'])
    # history_df.to_csv('lane_changes_history.csv')
    

