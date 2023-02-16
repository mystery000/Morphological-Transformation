
import cv2
import os
import numpy as np
import glob
import imutils
import datetime
from random import randint
import csv


# Build files list
cwd = os.getcwd()
file_list = glob.glob(cwd + "/images_set_test/*.png")

# Set firts run to True, it will be used to get pictures dimension and initialise empty arrays with the pictures size
first_run = True
ground_clutter_counter = 0

# Initialize value for erosion (remove noise) and dilatation (merge objects)
KEROD= np.ones((3,3),np.uint8)
KDETECT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
KCLUTTERE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
KCLUTTERD = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
RAIN_THICKNESS = 20
KRAINE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(RAIN_THICKNESS,RAIN_THICKNESS))
KRAIND = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
X_CENTER = 485
Y_CENTER = 535
RADIUS = 508
ANGLE = 50
SCOPE_MARGINE = 100
SCOPE_BUFFER = 10
GROUND_CLUTTER_DURATION = 20
TRAJECTORY_MAX_LENGHT = 1000
DOT_LENGHT = 5
TRAIL_FADE = 5


initial_date_str = '2020-04-01_00h00m00s.png'
initial_date = datetime.datetime.strptime(initial_date_str, "%Y-%m-%d_%Hh%Mm%Ss.png")
trajectories = np.zeros((1,TRAJECTORY_MAX_LENGHT),dtype = np.float32)
trajectories_to_record = np.zeros((1,TRAJECTORY_MAX_LENGHT),dtype = np.float32)
nb_trajectories = 0
counter = 0

DETECTION_CONTRAST = 5 # If a pixel as a value > DETECTION CONTRAST + stacked image, it wil be detected
FADE_FACTOR = 0.98 # multiplication cator applied to the stacked image in each loop
CSV_FILE_NAME = 'csv_trajectories.csv'

def random_color() : 
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    rand_color = (r, g, b)
    return rand_color

# Initilize several arrays with the same shape than the pictures to analyse
def initialize(img):
    height, width = img.shape
    black_img = np.zeros((height,width), dtype = np.float64) # black in the RGB uint8 space
    display = np.zeros((height,width,3), dtype = np.float64)
    first_run = False
    return height, width,first_run,black_img,display

def read_timestamp(file_path):
    _,file_name = os.path.split(file_path)
    raw_timestamp = datetime.datetime.strptime(file_name, "%Y-%m-%d_%Hh%Mm%Ss.png")
    raws_seconds = raw_timestamp - initial_date
    print('raw_timestamp', raw_timestamp)
    seconds_timestamp = raws_seconds.total_seconds()
    print('second_timestamp',seconds_timestamp)
    return seconds_timestamp


def draw_ground_clutter(img, ground_clutter,ground_clutter_counter):
    ground_clutter = np.where(img > 175,255,ground_clutter)
    ground_clutter_counter += 1
    if ground_clutter_counter == GROUND_CLUTTER_DURATION:
        ground_clutter = cv2.erode(ground_clutter,KCLUTTERE,iterations = 1)
        ground_clutter = cv2.dilate(ground_clutter,KCLUTTERD,iterations = 1)
    cv2.imshow('clutter',ground_clutter.astype(np.uint8))
    return ground_clutter,ground_clutter_counter


def draw_rain_clutter(ground_clutter,img,black_img,rain_clutter):
    raw_rain = np.where((ground_clutter<255) & (img>20),255,black_img)# Stack, produce an np.float64 even with an img 
    raw_rain = cv2.erode(raw_rain.astype(np.uint8),KRAINE,iterations = 1)
    raw_rain = cv2.dilate(raw_rain.astype(np.uint8),KRAIND,iterations = 1)
    rain_clutter = np.where(raw_rain>rain_clutter, 255,rain_clutter)
    rain_clutter = rain_clutter * FADE_FACTOR
    return rain_clutter


def detect(ground_clutter,rain_clutter,img,stacked,trails,seconds_timestamp):
    raw_detection = np.where((ground_clutter<255) & (rain_clutter<200) & (img>stacked+DETECTION_CONTRAST),255,black_img)
    eroded_detection = cv2.erode(raw_detection.astype(np.uint8),KEROD,iterations = 1)
    dilated_detection = cv2.dilate(eroded_detection,KDETECT,iterations = 1)
    contours, hierarchy = cv2.findContours(dilated_detection, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    nb_dots = len(contours)
    dot_list = np.zeros((nb_dots,TRAJECTORY_MAX_LENGHT),dtype = np.float32) # TRAJECTORY_MAX_LENGHT to allow later vstack with trajectories

    for j in range (0,nb_dots): 
        cnt = contours[j]
        area = cv2.contourArea(cnt)+1
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        status = 0

        box_x_center = int((box[0,0]+box[1,0]+box[2,0]+box[3,0])/4.0)
        box_y_center = int((box[0,1]+box[1,1]+box[2,1]+box[3,1])/4.0)
        
        dot_list[j,0] = seconds_timestamp
        dot_list[j,1] = box_x_center
        dot_list[j,2] = box_y_center
        dot_list[j,3] = area
        dot_list[j,4] = status
        trails = cv2.circle(trails,(box_x_center,box_y_center),int(np.sqrt(area)),(255,255,255),2 )

    return raw_detection,dilated_detection,contours,trails,dot_list,nb_dots


def clean_trajectories(trajectories,seconds_timestamp,trajectories_to_record):
    
    # Record old trajectories
    # to be improve to avoid to record short trajectories (15 ?)
    dead_long_trajectories = trajectories[(seconds_timestamp-trajectories[:, 0] >= 20) & (trajectories[:, 15] !=0)] # keep only the elements matching the condition high deltat time and NOT 0
    trajectories_to_record = np.vstack([trajectories_to_record,dead_long_trajectories])

    # Erase old trajectories
    trajectories = trajectories[seconds_timestamp-trajectories[:, 0] < 20] 
    nb_trajectories = (trajectories[:, 0] > 0).sum() # count the number of not empty raws - Replace by len ?

    return trajectories, nb_trajectories,trajectories_to_record


def control(raw_detection,dilated_detection,stacked_raw_detection,stacked_dilated_detection) : # Visual control
    stacked_raw_detection = np.where(raw_detection>stacked_raw_detection,255,stacked_raw_detection)
    stacked_raw_detection = stacked_raw_detection*FADE_FACTOR
    stacked_dilated_detection = np.where(dilated_detection>stacked_dilated_detection,255,stacked_dilated_detection)
    stacked_dilated_detection = stacked_dilated_detection*FADE_FACTOR
    return stacked_raw_detection,stacked_dilated_detection


def draw_display(ground_clutter,rain_clutter,contours,croped,trails):

    gray_croped = cv2.cvtColor(croped,cv2.COLOR_BGR2GRAY)

    inverted_gray_croped = 255 - gray_croped
    color_inverted_cropped = cv2.cvtColor(inverted_gray_croped.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    
    color_rain_clutter = cv2.cvtColor(rain_clutter.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    color_ground_clutter = cv2.cvtColor(ground_clutter.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    
    temp_color_rain_clutter = np.where(color_rain_clutter>200,(200,100,0),color_inverted_cropped)
    temp_color_rain_clutter_uint8 = temp_color_rain_clutter.astype(np.uint8)
    
    temp_color_ground_clutter = np.where(color_ground_clutter>200,(0,100,255),color_inverted_cropped)
    temp_color_ground_clutter_uint8 = temp_color_ground_clutter.astype(np.uint8)

    temp_out = cv2.addWeighted(temp_color_rain_clutter_uint8, 0.5, color_inverted_cropped, 1-0.5,0.0)
    temp_out = cv2.addWeighted(temp_color_ground_clutter_uint8, 0.5, temp_out, 1-0.5,0.0)
    
    red = np.full((trails.shape), 255,dtype = np.float64)
    red[:,:,0] = 100
    red[:,:,1] = 50
    red[:,:,2] = 50
    # Convert uint8 to float
    red = red.astype(float)
    temp_out = temp_out.astype(float)
     
    # Normalize the alpha mask to keep intensity between 0 and 1
    trails = trails.astype(float)/255
     
    # Multiply the foreground with the alpha matte
    red = cv2.multiply(trails, red)
     
    # Multiply the background with ( 1 - alpha )
    temp_out = cv2.multiply(1.0 - trails, temp_out)
     
    # Add the masked foreground and background.
    out = cv2.add(red, temp_out)
         
    out = np.where(scope>0,out,(255,255,255))
    cv2.imshow('EDGE',out.astype(np.uint8))

    return out


def convert_raster_to_wgs84_coordinates(x,y,WGS84_DATA):
    x_coord = 0
    y_coord = 0
    return x_coord,y_coord
 
   
def regenerate_timestamp_from_seconds(seconds):
    timestamp = 0
    return timestamp

'''
def create_exmpty_csv(CSV_FILE_NAME,headers):
    
    with open(CSV_FILE_NAME, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(headers)

    
def write_csv_data(CSV_FILE_NAME,trajectories_to_record):
    
    if not os.path.isfile(CSV_FILE_NAME):
        create_exmpty_csv(CSV_FILE_NAME)
        
    else : 
        np.savetxt(CSV_FILE_NAME, trajectories_to_record, delimiter=",")
        f=open(CSV_FILE_NAME,'ab')  
        np.savetxt(f,trajectories_to_record)
        f.close()
'''       
def convert_raster_to_wgs84_coordinates(x,y,WGS84_DATA):
    x_coord = 0
    y_coord = 0
    return x_coord,y_coord
    
def regenerate_timestamp_from_seconds(seconds):
    timestamp = 0
    return timestamp


for file in file_list :
    
    seconds_timestamp = read_timestamp(file)
    # Read a new picture 
    color_img = cv2.imread(file)
    rotated = imutils.rotate_bound(color_img, ANGLE)
    croped = rotated[450:1500,225:1250,:]
    # Keep only the green layer (/!\ OpenCV use BGR color, not RGB)
    img = croped[:,:,1]
    
    if first_run == True : 
        counter = 0
        height, width,first_run,black_img, display = initialize(img)
        stacked = black_img.copy()
        ground_clutter = black_img.copy()
        rain_clutter = black_img.copy()
        stacked_raw_detection = black_img.copy()
        stacked_dilated_detection = black_img.copy()
        scope = display.copy()
        scope = cv2.circle(scope,(X_CENTER,Y_CENTER),RADIUS-50,(255,255,255),-1)
        trails = display.copy()
    
    # Clutter -  the ground clutter is defined during the 20 first images
    if ground_clutter_counter < GROUND_CLUTTER_DURATION:
        ground_clutter,ground_clutter_counter = draw_ground_clutter(img, ground_clutter,ground_clutter_counter)
       
    # Rain
    rain_clutter  = draw_rain_clutter(ground_clutter,img,black_img,rain_clutter)   
    
    # Detection 
    raw_detection,dilated_detection,contours,trails,dot_list,nb_dots = detect(ground_clutter,rain_clutter,img,stacked,trails,seconds_timestamp)
    
    if nb_dots<200: #too much dot to handle a realistic detection 
    
        # Try to link dots with previous trajectories
        for trajectory in trajectories :
            display  = color_img.copy()
            
            # Test if the dot can be merged to the trajectory 
            # 1- Define distance between the dot and the trajectory
            for dot in dot_list : 
                a = np.array((dot[1], dot[2], 0))
                b = np.array((trajectory[1], trajectory[2], 0))
                dist = np.linalg.norm(a-b)
                
                if dist<10 :
                    # Angle analysis as to be added there
                    # distance / time ratio as to be added too
                    trajectory[DOT_LENGHT:] = trajectory[:-DOT_LENGHT] # Magic modification of the parent array :)
                    trajectory[:(DOT_LENGHT-1)] = dot[:(DOT_LENGHT-1)] # Magic modification of the parent array :)
                    dot[(DOT_LENGHT-1)] = 1 # write the sum of the last coordinates x,y of the trajectory in dot array to avoid to add it at a same trajectory
        
        # Delete dot already linked from dot_list
        dot_list = dot_list[dot_list[:, (DOT_LENGHT-1)] == 0]
        
        # Add the dot to the trajectories list
        trajectories = np.vstack([trajectories, dot_list])
        
        nb_trajectories = nb_trajectories+nb_dots
        
        # Clean trajectories
        trajectories, nb_trajectories,trajectories_to_record = clean_trajectories(trajectories, seconds_timestamp,trajectories_to_record)
        
    
    trails = np.where(trails>TRAIL_FADE, trails-TRAIL_FADE,0)
    trails = trails.astype(np.uint8)
    # Control
    stacked_raw_detection,stacked_dilated_detection = control(raw_detection,dilated_detection,stacked_raw_detection,stacked_dilated_detection)
    
    # Display
    display = draw_display(ground_clutter,rain_clutter,contours,croped,trails)
    cv2.putText(display,'Nb_dots = '+str(nb_dots), (50,40), cv2.FONT_HERSHEY_DUPLEX, 1, (200,200,200),2,lineType = cv2.LINE_AA)#, cv2.LINE_AA)
    cv2.putText(display,'Nb_trajectories = '+str(len(trajectories_to_record)), (50,70), cv2.FONT_HERSHEY_DUPLEX, 1, (200,200,200),2,lineType = cv2.LINE_AA)#, cv2.LINE_AA)
    
    trajectory_to_record_visual_check  = display.copy()                                                                     
    for trajectory in trajectories_to_record :
        
        for trajectory in trajectories_to_record : 
            color = random_color()   
            for i in range (0,int(TRAJECTORY_MAX_LENGHT/DOT_LENGHT)-1) :
                if trajectory[(1+i)*DOT_LENGHT]>0 :
                    cv2.line(trajectory_to_record_visual_check, (int(trajectory[1+DOT_LENGHT*i]), int(trajectory[2+DOT_LENGHT*i])), (int(trajectory[1+DOT_LENGHT*(i+1)]), int(trajectory[2+DOT_LENGHT*(i+1)])), (color), thickness=2)

    #write_csv_data(CSV_FILE_NAME,trajectories_to_record)
    #trajectories_to_record = np.zeros((1,TRAJECTORY_MAX_LENGHT),dtype = np.float32)
    #print('csv file created')

    cv2.imshow('Recorded',trajectory_to_record_visual_check.astype(np.uint8))
    key = cv2.waitKey(1)
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()

    # Stack (// trail but on the raw data) 
    stacked = np.where(img>stacked,img,stacked)

cv2.destroyAllWindows()




