import cv2
import numpy as np

# Create a function based on a CV2 Event (Left button click)

# mouse callback function
def get_line_points(event,x,y,flags,param):

    global line, num_clicks, mouse_moving

    # get mouse click on down and track center
    if event == cv2.EVENT_LBUTTONDOWN:
        if num_clicks == 0:
            line[0] = (x, y)

        num_clicks += 1
    
    if event == cv2.EVENT_MOUSEMOVE and num_clicks > 0:
        mouse_moving = True
        if num_clicks < 2:
            line[1] = (x, y)
        #clicked = True
   

        
# Haven't drawn anything yet!
line = [(0,0), (0,0)]
num_clicks = 0
mouse_moving = False

# Create a named window for connections
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# Bind draw_rectangle function to mouse cliks
cv2.setMouseCallback('frame', get_line_points)

#original_frame = cv2.imread("./TEST_IMGs_SESC/20201126_172219-SESC2.jpg")
original_frame = cv2.imread("./AGORA_IMGs/20141223_11.jpeg")


while True:
    frame = original_frame.copy()

    if num_clicks == 1 and mouse_moving:
        frame = cv2.line(frame, line[0], line[1], color=(255, 0, 0), thickness=2)
    
    elif num_clicks == 2:
        frame = cv2.line(frame, line[0], line[1], color=(255, 0, 0), thickness=2)
        original_frame = frame.copy()
        
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
    

    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("Line end-points: " + str(line))

distance_in_pixel = np.sqrt((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2)

print("Pixel distance: " + str(distance_in_pixel))
