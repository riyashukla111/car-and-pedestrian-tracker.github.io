import cv2

# our image and video
img_file = 'Car image.jpg'
# video = cv2.VideoCapture('Tesla Dashcam Accident.mp4')
# video = cv2.VideoCapture('Tesla Dashcam Tumbleweed.mp4')
# video = cv2.VideoCapture('Tesla Dashcam Highway.mp4')
# video = cv2.VideoCapture('Dashcam Pedestrians .mp4')
video = cv2.VideoCapture('car and pedestrian dashcam.mp4')

# our pre-trained car and pedestrian classifiers
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# create car and pedestrian classifiers
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# run forever until car stops
while True:

    # read the current frame
    (read_successful, frame) = video.read()

# safe coding
    if read_successful:
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # draw rectangle around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw rectangle around the pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # display the image with faces spotted
    cv2.imshow('car and pedestrian tracker', frame)

    # Don't auto close (wait here in th code and listen for a key press)
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key == 81 or key == 113:
        break

# release the video capture
video.release()
"""
# create opencv image
img = cv2.imread(img_file)

# convert to grayscale(needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# draw rectangle around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# display the image with faces spotted
cv2.imshow('car and pedestrian tracker', img)

# Don't auto close (wait here in th code and listen for a key press)
cv2.waitKey()"""

print("code completed")
