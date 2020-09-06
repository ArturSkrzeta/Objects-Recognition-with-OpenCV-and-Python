import cv2

def main():

    video = cv2.VideoCapture('traffic.mp4')
    car_tracker = cv2.CascadeClassifier('cars.xml')

    while True:

        (read_successful, frame) = video.read()

        if read_successful:
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            break

        cars = car_tracker.detectMultiScale(grayscaled_frame)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

        cv2.imshow('Detection', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
