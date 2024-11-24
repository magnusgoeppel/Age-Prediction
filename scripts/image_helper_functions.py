import cv2
import cv2.data
import os
import threading

cropped_image_count = 0
count_lock = threading.Lock()


def display_image(img, description):
    cv2.imshow(description, img)
    cv2.waitKey()  # waits for key-tap to close the window
    cv2.destroyAllWindows()  # closes the window


def get_gray_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_by_height(img, target_height):
    if img is None:
        return None

    (actual_height, actual_width) = img.shape[:2]

    aspect_ratio = actual_width / actual_height

    new_height = target_height
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(img, (new_width, new_height))


def get_image_from_path(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print('Error: Image not found or unable to load.')
        return None

    return img


def get_framed_cropped_face_image(grey_image):
    # ------------- Model for face detection---------#
    face_detector_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # -------------detecting the faces--------------#
    faces = face_detector_cascade.detectMultiScale(grey_image, 1.3, 5)
    # If no faces our detected
    # if not faces:
    #    print('No face detected')
    #   #skip picture
    # --------- Bounding Face ---------#
    for (x, y, w, h) in faces:
        framed_image = cv2.rectangle(grey_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        display_image(framed_image, 'framed face')

        cropped_image = grey_image[y:y + h, x:x + w]
        return cropped_image


def crop_to_face(img):
    face_detector_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detector_cascade.detectMultiScale(img, 1.3, 5)

    # no faces detected
    if len(faces) == 0:
        return None
    else:
        for (x, y, w, h) in faces:
            # cropping the img
            return img[y:y + h, x:x + w]


def process_folder(folder):
    global cropped_image_count

    # creating new folder
    new_folder = folder + "_prepared"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # cropping every image
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            file_path = os.path.join(folder, filename)
            actual_image = get_image_from_path(file_path)
            grey_image = get_gray_image(actual_image)
            cropped_image = crop_to_face(grey_image)
            resized_image = resize_by_height(cropped_image, 100)
            if resized_image is not None:
                new_file_path = os.path.join(new_folder, filename)
                cv2.imwrite(new_file_path, resized_image)

                with count_lock:
                    cropped_image_count += 1

                if cropped_image_count % 10 == 0:
                    print(f"Processed {cropped_image_count}")

    print("Thread finished for folder: " + folder)


def crop_all_images_multi_threaded():
    folders = ['data/part1', 'data/part2', 'data/part3']

    # list to store all the threads
    threads = []

    # create and start a new thread for each folder
    for folder in folders:
        thread = threading.Thread(target=process_folder, args=(folder,))
        threads.append(thread)
        thread.start()

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    print(f"Total images cropped: {cropped_image_count}")


# THIS DEMONSTRATES EXAMPLE USAGE - BUT IS NOT NEEDED
def show_image_framed():
    test_image = get_image_from_path('scripts/test.jpg')
    grey_image = get_gray_image(test_image)
    resized_grey_image = resize_by_height(grey_image, 400)
    display_image(resized_grey_image, 'grey and resized image')

    framed_image = get_framed_cropped_face_image(resized_grey_image)
    display_image(framed_image, 'framed and cropped image')
