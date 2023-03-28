import  tensorflow as tf
import  numpy as np
import cv2
import cv2
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf

#
# def prediction_with_model(model,image_path):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image,channels = 3)
#     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#     image = tf.image.resize(image,[128,128])#(60,60,3)
#     image = tf.expand_dims(image,axis=0)#(1,60,60,3)
#
#
#     prediction = model.predict(image)
#     print(prediction)
#
#     prediction = np.argmax(prediction)
#     return prediction


np.set_printoptions(suppress=True)
webcam = cv2.VideoCapture(0)

model = tf.keras.models.load_model("./model")
data_for_model = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while 1:

    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x, y, w, h) in faces:


        roi_color = img[y:y + h, x:x + w]
        cv2.imwrite('faces.jpg', roi_color)
        img1 = cv2.resize(roi_color, (128, 128))

        img1 = cv2.flip(img1, 1)
        normalized_img = (img1.astype(np.float32) / 128.0) - 1
        data_for_model[0] = normalized_img
        prediction = model.predict(data_for_model)
        prediction = np.argmax(prediction)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if prediction ==0:
            print("improper face mask")
            cv2.putText(img, "improper face mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
        elif prediction ==1:
            cv2.putText(img, "correct face mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
        elif prediction == 1:
            cv2.putText(img, "No face mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

    cv2.imshow('img', img)



    if cv2.waitKey('q') & 0xff:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()

