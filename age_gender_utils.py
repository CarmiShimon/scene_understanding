import cv2

class AgeGender:
  def __init__(self, global_path):

    faceProto=f'{global_path}/Gender-and-Age-Detection/opencv_face_detector.pbtxt'
    faceModel=f'{global_path}/Gender-and-Age-Detection/opencv_face_detector_uint8.pb'
    ageProto=f'{global_path}/Gender-and-Age-Detection/age_deploy.prototxt'
    ageModel=f'{global_path}/Gender-and-Age-Detection/age_net.caffemodel'
    genderProto=f'{global_path}/Gender-and-Age-Detection/gender_deploy.prototxt'
    genderModel=f'{global_path}/Gender-and-Age-Detection/gender_net.caffemodel'
    self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    self.ageList=[2, 5, 10, 17, 28, 40, 50, 70]
    self.genderList=['Male', 'Female']
    # models
    self.faceNet = cv2.dnn.readNet(faceModel, faceProto)
    self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
    self.genderNet = cv2.dnn.readNet(genderModel, genderProto)  
    self.padding = 20

  def face_detector(self, rgbframe, conf_threshold=0.5):
    net = self.faceNet
    frameOpencvDnn = rgbframe.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1=int(detections[0, 0, i, 3]*frameWidth)
            y1=int(detections[0, 0, i, 4]*frameHeight)
            x2=int(detections[0, 0, i, 5]*frameWidth)
            y2=int(detections[0, 0, i, 6]*frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return faceBoxes

  def age_estimator(self, rgbframe, faceBoxes):
    ages, genders = [], []
    for faceBox in faceBoxes:
      face = rgbframe[max(0,faceBox[1]-self.padding):min(faceBox[3]+self.padding,rgbframe.shape[0]-1),max(0,faceBox[0]-self.padding):
      min(faceBox[2] + self.padding, rgbframe.shape[1]-1)]
      if face.shape[0] == 0:
        continue

      blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), self.MODEL_MEAN_VALUES, swapRB=False)
      self.genderNet.setInput(blob)
      genderPreds=self.genderNet.forward()
      gender = self.genderList[genderPreds[0].argmax()]
      genders.append(gender)

      self.ageNet.setInput(blob)
      agePreds = self.ageNet.forward()
      age = self.ageList[agePreds[0].argmax()]
      ages.append(age)
      cv2.putText(rgbframe, f'{gender}, {str(age)}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    return ages, genders


