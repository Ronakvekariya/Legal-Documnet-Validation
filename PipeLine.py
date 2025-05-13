import cv2
import json
from datetime import datetime
import os
import socket
from ultralytics import YOLO
import pytesseract

class DocumentValidation:
    def __init__(self, Validation_model_path, Extraction_model_path):
        # This is the path for the model which is to indentify whether it is adhar card or not
        self.Validation_model_path = Validation_model_path
        # This is the path for the model which is to create bounding box the adhar card and pan card
        self.Extraction_model_path = Extraction_model_path
        # loading both the model
        self.validationYOLO = YOLO(Validation_model_path)
        self.ExtractionYOLO = YOLO(Extraction_model_path)

    def Validator(self, image_path):
        image = cv2.imread(image_path)
        results = self.validationYOLO(image)

        ReturnList = []
        
        # Check if results contain data
        if len(results) > 0:
            r = results[0]  # Get the first result
            
            for box in r.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert coordinates to integers
                confidence = box.conf.cpu().item()  # Confidence score
                class_id = int(box.cls.cpu().item())  # Class ID
                class_name = r.names[class_id]  # Class name
                
                # Crop the object from the frame
                cropped_object = image[y1:y2, x1:x2]
                
                # Save the cropped image which is to remove unnecessary part of the image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                cropped_image_path = f"cropped_objects/{class_name}_{timestamp}.jpg"
                os.makedirs("cropped_objects", exist_ok=True)
                cv2.imwrite(cropped_image_path, cropped_object)
                
                print(f"Saved cropped object: {cropped_image_path}, Confidence: {confidence}")

                # I have solved here divyang sir's doubt , by using confidence score
                if confidence > 0.96:
                    text = f"{class_name} {confidence:.3f}"
                    ReturnList.append([class_name , cropped_image_path])
                else:
                    text = "Unknown"
                
                # putting reactangle and name 
                cv2.rectangle(image , (x1,y1) , (x2,y2) , (0,255,0) , 2)
                cv2.putText(image , text ,(x1 , y1-10), cv2.FONT_HERSHEY_COMPLEX , 0.5 ,  (0,144,255) , 1 , cv2.LINE_AA )
                cv2.imshow('YOLOv8 Detection', image) 
            
            cv2.waitKey(0)

            # returing the list containing class name and cropped image path
            return ReturnList
        else:
            print("No objects detected")
            return None
        
    def Extractor(self , ImageList):
        ReturnList = []
        #imagelist = [class_name , cropped_image_path] , so we are taking imagelist[1]
        for image_path in ImageList:
            image = cv2.imread(image_path[1])
            # provding that image to second model
            results = self.ExtractionYOLO(image)
            
            # Check if results contain data
            if len(results) > 0:
                r = results[0]  # Get the first result
                
                for box in r.boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert coordinates to integers
                    confidence = box.conf.cpu().item()  # Confidence score
                    class_id = int(box.cls.cpu().item())  # Class ID
                    class_name = r.names[class_id]  # Class name
                    
                    # Crop the object from the frame
                    cropped_object = image[y1:y2, x1:x2]
                    
                    # Save the cropped image
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    cropped_image_path = f"cropped_objects/{class_name}_{timestamp}.jpg"
                    os.makedirs("cropped_objects", exist_ok=True)
                    cv2.imwrite(cropped_image_path, cropped_object)
                    
                    print(f"Saved cropped object: {cropped_image_path}, Confidence: {confidence}")
                    
                    # Annotate and display the frame
                    annotated_frame = results[0].plot()
                    cv2.imshow('YOLOv8 Detection', annotated_frame) 
                    # this will be used by TextExtractor function , the cropped image will be given to OCR 
                    ReturnList.append([class_name , cropped_image_path])
            else:
                print("No objects detected")
        
        return ReturnList
            
    def TextExtractor(self , ImageList):
        # this is takinf cropped image and trying to extract text from it , if image is not so blur
        for image_path in   ImageList:
            image = cv2.imread(image_path[1])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpened = cv2.GaussianBlur(gray, (0, 0), 3)
            sharpened = cv2.addWeighted(gray, 1.5, sharpened, -0.5, 0)
            text = pytesseract.image_to_string(sharpened)
            print(text)
        

if __name__ == "__main__":
    # Set up video capture (0 for USB webcam or Raspberry Pi Camera)
    cap = cv2.VideoCapture(0)

    # To hold the data temporarily
    predictions = []

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Display the current frame
        cv2.imshow('Camera Feed - Press "s" to simulate IoT signal', frame)
        
        # Check for keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # Simulate IoT signal with the "s" key
            print("IoT Signal Simulated with 's' key: Running YOLOv8 for object detection...")

            cv2.imwrite("./test.jpg" , frame)

            # Load the YOLOv8 model
            # intializing the model with model path , "./models/Validation.pt" this path can be different in your laptop , so change it okay
            extractor  = DocumentValidation("./models/Validation.pt" , "./models/Extractor.pt")
            ImageList = extractor.Validator('./test.jpg')
            print(ImageList)
            # Imagelist here will be adhar card and pan card
            ImageList = extractor.Extractor(ImageList)
            # image list here will be cropped image of field in adhar card and pan card such as father name , 12 digit number etc
            extractor.TextExtractor(ImageList)
        # Press 'q' to quit the program
        if key == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

        
    