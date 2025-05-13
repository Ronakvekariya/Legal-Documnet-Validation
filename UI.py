import gradio as gr
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
import os
# import spacy
import re
# from nameparser import HumanName

# Load YOLO models
IdentifyModel = YOLO("./models/NewValidator(1).pt")
AadharPanmodel = YOLO("./models/best (2).pt")
ResultIdentifyModel = YOLO("./models/results.pt")
ResultOCRModel = YOLO("./models/ResulrOCR.pt")

# Configure Pytesseract


AADHAAR_REGEX = {
    "uid": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "dob": r"\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d{2}\b",
    "gender": r"\b(MALE|Male|FEMALE|Female|Transgender)\b",
    "fullname" : r"\b[A-Z][a-z]+(\s[A-Z][a-z]+)+\b"
}

PAN_REGEX = {
    "pannumber" : r"^[A-Z]{5}[0-9]{4}[A-Z]$",
    "fullname" : r"^[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})+$",
    "fathersname" : r"^[A-Za-z]{2,}(?:\s+[A-Za-z]{2,})+$",
    "dob" : r"\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d{2}\b"
}


# def ner_based_extraction(text):
#     """Fallback using spaCy's named entity recognition"""
#     doc = nlp(text)
    
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             return {"name": ent.text, "confidence": "medium"}
    
#     return {"name": None, "confidence": "low"}

# def is_valid_indian_name(name):
#     """
#     Heuristic validation to decide if a line looks like a valid Indian name.
#     """
#     # Remove extra spaces and unwanted punctuation
#     name = name.strip()
    
#     # Split the line into tokens
#     tokens = name.split()
    
#     # Require at least two tokens (first and last name)
#     if len(tokens) < 2:
#         return False
    
#     # Ensure each token is alphabetic (this removes tokens with numbers or symbols)
#     if not all(token.isalpha() for token in tokens):
#         return False
    
#     # Use nameparser to see if the name has a first and last component
#     parsed = HumanName(name)
#     if not parsed.first or not parsed.last:
#         return False
    
#     # Optional: Check average token length (adjust thresholds as needed)
#     avg_length = sum(len(token) for token in tokens) / len(tokens)
#     if avg_length < 3 or avg_length > 12:
#         return False
    
#     return True

# def filter_valid_names(text):
#     """
#     Given a multiline string, return only the lines that appear to be valid names.
#     """
#     lines = text.splitlines()
#     valid_lines = [line for line in lines if is_valid_indian_name(line)]
#     return "\n".join(valid_lines)


def parse_full_image(image, DocumentType):
    """Custom OCR processing for Legal Document (full image approach)"""
    try:
        # Convert to PIL Image for Tesseract
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_img_np = np.array(pil_img)
        denoised = cv2.bilateralFilter(pil_img_np, d=9, sigmaColor=75, sigmaSpace=75)
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)  # Create a blurred version
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)  # Sharpen
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        _, binary = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        custom_config = r'--oem 3 --psm 13'
        # Custom OCR logic
        # data = pytesseract.image_to_data(pil_img, output_type=Output.DICT)
        text = pytesseract.image_to_string(binary, config=custom_config)

        # Initialize variables for grouping lines
        # n_boxes = len(data['text'])
        # current_line = None
        # line_text = ""
        full_name = None

        # # Iterate over each detected element
        # for i in range(n_boxes):
        #     # Filter out weak confidence detections (optional)
        #     if int(data['conf'][i]) > 60:
        #         line_num = data['line_num'][i]
        #         # Check if we are still on the same line
        #         if current_line is None:
        #             current_line = line_num
                
        #         if line_num == current_line:
        #             line_text += data['text'][i] + " "
        #         else:
        #             # When the line changes, print the current line and start a new one
        #             print(f"Line {current_line}: {line_text.strip()}")
        #             line_text = data['text'][i] + " "

        #             flag = ner_based_extraction(line_text.strip())
        #             if flag["confidence"] == "medium":
        #                 full_name = flag["name"]
        #                 break


        #             current_line = line_num
        
        # if full_name is None:
        #     full_name = ner_based_extraction(text)["name"]
        
        # Extract Aadhaar, DOB, and Gender with string conversion if match is found
        if DocumentType == "Aadhar Card":
            full_name_match = re.search(AADHAAR_REGEX["fullname"], text)
            full_name = full_name_match.group(0) if full_name_match else None

            aadhaar_match = re.search(AADHAAR_REGEX["uid"], text)
            aadhaar_number = aadhaar_match.group(0) if aadhaar_match else None
                        
            dob_match = re.search(AADHAAR_REGEX["dob"], text)
            dob = dob_match.group(0) if dob_match else None
                        
            gender_match = re.search(AADHAAR_REGEX["gender"], text)
            gender = gender_match.group(0) if gender_match else None

            return [full_name, aadhaar_number, dob, gender]
        else:
            full_name_match = re.search(PAN_REGEX["fullname"], text)
            full_name = full_name_match.group(0) if full_name_match else None

            father_name_match = re.search(PAN_REGEX["fathersname"], text)
            father_name = father_name_match.group(0) if father_name_match else None

            pannumber_match = re.search(PAN_REGEX["pannumber"], text)
            pannumber = pannumber_match.group(0) if pannumber_match else None
                        
            dob_match = re.search(PAN_REGEX["dob"], text)
            dob = dob_match.group(0) if dob_match else None
                        

            return [full_name, father_name, dob, pannumber]
    except Exception as e:
        print(f"OCR Error in full image processing: {str(e)}")
        return ["", f"OCR Error: {str(e)}"]


def parse_image(cropped_image , class_name):
    """Custom OCR processing for cropped image sections"""
    try:
            pil_img = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            pil_img_np = np.array(pil_img)
            denoised = cv2.bilateralFilter(pil_img_np, d=9, sigmaColor=75, sigmaSpace=75)
            blurred = cv2.GaussianBlur(denoised, (5, 5), 0)  # Create a blurred version
            sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)  # Sharpen
            gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
            _, binary = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            custom_config = r'--oem 3 --psm 13'
            text = pytesseract.image_to_string(binary, config=custom_config)
            # Custom OCR logic here
        
            return text
    except Exception as e:
        print(f"OCR Error in cropped image processing: {str(e)}")
        return f"OCR Error: {str(e)}"

def predict_document_type(img , model):
    """Predict document type using a pre-trained model"""
    # Load the pre-trained model
    class_name_ = None
    confidence_score = None
    x1_ , y1_ , x2_ , y2_ = None , None , None , None

    results = model(img)
    if len(results) > 0:
            r = results[0]  # Get the first result
            for box in r.boxes:
                # Extract bounding box coordinates and details
                if box is None:
                    return [None, "Error: Invalid image file"]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf.cpu().item()  # Confidence score
                class_id = int(box.cls.cpu().item())  # Class ID
                class_name = r.names[class_id]  # Class name
                print(confidence, class_name)
                if confidence_score is None :
                    confidence_score = confidence
                    class_name_ = class_name
                    x1_ , y1_ , x2_ , y2_ = x1 , y1 , x2 , y2
                else:
                    if confidence > confidence_score:
                        confidence_score = confidence
                        class_name_ = class_name
                        x1_ , y1_ , x2_ , y2_ = x1 , y1 , x2 , y2
    print(class_name_ , confidence_score , x1_ , y1_ , x2_ , y2_)
        
    return [class_name_ , confidence_score , x1_ , y1_ , x2_ , y2_]



def process_documnet(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Error: Invalid image file"
        DocumentType = "UnKnown"
        lst = predict_document_type(img , IdentifyModel)
        if lst is not None:
            print("hello")
            if lst[0] is None:
                return img, "No Legal Document components detected"
            class_name_ , confidence_score , x1_ , y1_ , x2_ , y2_ = lst[0] , lst[1] , lst[2] , lst[3] , lst[4] , lst[5]
            print(class_name_ , confidence_score , x1_ , y1_ , x2_ , y2_)

            if class_name_ == "Unknown Document" or confidence_score < 0.85:
                lst = predict_document_type(img , ResultIdentifyModel)
                if lst is not None:
                    if lst[0] is None:
                        return img, "No Legal Document components detected"
                    class_name_ , confidence_score , x1_ , y1_ , x2_ , y2_ = lst[0] , lst[1] , lst[2] , lst[3] , lst[4] , lst[5]
                    print(class_name_ , confidence_score , x1_ , y1_ , x2_ , y2_)

                    if class_name_ == "Unknown Document":
                        return None, "Unknown Dcoument Type"
                    else:
                        DocumentType = class_name_
                        img = img[y1_:y2_, x1_:x2_]
            else:
                DocumentType = class_name_
                img = img[y1_:y2_ , x1_:x2_]
          

        results = None

        if DocumentType in ["Aadhar Card", "Pan Card"]:
            results = AadharPanmodel(img)
        elif DocumentType in ["10 Result" , "12 Result"]:
            results = ResultOCRModel(img)


        
        print("hello")
        print(results)
        if not results[0].boxes:
            return img, "No Legal Document components detected"
        # Process detections
        annotated_img = results[0].plot()
        extracted_data = []

        if len(results[0].boxes) < 4:
            full_image_result = parse_full_image(img, DocumentType)
            for i, field in enumerate(full_image_result):
                # if i == 0:
                #     field = extract_name_from_noisy_text(field)
                extracted_data.append({
                    "coordinates": None,
                    "text": field,
                    "confidence": None  # Confidence is not available in full image OCR
                })
        else:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if DocumentType == "Pan Card":
                    cropped = img[y1-5:y2+5, x1-5:x2+5]
                else:
                    cropped = img[y1:y2, x1:x2]
                class_id = int(box.cls.cpu().item())  # Class ID
                class_name = results[0].names[class_id]
                print(class_name)
                # OCR Processing for cropped image
                raw_text = parse_image(cropped , class_name )
                print(raw_text)
                if raw_text:
                    extracted_data.append({
                        "coordinates": (x1, y1, x2, y2),
                        "text": raw_text,
                        "confidence": box.conf.item()
                    })
        
        # Format results with confidence handling ("N/A" when not available)
        result_text = "\n\n".join([
            f"Document Type {DocumentType} 🔍 Field {i+1} (Confidence: {d['confidence'] if d['confidence'] is not None else 'N/A'}):\n{d['text']}"
            for i, d in enumerate(extracted_data)
        ]) if extracted_data else "No valid Legal Document data found"
        
        return annotated_img, result_text
        
    except Exception as e:
        error_message = f"Processing Error: {str(e)}"
        print(error_message)
        return None, error_message


# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Legal Document OCR Processor") as app:
    gr.Markdown("# 🆔 Legal Document OCR System")
    gr.Markdown("Upload a scanned Legal Document image for automated verification")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Legal Document Card")
            upload_btn = gr.UploadButton("📁 Upload Image", file_types=["image"])
        with gr.Column():
            output_image = gr.Image(label="Detected Components", interactive=False)
            output_text = gr.Textbox(label="Extracted Information", lines=10)
    
    # Examples
    gr.Examples(
        examples=["./65093343-b94b-42c6-89aa-e38f0a266678_jpeg.rf.73526af7f0955413cd43f1caebf181cb - Copy.jpg", "./ronakaadha.jpg" , "./WhatsApp Image 2025-03-27 at 16.45.23_8d6071c0 - Copy.jpg" , "./3a5dc768-58cd07ba187f99551bc1fa5e_pan_jpg.rf.548972cc014046cad43209d240980541.jpg" , "./3def06d4-58d0d5d17b8d86a57f7f61f2_pan_jpg.rf.aa7611a5b549a3fcb56d14ba46dd04cb.jpg"],
        inputs=input_image,
        outputs=[output_image, output_text],
        fn=process_documnet,
        cache_examples=True
    )
    
    # Event handlers
    upload_btn.upload(
        fn=lambda file: file.name,
        inputs=upload_btn,
        outputs=input_image
    )
    
    input_image.change(
        fn=process_documnet,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    app.launch(
        server_port=7860,
        share=False,
        show_error=True,
        debug=False
    )
