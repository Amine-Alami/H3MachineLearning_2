import streamlit as st
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VitsModel, AutoTokenizer
import torch
import yolov5

# Load YOLOv5 model
def load_model():
    return yolov5.load('keremberke/yolov5m-license-plate')

# Load TR-OCR model
def load_ocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

# Load TTS model
def load_tts_model():
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    return model, tokenizer

# Main function for Streamlit app
def main():
    st.title("License Plate Recognition App")

    # Static test image
    test_image_path = "test_image.jpg"
    test_image = Image.open(test_image_path)

    # Upload file
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
    else:
        img = test_image

    st.image(img, caption='Image', use_column_width=True)

    if st.button("Run Inference"):
        # Load models on startup
        model = load_model()
        processor, ocr_model = load_ocr_model()
        tts_model, tokenizer = load_tts_model()

        results = model(img, size=640)
        # results.show()
        predictions = results.pred[0]
        boxes = predictions[:, :4] # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # Crop the image of the license plate
        cropped_image = img.crop(tuple(results.xyxy[0][0, :4].squeeze().tolist()[:4]))
        st.image(cropped_image, caption='Plate detected')

        # Extract text from the image
        pixel_values = processor(cropped_image, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.write("Detected License Plate Text:", generated_text)

        # Convert the text to audio
        inputs = tokenizer(generated_text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
        st.audio(output.numpy(), format="audio/wav", sample_rate=tts_model.config.sampling_rate)

if __name__ == "__main__":
    main()