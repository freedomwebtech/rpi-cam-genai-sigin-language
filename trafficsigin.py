import cv2
import base64
import os
import time
import threading
from picamera2 import Picamera2
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Set up Google API Key
os.environ["GOOGLE_API_KEY"] = ""

# ✅ Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ✅ Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

# ✅ Global variable to store the latest frame
latest_frame = None
frame_lock = threading.Lock()  # Lock to handle thread safety

# ✅ Function to continuously capture images for display
def capture_latest_frame():
    global latest_frame
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with frame_lock:
            latest_frame = frame_bgr.copy()

        cv2.imshow("Live Video Feed - Press 'c' to capture, 'q' to quit", frame_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("📸 'c' key pressed - Capturing and analyzing image...")
            response_content = analyze_vehicle_frame()
            save_vehicle_data(response_content)
            print(f"✅ Gemini Response Saved:\n{response_content}\n")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.close()

# ✅ Function to analyze vehicle-related information
def analyze_vehicle_frame():
    global latest_frame

    with frame_lock:
        if latest_frame is None:
            return "No image available yet. Please wait a moment."
        frame_bgr = latest_frame.copy()

    # Convert to base64
    _, img_buffer = cv2.imencode('.jpg', frame_bgr)
    image_data = base64.b64encode(img_buffer).decode('utf-8')

    # AI prompt for vehicle analysis
    prompt = """

    You are a traffic surveillance AI. Analyze the given image and **only** detect and describe visible **traffic signs**.

    🔍 For each traffic sign you see:
    - Identify the **type of sign** (e.g., stop, speed limit, yield, no entry, pedestrian crossing).
    - Explain the **meaning or instruction** of the sign.
    - Include any **visible text or numbers** (e.g., speed limit value).
    - Mention its **approximate location** in the image (e.g., top-left, center-right).

    🚫 Do not mention any vehicles, people, or number plates.

    Provide a clean, bullet-point format response.

    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    )

    response = model.invoke([message])
    return response.content

# ✅ Function to save vehicle data to a file
def save_vehicle_data(data):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n[{timestamp}]\n{data}\n{'='*50}\n"
    with open("vehicle_data.txt", "a") as file:
        file.write(entry)

# ✅ Simple chatbot (optional)
def ai_chatbot():
    print("\n💬 Type 'describe' to analyze latest frame manually or 'exit' to quit.\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break
        print("📸 Analyzing latest frame...")
        response = analyze_vehicle_frame()
        print(f"Gemini Response:\n{response}\n")

# ✅ Main runner
if __name__ == "__main__":
    print("🚗 Live Video Feed Started - Press 'c' to capture vehicle image, 'q' to quit.")
    threading.Thread(target=capture_latest_frame, daemon=True).start()
    # Optionally enable chatbot: ai_chatbot()
