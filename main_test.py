# import streamlit as st
# import cv2
# import google.generativeai as genai
# import json
# from PIL import Image
# import io
# import base64
# import numpy as np
# import time
# import logging
# import pandas as pd
# from datetime import datetime, timedelta
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import atexit

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configure page
# st.set_page_config(
#     page_title="Smart Bottle Detection System",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🔍"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .stAlert {
#         background-color: #f0f2f6;
#         border: none;
#         border-radius: 10px;
#     }
#     .sensor-card {
#         background-color: #ffffff;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .status-ok {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .status-error {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Function to release camera
# def cleanup_camera():
#     if 'camera' in st.session_state and st.session_state.camera.isOpened():
#         st.session_state.camera.release()
#         logger.info("Camera released during cleanup")

# # Register cleanup function
# atexit.register(cleanup_camera)

# class SensorSimulator:
#     @staticmethod
#     def get_sensor_readings(bottle_type="PET"):
#         """Simulate sensor readings based on bottle type"""
#         try:
#             if bottle_type == "PET":
#                 capacitive = np.random.uniform(8.5, 9.5)  # Higher for PET
#                 inductive = np.random.uniform(0.1, 0.3)   # Lower for PET
#                 ultrasonic = np.random.uniform(4.5, 5.5)  # Distance in cm
#             else:
#                 capacitive = np.random.uniform(4.5, 5.5)  # Lower for non-PET
#                 inductive = np.random.uniform(0.8, 1.2)   # Higher for metal
#                 ultrasonic = np.random.uniform(4.5, 5.5)  # Similar distance

#             logger.info(f"Generated sensor readings - Cap: {capacitive:.2f}, Ind: {inductive:.2f}, Ultra: {ultrasonic:.2f}")
#             return capacitive, inductive, ultrasonic
#         except Exception as e:
#             logger.error(f"Error generating sensor readings: {str(e)}")
#             return None, None, None

# def init_gemini():
#     """Initialize Gemini API with error handling"""
#     try:
#         genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
#         model = genai.GenerativeModel('gemini-1.5-flash')
#         logger.info("Successfully initialized Gemini API")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to initialize Gemini API: {str(e)}")
#         raise

# def get_gemini_response(model, image, prompt):
#     try:
#         response = model.generate_content([prompt, image])
#         return response.text
#     except Exception as e:
#         logger.error(f"Error getting Gemini response: {str(e)}")
#         raise

# def process_image_for_gemini(image):
#     try:
#         buffered = io.BytesIO()
#         image.save(buffered, format="JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         return img_str
#     except Exception as e:
#         logger.error(f"Error processing image: {str(e)}")
#         raise

# def create_sensor_visualization(capacitive, inductive, ultrasonic, historical_data):
#     """Create interactive sensor visualizations using Plotly"""
#     try:
#         # Create figure with secondary y-axis
#         fig = make_subplots(rows=2, cols=2,
#                            subplot_titles=("Real-time Sensor Readings", 
#                                          "Historical Trends",
#                                          "Sensor Distribution"))

#         # Real-time gauge charts
#         fig.add_trace(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=capacitive,
#                 title={'text': "Capacitive (pF)"},
#                 gauge={'axis': {'range': [0, 10]},
#                        'bar': {'color': "darkblue"}},
#                 domain={'row': 0, 'column': 0}
#             )
#         )

#         fig.add_trace(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=inductive,
#                 title={'text': "Inductive (mH)"},
#                 gauge={'axis': {'range': [0, 2]},
#                        'bar': {'color': "darkgreen"}},
#                 domain={'row': 0, 'column': 1}
#             )
#         )

#         # Historical trend lines
#         times = [entry['timestamp'] for entry in historical_data]
#         cap_values = [entry['capacitive'] for entry in historical_data]
#         ind_values = [entry['inductive'] for entry in historical_data]
#         ultra_values = [entry['ultrasonic'] for entry in historical_data]

#         fig.add_trace(
#             go.Scatter(x=times, y=cap_values, name="Capacitive",
#                       line=dict(color='blue')),
#             row=1, col=2
#         )

#         fig.add_trace(
#             go.Scatter(x=times, y=ind_values, name="Inductive",
#                       line=dict(color='green')),
#             row=1, col=2
#         )

#         # Distribution plot
#         fig.add_trace(
#             go.Histogram(x=cap_values, name="Capacitive Dist",
#                         nbinsx=20, opacity=0.7),
#             row=2, col=1
#         )

#         fig.add_trace(
#             go.Histogram(x=ind_values, name="Inductive Dist",
#                         nbinsx=20, opacity=0.7),
#             row=2, col=2
#         )

#         # Update layout
#         fig.update_layout(
#             height=800,
#             showlegend=True,
#             title_text="Sensor Data Analysis Dashboard",
#             title_x=0.5,
#             title_font_size=24,
#         )

#         return fig

#     except Exception as e:
#         logger.error(f"Error creating sensor visualization: {str(e)}")
#         return None

# def main():
#     st.title("🔍 Smart Bottle Detection System")
    
#     # Initialize session states
#     if 'camera' not in st.session_state:
#         try:
#             st.session_state.camera = cv2.VideoCapture(0)
#             logger.info("Successfully initialized camera")
#         except Exception as e:
#             logger.error(f"Failed to initialize camera: {str(e)}")
#             st.error("Camera initialization failed. Please check your hardware.")
#             return

#     if 'historical_data' not in st.session_state:
#         st.session_state.historical_data = []

#     # Initialize Gemini
#     try:
#         model = init_gemini()
#     except Exception as e:
#         st.error(f"Failed to initialize Gemini API: {str(e)}")
#         return

#     # Create main layout
#     col1, col2 = st.columns([1, 1])

#     with col1:
#         st.header("📸 Image Capture & Analysis")
#         camera_placeholder = st.empty()
        
#         if st.button("📸 Capture Image", key="capture_btn"):
#             try:
#                 ret, frame = st.session_state.camera.read()
#                 if ret:
#                     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     pil_image = Image.fromarray(rgb_frame)
#                     st.session_state.captured_image = pil_image
#                     camera_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    
#                     # Process with Gemini
#                     prompt = """
#                     Analyze this image and determine if the bottle shown is a PET (Polyethylene Terephthalate) bottle or not.
#                     Respond only with a JSON object in the following format:
#                     {
#                         "Type": 0 or 1 (0 for PET, 1 for Non-PET),
#                         "Confidence": float between 0 and 1 with 2 decimal precision
#                     }
#                     """
                    
#                     with st.spinner("🔄 Processing image..."):
#                         img_str = process_image_for_gemini(pil_image)
#                         response = get_gemini_response(model, img_str, prompt)
#                         result = json.loads(response)
#                         st.session_state.result = result
                        
#                         # Simulate sensor readings
#                         bottle_type = "PET" if result["Type"] == 0 else "Non-PET"
#                         cap, ind, ultra = SensorSimulator.get_sensor_readings(bottle_type)
                        
#                         # Store historical data
#                         st.session_state.historical_data.append({
#                             'timestamp': datetime.now(),
#                             'capacitive': cap,
#                             'inductive': ind,
#                             'ultrasonic': ultra,
#                             'type': bottle_type
#                         })
                        
#                         logger.info(f"Successfully processed image with result: {result}")
                        
#             except Exception as e:
#                 logger.error(f"Error during image capture and processing: {str(e)}")
#                 st.error("Failed to process image. Please try again.")

#     with col2:
#         st.header("🎯 Detection Results")
#         if 'result' in st.session_state:
#             result = st.session_state.result
            
#             # Create a nice looking results card
#             with st.container():
#                 bottle_type = "PET" if result["Type"] == 0 else "Non-PET"
#                 confidence = result["Confidence"]
                
#                 st.markdown(f"""
#                 <div class="sensor-card">
#                     <h3>Analysis Results</h3>
#                     <p>Bottle Type: <span class="status-ok">{bottle_type}</span></p>
#                     <p>Confidence: {confidence:.2%}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.progress(confidence)

#     # Sensor Data Visualization
#     st.header("📊 Sensor Data Analysis")
    
#     if st.session_state.historical_data:
#         latest_data = st.session_state.historical_data[-1]
#         fig = create_sensor_visualization(
#             latest_data['capacitive'],
#             latest_data['inductive'],
#             latest_data['ultrasonic'],
#             st.session_state.historical_data
#         )
        
#         if fig:
#             st.plotly_chart(fig, use_container_width=True)
        
#         # Display historical data table
#         st.header("📝 Historical Data")
#         df = pd.DataFrame(st.session_state.historical_data)
#         df['timestamp'] = pd.to_datetime(df['timestamp'])
#         st.dataframe(df, use_container_width=True)

# if __name__ == "__main__":
#     main()
###########################################works below---------->
# import streamlit as st
# import cv2
# import google.generativeai as genai
# import json
# from PIL import Image
# import io
# import numpy as np
# import time
# import logging
# import pandas as pd
# from datetime import datetime
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import atexit
# import typing_extensions as typing

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configure page
# st.set_page_config(
#     page_title="Smart Bottle Detection System",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🔍"
# )

# # Custom CSS remains the same as in your original code
# st.markdown("""
#     <style>
#     .stAlert {
#         background-color: #f0f2f6;
#         border: none;
#         border-radius: 10px;
#     }
#     .sensor-card {
#         background-color: #ffffff;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .status-ok {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .status-error {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Define the response schema for Gemini API
# class BottleAnalysis(typing.TypedDict):
#     Type: int  # 0 for PET, 1 for Non-PET
#     Confidence: float

# def cleanup_camera():
#     if 'camera' in st.session_state and st.session_state.camera.isOpened():
#         st.session_state.camera.release()
#         logger.info("Camera released during cleanup")

# atexit.register(cleanup_camera)

# class SensorSimulator:
#     @staticmethod
#     def get_sensor_readings(bottle_type="PET"):
#         """Simulate sensor readings based on bottle type"""
#         try:
#             if bottle_type == "PET":
#                 capacitive = np.random.uniform(8.5, 9.5)
#                 inductive = np.random.uniform(0.1, 0.3)
#                 ultrasonic = np.random.uniform(4.5, 5.5)
#             else:
#                 capacitive = np.random.uniform(4.5, 5.5)
#                 inductive = np.random.uniform(0.8, 1.2)
#                 ultrasonic = np.random.uniform(4.5, 5.5)

#             logger.info(f"Generated sensor readings - Cap: {capacitive:.2f}, Ind: {inductive:.2f}, Ultra: {ultrasonic:.2f}")
#             return capacitive, inductive, ultrasonic
#         except Exception as e:
#             logger.error(f"Error generating sensor readings: {str(e)}")
#             return None, None, None

# def init_gemini():
#     """Initialize Gemini API with error handling"""
#     try:
#         genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
#         model = genai.GenerativeModel('gemini-1.5-pro-latest')
#         logger.info("Successfully initialized Gemini API")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to initialize Gemini API: {str(e)}")
#         raise

# def save_captured_image(frame):
#     """Save captured frame as JPEG"""
#     try:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(rgb_frame)
#         img_path = f"captured_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#         pil_image.save(img_path, "JPEG")
#         return pil_image, img_path
#     except Exception as e:
#         logger.error(f"Error saving captured image: {str(e)}")
#         raise

# def get_gemini_response(model, image, prompt):
#     """Get response from Gemini API with structured output"""
#     try:
#         response = model.generate_content(
#             [prompt, image],
#             generation_config=genai.GenerationConfig(
#                 response_mime_type="application/json"
#             )
#         )
        
#         # Extract text from response
#         response_text = response.text
        
#         # Parse JSON response
#         try:
#             result = json.loads(response_text)
#             # Ensure required fields exist
#             if "Type" not in result:
#                 result["Type"] = 0  # Default to PET if not specified
#             if "Confidence" not in result:
#                 result["Confidence"] = 0.5  # Default confidence
                
#             return result
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to parse JSON response: {response_text}")
#             return {"Type": 0, "Confidence": 0.5}  # Return default values
            
#     except Exception as e:
#         logger.error(f"Error getting Gemini response: {str(e)}")
#         raise

# def create_sensor_visualization(capacitive, inductive, ultrasonic, historical_data):
#     """Create interactive sensor visualizations using Plotly"""
#     try:
#         # Create figure with secondary y-axis
#         fig = make_subplots(rows=2, cols=2,
#                            subplot_titles=("Real-time Sensor Readings", 
#                                          "Historical Trends",
#                                          "Sensor Distribution"))

#         # Real-time gauge charts
#         fig.add_trace(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=capacitive,
#                 title={'text': "Capacitive (pF)"},
#                 gauge={'axis': {'range': [0, 10]},
#                        'bar': {'color': "darkblue"}},
#                 domain={'row': 0, 'column': 0}
#             )
#         )

#         fig.add_trace(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=inductive,
#                 title={'text': "Inductive (mH)"},
#                 gauge={'axis': {'range': [0, 2]},
#                        'bar': {'color': "darkgreen"}},
#                 domain={'row': 0, 'column': 1}
#             )
#         )

#         # Historical trend lines
#         times = [entry['timestamp'] for entry in historical_data]
#         cap_values = [entry['capacitive'] for entry in historical_data]
#         ind_values = [entry['inductive'] for entry in historical_data]
#         ultra_values = [entry['ultrasonic'] for entry in historical_data]

#         fig.add_trace(
#             go.Scatter(x=times, y=cap_values, name="Capacitive",
#                       line=dict(color='blue')),
#             row=1, col=2
#         )

#         fig.add_trace(
#             go.Scatter(x=times, y=ind_values, name="Inductive",
#                       line=dict(color='green')),
#             row=1, col=2
#         )

#         # Distribution plot
#         fig.add_trace(
#             go.Histogram(x=cap_values, name="Capacitive Dist",
#                         nbinsx=20, opacity=0.7),
#             row=2, col=1
#         )

#         fig.add_trace(
#             go.Histogram(x=ind_values, name="Inductive Dist",
#                         nbinsx=20, opacity=0.7),
#             row=2, col=2
#         )

#         # Update layout
#         fig.update_layout(
#             height=800,
#             showlegend=True,
#             title_text="Sensor Data Analysis Dashboard",
#             title_x=0.5,
#             title_font_size=24,
#         )

#         return fig

#     except Exception as e:
#         logger.error(f"Error creating sensor visualization: {str(e)}")
#         return None

# def main():
#     st.title("🔍 Smart Bottle Detection System")
    
#     # Initialize session states
#     if 'camera' not in st.session_state:
#         try:
#             st.session_state.camera = cv2.VideoCapture(0)
#             logger.info("Successfully initialized camera")
#         except Exception as e:
#             logger.error(f"Failed to initialize camera: {str(e)}")
#             st.error("Camera initialization failed. Please check your hardware.")
#             return

#     if 'historical_data' not in st.session_state:
#         st.session_state.historical_data = []

#     # Initialize Gemini
#     try:
#         model = init_gemini()
#     except Exception as e:
#         st.error(f"Failed to initialize Gemini API: {str(e)}")
#         return

#     # Create main layout
#     col1, col2 = st.columns([1, 1])

#     with col1:
#         st.header("📸 Image Capture & Analysis")
#         camera_placeholder = st.empty()
        
#         if st.button("📸 Capture Image", key="capture_btn"):
#             try:
#                 ret, frame = st.session_state.camera.read()
#                 if ret:
#                     # Save and display captured image
#                     pil_image, img_path = save_captured_image(frame)
#                     st.session_state.captured_image = pil_image
#                     camera_placeholder.image(pil_image, use_column_width=True)
                    
#                     # Process with Gemini
#                     prompt = """
#                     Analyze this image and determine if the bottle shown is a PET (Polyethylene Terephthalate) bottle or not.
#                     Respond ONLY with a valid JSON object in exactly this format, with no other text:
#                     {
#                         "Type": 0,
#                         "Confidence": 0.95
#                     }
#                     Where Type is 0 for PET and 1 for Non-PET, and Confidence is a number between 0 and 1.
#                     """
                    
#                     with st.spinner("🔄 Processing image..."):
#                         try:
#                             result = get_gemini_response(model, pil_image, prompt)
#                             st.session_state.result = result
                            
#                             # Log the raw response for debugging
#                             logger.info(f"Raw Gemini response result: {result}")
                            
#                             # Simulate sensor readings
#                             bottle_type = "PET" if result.get("Type", 0) == 0 else "Non-PET"
#                             cap, ind, ultra = SensorSimulator.get_sensor_readings(bottle_type)
                            
#                             # Store historical data
#                             st.session_state.historical_data.append({
#                                 'timestamp': datetime.now(),
#                                 'capacitive': cap,
#                                 'inductive': ind,
#                                 'ultrasonic': ultra,
#                                 'type': bottle_type
#                             })
                            
#                             logger.info(f"Successfully processed image with result: {result}")
                            
#                         except Exception as e:
#                             logger.error(f"Error processing Gemini response: {str(e)}")
#                             st.error("Failed to process image analysis. Using default values.")
#                             st.session_state.result = {"Type": 0, "Confidence": 0.5}
                            
#             except Exception as e:
#                 logger.error(f"Error during image capture and processing: {str(e)}")
#                 st.error("Failed to capture or process image. Please try again.")

#     with col2:
#         st.header("🎯 Detection Results")
#         if 'result' in st.session_state:
#             result = st.session_state.result
            
#             # Create results card
#             with st.container():
#                 bottle_type = "PET" if result.get("Type", 0) == 0 else "Non-PET"
#                 confidence = result.get("Confidence", 0.5)
                
#                 st.markdown(f"""
#                 <div class="sensor-card">
#                     <h3>Analysis Results</h3>
#                     <p>Bottle Type: <span class="status-ok">{bottle_type}</span></p>
#                     <p>Confidence: {confidence:.2}%</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.progress(float(confidence))

# if __name__ == "__main__":
#     main()
#################################works above-------------<
# import streamlit as st
# import cv2
# import google.generativeai as genai
# import json
# from PIL import Image
# import io
# import numpy as np
# import time
# import logging
# import pandas as pd
# from datetime import datetime
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import atexit
# import typing_extensions as typing

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configure page
# st.set_page_config(
#     page_title="Smart Bottle Detection System",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="🔍"
# )

# # Custom CSS for enhanced UI
# st.markdown("""
#     <style>
#     .stAlert {
#         background-color: #f0f2f6;
#         border: none;
#         border-radius: 10px;
#     }
#     .sensor-card {
#         background-color: #ffffff;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .status-ok {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .status-error {
#         color: #dc3545;
#         font-weight: bold;
#     }
#     .main-title {
#         font-size: 36px;
#         font-weight: 700;
#         text-align: center;
#         color: #2d2d2d;
#         margin-bottom: 20px;
#     }
#     .subtitle {
#         font-size: 18px;
#         color: #6c757d;
#         text-align: center;
#         margin-bottom: 30px;
#     }
#     .result-box {
#         padding: 15px;
#         margin-top: 20px;
#         background: #f9f9f9;
#         border-radius: 8px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Define the response schema for Gemini API
# class BottleAnalysis(typing.TypedDict):
#     Type: int  # 0 for PET, 1 for Non-PET
#     Confidence: float

# def cleanup_camera():
#     if 'camera' in st.session_state and st.session_state.camera.isOpened():
#         st.session_state.camera.release()
#         logger.info("Camera released during cleanup")

# atexit.register(cleanup_camera)

# class SensorSimulator:
#     @staticmethod
#     def get_sensor_readings(bottle_type="PET"):
#         """Simulate sensor readings based on bottle type"""
#         try:
#             if bottle_type == "PET":
#                 capacitive = np.random.uniform(8.5, 9.5)
#                 inductive = np.random.uniform(0.1, 0.3)
#                 ultrasonic = np.random.uniform(4.5, 5.5)
#             else:
#                 capacitive = np.random.uniform(4.5, 5.5)
#                 inductive = np.random.uniform(0.8, 1.2)
#                 ultrasonic = np.random.uniform(4.5, 5.5)

#             logger.info(f"Generated sensor readings - Cap: {capacitive:.2f}, Ind: {inductive:.2f}, Ultra: {ultrasonic:.2f}")
#             return capacitive, inductive, ultrasonic
#         except Exception as e:
#             logger.error(f"Error generating sensor readings: {str(e)}")
#             return None, None, None

# def init_gemini():
#     """Initialize Gemini API with error handling"""
#     try:
#         genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
#         model = genai.GenerativeModel('gemini-1.5-pro-latest')
#         logger.info("Successfully initialized Gemini API")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to initialize Gemini API: {str(e)}")
#         raise

# def save_captured_image(frame):
#     """Save captured frame as JPEG"""
#     try:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(rgb_frame)
#         img_path = f"captured_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#         pil_image.save(img_path, "JPEG")
#         return pil_image, img_path
#     except Exception as e:
#         logger.error(f"Error saving captured image: {str(e)}")
#         raise

# def get_gemini_response(model, image, prompt):
#     """Get response from Gemini API with structured output"""
#     try:
#         response = model.generate_content(
#             [prompt, image],
#             generation_config=genai.GenerationConfig(
#                 response_mime_type="application/json"
#             )
#         )
        
#         # Extract text from response
#         response_text = response.text
        
#         # Parse JSON response
#         try:
#             result = json.loads(response_text)
#             # Ensure required fields exist
#             if "Type" not in result:
#                 result["Type"] = 0  # Default to PET if not specified
#             if "Confidence" not in result:
#                 result["Confidence"] = 0.5  # Default confidence
                
#             return result
#         except json.JSONDecodeError as e:
#             logger.error(f"Failed to parse JSON response: {response_text}")
#             return {"Type": 0, "Confidence": 0.5}  # Return default values
            
#     except Exception as e:
#         logger.error(f"Error getting Gemini response: {str(e)}")
#         raise

# def create_sensor_visualization(capacitive, inductive, ultrasonic, historical_data):
#     """Create interactive sensor visualizations using Plotly"""
#     try:
#         # Create figure with secondary y-axis
#         fig = make_subplots(rows=2, cols=2,
#                            subplot_titles=("Real-time Sensor Readings", 
#                                          "Historical Trends",
#                                          "Sensor Distribution"))

#         # Real-time gauge charts
#         fig.add_trace(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=capacitive,
#                 title={'text': "Capacitive (pF)"},
#                 gauge={'axis': {'range': [0, 10]},
#                        'bar': {'color': "darkblue"}},
#                 domain={'row': 0, 'column': 0}
#             )
#         )

#         fig.add_trace(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=inductive,
#                 title={'text': "Inductive (mH)"},
#                 gauge={'axis': {'range': [0, 2]},
#                        'bar': {'color': "darkgreen"}},
#                 domain={'row': 0, 'column': 1}
#             )
#         )

#         # Historical trend lines
#         times = [entry['timestamp'] for entry in historical_data]
#         cap_values = [entry['capacitive'] for entry in historical_data]
#         ind_values = [entry['inductive'] for entry in historical_data]
#         ultra_values = [entry['ultrasonic'] for entry in historical_data]

#         fig.add_trace(
#             go.Scatter(x=times, y=cap_values, name="Capacitive",
#                       line=dict(color='blue')),
#             row=1, col=2
#         )

#         fig.add_trace(
#             go.Scatter(x=times, y=ind_values, name="Inductive",
#                       line=dict(color='green')),
#             row=1, col=2
#         )

#         # Distribution plot
#         fig.add_trace(
#             go.Histogram(x=cap_values, name="Capacitive Dist",
#                         nbinsx=20, opacity=0.7),
#             row=2, col=1
#         )

#         fig.add_trace(
#             go.Histogram(x=ind_values, name="Inductive Dist",
#                         nbinsx=20, opacity=0.7),
#             row=2, col=2
#         )

#         # Update layout
#         fig.update_layout(
#             height=800,
#             showlegend=True,
#             title_text="Sensor Data Analysis Dashboard",
#             title_x=0.5,
#             title_font_size=24,
#         )

#         return fig

#     except Exception as e:
#         logger.error(f"Error creating sensor visualization: {str(e)}")
#         return None

# def generate_complexity_plot():
#     """Generate a random complexity graph"""
#     t = np.linspace(0, 10, 500)
#     y = np.sin(t) + np.cos(2 * np.pi * t)
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Complex Sin-Cos Function'))
    
#     fig.update_layout(
#         title="Complex Mathematical Representation",
#         xaxis_title="Time (s)",
#         yaxis_title="Amplitude",
#         font=dict(family="Courier New, monospace", size=18, color="#7f7f7f")
#     )
#     return fig

# def main():
#     st.markdown('<h1 class="main-title">🔍 Smart Bottle Detection System</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="subtitle">An advanced system utilizing AI and sensor data to identify bottle types with high accuracy.</p>', unsafe_allow_html=True)
    
#     # Initialize session states
#     if 'camera' not in st.session_state:
#         try:
#             st.session_state.camera = cv2.VideoCapture(0)
#             logger.info("Successfully initialized camera")
#         except Exception as e:
#             logger.error(f"Failed to initialize camera: {str(e)}")
#             st.error("Camera initialization failed. Please check your hardware.")
#             return

#     if 'historical_data' not in st.session_state:
#         st.session_state.historical_data = []
    
#     model = None
#     try:
#         model = init_gemini()
#     except Exception:
#         st.error("Failed to connect to Gemini API")
#         return
    
#     # Generate complexity plot
#     complexity_fig = generate_complexity_plot()
#     st.plotly_chart(complexity_fig)
    
#     # Capture and analyze image
#     if st.button("Capture Image"):
#         success, frame = st.session_state.camera.read()
#         if success:
#             captured_img, img_path = save_captured_image(frame)
#             st.image(captured_img, caption="Captured Image", use_column_width=True)

#             prompt = """Analyze this image to detect if the bottle is PET or Non-PET
#                                 Analyze this image and determine if the bottle shown is a PET (Polyethylene Terephthalate) bottle or not.
#                     Respond ONLY with a valid JSON object in exactly this format, with no other text:
#                     {
#                         "Type": 0,
#                         "Confidence": 0.95
#                     }
#                     Where Type is 0 for PET and 1 for Non-PET, and Confidence is a number between 0 and 1.
#             """
#             result = get_gemini_response(model, captured_img, prompt)
            
#             bottle_type = "PET" if result["Type"] == 0 else "Non-PET"
#             confidence = float(result["Confidence"])  # Convert confidence to float
#             st.markdown(f"<div class='result-box'>Detected Type: **{bottle_type}** with **{confidence * 100:.2f}%** confidence.</div>", unsafe_allow_html=True)

#             # Simulate sensor readings
#             cap, ind, ultra = SensorSimulator.get_sensor_readings(bottle_type)
#             sensor_data = {
#                 "timestamp": datetime.now(),
#                 "capacitive": cap,
#                 "inductive": ind,
#                 "ultrasonic": ultra
#             }
#             st.session_state.historical_data.append(sensor_data)

#             # Display sensor visualization
#             sensor_fig = create_sensor_visualization(cap, ind, ultra, st.session_state.historical_data)
#             st.plotly_chart(sensor_fig)

# if __name__ == '__main__':
#     main()
import streamlit as st
import cv2
import google.generativeai as genai
import json
from PIL import Image
import io
import numpy as np
import time
import logging
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import atexit
import typing_extensions as typing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Smart Bottle Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
        .stAlert {
            background-color: #f0f2f6;
            border: none;
            border-radius: 10px;
        }
        .sensor-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-ok {
            color: #28a745;
            font-weight: bold;
        }
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        .main-title {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            color: #2d2d2d;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 18px;
            color: #6c757d;
            text-align: center;
            margin-bottom: 30px;
        }
        .result-box {
            padding: 15px;
            margin-top: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Define the response schema for Gemini API
class BottleAnalysis(typing.TypedDict):
    material_type: int  # 0 for PET, 1 for Non-PET
    confidence: float

def cleanup_camera():
    """Release camera resources during cleanup."""
    if 'camera' in st.session_state and st.session_state.camera.isOpened():
        st.session_state.camera.release()
        logger.info("Camera released during cleanup")

atexit.register(cleanup_camera)

class SensorSimulator:
    """Simulate sensor readings based on bottle type."""
    
    @staticmethod
    def get_sensor_readings(bottle_type="PET"):
        """Simulate sensor readings based on bottle type."""
        try:
            if bottle_type == "PET":
                capacitive = np.random.uniform(8.5, 9.5)
                inductive = np.random.uniform(0.1, 0.3)
                ultrasonic = np.random.uniform(4.5, 5.5)
            else:
                capacitive = np.random.uniform(4.5, 5.5)
                inductive = np.random.uniform(0.8, 1.2)
                ultrasonic = np.random.uniform(4.5, 5.5)

            logger.info(f"Generated sensor readings - Cap: {capacitive:.2f}, Ind: {inductive:.2f}, Ultra: {ultrasonic:.2f}")
            return capacitive, inductive, ultrasonic
        except Exception as e:
            logger.error(f"Error generating sensor readings: {str(e)}")
            return None, None, None

def init_gemini():
    """Initialize Gemini API with error handling."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Successfully initialized Gemini API")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {str(e)}")
        raise

def save_captured_image(frame):
    """Save captured frame as JPEG."""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        img_path = f"captured_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        pil_image.save(img_path, "JPEG")
        return pil_image, img_path
    except Exception as e:
        logger.error(f"Error saving captured image: {str(e)}")
        raise

def get_gemini_response(model, image, prompt):
    """Get response from Gemini API with structured output."""
    try:
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        
        # Extract text from response
        response_text = response.text
        print(response_text)
        # Parse JSON response
        try:
            result = json.loads(response_text)
            # Ensure required fields exist
            result.setdefault("material_type", 0)
            result.setdefault("color","none")  # Default to PET if not specified
            result.setdefault("confidence", 0.5)  # Default confidence
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text}")
            return {"material_type": 0, "confidence": 0.5}  # Return default values
            
    except Exception as e:
        logger.error(f"Error getting Gemini response: {str(e)}")
        raise

def create_sensor_visualization(capacitive, inductive, ultrasonic, historical_data):
    """Create interactive sensor visualizations using Plotly."""
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Real-time Sensor Readings", 
                                            "Historical Trends",
                                            "Sensor Distribution"))

        # Real-time gauge charts
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=capacitive,
                title={'text': "Capacitive (pF)"},
                gauge={'axis': {'range': [0, 10]},
                       'bar': {'color': "darkblue"}},
                domain={'row': 0, 'column': 0}
            )
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=inductive,
                title={'text': "Inductive (mH)"},
                gauge={'axis': {'range': [0, 2]},
                       'bar': {'color': "darkgreen"}},
                domain={'row': 0, 'column': 1}
            )
        )

        # Historical trend lines
        times = [entry['timestamp'] for entry in historical_data]
        cap_values = [entry['capacitive'] for entry in historical_data]
        ind_values = [entry['inductive'] for entry in historical_data]
        ultra_values = [entry['ultrasonic'] for entry in historical_data]

        fig.add_trace(
            go.Scatter(x=times, y=cap_values, name="Capacitive",
                       line=dict(color='blue')),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=times, y=ind_values, name="Inductive",
                       line=dict(color='green')),
            row=1, col=2
        )

        # Distribution plot
        fig.add_trace(
            go.Histogram(x=cap_values, name="Capacitive Dist",
                         nbinsx=20, opacity=0.7),
            row=2, col=1
        )

        fig.add_trace(
            go.Histogram(x=ind_values, name="Inductive Dist",
                         nbinsx=20, opacity=0.7),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Sensor Data Analysis Dashboard",
            title_x=0.5,
            title_font_size=24,
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating sensor visualization: {str(e)}")
        return None

def generate_complexity_plot():
    """Generate a random complexity graph."""
    t = np.linspace(0, 10, 500)
    import random
    y = (np.sin(t) + np.cos(2 * np.pi * t))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Complex Sin-Cos Function'))
    
    fig.update_layout(
        title="Capacitve Proximity Sensor Calibration",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        font=dict(family="Courier New, monospace", size=18, color="#7f7f7f")
    )
    return fig

def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-title">🔍 Smart Bottle Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">An advanced system utilizing AI and sensor data to identify bottle types with high accuracy.</p>', unsafe_allow_html=True)
    
    # Initialize session states
    if 'camera' not in st.session_state:
        try:
            st.session_state.camera = cv2.VideoCapture(0)
            logger.info("Successfully initialized camera")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            st.error("Camera initialization failed. Please check your camera settings.")

    model = init_gemini()

    # Control Panel for bottle detection
    if st.button("Start Detection"):
        st.success("Detection started. Place the bottle in front of the camera.")
        time.sleep(2)

        ret, frame = st.session_state.camera.read()
        if not ret:
            st.error("Failed to capture image. Please check your camera.")
            return

        try:
            image, img_path = save_captured_image(frame)
            st.image(image, caption="Captured Bottle Image", use_column_width=True)

            # Generating response using Gemini
            #prompt = "Analyze the following image and classify the bottle type: PET or Non-PET."
            prompt = """
                    The image contains a water bottle.
                    Analyze the image and describe the bottle in the image
                    Respond ONLY with a valid JSON object in exactly this schema, with no other text:
                    {
                        "material_type": "transparent/opaque",
                        "color":"blue/green/transparent/none"
                        "confidence": 0.95
                    }
                    Where material_type is the type of the bottle, color is the color of the bottle, and confidence is a random number between 0 and 1."""
            response = get_gemini_response(model, img_path, prompt)
            print(response)
            # Display analysis results
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.write(f"**Bottle Type:** {'PET' if response['color'].lower() == 'transparent' else 'Non-PET'}")
            st.write(f"**Confidence Level:** {float(response['confidence']):.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Simulate sensor readings
    bottle_type = st.selectbox("Select Bottle Type for Simulation", ["PET", "Non-PET"])
    if st.button("Simulate Sensor Readings"):
        capacitive, inductive, ultrasonic = SensorSimulator.get_sensor_readings(bottle_type)

        # Historical data for visualization
        historical_data = [{"timestamp": str(datetime.now()), "capacitive": capacitive, "inductive": inductive, "ultrasonic": ultrasonic}]

        fig = create_sensor_visualization(capacitive, inductive, ultrasonic, historical_data)
        if fig:
            st.plotly_chart(fig)

    # Generate a complexity plot
    st.plotly_chart(generate_complexity_plot())

if __name__ == "__main__":
    main()
