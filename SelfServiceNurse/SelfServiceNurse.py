from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import os
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
import cv2
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import csv

# Define your screens
class SignInScreen(Screen):
    def verify_credentials(self):
        # Placeholder for authentication logic
        if self.ids.username_input.text == "Username" and self.ids.password_input.text == "Password":
            self.manager.current = 'triage'
        else:
            print("Incorrect credentials")

class TriageScreen(Screen):
    def start_ultrasonic_sensor(self):
        # Placeholder function - integrate with your sensor here
        print("Ultrasonic sensor started")

    def start_heart_rate_monitor(self):
        # Placeholder function - integrate with your monitor here
        print("Heart rate monitor started")

    def start_temperature_gauge(self):
        # Placeholder function - integrate with your gauge here
        print("Temperature gauge started")

class HabitsScreen(Screen):
    def submit_habits(self, bad_habits, medications, diseases):
        # Path to save the file, you might want to adjust this according to your file structure
        file_path = 'habits_info.txt'
        with open(file_path, 'a') as file:
            file.write(f"Bad Habits: {bad_habits}, Medications: {medications}, Diseases: {diseases}\n")
        print("Information saved to file.")
        # Optional: Automatically navigate to the next screen after saving
        self.manager.current = 'symptoms'

class CameraWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.camera = cv2.VideoCapture(0)  # 0 is the default camera
        self.image = Image()
        self.add_widget(self.image)
        self.capture_button = Button(text='Capture')
        self.capture_button.bind(on_press=self.capture)
        self.add_widget(self.capture_button)
        self.details_input = TextInput(hint_text='Enter details about the rash...', size_hint_y=0.2)
        self.add_widget(self.details_input)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # update at 30Hz

    def update(self, dt):
        ret, frame = self.camera.read()
        if ret:
            # Convert it to texture
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def capture(self, *args):
        ret, frame = self.camera.read()
        if ret:
            # Save captured image for further processing
            cv2.imwrite('captured_rash.jpg', frame)
            print("Captured image saved as captured_rash.jpg")
            # Optionally stop the camera or indicate capture is done
            self.capture_button.disabled = True
            # You might want to add logic to pass the image to your CNN model here

class SkinRashesScreen(Screen):
    def __init__(self, **kwargs):
        super(SkinRashesScreen, self).__init__(**kwargs)
        self.add_widget(CameraWidget())



class ReportScreen(Screen):
    pass
kv = """
ScreenManager:
    SignInScreen:
    TriageScreen:
    HabitsScreen:
    SymptomsScreen:
    SkinRashesScreen:
    ReportScreen:

<SignInScreen>:
    name: 'signin'
    # Add your sign-in screen layout here

<TriageScreen>:
    name: 'triage'
    BoxLayout:
        orientation: 'vertical'
        Button:
            text: 'Start Ultrasonic Sensor'
            on_press: root.start_ultrasonic_sensor()
        Button:
            text: 'Start Heart Rate Monitor'
            on_press: root.start_heart_rate_monitor()
        Button:
            text: 'Start Temperature Gauge'
            on_press: root.start_temperature_gauge()
        Button:
            text: 'Next'
            on_press: app.root.current = 'habits'
    # Continue defining other screens similarly...

<HabitsScreen>:
    name: 'habits'
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Enter your bad habits:'
        TextInput:
            id: bad_habits
            multiline: False
        Label:
            text: 'Current medications:'
        TextInput:
            id: current_medications
            multiline: False
        Label:
            text: 'Family diseases:'
        TextInput:
            id: family_diseases
            multiline: False
        Button:
            text: 'Submit'
            on_press: root.submit_habits(bad_habits.text, current_medications.text, family_diseases.text)
        Button:
            text: 'Next'
            on_press: app.root.current = 'symptoms'

<SymptomsScreen>:
    name: 'symptoms'
    ScrollView:
        do_scroll_x: False
        do_scroll_y: True
        BoxLayout:
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height

            Label:
                text: 'Please select your symptoms:'
                size_hint_y: None
                height: '48dp'

            # Example Checkbox Items for Symptoms
            BoxLayout:
                size_hint_y: None
                height: '48dp'
                CheckBox:
                    id: fever
                    group: "symptoms"
                Label:
                    text: "Fever"

            BoxLayout:
                size_hint_y: None
                height: '48dp'
                CheckBox:
                    id: cough
                    group: "symptoms"
                Label:
                    text: "Cough"

            # Add more symptoms here in similar fashion

            TextInput:
                id: details_input
                hint_text: 'Enter any additional details here...'
                size_hint_y: None
                height: '120dp'
                multiline: True

            Button:
                text: 'Submit'
                size_hint_y: None
                height: '48dp'
                on_press: root.submit_symptoms()

<SkinRashesScreen>:
    BoxLayout:
        orientation: 'vertical'
        CameraWidget:
            id: camera_widget
            # Assuming CameraWidget is a custom widget you've created for displaying the camera feed
        TextInput:
            id: details_input
            hint_text: 'Enter details about the rash...'
            size_hint_y: 0.2
            multiline: True
        Button:
            text: 'Capture Image'
            size_hint_y: 0.1
            on_press: root.capture_image()
        Button:
            text: 'Submit'
            size_hint_y: 0.1
            on_press: root.submit_details(details_input.text)

<ReportScreen>:
    # Define layout
"""
class SymptomsScreen(Screen):
    def __init__(self, **kwargs):
        super(SymptomsScreen, self).__init__(**kwargs)
        self.symptoms_list = [
            "Abdominal Pain", "Abdominal Swelling", "Angina/Heart Problem", "Anxiety", "Back Pain", "Being Sick",
            "Blood in Stool", "Bloating", "Breast Discharge", "Breast Lump", "Bulge in the Abdomen", "Bulging",
            "Burning Mouth/Skin", "Chest Pain", "Chills", "Conjuctivitis", "Sweaty and Clammy skin", "Cold Intolerance",
            "Constipation", "Cough", "Discharge Smell", "Depression", "Diarrhea", "Dizziness", "Dry Skin", "Dry mouth",
            "Dry eye", "Ear pain", "Ear Infection", "Vertigo", "Tinnitus", "Easily Bleeds", "Easily Bruises", "Edema",
            "Excessive Hunger", "Excessive Thirst", "Excessive Urination", "Extremity Numbness", "Extremity Weakness",
            "Fatigue", "Faint", "Fever", "Feeling of guilt", "Feeling of Fullness", "Food Allergies", "Scary Thoughts",
            "Hair Loss", "Headaches", "Hearing Loss", "Heartburn", "Heat Intolerance", "Hives", "Indigestion", "Infertility",
            "Itchy Skin", "Issues with Blood Clots", "Joint Pain", "Joint Swelling", "Loss of Appetite", "Loss of Consciousness",
            "Loss of Interest", "Loss of Interest in Sex", "Loss of taste", "Loss of vision", "Lymphedema", "Mole Changes",
            "Muscle Weakness", "Nausea", "Neck Pain", "Night Sweats", "Nose bleed", "Pain while Walking", "Pale Skin",
            "Painful Urination", "Palpitations", "Rash", "Urinary Infections", "Red Palms", "Red or Purple spots",
            "Sadness/Low mood", "Seasonal Allergies", "Seizures", "Severe Stomach Pain", "Shortness of Breath",
            "Sinus Pressure", "Skin Lesion", "Soreness/Redness", "Stomach Pain", "Swollen lymph nodes", "Sneezing",
            "Tremors", "Unexpected Weight Gain", "Unexpected Weight Loss", "Urinary Frequency", "Visual Changes", "Vomiting",
            "Wheezing", "Yellowing of skin and whites of the eyes", "Heavy periods", "Period pain", "Craving for salty foods",
            "Dehydration", "Shivering attacks", "Clubbed fingers", "Memory problems", "Insomnia", "Internal bleeding",
            "Increased sensitivity", "Crumbly nails", "Tonsils/Throat infection", "Toothache", "Swollen gums",
            "Bad Breath/mouth", "Pain in testicles", "Heavy Testicles", "Swelling in testicles", "label"
        ]
        self.layout = GridLayout(cols=2, spacing=10, size_hint_y=None)
        self.layout.bind(minimum_height=self.layout.setter('height'))
        self.add_widgets()
        self.add_widget(self.layout)
        self.details_input = TextInput(size_hint_y=None, height=30, multiline=False, hint_text='Additional details here...')
        self.add_widget(self.details_input)
        self.submit_btn = Button(text='Submit', size_hint_y=None, height=50, on_press=self.submit_symptoms)
        self.add_widget(self.submit_btn)

    def checkbox_click(self, instance, value, symptom):
        if value:
            if symptom not in self.selected_symptoms:
                self.selected_symptoms.append(symptom)
        else:
            if symptom in self.selected_symptoms:
                self.selected_symptoms.remove(symptom)

    def add_widgets(self):
        for symptom in self.symptoms_list:
            cb = CheckBox(size_hint_y=None, height=30)
            cb.bind(active=lambda instance, value, s=symptom: self.checkbox_click(instance, value, s))
            self.layout.add_widget(cb)
            self.layout.add_widget(Label(text=symptom, size_hint_y=None, height=30))

    def submit_symptoms(self, instance):
        # Here you would handle the selected symptoms and additional info.
        # For example, saving to a file, sending to an API, or processing further.
        print(f"Selected Symptoms: {self.selected_symptoms}")
        print(f"Additional Details: {self.details_input.text}")
        # Navigate to another screen or show a confirmation as needed
        # self.manager.current = 'next_screen_name'

class MyApp(App):
    def build(self):
        return Builder.load_string(kv)

# Start the application
if __name__ == '__main__':
    MyApp().run()