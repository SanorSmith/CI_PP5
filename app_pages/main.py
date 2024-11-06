import streamlit as st
from app_pages.main import MultiPage

# load pages scripts

app = MultiPage(app_name="Leaf Disease Detector")  # Create an instance of the app

# Add your app pages here using .add_page()


app.run()  # Run the app