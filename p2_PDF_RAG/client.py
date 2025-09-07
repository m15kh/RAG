import requests
import streamlit as st



st.write("Upload a file to FastAPI")
file = st.file_uploader("Choose a file", type=["pdf"])

if st.button("Submit File"):
    if file is not None:
        files = {"file": (file.name, file, file.type)}
        response = requests.post("http://localhost:8000/upload", files=files)
        st.write(response.text)
    else:
        st.write("No file uploaded.")

# Add a section for user prompt and question
st.write("Generate text from your prompt")
prompt = st.text_input("Enter your prompt")

if st.button("Submit Prompt"):
    if prompt:
        payload = {"prompt": prompt, "temperature": 0.7}
        response = requests.post("http://localhost:8000/generate/text", json=payload)
        if response.status_code == 200:
            st.write("Generated Text:")
            st.write(response.json().get("content", "No content returned"))
        else:
            st.write(f"Error: {response.status_code} - {response.text}")
    else:
        st.write("Please enter a prompt.")