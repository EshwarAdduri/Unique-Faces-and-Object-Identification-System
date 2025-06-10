import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import os

# Initialize the Wasabi client
def get_wasabi_client(access_key, secret_key):
    return boto3.client(
        's3',
        endpoint_url='https://s3.wasabisys.com',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

st.title("Wasabi Bucket Interface")
 
# Sidebar for user input
st.sidebar.header("Credentials")
access_key = st.sidebar.text_input("Access Key ID", type="password")
secret_key = st.sidebar.text_input("Secret Access Key", type="password")
bucket_name = st.sidebar.text_input("Bucket Name", "loadfiledownloads")

if access_key and secret_key and bucket_name:
    client = get_wasabi_client(access_key, secret_key)

    st.header("Upload File to Wasabi")
    uploaded_file = st.file_uploader("Choose a file to upload")
 
    if uploaded_file:
        try:
            client.upload_fileobj(uploaded_file, bucket_name, uploaded_file.name)
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        except NoCredentialsError:
            st.error("Invalid credentials. Please check your access key and secret key.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.header("List Files in Bucket")
    if st.button("List Files"):
        try:
            response = client.list_objects_v2(Bucket=bucket_name)
            if 'Contents' in response:
                for obj in response['Contents']:
                    st.write(f"{obj['Key']} (Size: {obj['Size']} bytes)")
            else:
                st.info("No files found in the bucket.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.header("Download File from Bucket")
    file_to_download = st.text_input("Enter the file name to download")

    if st.button("Download"):
        if file_to_download:
            try:
            # Ensure the local path exists
                local_directory = os.path.dirname(file_to_download)
                if local_directory and not os.path.exists(local_directory):
                    os.makedirs(local_directory)

            # Download the file from Wasabi
                with open(file_to_download, 'wb') as f:
                    client.download_fileobj(bucket_name, file_to_download, f)
                st.success(f"File '{file_to_download}' downloaded successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a file name to download.")

else:
    st.warning("Please enter your Wasabi credentials and bucket name.")
