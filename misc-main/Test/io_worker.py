import os
import time
import requests

HOST= os.getenv("HOST","0.0.0.0")
PORT = os.getenv("PORT","8888")

time.sleep(2) # Wait for the inference_server to become ready
 
# If the environment variable HOST is set to '::' (which listens on both IPv4 and IPv6), the URL becomes http://[::1]:{PORT}/ for IPv6. 
# Otherwise ('0.0.0.0'), it defaults to http://127.0.0.1:{PORT}/ for IPv4.
URL = f"http://[::1]:{PORT}/" if (HOST == "::") else f"http://127.0.0.1:{PORT}/" 
# URL = f"http://localhost:{PORT}/"

while True:

    # Retrieve a job from a queue service (such as AWS SQS or others) that provides a reference of the job input stored in cloud storage
    # Download the job input from cloud storage
    print(80 * "*")
    print("Retrieve a job and download its input")
 
    # Call the inference server locally, and get the output and result
    print("the io_worker calls the inference_server: " + URL)     
    time.sleep(1)
    response = requests.get(URL)
    print("The response from the inference_server: " + response.content.decode('utf-8'))

    # Upload the job output to cloud storage
    # Return the job result to the queue - success or failure
    print("Upload the job output and return its result")