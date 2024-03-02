# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run cli_demo_sat.py when the container launches
# Note: This command may need to be updated based on the specific entry point for the project.
CMD ["python", "cli_demo_sat.py", "--from_pretrained", "cogvlm-chat", "--version", "chat_old", "--bf16", "--stream_chat"]
