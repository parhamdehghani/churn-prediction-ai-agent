# Use the official Jenkins image as a base
FROM jenkins/jenkins:lts-jdk17

# Switch to the root user to install tools
USER root

# Install prerequisites for Docker and gcloud
RUN apt-get update && apt-get install -y lsb-release curl gpg apt-transport-https ca-certificates

# --- Install Docker CLI ---
RUN curl -fsSLo /usr/share/keyrings/docker-archive-keyring.asc \
  https://download.docker.com/linux/debian/gpg
RUN echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/docker-archive-keyring.asc] \
  https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list

# --- Install Google Cloud SDK ---
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# --- Run final install for all tools ---
RUN apt-get update && apt-get install -y docker-ce-cli python3-pip google-cloud-sdk

# Use pip to install DVC with GCS support
RUN pip install "dvc[gcs]" "dvc-gs" --break-system-packages

# Switch back to the jenkins user
USER jenkins