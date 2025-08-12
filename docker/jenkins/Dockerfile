# Use the official Jenkins LTS image with Java 17 as the base
FROM jenkins/jenkins:lts-jdk17

# Switch to the root user to install system packages
USER root

# Install prerequisites and the Docker CLI
RUN apt-get update && apt-get install -y lsb-release curl gpg
RUN curl -fsSLo /usr/share/keyrings/docker-archive-keyring.asc \
  https://download.docker.com/linux/debian/gpg
RUN echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/docker-archive-keyring.asc] \
  https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list
RUN apt-get update && apt-get install -y docker-ce-cli python3-pip

# Use pip to install DVC with GCS support
RUN pip install "dvc[gcs]" --break-system-packages

# Switch back to the default jenkins user
USER jenkins
